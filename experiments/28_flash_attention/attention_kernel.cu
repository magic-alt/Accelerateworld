#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kThreads / kWarpSize;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float BlockReduceSum(float value, float* scratch) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  value = WarpReduceSum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  float block_value = 0.0f;
  if (warp == 0) {
    block_value = lane < kMaxWarps ? scratch[lane] : 0.0f;
    block_value = WarpReduceSum(block_value);
    if (lane == 0) {
      scratch[0] = block_value;
    }
  }
  __syncthreads();
  return scratch[0];
}

template <typename T>
__device__ __forceinline__ float LoadValue(const T* ptr);

template <>
__device__ __forceinline__ float LoadValue<__half>(const __half* ptr) {
  return __half2float(*ptr);
}

template <>
__device__ __forceinline__ float LoadValue<__nv_bfloat16>(const __nv_bfloat16* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(*ptr);
#else
  return 0.0f;
#endif
}

template <typename T>
__device__ __forceinline__ void StoreValue(T* ptr, float value);

template <>
__device__ __forceinline__ void StoreValue<__half>(__half* ptr, float value) {
  *ptr = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void StoreValue<__nv_bfloat16>(__nv_bfloat16* ptr, float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  *ptr = __float2bfloat16_rn(value);
#endif
}

template <typename T>
__global__ void FlashAttentionKernel(
    const T* q,
    const T* k,
    const T* v,
    T* output,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    bool causal,
    int64_t query_start,
    float scale) {
  const int64_t row = blockIdx.x;
  const int64_t rows = batch * q_heads * query_length;
  if (row >= rows) {
    return;
  }

  const int64_t query_index = row % query_length;
  const int64_t tmp = row / query_length;
  const int64_t q_head = tmp % q_heads;
  const int64_t batch_index = tmp / q_heads;
  const int64_t group_size = q_heads / kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t absolute_query = query_start + query_index;

  const int64_t q_base = ((batch_index * q_heads + q_head) * query_length + query_index) * head_dim;
  const int tid = threadIdx.x;
  const bool active_dim = tid < head_dim;
  const float q_component = active_dim ? LoadValue(q + q_base + tid) : 0.0f;
  float output_acc = 0.0f;

  __shared__ float reduction[kMaxWarps];
  __shared__ float running_m;
  __shared__ float running_l;
  __shared__ float alpha;
  __shared__ float beta;

  if (tid == 0) {
    running_m = -CUDART_INF_F;
    running_l = 0.0f;
  }
  __syncthreads();

  for (int64_t key_index = 0; key_index < key_length; ++key_index) {
    const bool valid = !causal || key_index <= absolute_query;
    if (!valid) {
      continue;
    }

    const int64_t kv_base =
        ((batch_index * kv_heads + kv_head) * key_length + key_index) * head_dim;
    const float partial =
        active_dim ? q_component * LoadValue(k + kv_base + tid) : 0.0f;
    const float score = BlockReduceSum(partial, reduction) * scale;

    if (tid == 0) {
      const float new_m = fmaxf(running_m, score);
      alpha = running_l == 0.0f ? 0.0f : expf(running_m - new_m);
      beta = expf(score - new_m);
      running_l = running_l * alpha + beta;
      running_m = new_m;
    }
    __syncthreads();

    if (active_dim) {
      const float value = LoadValue(v + kv_base + tid);
      output_acc = output_acc * alpha + beta * value;
    }
    __syncthreads();
  }

  if (active_dim) {
    StoreValue(output + q_base + tid, output_acc / running_l);
  }
}

void ValidateContract(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& output,
    bool causal,
    int64_t query_start) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && output.is_cuda(),
              "Q/K/V/output must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "expected Q [B,QH,Q,D], K/V [B,KVH,K,D]");
  TORCH_CHECK(k.sizes() == v.sizes(), "K and V shapes must match");
  TORCH_CHECK(output.sizes() == q.sizes(), "output shape must match Q");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
              "Q/K/V batch and head_dim must match");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128,
              "v0 FlashAttention CUDA supports head_dim 64 or 128");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "FlashAttention supports float16 or bfloat16");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                  output.scalar_type() == q.scalar_type(),
              "Q/K/V/output dtype must match");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && output.is_contiguous(),
              "Q/K/V/output must be contiguous");
  TORCH_CHECK(query_start >= 0, "query_start must be non-negative");
  if (causal) {
    TORCH_CHECK(query_start + q.size(2) <= k.size(2),
                "causal query positions must fit key_length");
  }
}

template <typename T>
void LaunchTyped(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor output,
    bool causal,
    int64_t query_start) {
  const int64_t rows = q.size(0) * q.size(1) * q.size(2);
  const float scale = 1.0f / std::sqrt(static_cast<float>(q.size(3)));
  FlashAttentionKernel<<<static_cast<unsigned int>(rows), kThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(q.data_ptr()),
      reinterpret_cast<const T*>(k.data_ptr()),
      reinterpret_cast<const T*>(v.data_ptr()),
      reinterpret_cast<T*>(output.data_ptr()),
      q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3),
      causal, query_start, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void Launch(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor output,
    bool causal,
    int64_t query_start) {
  ValidateContract(q, k, v, output, causal, query_start);
  c10::cuda::CUDAGuard guard(q.device());
  if (q.scalar_type() == at::kHalf) {
    LaunchTyped<__half>(q, k, v, output, causal, query_start);
    return;
  }

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, q.get_device()));
  TORCH_CHECK(properties.major >= 8,
              "BF16 FlashAttention requires compute capability >= 8.0; current device is ",
              properties.major, ".", properties.minor);
  LaunchTyped<__nv_bfloat16>(q, k, v, output, causal, query_start);
}

}  // namespace

void flash_attention_cuda_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor output,
    bool causal,
    int64_t query_start) {
  Launch(q, k, v, output, causal, query_start);
}

at::Tensor flash_attention_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    int64_t query_start) {
  auto output = at::empty_like(q);
  Launch(q, k, v, output, causal, query_start);
  return output;
}
