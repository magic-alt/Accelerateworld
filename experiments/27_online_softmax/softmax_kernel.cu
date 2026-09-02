#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kThreads / kWarpSize;
constexpr unsigned kFullMask = 0xffffffffu;

struct OnlineState {
  float m;
  float l;
};

__device__ __forceinline__ OnlineState CombineState(OnlineState a, OnlineState b) {
  if (a.l == 0.0f) {
    return b;
  }
  if (b.l == 0.0f) {
    return a;
  }
  const float m = fmaxf(a.m, b.m);
  return OnlineState{m, a.l * expf(a.m - m) + b.l * expf(b.m - m)};
}

__device__ __forceinline__ float WarpReduceMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullMask, value, offset));
  }
  return value;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ OnlineState WarpReduceOnline(OnlineState state) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const OnlineState other{
        __shfl_down_sync(kFullMask, state.m, offset),
        __shfl_down_sync(kFullMask, state.l, offset)};
    state = CombineState(state, other);
  }
  return state;
}

__device__ __forceinline__ float BlockReduceMax(float value, float* scratch) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  value = WarpReduceMax(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  float block_value = -CUDART_INF_F;
  if (warp == 0) {
    block_value = lane < kMaxWarps ? scratch[lane] : -CUDART_INF_F;
    block_value = WarpReduceMax(block_value);
    if (lane == 0) {
      scratch[0] = block_value;
    }
  }
  __syncthreads();
  return scratch[0];
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

__device__ __forceinline__ OnlineState BlockReduceOnline(
    OnlineState state,
    float* scratch_m,
    float* scratch_l) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  state = WarpReduceOnline(state);
  if (lane == 0) {
    scratch_m[warp] = state.m;
    scratch_l[warp] = state.l;
  }
  __syncthreads();

  if (warp == 0) {
    OnlineState block_state = lane < kMaxWarps
        ? OnlineState{scratch_m[lane], scratch_l[lane]}
        : OnlineState{-CUDART_INF_F, 0.0f};
    block_state = WarpReduceOnline(block_state);
    if (lane == 0) {
      scratch_m[0] = block_state.m;
      scratch_l[0] = block_state.l;
    }
  }
  __syncthreads();
  return OnlineState{scratch_m[0], scratch_l[0]};
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

__device__ __forceinline__ bool IsValidColumn(
    int64_t col,
    int64_t query_index,
    int64_t query_start,
    bool causal) {
  return !causal || col <= query_start + query_index;
}

template <typename T>
__global__ void TwoPassSoftmaxKernel(
    const T* input,
    T* output,
    int64_t rows,
    int64_t query_length,
    int64_t key_length,
    bool causal,
    int64_t query_start) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t query_index = row % query_length;
  const int64_t row_base = row * key_length;
  __shared__ float scratch[kMaxWarps];

  float local_max = -CUDART_INF_F;
  for (int64_t col = threadIdx.x; col < key_length; col += blockDim.x) {
    if (IsValidColumn(col, query_index, query_start, causal)) {
      local_max = fmaxf(local_max, LoadValue(input + row_base + col));
    }
  }
  const float row_max = BlockReduceMax(local_max, scratch);

  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < key_length; col += blockDim.x) {
    if (IsValidColumn(col, query_index, query_start, causal)) {
      local_sum += expf(LoadValue(input + row_base + col) - row_max);
    }
  }
  const float denominator = BlockReduceSum(local_sum, scratch);

  for (int64_t col = threadIdx.x; col < key_length; col += blockDim.x) {
    const bool valid = IsValidColumn(col, query_index, query_start, causal);
    const float probability = valid
        ? expf(LoadValue(input + row_base + col) - row_max) / denominator
        : 0.0f;
    StoreValue(output + row_base + col, probability);
  }
}

template <typename T>
__global__ void OnlineSoftmaxKernel(
    const T* input,
    T* output,
    int64_t rows,
    int64_t query_length,
    int64_t key_length,
    bool causal,
    int64_t query_start) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t query_index = row % query_length;
  const int64_t row_base = row * key_length;
  __shared__ float scratch_m[kMaxWarps];
  __shared__ float scratch_l[kMaxWarps];

  OnlineState local{-CUDART_INF_F, 0.0f};
  for (int64_t col = threadIdx.x; col < key_length; col += blockDim.x) {
    if (IsValidColumn(col, query_index, query_start, causal)) {
      const float x = LoadValue(input + row_base + col);
      local = CombineState(local, OnlineState{x, 1.0f});
    }
  }
  const OnlineState row_state = BlockReduceOnline(local, scratch_m, scratch_l);

  for (int64_t col = threadIdx.x; col < key_length; col += blockDim.x) {
    const bool valid = IsValidColumn(col, query_index, query_start, causal);
    const float probability = valid
        ? expf(LoadValue(input + row_base + col) - row_state.m) / row_state.l
        : 0.0f;
    StoreValue(output + row_base + col, probability);
  }
}

void ValidateContract(
    const at::Tensor& scores,
    const at::Tensor& output,
    bool causal,
    int64_t query_start) {
  TORCH_CHECK(scores.is_cuda() && output.is_cuda(), "scores/output must be CUDA tensors");
  TORCH_CHECK(scores.dim() == 4 && output.sizes() == scores.sizes(),
              "scores/output must be matching [batch, heads, query_length, key_length]");
  TORCH_CHECK(scores.scalar_type() == at::kHalf || scores.scalar_type() == at::kBFloat16,
              "online softmax supports float16 or bfloat16 scores");
  TORCH_CHECK(output.scalar_type() == scores.scalar_type(), "output dtype must match scores");
  TORCH_CHECK(scores.is_contiguous() && output.is_contiguous(), "scores/output must be contiguous");
  TORCH_CHECK(scores.numel() > 0, "scores must be non-empty");
  TORCH_CHECK(query_start >= 0, "query_start must be non-negative");
  if (causal) {
    TORCH_CHECK(query_start + scores.size(2) <= scores.size(3),
                "causal query positions must fit the key dimension");
  }
}

template <bool Online>
void Launch(
    const at::Tensor& scores,
    at::Tensor output,
    bool causal,
    int64_t query_start) {
  ValidateContract(scores, output, causal, query_start);
  c10::cuda::CUDAGuard guard(scores.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = scores.size(0) * scores.size(1) * scores.size(2);
  const int64_t query_length = scores.size(2);
  const int64_t key_length = scores.size(3);
  const dim3 grid(static_cast<unsigned int>(rows));

  if (scores.scalar_type() == at::kHalf) {
    const auto* input = reinterpret_cast<const __half*>(scores.data_ptr<at::Half>());
    auto* out = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    if constexpr (Online) {
      OnlineSoftmaxKernel<<<grid, kThreads, 0, stream>>>(
          input, out, rows, query_length, key_length, causal, query_start);
    } else {
      TwoPassSoftmaxKernel<<<grid, kThreads, 0, stream>>>(
          input, out, rows, query_length, key_length, causal, query_start);
    }
  } else {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, scores.get_device()));
    TORCH_CHECK(properties.major >= 8,
                "BF16 online softmax requires compute capability >= 8.0; current device is ",
                properties.major, ".", properties.minor);
    const auto* input = reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr<at::BFloat16>());
    auto* out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    if constexpr (Online) {
      OnlineSoftmaxKernel<<<grid, kThreads, 0, stream>>>(
          input, out, rows, query_length, key_length, causal, query_start);
    } else {
      TwoPassSoftmaxKernel<<<grid, kThreads, 0, stream>>>(
          input, out, rows, query_length, key_length, causal, query_start);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void softmax_two_pass_cuda_out(
    const at::Tensor& scores,
    at::Tensor output,
    bool causal,
    int64_t query_start) {
  Launch<false>(scores, output, causal, query_start);
}

at::Tensor softmax_two_pass_cuda(const at::Tensor& scores, bool causal, int64_t query_start) {
  auto output = at::empty_like(scores);
  softmax_two_pass_cuda_out(scores, output, causal, query_start);
  return output;
}

void softmax_online_cuda_out(
    const at::Tensor& scores,
    at::Tensor output,
    bool causal,
    int64_t query_start) {
  Launch<true>(scores, output, causal, query_start);
}

at::Tensor softmax_online_cuda(const at::Tensor& scores, bool causal, int64_t query_start) {
  auto output = at::empty_like(scores);
  softmax_online_cuda_out(scores, output, causal, query_start);
  return output;
}
