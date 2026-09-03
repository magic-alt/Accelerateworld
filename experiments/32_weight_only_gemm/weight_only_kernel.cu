#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kTile = 16;

__device__ __forceinline__ int64_t ParamIndex(
    int n, int k, int granularity, int group_size, int groups_per_row) {
  if (granularity == 0) return 0;
  if (granularity == 1) return n;
  return static_cast<int64_t>(n) * groups_per_row + k / group_size;
}

__device__ __forceinline__ int8_t DecodeInt4(uint8_t byte, int k) {
  const int nibble = (k & 1) == 0 ? (byte & 0xF) : ((byte >> 4) & 0xF);
  return static_cast<int8_t>(nibble >= 8 ? nibble - 16 : nibble);
}

template <typename T>
__device__ __forceinline__ float LoadFloat(const T* ptr);

template <>
__device__ __forceinline__ float LoadFloat<__half>(const __half* ptr) {
  return __half2float(*ptr);
}

template <>
__device__ __forceinline__ float LoadFloat<__nv_bfloat16>(const __nv_bfloat16* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(*ptr);
#else
  return 0.0f;
#endif
}

template <typename T>
__device__ __forceinline__ void StoreFloat(T* ptr, float value);

template <>
__device__ __forceinline__ void StoreFloat<__half>(__half* ptr, float value) {
  *ptr = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void StoreFloat<__nv_bfloat16>(__nv_bfloat16* ptr, float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  *ptr = __float2bfloat16_rn(value);
#endif
}

template <typename scalar_t, bool kPackedInt4>
__global__ void WeightOnlyGemmKernel(
    const scalar_t* activation,
    const void* qweight,
    const float* scales,
    const int32_t* zero_points,
    scalar_t* output,
    int M, int N, int K,
    int granularity, int group_size, int groups_per_row) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float w_tile[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m = blockIdx.y * kTile + ty;
  const int n = blockIdx.x * kTile + tx;
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kTile) {
    const int a_k = k0 + tx;
    a_tile[ty][tx] = (m < M && a_k < K)
        ? LoadFloat(activation + static_cast<int64_t>(m) * K + a_k)
        : 0.0f;

    const int w_k = k0 + ty;
    float w = 0.0f;
    if (n < N && w_k < K) {
      int q;
      if constexpr (kPackedInt4) {
        const int packed_k = (K + 1) / 2;
        const auto* packed = static_cast<const uint8_t*>(qweight);
        const uint8_t byte = packed[static_cast<int64_t>(n) * packed_k + w_k / 2];
        q = static_cast<int>(DecodeInt4(byte, w_k));
      } else {
        const auto* q8 = static_cast<const int8_t*>(qweight);
        q = static_cast<int>(q8[static_cast<int64_t>(n) * K + w_k]);
      }
      const int64_t pidx = ParamIndex(n, w_k, granularity, group_size, groups_per_row);
      w = (static_cast<float>(q) - static_cast<float>(zero_points[pidx])) * scales[pidx];
    }
    w_tile[ty][tx] = w;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk) {
      acc = fmaf(a_tile[ty][kk], w_tile[kk][tx], acc);
    }
    __syncthreads();
  }

  if (m < M && n < N) {
    StoreFloat(output + static_cast<int64_t>(m) * N + n, acc);
  }
}

int64_t ExpectedQParams(int N, int K, int granularity, int group_size) {
  if (granularity == 0) return 1;
  if (granularity == 1) return N;
  TORCH_CHECK(granularity == 2, "granularity must be 0, 1 or 2");
  TORCH_CHECK(group_size > 0, "group_size must be positive in group mode");
  return static_cast<int64_t>(N) * ((K + group_size - 1) / group_size);
}

void Validate(
    const at::Tensor& activation,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t format_id,
    int64_t granularity,
    int64_t group_size) {
  TORCH_CHECK(activation.is_cuda() && qweight.is_cuda() && scales.is_cuda() && zero_points.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(activation.dim() == 2, "activation must be [M,K]");
  TORCH_CHECK(activation.scalar_type() == at::kHalf || activation.scalar_type() == at::kBFloat16,
              "activation must be float16 or bfloat16");
  TORCH_CHECK(format_id >= 0 && format_id <= 2, "format_id must be 0=int8_sym, 1=int8_asym or 2=int4_sym");
  TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");
  TORCH_CHECK(zero_points.scalar_type() == at::kInt, "zero_points must be int32");
  TORCH_CHECK(activation.is_contiguous() && qweight.is_contiguous() &&
              scales.is_contiguous() && zero_points.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(activation.device() == qweight.device() &&
              activation.device() == scales.device() &&
              activation.device() == zero_points.device(),
              "all tensors must be on the same device");

  const int K = static_cast<int>(activation.size(1));
  const int N = static_cast<int>(qweight.size(0));
  if (format_id == 2) {
    TORCH_CHECK(qweight.scalar_type() == at::kByte, "INT4 weight must be uint8 packed storage");
    TORCH_CHECK(qweight.dim() == 2 && qweight.size(1) == (K + 1) / 2,
                "INT4 weight must be [N, ceil(K/2)]");
  } else {
    TORCH_CHECK(qweight.scalar_type() == at::kChar, "INT8 weight must be int8");
    TORCH_CHECK(qweight.dim() == 2 && qweight.size(1) == K, "INT8 weight must be [N,K]");
  }
  const int64_t expected = ExpectedQParams(N, K, static_cast<int>(granularity), static_cast<int>(group_size));
  TORCH_CHECK(scales.numel() == expected && zero_points.numel() == expected,
              "qparam count mismatch");
}

}  // namespace

at::Tensor weight_only_gemm_cuda(
    const at::Tensor& activation,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t format_id,
    int64_t granularity,
    int64_t group_size) {
  Validate(activation, qweight, scales, zero_points, format_id, granularity, group_size);
  c10::cuda::CUDAGuard guard(activation.device());

  const int M = static_cast<int>(activation.size(0));
  const int K = static_cast<int>(activation.size(1));
  const int N = static_cast<int>(qweight.size(0));
  auto output = at::empty({M, N}, activation.options());
  if (M == 0 || N == 0 || K == 0) return output;

  if (activation.scalar_type() == at::kBFloat16) {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, activation.get_device()));
    TORCH_CHECK(properties.major >= 8,
                "BF16 weight-only GEMM requires compute capability >= 8.0; current device is ",
                properties.major, ".", properties.minor);
  }

  const dim3 block(kTile, kTile);
  const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  const int groups_per_row = (K + std::max<int64_t>(group_size, 1) - 1) /
                             std::max<int64_t>(group_size, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (activation.scalar_type() == at::kHalf) {
    const auto* a = reinterpret_cast<const __half*>(activation.data_ptr<at::Half>());
    auto* c = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    if (format_id == 2) {
      WeightOnlyGemmKernel<__half, true><<<grid, block, 0, stream>>>(
          a, qweight.data_ptr(), scales.data_ptr<float>(), zero_points.data_ptr<int32_t>(),
          c, M, N, K, granularity, group_size, groups_per_row);
    } else {
      WeightOnlyGemmKernel<__half, false><<<grid, block, 0, stream>>>(
          a, qweight.data_ptr(), scales.data_ptr<float>(), zero_points.data_ptr<int32_t>(),
          c, M, N, K, granularity, group_size, groups_per_row);
    }
  } else {
    const auto* a = reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr<at::BFloat16>());
    auto* c = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    if (format_id == 2) {
      WeightOnlyGemmKernel<__nv_bfloat16, true><<<grid, block, 0, stream>>>(
          a, qweight.data_ptr(), scales.data_ptr<float>(), zero_points.data_ptr<int32_t>(),
          c, M, N, K, granularity, group_size, groups_per_row);
    } else {
      WeightOnlyGemmKernel<__nv_bfloat16, false><<<grid, block, 0, stream>>>(
          a, qweight.data_ptr(), scales.data_ptr<float>(), zero_points.data_ptr<int32_t>(),
          c, M, N, K, granularity, group_size, groups_per_row);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
