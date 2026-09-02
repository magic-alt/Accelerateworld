#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float SiluMulFloat(float gate, float up) {
  const float sigmoid = 1.0f / (1.0f + expf(-gate));
  return gate * sigmoid * up;
}

__global__ void SwiGLUHalf2Kernel(const __half* packed,
                                  __half* output,
                                  int64_t rows,
                                  int64_t intermediate) {
  const int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = intermediate / 2;
  const int64_t total_pairs = rows * pairs_per_row;
  if (pair_index >= total_pairs) {
    return;
  }

  const int64_t row = pair_index / pairs_per_row;
  const int64_t pair_col = pair_index - row * pairs_per_row;
  const int64_t col = pair_col * 2;
  const int64_t packed_row = row * (2 * intermediate);
  const int64_t output_row = row * intermediate;

  const __half2 gate2 = *reinterpret_cast<const __half2*>(packed + packed_row + col);
  const __half2 up2 = *reinterpret_cast<const __half2*>(packed + packed_row + intermediate + col);
  const float2 gate = __half22float2(gate2);
  const float2 up = __half22float2(up2);
  const __half2 result = __floats2half2_rn(
      SiluMulFloat(gate.x, up.x), SiluMulFloat(gate.y, up.y));
  *reinterpret_cast<__half2*>(output + output_row + col) = result;
}

__global__ void SwiGLUHalfScalarKernel(const __half* packed,
                                       __half* output,
                                       int64_t rows,
                                       int64_t intermediate) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * intermediate;
  if (index >= total) {
    return;
  }
  const int64_t row = index / intermediate;
  const int64_t col = index - row * intermediate;
  const int64_t packed_row = row * (2 * intermediate);
  const float gate = __half2float(packed[packed_row + col]);
  const float up = __half2float(packed[packed_row + intermediate + col]);
  output[index] = __float2half_rn(SiluMulFloat(gate, up));
}

__global__ void SwiGLUBFloat162Kernel(const __nv_bfloat16* packed,
                                      __nv_bfloat16* output,
                                      int64_t rows,
                                      int64_t intermediate) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = intermediate / 2;
  const int64_t total_pairs = rows * pairs_per_row;
  if (pair_index >= total_pairs) {
    return;
  }

  const int64_t row = pair_index / pairs_per_row;
  const int64_t pair_col = pair_index - row * pairs_per_row;
  const int64_t col = pair_col * 2;
  const int64_t packed_row = row * (2 * intermediate);
  const int64_t output_row = row * intermediate;

  const __nv_bfloat162 gate2 =
      *reinterpret_cast<const __nv_bfloat162*>(packed + packed_row + col);
  const __nv_bfloat162 up2 =
      *reinterpret_cast<const __nv_bfloat162*>(packed + packed_row + intermediate + col);
  const float2 gate = __bfloat1622float2(gate2);
  const float2 up = __bfloat1622float2(up2);
  const __nv_bfloat162 result = __floats2bfloat162_rn(
      SiluMulFloat(gate.x, up.x), SiluMulFloat(gate.y, up.y));
  *reinterpret_cast<__nv_bfloat162*>(output + output_row + col) = result;
#endif
}

__global__ void SwiGLUBFloat16ScalarKernel(const __nv_bfloat16* packed,
                                           __nv_bfloat16* output,
                                           int64_t rows,
                                           int64_t intermediate) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * intermediate;
  if (index >= total) {
    return;
  }
  const int64_t row = index / intermediate;
  const int64_t col = index - row * intermediate;
  const int64_t packed_row = row * (2 * intermediate);
  const float gate = __bfloat162float(packed[packed_row + col]);
  const float up = __bfloat162float(packed[packed_row + intermediate + col]);
  output[index] = __float2bfloat16_rn(SiluMulFloat(gate, up));
#endif
}

void Validate(const at::Tensor& packed, const at::Tensor& output) {
  TORCH_CHECK(packed.is_cuda() && output.is_cuda(), "packed and output must be CUDA tensors");
  TORCH_CHECK(packed.dim() == 2 && output.dim() == 2, "packed and output must be rank-2");
  TORCH_CHECK(packed.size(1) % 2 == 0, "packed last dimension must be 2 * intermediate");
  const int64_t rows = packed.size(0);
  const int64_t intermediate = packed.size(1) / 2;
  TORCH_CHECK(output.size(0) == rows && output.size(1) == intermediate,
              "output must have shape [rows, intermediate]");
  TORCH_CHECK(packed.scalar_type() == at::kHalf || packed.scalar_type() == at::kBFloat16,
              "packed must be float16 or bfloat16");
  TORCH_CHECK(output.scalar_type() == packed.scalar_type(), "output dtype must match packed dtype");
  TORCH_CHECK(packed.is_contiguous() && output.is_contiguous(),
              "packed and output must be contiguous");
  TORCH_CHECK(packed.device() == output.device(), "packed and output must be on the same CUDA device");
}

}  // namespace

void swiglu_cuda_out(const at::Tensor& packed, at::Tensor output) {
  Validate(packed, output);
  c10::cuda::CUDAGuard device_guard(packed.device());

  const int64_t rows = packed.size(0);
  const int64_t intermediate = packed.size(1) / 2;
  const int64_t total = rows * intermediate;
  if (total == 0) {
    return;
  }

  constexpr int kThreads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (packed.scalar_type() == at::kHalf) {
    const auto* input = reinterpret_cast<const __half*>(packed.data_ptr<at::Half>());
    auto* out = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    if (intermediate % 2 == 0) {
      const int64_t pairs = total / 2;
      const int blocks = static_cast<int>((pairs + kThreads - 1) / kThreads);
      SwiGLUHalf2Kernel<<<blocks, kThreads, 0, stream>>>(input, out, rows, intermediate);
    } else {
      const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
      SwiGLUHalfScalarKernel<<<blocks, kThreads, 0, stream>>>(input, out, rows, intermediate);
    }
  } else {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, packed.get_device()));
    TORCH_CHECK(properties.major >= 8,
                "BF16 SwiGLU requires compute capability >= 8.0; current device is ",
                properties.major, ".", properties.minor);

    const auto* input =
        reinterpret_cast<const __nv_bfloat16*>(packed.data_ptr<at::BFloat16>());
    auto* out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    if (intermediate % 2 == 0) {
      const int64_t pairs = total / 2;
      const int blocks = static_cast<int>((pairs + kThreads - 1) / kThreads);
      SwiGLUBFloat162Kernel<<<blocks, kThreads, 0, stream>>>(input, out, rows, intermediate);
    } else {
      const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
      SwiGLUBFloat16ScalarKernel<<<blocks, kThreads, 0, stream>>>(input, out, rows, intermediate);
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor swiglu_cuda(const at::Tensor& packed) {
  TORCH_CHECK(packed.dim() == 2, "packed must be rank-2");
  TORCH_CHECK(packed.size(1) % 2 == 0, "packed last dimension must be 2 * intermediate");
  auto output = at::empty({packed.size(0), packed.size(1) / 2}, packed.options());
  swiglu_cuda_out(packed, output);
  return output;
}
