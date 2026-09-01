#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

__global__ void SiluMulKernel(const float* gate, const float* up, float* output, int64_t n) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    const float x = gate[index];
    const float silu = x / (1.0f + expf(-x));
    output[index] = silu * up[index];
  }
}

}  // namespace

at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "gate and up must be CUDA tensors");
  TORCH_CHECK(gate.scalar_type() == at::kFloat && up.scalar_type() == at::kFloat,
              "v0 supports float32 tensors only");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "gate and up must be contiguous");

  c10::cuda::CUDAGuard device_guard(gate.device());
  auto output = at::empty_like(gate);
  const int64_t n = gate.numel();
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  SiluMulKernel<<<blocks, kThreads, 0, stream>>>(gate.data_ptr<float>(), up.data_ptr<float>(),
                                                 output.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
