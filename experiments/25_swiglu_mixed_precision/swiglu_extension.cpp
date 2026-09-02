#include <torch/extension.h>

at::Tensor swiglu_cuda(const at::Tensor& packed);
void swiglu_cuda_out(const at::Tensor& packed, at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu", &swiglu_cuda, "Mixed-precision fused SwiGLU (CUDA)");
  m.def("swiglu_out", &swiglu_cuda_out, "Mixed-precision fused SwiGLU into output (CUDA)");
}
