#include <torch/extension.h>
#include <torch/library.h>

at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& up);

TORCH_LIBRARY(accelerateworld, m) {
  m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
}

TORCH_LIBRARY_IMPL(accelerateworld, CUDA, m) {
  m.impl("silu_mul", &silu_mul_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
