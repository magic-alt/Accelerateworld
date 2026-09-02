#include <torch/extension.h>

#include <tuple>

std::tuple<at::Tensor, at::Tensor> rope_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& positions,
    int64_t rotary_dim,
    int64_t layout);

void rope_cuda_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& positions,
    at::Tensor q_out,
    at::Tensor k_out,
    int64_t rotary_dim,
    int64_t layout);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope", &rope_cuda, "Mixed-precision Q/K RoPE (CUDA)");
  m.def("rope_out", &rope_cuda_out, "Mixed-precision Q/K RoPE into outputs (CUDA)");
}
