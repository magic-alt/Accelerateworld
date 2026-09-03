#include <torch/extension.h>

at::Tensor flash_attention_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal,
    int64_t query_start);

void flash_attention_cuda_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor output,
    bool causal,
    int64_t query_start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("flash_attention", &flash_attention_cuda,
             "Score-matrix-free FlashAttention-style CUDA forward");
  module.def("flash_attention_out", &flash_attention_cuda_out,
             "Score-matrix-free FlashAttention-style CUDA forward (out)");
}
