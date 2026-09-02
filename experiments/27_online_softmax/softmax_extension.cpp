#include <torch/extension.h>

#include <cstdint>

at::Tensor softmax_two_pass_cuda(const at::Tensor& scores, bool causal, int64_t query_start);
void softmax_two_pass_cuda_out(
    const at::Tensor& scores,
    at::Tensor output,
    bool causal,
    int64_t query_start);
at::Tensor softmax_online_cuda(const at::Tensor& scores, bool causal, int64_t query_start);
void softmax_online_cuda_out(
    const at::Tensor& scores,
    at::Tensor output,
    bool causal,
    int64_t query_start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_two_pass", &softmax_two_pass_cuda,
        "Stable two-pass row softmax (CUDA)");
  m.def("softmax_two_pass_out", &softmax_two_pass_cuda_out,
        "Stable two-pass row softmax into preallocated output (CUDA)");
  m.def("softmax_online", &softmax_online_cuda,
        "Online max/sum row softmax (CUDA)");
  m.def("softmax_online_out", &softmax_online_cuda_out,
        "Online max/sum row softmax into preallocated output (CUDA)");
}
