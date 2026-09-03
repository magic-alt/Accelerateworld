#include <torch/extension.h>

at::Tensor weight_only_gemm_cuda(
    const at::Tensor& activation,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t format_id,
    int64_t granularity,
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("weight_only_gemm", &weight_only_gemm_cuda,
             "INT8/packed-INT4 weight-only GEMM CUDA");
}
