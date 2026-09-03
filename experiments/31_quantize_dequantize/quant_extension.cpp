#include <torch/extension.h>

at::Tensor quantize_int8_cuda(const at::Tensor& input, const at::Tensor& scales, const at::Tensor& zero_points, int64_t granularity, int64_t group_size, bool asymmetric);
at::Tensor dequantize_int8_cuda(const at::Tensor& q, const at::Tensor& scales, const at::Tensor& zero_points, int64_t granularity, int64_t group_size, int64_t output_dtype);
at::Tensor quantize_int4_cuda(const at::Tensor& input, const at::Tensor& scales, int64_t granularity, int64_t group_size);
at::Tensor dequantize_int4_cuda(const at::Tensor& packed, const at::Tensor& scales, int64_t rows, int64_t cols, int64_t granularity, int64_t group_size, int64_t output_dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("quantize_int8", &quantize_int8_cuda, "INT8 quantize CUDA");
  module.def("dequantize_int8", &dequantize_int8_cuda, "INT8 dequantize CUDA");
  module.def("quantize_int4", &quantize_int4_cuda, "packed signed INT4 quantize CUDA");
  module.def("dequantize_int4", &dequantize_int4_cuda, "packed signed INT4 dequantize CUDA");
}
