#include <torch/extension.h>

#include <vector>

void paged_kv_update_cuda(
    at::Tensor page_k,
    at::Tensor page_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    int64_t page_size,
    bool check_bounds);

std::vector<at::Tensor> paged_kv_read_cuda(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    int64_t page_size,
    bool check_bounds);

void paged_kv_read_cuda_out(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    at::Tensor out_k,
    at::Tensor out_v,
    int64_t page_size,
    bool check_bounds);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("update", &paged_kv_update_cuda,
             "Paged KV-cache update using logical block table");
  module.def("read", &paged_kv_read_cuda,
             "Paged KV-cache attention-compatible read");
  module.def("read_out", &paged_kv_read_cuda_out,
             "Paged KV-cache attention-compatible read (out)");
}
