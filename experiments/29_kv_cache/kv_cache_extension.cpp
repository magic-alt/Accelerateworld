#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

void kv_cache_update_cuda(
    at::Tensor cache_k,
    at::Tensor cache_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    int64_t layout,
    bool check_bounds);

std::vector<at::Tensor> kv_cache_read_cuda(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& positions,
    int64_t layout,
    bool check_bounds);

void kv_cache_read_cuda_out(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& positions,
    at::Tensor out_k,
    at::Tensor out_v,
    int64_t layout,
    bool check_bounds);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &kv_cache_update_cuda,
        py::arg("cache_k"), py::arg("cache_v"), py::arg("new_k"), py::arg("new_v"),
        py::arg("positions"), py::arg("layout"), py::arg("check_bounds") = true);
  m.def("read", &kv_cache_read_cuda,
        py::arg("cache_k"), py::arg("cache_v"), py::arg("positions"),
        py::arg("layout"), py::arg("check_bounds") = true);
  m.def("read_out", &kv_cache_read_cuda_out,
        py::arg("cache_k"), py::arg("cache_v"), py::arg("positions"),
        py::arg("out_k"), py::arg("out_v"), py::arg("layout"),
        py::arg("check_bounds") = true);
}
