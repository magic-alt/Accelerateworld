#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int64_t PagePairOffset(
    int64_t physical_page,
    int64_t head,
    int64_t slot,
    int64_t pair,
    int64_t kv_heads,
    int64_t page_size,
    int64_t pairs_per_head) {
  return (((physical_page * kv_heads + head) * page_size + slot) * pairs_per_head + pair);
}

__global__ void PagedKvUpdatePairsKernel(
    uint32_t* page_k,
    uint32_t* page_v,
    const uint32_t* new_k,
    const uint32_t* new_v,
    const int64_t* positions,
    const int64_t* block_table,
    int64_t batch,
    int64_t kv_heads,
    int64_t tokens,
    int64_t num_pages,
    int64_t max_blocks,
    int64_t page_size,
    int64_t pairs_per_head) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_pairs = batch * kv_heads * tokens * pairs_per_head;
  if (index >= total_pairs) {
    return;
  }

  const int64_t pair = index % pairs_per_head;
  int64_t row = index / pairs_per_head;
  const int64_t token = row % tokens;
  row /= tokens;
  const int64_t head = row % kv_heads;
  const int64_t batch_index = row / kv_heads;

  const int64_t position = positions[batch_index * tokens + token];
  if (position < 0) {
    return;
  }
  const int64_t logical_block = position / page_size;
  if (logical_block < 0 || logical_block >= max_blocks) {
    return;
  }
  const int64_t slot = position % page_size;
  const int64_t physical_page = block_table[batch_index * max_blocks + logical_block];
  if (physical_page < 0 || physical_page >= num_pages) {
    return;
  }

  const int64_t source =
      (((batch_index * kv_heads + head) * tokens + token) * pairs_per_head + pair);
  const int64_t target = PagePairOffset(
      physical_page, head, slot, pair, kv_heads, page_size, pairs_per_head);
  page_k[target] = new_k[source];
  page_v[target] = new_v[source];
}

__global__ void PagedKvReadPairsKernel(
    const uint32_t* page_k,
    const uint32_t* page_v,
    const int64_t* positions,
    const int64_t* block_table,
    uint32_t* out_k,
    uint32_t* out_v,
    int64_t batch,
    int64_t kv_heads,
    int64_t tokens,
    int64_t num_pages,
    int64_t max_blocks,
    int64_t page_size,
    int64_t pairs_per_head) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_pairs = batch * kv_heads * tokens * pairs_per_head;
  if (index >= total_pairs) {
    return;
  }

  const int64_t pair = index % pairs_per_head;
  int64_t row = index / pairs_per_head;
  const int64_t token = row % tokens;
  row /= tokens;
  const int64_t head = row % kv_heads;
  const int64_t batch_index = row / kv_heads;

  const int64_t position = positions[batch_index * tokens + token];
  if (position < 0) {
    return;
  }
  const int64_t logical_block = position / page_size;
  if (logical_block < 0 || logical_block >= max_blocks) {
    return;
  }
  const int64_t slot = position % page_size;
  const int64_t physical_page = block_table[batch_index * max_blocks + logical_block];
  if (physical_page < 0 || physical_page >= num_pages) {
    return;
  }

  const int64_t source = PagePairOffset(
      physical_page, head, slot, pair, kv_heads, page_size, pairs_per_head);
  const int64_t target =
      (((batch_index * kv_heads + head) * tokens + token) * pairs_per_head + pair);
  out_k[target] = page_k[source];
  out_v[target] = page_v[source];
}

void ValidateBf16Runtime(const at::Tensor& tensor) {
  if (tensor.scalar_type() != at::kBFloat16) {
    return;
  }
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(
      properties.major >= 8,
      "BF16 paged KV-cache runtime validation requires compute capability >= 8.0; current device is ",
      properties.major, ".", properties.minor);
}

void ValidatePagePair(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    int64_t page_size) {
  TORCH_CHECK(page_k.is_cuda() && page_v.is_cuda(), "page K/V must be CUDA tensors");
  TORCH_CHECK(
      page_k.dim() == 4 && page_v.sizes() == page_k.sizes(),
      "page K/V must be matching [pages, kv_heads, page_size, head_dim]");
  TORCH_CHECK(
      page_k.scalar_type() == at::kHalf || page_k.scalar_type() == at::kBFloat16,
      "paged KV cache supports float16 or bfloat16 storage");
  TORCH_CHECK(page_v.scalar_type() == page_k.scalar_type(), "page K/V dtype must match");
  TORCH_CHECK(
      page_k.is_contiguous() && page_v.is_contiguous(),
      "physical page tensors must be contiguous");
  TORCH_CHECK(page_k.size(0) > 0 && page_k.size(1) > 0, "page pool and kv_heads must be positive");
  TORCH_CHECK(page_size > 0 && page_k.size(2) == page_size, "page_size must match page tensor");
  TORCH_CHECK(
      page_k.size(3) == 64 || page_k.size(3) == 128,
      "v0 paged KV cache supports head_dim 64 or 128");
  TORCH_CHECK(page_k.size(3) % 2 == 0, "head_dim must be even for 32-bit pair copies");
  ValidateBf16Runtime(page_k);
}

void ValidateMetadata(
    const at::Tensor& positions,
    const at::Tensor& block_table,
    const at::Tensor& page_k,
    int64_t page_size,
    bool check_bounds) {
  TORCH_CHECK(
      positions.is_cuda() && positions.scalar_type() == at::kLong && positions.dim() == 2,
      "positions must be CUDA int64 [batch, tokens]");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(
      block_table.is_cuda() && block_table.scalar_type() == at::kLong && block_table.dim() == 2,
      "block_table must be CUDA int64 [batch, max_blocks]");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(
      positions.size(0) == block_table.size(0),
      "positions batch must match block_table batch");
  TORCH_CHECK(block_table.size(1) > 0, "block_table must have at least one logical block");

  if (check_bounds) {
    const int64_t minimum = positions.min().item<int64_t>();
    const int64_t maximum = positions.max().item<int64_t>();
    const int64_t max_tokens = block_table.size(1) * page_size;
    TORCH_CHECK(
        minimum >= 0 && maximum < max_tokens,
        "logical positions [", minimum, ", ", maximum,
        "] exceed block-table capacity ", max_tokens);

    auto logical_blocks = at::floor_divide(positions, page_size);
    auto physical_pages = block_table.gather(1, logical_blocks);
    const int64_t minimum_page = physical_pages.min().item<int64_t>();
    const int64_t maximum_page = physical_pages.max().item<int64_t>();
    TORCH_CHECK(
        minimum_page >= 0 && maximum_page < page_k.size(0),
        "requested logical block maps to physical page range [",
        minimum_page, ", ", maximum_page, "] outside page pool ", page_k.size(0));
  }
}

void ValidateUpdate(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    int64_t page_size,
    bool check_bounds) {
  ValidatePagePair(page_k, page_v, page_size);
  ValidateMetadata(positions, block_table, page_k, page_size, check_bounds);
  TORCH_CHECK(
      new_k.is_cuda() && new_v.is_cuda() && new_k.dim() == 4 && new_v.sizes() == new_k.sizes(),
      "new K/V must be matching CUDA [batch, kv_heads, tokens, head_dim]");
  TORCH_CHECK(
      new_k.is_contiguous() && new_v.is_contiguous(),
      "new K/V must be contiguous");
  TORCH_CHECK(
      new_k.scalar_type() == page_k.scalar_type() && new_v.scalar_type() == page_k.scalar_type(),
      "new K/V dtype must match physical pages");
  TORCH_CHECK(
      new_k.size(0) == positions.size(0) &&
          new_k.size(1) == page_k.size(1) &&
          new_k.size(2) == positions.size(1) &&
          new_k.size(3) == page_k.size(3),
      "new K/V dimensions must match page cache and positions");
}

void LaunchUpdate(
    at::Tensor page_k,
    at::Tensor page_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    int64_t page_size,
    bool check_bounds) {
  ValidateUpdate(
      page_k, page_v, new_k, new_v, positions, block_table, page_size, check_bounds);
  c10::cuda::CUDAGuard guard(page_k.device());
  const int64_t pairs = page_k.size(3) / 2;
  const int64_t total = new_k.size(0) * new_k.size(1) * new_k.size(2) * pairs;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  PagedKvUpdatePairsKernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t*>(page_k.data_ptr()),
      reinterpret_cast<uint32_t*>(page_v.data_ptr()),
      reinterpret_cast<const uint32_t*>(new_k.data_ptr()),
      reinterpret_cast<const uint32_t*>(new_v.data_ptr()),
      positions.data_ptr<int64_t>(),
      block_table.data_ptr<int64_t>(),
      new_k.size(0),
      new_k.size(1),
      new_k.size(2),
      page_k.size(0),
      block_table.size(1),
      page_size,
      pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ValidateRead(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    const at::Tensor& out_k,
    const at::Tensor& out_v,
    int64_t page_size,
    bool check_bounds) {
  ValidatePagePair(page_k, page_v, page_size);
  ValidateMetadata(positions, block_table, page_k, page_size, check_bounds);
  const std::vector<int64_t> expected = {
      positions.size(0), page_k.size(1), positions.size(1), page_k.size(3)};
  TORCH_CHECK(
      out_k.sizes().vec() == expected && out_v.sizes().vec() == expected,
      "read output must be [batch, kv_heads, tokens, head_dim]");
  TORCH_CHECK(
      out_k.scalar_type() == page_k.scalar_type() && out_v.scalar_type() == page_k.scalar_type(),
      "read output dtype must match page cache");
  TORCH_CHECK(
      out_k.is_cuda() && out_v.is_cuda() && out_k.is_contiguous() && out_v.is_contiguous(),
      "read outputs must be contiguous CUDA tensors");
}

void LaunchRead(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    at::Tensor out_k,
    at::Tensor out_v,
    int64_t page_size,
    bool check_bounds) {
  ValidateRead(
      page_k, page_v, positions, block_table, out_k, out_v, page_size, check_bounds);
  c10::cuda::CUDAGuard guard(page_k.device());
  const int64_t pairs = page_k.size(3) / 2;
  const int64_t total =
      positions.size(0) * page_k.size(1) * positions.size(1) * pairs;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  PagedKvReadPairsKernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(page_k.data_ptr()),
      reinterpret_cast<const uint32_t*>(page_v.data_ptr()),
      positions.data_ptr<int64_t>(),
      block_table.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(out_k.data_ptr()),
      reinterpret_cast<uint32_t*>(out_v.data_ptr()),
      positions.size(0),
      page_k.size(1),
      positions.size(1),
      page_k.size(0),
      block_table.size(1),
      page_size,
      pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void paged_kv_update_cuda(
    at::Tensor page_k,
    at::Tensor page_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    int64_t page_size,
    bool check_bounds) {
  LaunchUpdate(
      page_k, page_v, new_k, new_v, positions, block_table, page_size, check_bounds);
}

std::vector<at::Tensor> paged_kv_read_cuda(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    int64_t page_size,
    bool check_bounds) {
  auto out_k = at::empty(
      {positions.size(0), page_k.size(1), positions.size(1), page_k.size(3)},
      page_k.options());
  auto out_v = at::empty_like(out_k);
  LaunchRead(
      page_k, page_v, positions, block_table, out_k, out_v, page_size, check_bounds);
  return {out_k, out_v};
}

void paged_kv_read_cuda_out(
    const at::Tensor& page_k,
    const at::Tensor& page_v,
    const at::Tensor& positions,
    const at::Tensor& block_table,
    at::Tensor out_k,
    at::Tensor out_v,
    int64_t page_size,
    bool check_bounds) {
  LaunchRead(
      page_k, page_v, positions, block_table, out_k, out_v, page_size, check_bounds);
}
