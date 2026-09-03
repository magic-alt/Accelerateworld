#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int64_t kTokenMajor = 0;
constexpr int64_t kHeadMajor = 1;

__device__ __forceinline__ int64_t CachePairOffset(
    int64_t batch_index,
    int64_t head,
    int64_t position,
    int64_t pair,
    int64_t kv_heads,
    int64_t capacity,
    int64_t pairs_per_head,
    int64_t layout) {
  if (layout == kTokenMajor) {
    return (((batch_index * capacity + position) * kv_heads + head) * pairs_per_head + pair);
  }
  return (((batch_index * kv_heads + head) * capacity + position) * pairs_per_head + pair);
}

__global__ void KvCacheUpdatePairsKernel(
    uint32_t* cache_k,
    uint32_t* cache_v,
    const uint32_t* new_k,
    const uint32_t* new_v,
    const int64_t* positions,
    int64_t batch,
    int64_t kv_heads,
    int64_t tokens,
    int64_t capacity,
    int64_t pairs_per_head,
    int64_t layout) {
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
  if (position < 0 || position >= capacity) {
    return;
  }

  const int64_t source =
      (((batch_index * kv_heads + head) * tokens + token) * pairs_per_head + pair);
  const int64_t target = CachePairOffset(
      batch_index, head, position, pair, kv_heads, capacity, pairs_per_head, layout);
  cache_k[target] = new_k[source];
  cache_v[target] = new_v[source];
}

__global__ void KvCacheReadPairsKernel(
    const uint32_t* cache_k,
    const uint32_t* cache_v,
    const int64_t* positions,
    uint32_t* out_k,
    uint32_t* out_v,
    int64_t batch,
    int64_t kv_heads,
    int64_t tokens,
    int64_t capacity,
    int64_t pairs_per_head,
    int64_t layout) {
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
  if (position < 0 || position >= capacity) {
    return;
  }

  const int64_t source = CachePairOffset(
      batch_index, head, position, pair, kv_heads, capacity, pairs_per_head, layout);
  const int64_t target =
      (((batch_index * kv_heads + head) * tokens + token) * pairs_per_head + pair);
  out_k[target] = cache_k[source];
  out_v[target] = cache_v[source];
}

int64_t Capacity(const at::Tensor& cache, int64_t layout) {
  return layout == kTokenMajor ? cache.size(1) : cache.size(2);
}

int64_t KvHeads(const at::Tensor& cache, int64_t layout) {
  return layout == kTokenMajor ? cache.size(2) : cache.size(1);
}

void ValidateLayout(int64_t layout) {
  TORCH_CHECK(layout == kTokenMajor || layout == kHeadMajor,
              "layout must be 0 (token_major) or 1 (head_major)");
}

void ValidateCachePair(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    int64_t layout) {
  ValidateLayout(layout);
  TORCH_CHECK(cache_k.is_cuda() && cache_v.is_cuda(), "cache K/V must be CUDA tensors");
  TORCH_CHECK(cache_k.dim() == 4 && cache_v.sizes() == cache_k.sizes(),
              "cache K/V must be matching rank-4 tensors");
  TORCH_CHECK(cache_k.scalar_type() == at::kHalf || cache_k.scalar_type() == at::kBFloat16,
              "KV cache supports float16 or bfloat16 storage");
  TORCH_CHECK(cache_v.scalar_type() == cache_k.scalar_type(), "cache K/V dtype must match");
  TORCH_CHECK(cache_k.is_contiguous() && cache_v.is_contiguous(), "cache K/V must be contiguous");
  TORCH_CHECK(cache_k.size(3) == 64 || cache_k.size(3) == 128,
              "v0 KV cache supports head_dim 64 or 128");
  TORCH_CHECK(cache_k.size(3) % 2 == 0, "head_dim must be even for 32-bit pair copies");
  TORCH_CHECK(Capacity(cache_k, layout) > 0, "cache capacity must be positive");
}

void ValidatePositions(
    const at::Tensor& positions,
    const at::Tensor& cache,
    int64_t layout,
    bool check_bounds) {
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.dim() == 2,
              "positions must be int64 [batch, tokens]");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(positions.size(0) == cache.size(0), "positions batch must match cache batch");
  if (check_bounds) {
    const int64_t minimum = positions.min().item<int64_t>();
    const int64_t maximum = positions.max().item<int64_t>();
    TORCH_CHECK(minimum >= 0 && maximum < Capacity(cache, layout),
                "KV-cache positions [", minimum, ", ", maximum,
                "] exceed capacity ", Capacity(cache, layout));
  }
}

void ValidateBf16Runtime(const at::Tensor& tensor) {
  if (tensor.scalar_type() != at::kBFloat16) {
    return;
  }
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(properties.major >= 8,
              "BF16 KV-cache runtime validation requires compute capability >= 8.0; current device is ",
              properties.major, ".", properties.minor);
}

void ValidateUpdate(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    int64_t layout,
    bool check_bounds) {
  ValidateCachePair(cache_k, cache_v, layout);
  TORCH_CHECK(new_k.is_cuda() && new_v.is_cuda(), "new K/V must be CUDA tensors");
  TORCH_CHECK(new_k.dim() == 4 && new_v.sizes() == new_k.sizes(),
              "new K/V must be matching [batch, kv_heads, tokens, head_dim]");
  TORCH_CHECK(new_k.scalar_type() == cache_k.scalar_type() && new_v.scalar_type() == cache_k.scalar_type(),
              "new K/V dtype must match cache dtype");
  TORCH_CHECK(new_k.is_contiguous() && new_v.is_contiguous(), "new K/V must be contiguous");
  TORCH_CHECK(new_k.size(0) == cache_k.size(0) && new_k.size(1) == KvHeads(cache_k, layout) &&
                  new_k.size(3) == cache_k.size(3),
              "new K/V batch, kv_heads and head_dim must match cache");
  TORCH_CHECK(positions.size(1) == new_k.size(2), "positions token count must match new K/V");
  ValidatePositions(positions, cache_k, layout, check_bounds);
  ValidateBf16Runtime(cache_k);
}

void LaunchUpdate(
    at::Tensor cache_k,
    at::Tensor cache_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    int64_t layout,
    bool check_bounds) {
  ValidateUpdate(cache_k, cache_v, new_k, new_v, positions, layout, check_bounds);
  c10::cuda::CUDAGuard guard(cache_k.device());
  const int64_t pairs = cache_k.size(3) / 2;
  const int64_t total = new_k.size(0) * new_k.size(1) * new_k.size(2) * pairs;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  KvCacheUpdatePairsKernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t*>(cache_k.data_ptr()),
      reinterpret_cast<uint32_t*>(cache_v.data_ptr()),
      reinterpret_cast<const uint32_t*>(new_k.data_ptr()),
      reinterpret_cast<const uint32_t*>(new_v.data_ptr()),
      positions.data_ptr<int64_t>(),
      new_k.size(0), new_k.size(1), new_k.size(2), Capacity(cache_k, layout), pairs, layout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ValidateRead(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& positions,
    const at::Tensor& out_k,
    const at::Tensor& out_v,
    int64_t layout,
    bool check_bounds) {
  ValidateCachePair(cache_k, cache_v, layout);
  ValidatePositions(positions, cache_k, layout, check_bounds);
  ValidateBf16Runtime(cache_k);
  const auto expected = std::vector<int64_t>{
      cache_k.size(0), KvHeads(cache_k, layout), positions.size(1), cache_k.size(3)};
  TORCH_CHECK(out_k.sizes().vec() == expected && out_v.sizes().vec() == expected,
              "read output must be [batch, kv_heads, tokens, head_dim]");
  TORCH_CHECK(out_k.scalar_type() == cache_k.scalar_type() && out_v.scalar_type() == cache_k.scalar_type(),
              "read output dtype must match cache");
  TORCH_CHECK(out_k.is_cuda() && out_v.is_cuda() && out_k.is_contiguous() && out_v.is_contiguous(),
              "read outputs must be contiguous CUDA tensors");
}

void LaunchRead(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& positions,
    at::Tensor out_k,
    at::Tensor out_v,
    int64_t layout,
    bool check_bounds) {
  ValidateRead(cache_k, cache_v, positions, out_k, out_v, layout, check_bounds);
  c10::cuda::CUDAGuard guard(cache_k.device());
  const int64_t pairs = cache_k.size(3) / 2;
  const int64_t total = cache_k.size(0) * KvHeads(cache_k, layout) * positions.size(1) * pairs;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  KvCacheReadPairsKernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(cache_k.data_ptr()),
      reinterpret_cast<const uint32_t*>(cache_v.data_ptr()),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(out_k.data_ptr()),
      reinterpret_cast<uint32_t*>(out_v.data_ptr()),
      cache_k.size(0), KvHeads(cache_k, layout), positions.size(1), Capacity(cache_k, layout), pairs, layout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void kv_cache_update_cuda(
    at::Tensor cache_k,
    at::Tensor cache_v,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    const at::Tensor& positions,
    int64_t layout,
    bool check_bounds) {
  LaunchUpdate(cache_k, cache_v, new_k, new_v, positions, layout, check_bounds);
}

std::vector<at::Tensor> kv_cache_read_cuda(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& positions,
    int64_t layout,
    bool check_bounds) {
  const int64_t kv_heads = KvHeads(cache_k, layout);
  auto out_k = at::empty({cache_k.size(0), kv_heads, positions.size(1), cache_k.size(3)}, cache_k.options());
  auto out_v = at::empty_like(out_k);
  LaunchRead(cache_k, cache_v, positions, out_k, out_v, layout, check_bounds);
  return {out_k, out_v};
}

void kv_cache_read_cuda_out(
    const at::Tensor& cache_k,
    const at::Tensor& cache_v,
    const at::Tensor& positions,
    at::Tensor out_k,
    at::Tensor out_v,
    int64_t layout,
    bool check_bounds) {
  LaunchRead(cache_k, cache_v, positions, out_k, out_v, layout, check_bounds);
}
