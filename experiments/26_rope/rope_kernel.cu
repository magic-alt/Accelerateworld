#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kInterleaved = 0;
constexpr int kHalfSplit = 1;
constexpr int kThreads = 256;

__global__ void RopeHalfKernel(
    const __half* input,
    __half* output,
    const float* cos,
    const float* sin,
    const int64_t* positions,
    int64_t total_units,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t cache_stride,
    int layout) {
  const int64_t unit = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (unit >= total_units) {
    return;
  }

  const int64_t pair_count = rotary_dim / 2;
  const int64_t units_per_head = pair_count + (head_dim - rotary_dim);
  const int64_t row_head = unit / units_per_head;
  const int64_t local = unit - row_head * units_per_head;
  const int64_t token = row_head / heads;
  const int64_t base = row_head * head_dim;

  if (local < pair_count) {
    const int64_t position = positions[token];
    const int64_t cache_index = position * cache_stride + local;
    const float c = cos[cache_index];
    const float s = sin[cache_index];

    if (layout == kInterleaved) {
      const int64_t dim0 = local * 2;
      const __half2 pair = *reinterpret_cast<const __half2*>(input + base + dim0);
      const float2 values = __half22float2(pair);
      const __half2 rotated = __floats2half2_rn(
          values.x * c - values.y * s,
          values.x * s + values.y * c);
      *reinterpret_cast<__half2*>(output + base + dim0) = rotated;
    } else {
      const int64_t dim0 = local;
      const int64_t dim1 = local + pair_count;
      const float x0 = __half2float(input[base + dim0]);
      const float x1 = __half2float(input[base + dim1]);
      output[base + dim0] = __float2half_rn(x0 * c - x1 * s);
      output[base + dim1] = __float2half_rn(x0 * s + x1 * c);
    }
  } else {
    const int64_t tail_dim = rotary_dim + (local - pair_count);
    output[base + tail_dim] = input[base + tail_dim];
  }
}

__global__ void RopeBFloat16Kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const float* cos,
    const float* sin,
    const int64_t* positions,
    int64_t total_units,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t cache_stride,
    int layout) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int64_t unit = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (unit >= total_units) {
    return;
  }

  const int64_t pair_count = rotary_dim / 2;
  const int64_t units_per_head = pair_count + (head_dim - rotary_dim);
  const int64_t row_head = unit / units_per_head;
  const int64_t local = unit - row_head * units_per_head;
  const int64_t token = row_head / heads;
  const int64_t base = row_head * head_dim;

  if (local < pair_count) {
    const int64_t position = positions[token];
    const int64_t cache_index = position * cache_stride + local;
    const float c = cos[cache_index];
    const float s = sin[cache_index];

    if (layout == kInterleaved) {
      const int64_t dim0 = local * 2;
      const __nv_bfloat162 pair =
          *reinterpret_cast<const __nv_bfloat162*>(input + base + dim0);
      const float2 values = __bfloat1622float2(pair);
      const __nv_bfloat162 rotated = __floats2bfloat162_rn(
          values.x * c - values.y * s,
          values.x * s + values.y * c);
      *reinterpret_cast<__nv_bfloat162*>(output + base + dim0) = rotated;
    } else {
      const int64_t dim0 = local;
      const int64_t dim1 = local + pair_count;
      const float x0 = __bfloat162float(input[base + dim0]);
      const float x1 = __bfloat162float(input[base + dim1]);
      output[base + dim0] = __float2bfloat16_rn(x0 * c - x1 * s);
      output[base + dim1] = __float2bfloat16_rn(x0 * s + x1 * c);
    }
  } else {
    const int64_t tail_dim = rotary_dim + (local - pair_count);
    output[base + tail_dim] = input[base + tail_dim];
  }
#endif
}

void ValidateTensorContract(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& positions,
    int64_t rotary_dim) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
  TORCH_CHECK(input.dim() == 4 && output.sizes() == input.sizes(),
              "input/output must be matching [batch, sequence, heads, head_dim]");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "RoPE supports float16 or bfloat16 inputs");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input/output must be contiguous");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= input.size(3),
              "rotary_dim must be positive, even and <= head_dim");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && positions.is_cuda(),
              "cos/sin/positions must be CUDA tensors");
  TORCH_CHECK(cos.device() == input.device() && sin.device() == input.device() &&
                  positions.device() == input.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "cos/sin cache must be float32");
  TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes() && cos.size(1) >= rotary_dim / 2,
              "cos/sin cache must be [max_position, rotary_dim/2 or larger]");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.dim() == 2 &&
                  positions.size(0) == input.size(0) && positions.size(1) == input.size(1),
              "positions must be int64 [batch, sequence]");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous() && positions.is_contiguous(),
              "cos/sin/positions must be contiguous");
}

void LaunchTensor(
    const at::Tensor& input,
    at::Tensor output,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& positions,
    int64_t rotary_dim,
    int layout,
    cudaStream_t stream) {
  ValidateTensorContract(input, output, cos, sin, positions, rotary_dim);
  const int64_t pair_count = rotary_dim / 2;
  const int64_t units_per_head = pair_count + (input.size(3) - rotary_dim);
  const int64_t total_units = input.size(0) * input.size(1) * input.size(2) * units_per_head;
  if (total_units == 0) {
    return;
  }
  const int blocks = static_cast<int>((total_units + kThreads - 1) / kThreads);

  if (input.scalar_type() == at::kHalf) {
    RopeHalfKernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        total_units,
        input.size(2),
        input.size(3),
        rotary_dim,
        cos.size(1),
        layout);
  } else {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
    TORCH_CHECK(properties.major >= 8,
                "BF16 RoPE requires compute capability >= 8.0; current device is ",
                properties.major, ".", properties.minor);
    RopeBFloat16Kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        total_units,
        input.size(2),
        input.size(3),
        rotary_dim,
        cos.size(1),
        layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ValidateQK(const at::Tensor& q, const at::Tensor& k, int64_t layout) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3),
              "q/k must share batch, sequence and head_dim");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype must match");
  TORCH_CHECK(q.device() == k.device(), "q/k device must match");
  TORCH_CHECK(layout == kInterleaved || layout == kHalfSplit,
              "layout must be 0 (interleaved) or 1 (half_split)");
}

}  // namespace

void rope_cuda_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& positions,
    at::Tensor q_out,
    at::Tensor k_out,
    int64_t rotary_dim,
    int64_t layout) {
  ValidateQK(q, k, layout);
  c10::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  LaunchTensor(q, q_out, cos, sin, positions, rotary_dim, static_cast<int>(layout), stream);
  LaunchTensor(k, k_out, cos, sin, positions, rotary_dim, static_cast<int>(layout), stream);
}

std::tuple<at::Tensor, at::Tensor> rope_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& positions,
    int64_t rotary_dim,
    int64_t layout) {
  auto q_out = at::empty_like(q);
  auto k_out = at::empty_like(k);
  rope_cuda_out(q, k, cos, sin, positions, q_out, k_out, rotary_dim, layout);
  return std::make_tuple(q_out, k_out);
}
