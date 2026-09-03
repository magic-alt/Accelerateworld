#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
constexpr int64_t kPerTensor = 0;
constexpr int64_t kPerChannel = 1;
constexpr int64_t kGroup = 2;
constexpr int64_t kOutputHalf = 0;
constexpr int64_t kOutputBFloat16 = 1;

__device__ __forceinline__ int64_t ParamIndex(int64_t row, int64_t col, int64_t granularity, int64_t group_size, int64_t groups_per_row) {
  if (granularity == kPerTensor) return 0;
  if (granularity == kPerChannel) return row;
  return row * groups_per_row + col / group_size;
}
__device__ __forceinline__ int RoundHalfAwayFromZero(float value) {
  return value >= 0.0f ? static_cast<int>(floorf(value + 0.5f)) : static_cast<int>(ceilf(value - 0.5f));
}
template <typename T> __device__ __forceinline__ float LoadFloat(T value);
template <> __device__ __forceinline__ float LoadFloat<__half>(__half value) { return __half2float(value); }
template <> __device__ __forceinline__ float LoadFloat<__nv_bfloat16>(__nv_bfloat16 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(value);
#else
  return 0.0f;
#endif
}
template <typename T> __device__ __forceinline__ T StoreFloat(float value);
template <> __device__ __forceinline__ __half StoreFloat<__half>(float value) { return __float2half_rn(value); }
template <> __device__ __forceinline__ __nv_bfloat16 StoreFloat<__nv_bfloat16>(float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __float2bfloat16_rn(value);
#else
  return __nv_bfloat16{};
#endif
}

template <typename InputT>
__global__ void QuantizeInt8Kernel(const InputT* input, const float* scales, const int32_t* zero_points, int8_t* output, int64_t rows, int64_t cols, int64_t granularity, int64_t group_size, int64_t groups_per_row, int qmin, int qmax) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  const int64_t row = index / cols;
  const int64_t col = index - row * cols;
  const int64_t pidx = ParamIndex(row, col, granularity, group_size, groups_per_row);
  const float qf = LoadFloat(input[index]) / scales[pidx] + static_cast<float>(zero_points[pidx]);
  int q = RoundHalfAwayFromZero(qf);
  q = max(qmin, min(qmax, q));
  output[index] = static_cast<int8_t>(q);
}

template <typename OutputT>
__global__ void DequantizeInt8Kernel(const int8_t* input, const float* scales, const int32_t* zero_points, OutputT* output, int64_t rows, int64_t cols, int64_t granularity, int64_t group_size, int64_t groups_per_row) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  const int64_t row = index / cols;
  const int64_t col = index - row * cols;
  const int64_t pidx = ParamIndex(row, col, granularity, group_size, groups_per_row);
  const float value = (static_cast<float>(input[index]) - static_cast<float>(zero_points[pidx])) * scales[pidx];
  output[index] = StoreFloat<OutputT>(value);
}

template <typename InputT>
__global__ void QuantizeInt4Kernel(const InputT* input, const float* scales, uint8_t* output, int64_t rows, int64_t cols, int64_t packed_cols, int64_t granularity, int64_t group_size, int64_t groups_per_row) {
  const int64_t byte_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (byte_index >= rows * packed_cols) return;
  const int64_t row = byte_index / packed_cols;
  const int64_t byte_col = byte_index - row * packed_cols;
  const int64_t col0 = byte_col * 2;
  const int64_t col1 = col0 + 1;
  const int64_t p0 = ParamIndex(row, col0, granularity, group_size, groups_per_row);
  int q0 = RoundHalfAwayFromZero(LoadFloat(input[row * cols + col0]) / scales[p0]);
  q0 = max(-8, min(7, q0));
  const uint8_t low = static_cast<uint8_t>(q0 & 0xF);
  uint8_t high = 0;
  if (col1 < cols) {
    const int64_t p1 = ParamIndex(row, col1, granularity, group_size, groups_per_row);
    int q1 = RoundHalfAwayFromZero(LoadFloat(input[row * cols + col1]) / scales[p1]);
    q1 = max(-8, min(7, q1));
    high = static_cast<uint8_t>((q1 & 0xF) << 4);
  }
  output[byte_index] = static_cast<uint8_t>(low | high);
}

template <typename OutputT>
__global__ void DequantizeInt4Kernel(const uint8_t* input, const float* scales, OutputT* output, int64_t rows, int64_t cols, int64_t packed_cols, int64_t granularity, int64_t group_size, int64_t groups_per_row) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  const int64_t row = index / cols;
  const int64_t col = index - row * cols;
  const uint8_t byte = input[row * packed_cols + col / 2];
  const int nibble = (col & 1) == 0 ? (byte & 0xF) : ((byte >> 4) & 0xF);
  const int q = nibble >= 8 ? nibble - 16 : nibble;
  const int64_t pidx = ParamIndex(row, col, granularity, group_size, groups_per_row);
  output[index] = StoreFloat<OutputT>(static_cast<float>(q) * scales[pidx]);
}

void ValidateGranularity(int64_t granularity, int64_t group_size) {
  TORCH_CHECK(granularity == kPerTensor || granularity == kPerChannel || granularity == kGroup, "invalid granularity");
  if (granularity == kGroup) TORCH_CHECK(group_size > 0, "group_size must be positive");
}
int64_t ParameterCount(int64_t rows, int64_t cols, int64_t granularity, int64_t group_size) {
  if (granularity == kPerTensor) return 1;
  if (granularity == kPerChannel) return rows;
  return rows * ((cols + group_size - 1) / group_size);
}
void ValidateQParams(const at::Tensor& scales, const at::Tensor& zero_points, int64_t rows, int64_t cols, int64_t granularity, int64_t group_size) {
  ValidateGranularity(granularity, group_size);
  TORCH_CHECK(scales.is_cuda() && zero_points.is_cuda(), "qparams must be CUDA");
  TORCH_CHECK(scales.scalar_type() == at::kFloat && zero_points.scalar_type() == at::kInt, "scales float32 / zero_points int32 required");
  TORCH_CHECK(scales.dim() == 1 && zero_points.dim() == 1 && scales.is_contiguous() && zero_points.is_contiguous(), "qparams must be contiguous rank-1");
  const int64_t expected = ParameterCount(rows, cols, granularity, group_size);
  TORCH_CHECK(scales.numel() == expected && zero_points.numel() == expected, "qparam count mismatch");
}
void ValidateInput(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(), "input must be contiguous CUDA rank-2");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16, "input must be fp16/bf16");
}
void ValidateBFloat16Runtime(const at::Tensor& tensor) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(properties.major >= 8, "BF16 quantization requires compute capability >= 8.0");
}
int64_t GroupsPerRow(int64_t cols, int64_t granularity, int64_t group_size) { return granularity == kGroup ? (cols + group_size - 1) / group_size : 1; }
}  // namespace

at::Tensor quantize_int8_cuda(const at::Tensor& input, const at::Tensor& scales, const at::Tensor& zero_points, int64_t granularity, int64_t group_size, bool asymmetric) {
  ValidateInput(input); ValidateQParams(scales, zero_points, input.size(0), input.size(1), granularity, group_size);
  TORCH_CHECK(input.device() == scales.device() && scales.device() == zero_points.device(), "input/qparams device mismatch");
  if (input.scalar_type() == at::kBFloat16) ValidateBFloat16Runtime(input);
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty(input.sizes(), input.options().dtype(at::kChar));
  const int blocks = static_cast<int>((input.numel() + kThreads - 1) / kThreads);
  const int64_t groups = GroupsPerRow(input.size(1), granularity, group_size);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (input.scalar_type() == at::kHalf) QuantizeInt8Kernel<<<blocks,kThreads,0,stream>>>(reinterpret_cast<const __half*>(input.data_ptr<at::Half>()), scales.data_ptr<float>(), zero_points.data_ptr<int32_t>(), output.data_ptr<int8_t>(), input.size(0), input.size(1), granularity, group_size, groups, asymmetric ? -128 : -127, 127);
  else QuantizeInt8Kernel<<<blocks,kThreads,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()), scales.data_ptr<float>(), zero_points.data_ptr<int32_t>(), output.data_ptr<int8_t>(), input.size(0), input.size(1), granularity, group_size, groups, asymmetric ? -128 : -127, 127);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return output;
}

at::Tensor dequantize_int8_cuda(const at::Tensor& q, const at::Tensor& scales, const at::Tensor& zero_points, int64_t granularity, int64_t group_size, int64_t output_dtype) {
  TORCH_CHECK(q.is_cuda() && q.dim()==2 && q.scalar_type()==at::kChar && q.is_contiguous(), "q must be CUDA int8 rank-2");
  ValidateQParams(scales, zero_points, q.size(0), q.size(1), granularity, group_size);
  TORCH_CHECK(output_dtype==kOutputHalf || output_dtype==kOutputBFloat16, "invalid output dtype");
  if (output_dtype==kOutputBFloat16) ValidateBFloat16Runtime(q);
  c10::cuda::CUDAGuard guard(q.device());
  auto output = at::empty(q.sizes(), q.options().dtype(output_dtype==kOutputHalf ? at::kHalf : at::kBFloat16));
  const int blocks = static_cast<int>((q.numel()+kThreads-1)/kThreads); const int64_t groups=GroupsPerRow(q.size(1),granularity,group_size); auto stream=at::cuda::getCurrentCUDAStream();
  if (output_dtype==kOutputHalf) DequantizeInt8Kernel<<<blocks,kThreads,0,stream>>>(q.data_ptr<int8_t>(),scales.data_ptr<float>(),zero_points.data_ptr<int32_t>(),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),q.size(0),q.size(1),granularity,group_size,groups);
  else DequantizeInt8Kernel<<<blocks,kThreads,0,stream>>>(q.data_ptr<int8_t>(),scales.data_ptr<float>(),zero_points.data_ptr<int32_t>(),reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),q.size(0),q.size(1),granularity,group_size,groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return output;
}

at::Tensor quantize_int4_cuda(const at::Tensor& input, const at::Tensor& scales, int64_t granularity, int64_t group_size) {
  ValidateInput(input); ValidateGranularity(granularity,group_size);
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type()==at::kFloat && scales.is_contiguous(), "scales must be contiguous CUDA float32");
  TORCH_CHECK(scales.numel()==ParameterCount(input.size(0),input.size(1),granularity,group_size), "qparam count mismatch");
  if (input.scalar_type()==at::kBFloat16) ValidateBFloat16Runtime(input);
  c10::cuda::CUDAGuard guard(input.device()); const int64_t packed_cols=(input.size(1)+1)/2;
  auto output=at::empty({input.size(0),packed_cols},input.options().dtype(at::kByte)); const int64_t total=input.size(0)*packed_cols; const int blocks=static_cast<int>((total+kThreads-1)/kThreads); const int64_t groups=GroupsPerRow(input.size(1),granularity,group_size); auto stream=at::cuda::getCurrentCUDAStream();
  if (input.scalar_type()==at::kHalf) QuantizeInt4Kernel<<<blocks,kThreads,0,stream>>>(reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),scales.data_ptr<float>(),output.data_ptr<uint8_t>(),input.size(0),input.size(1),packed_cols,granularity,group_size,groups);
  else QuantizeInt4Kernel<<<blocks,kThreads,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),scales.data_ptr<float>(),output.data_ptr<uint8_t>(),input.size(0),input.size(1),packed_cols,granularity,group_size,groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return output;
}

at::Tensor dequantize_int4_cuda(const at::Tensor& packed, const at::Tensor& scales, int64_t rows, int64_t cols, int64_t granularity, int64_t group_size, int64_t output_dtype) {
  TORCH_CHECK(packed.is_cuda() && packed.dim()==2 && packed.scalar_type()==at::kByte && packed.is_contiguous(), "packed must be CUDA uint8 rank-2");
  TORCH_CHECK(packed.size(0)==rows && packed.size(1)==(cols+1)/2, "packed INT4 shape mismatch"); ValidateGranularity(granularity,group_size);
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type()==at::kFloat && scales.is_contiguous() && scales.numel()==ParameterCount(rows,cols,granularity,group_size), "invalid scales");
  TORCH_CHECK(output_dtype==kOutputHalf || output_dtype==kOutputBFloat16, "invalid output dtype"); if (output_dtype==kOutputBFloat16) ValidateBFloat16Runtime(packed);
  c10::cuda::CUDAGuard guard(packed.device()); auto output=at::empty({rows,cols},packed.options().dtype(output_dtype==kOutputHalf?at::kHalf:at::kBFloat16)); const int64_t total=rows*cols; const int blocks=static_cast<int>((total+kThreads-1)/kThreads); const int64_t groups=GroupsPerRow(cols,granularity,group_size); auto stream=at::cuda::getCurrentCUDAStream();
  if (output_dtype==kOutputHalf) DequantizeInt4Kernel<<<blocks,kThreads,0,stream>>>(packed.data_ptr<uint8_t>(),scales.data_ptr<float>(),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),rows,cols,packed.size(1),granularity,group_size,groups);
  else DequantizeInt4Kernel<<<blocks,kThreads,0,stream>>>(packed.data_ptr<uint8_t>(),scales.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),rows,cols,packed.size(1),granularity,group_size,groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return output;
}
