#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float)>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x, param) * y;
  }
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

// Helper functions for safe type conversion between c10 and CUDA native types
#ifndef USE_ROCM
__device__ __forceinline__ __half c10_half_to_cuda_half(const c10::Half& h) {
  __half result;
  memcpy(&result, &h, sizeof(__half));
  return result;
}

__device__ __forceinline__ c10::Half cuda_half_to_c10_half(const __half& h) {
  c10::Half result;
  memcpy(&result, &h, sizeof(c10::Half));
  return result;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ __nv_bfloat16 c10_bfloat16_to_cuda_bfloat16(const c10::BFloat16& bf) {
  __nv_bfloat16 result;
  memcpy(&result, &bf, sizeof(__nv_bfloat16));
  return result;
}

__device__ __forceinline__ c10::BFloat16 cuda_bfloat16_to_c10_bfloat16(const __nv_bfloat16& bf) {
  c10::BFloat16 result;
  memcpy(&result, &bf, sizeof(c10::BFloat16));
  return result;
}
#endif  // __CUDA_ARCH__ >= 800
#endif  // !USE_ROCM

// Vector type traits for different scalar types
// Provides vectorized load/store operations for 2-element vectors
template <typename scalar_t>
struct VecTypeTraits {
  static constexpr bool can_vectorize = false;
};

template <>
struct VecTypeTraits<float> {
  static constexpr bool can_vectorize = true;
  using Vec2_t = float2;
  
  // Requires: ptr must be 8-byte aligned for float2 access
  __device__ static __forceinline__ Vec2_t load2(const float* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
  }
  
  // Requires: ptr must be 8-byte aligned for float2 access
  __device__ static __forceinline__ void store2(float* ptr, float x, float y) {
    float2 vec = make_float2(x, y);
    *reinterpret_cast<float2*>(ptr) = vec;
  }
};

#ifndef USE_ROCM
template <>
struct VecTypeTraits<c10::Half> {
  static constexpr bool can_vectorize = true;
  using Vec2_t = __half2;
  
  // Requires: ptr must be 4-byte aligned for __half2 access
  __device__ static __forceinline__ Vec2_t load2(const c10::Half* ptr) {
    return *reinterpret_cast<const __half2*>(ptr);
  }
  
  // Requires: ptr must be 4-byte aligned for __half2 access
  __device__ static __forceinline__ void store2(c10::Half* ptr, c10::Half x, c10::Half y) {
    __half hx = c10_half_to_cuda_half(x);
    __half hy = c10_half_to_cuda_half(y);
    __half2 vec = __halves2half2(hx, hy);
    *reinterpret_cast<__half2*>(ptr) = vec;
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
struct VecTypeTraits<c10::BFloat16> {
  static constexpr bool can_vectorize = true;
  using Vec2_t = __nv_bfloat162;
  
  // Requires: ptr must be 4-byte aligned for __nv_bfloat162 access
  __device__ static __forceinline__ Vec2_t load2(const c10::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
  }
  
  // Requires: ptr must be 4-byte aligned for __nv_bfloat162 access
  __device__ static __forceinline__ void store2(c10::BFloat16* ptr, c10::BFloat16 x, c10::BFloat16 y) {
    __nv_bfloat16 bx = c10_bfloat16_to_cuda_bfloat16(x);
    __nv_bfloat16 by = c10_bfloat16_to_cuda_bfloat16(y);
    __nv_bfloat162 vec = __halves2bfloat162(bx, by);
    *reinterpret_cast<__nv_bfloat162*>(ptr) = vec;
  }
};
#endif  // __CUDA_ARCH__ >= 800
#endif  // !USE_ROCM

// Helper to extract individual elements from packed vector types
template <typename scalar_t>
struct VecExtractor {
  using Vec2_t = typename VecTypeTraits<scalar_t>::Vec2_t;
  __device__ static __forceinline__ void extract2(Vec2_t vec, scalar_t& x, scalar_t& y);
};

template <>
struct VecExtractor<float> {
  __device__ static __forceinline__ void extract2(float2 vec, float& x, float& y) {
    x = vec.x;
    y = vec.y;
  }
};

#ifndef USE_ROCM
template <>
struct VecExtractor<c10::Half> {
  __device__ static __forceinline__ void extract2(__half2 vec, c10::Half& x, c10::Half& y) {
    __half hx = __low2half(vec);
    __half hy = __high2half(vec);
    x = cuda_half_to_c10_half(hx);
    y = cuda_half_to_c10_half(hy);
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
struct VecExtractor<c10::BFloat16> {
  __device__ static __forceinline__ void extract2(__nv_bfloat162 vec, c10::BFloat16& x, c10::BFloat16& y) {
    __nv_bfloat16 bx = __low2bfloat16(vec);
    __nv_bfloat16 by = __high2bfloat16(vec);
    x = cuda_bfloat16_to_c10_bfloat16(bx);
    y = cuda_bfloat16_to_c10_bfloat16(by);
  }
};
#endif  // __CUDA_ARCH__ >= 800
#endif  // !USE_ROCM

// Vectorized kernel implementation
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float),
          bool can_vectorize = VecTypeTraits<scalar_t>::can_vectorize>
struct SwigluOAIKernelImpl {
  __device__ static void run(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const int d, const float alpha, const float limit,
      const int64_t token_idx) {
    // Fallback scalar implementation
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
      const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);
      out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
    }
  }
};

// Specialized vectorized implementation
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
struct SwigluOAIKernelImpl<scalar_t, ACT_FN, true> {
  __device__ static void run(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const int d, const float alpha, const float limit,
      const int64_t token_idx) {
    
    using Traits = VecTypeTraits<scalar_t>;
    using Vec2_t = typename Traits::Vec2_t;
    using Extractor = VecExtractor<scalar_t>;
    
    constexpr int VEC_SIZE = 2;
    const int64_t vec_d = d / VEC_SIZE;
    const int64_t base_addr = token_idx * 2 * d;
    
    // Vectorized loop: process 2 output elements per iteration
    for (int64_t vec_idx = threadIdx.x; vec_idx < vec_d; vec_idx += blockDim.x) {
      // Input layout: [gate0, up0, gate1, up1, ...]
      // Load two (gate, up) pairs = 4 scalar_t values = 2 Vec2_t loads
      const int64_t input_offset = base_addr + vec_idx * 2 * VEC_SIZE;
      
      // Load first (gate, up) pair
      Vec2_t pair0 = Traits::load2(&input[input_offset]);
      scalar_t gate0, up0;
      Extractor::extract2(pair0, gate0, up0);
      
      // Load second (gate, up) pair
      Vec2_t pair1 = Traits::load2(&input[input_offset + 2]);
      scalar_t gate1, up1;
      Extractor::extract2(pair1, gate1, up1);
      
      // Compute activation for both pairs
      const scalar_t out0 = ACT_FN(gate0, up0, alpha, limit);
      const scalar_t out1 = ACT_FN(gate1, up1, alpha, limit);
      
      // Store 2 outputs as a vector
      const int64_t output_offset = token_idx * d + vec_idx * VEC_SIZE;
      Traits::store2(&out[output_offset], out0, out1);
    }
    
    // Scalar tail loop for remaining elements (when d is odd)
    const int64_t remainder_start = vec_d * VEC_SIZE;
    for (int64_t idx = remainder_start + threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t gate = VLLM_LDG(&input[base_addr + 2 * idx]);
      const scalar_t up = VLLM_LDG(&input[base_addr + 2 * idx + 1]);
      out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
    }
  }
};

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float alpha, const float limit) {
  const int64_t token_idx = blockIdx.x;
  SwigluOAIKernelImpl<scalar_t, ACT_FN>::run(out, input, d, alpha, limit, token_idx);
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)         \
  int d = input.size(-1) / 2;                                           \
  int64_t num_tokens = input.numel() / input.size(-1);                  \
  dim3 grid(num_tokens);                                                \
  dim3 block(std::min(d, 1024));                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));     \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();         \
  VLLM_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {       \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>> \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),      \
                                         input.data_ptr<scalar_t>(), d, \
                                         PARAM);                        \
      });

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                          \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {            \
        vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),             \
                                         input.data_ptr<scalar_t>(), d, ALPHA, \
                                         LIMIT);                               \
      });

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(vllm::fatrelu_kernel, threshold);
}
void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}
