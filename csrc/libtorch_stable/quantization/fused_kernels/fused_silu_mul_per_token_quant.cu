// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"

#include <limits>
#include <type_traits>

#include "../../../cuda_compat.h"

namespace vllm {

// One block per token. The bf16 SiLU result is staged in shared memory when
// the row fits, so the absmax and the quant read it from the SM instead of
// from a global bf16 activation buffer. Otherwise SiLU is recomputed.

template <typename scalar_out_t>
__device__ __forceinline__ float scale_from_absmax(float amax,
                                                   float const* scale_ub) {
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    return fmaxf(amax, 1e-10f) / 127.0f;
  } else {
    float capped = scale_ub != nullptr ? fminf(amax, *scale_ub) : amax;
    return fmaxf(capped / quant_type_max_v<scalar_out_t>,
                 min_scaling_factor<scalar_out_t>::val());
  }
}

template <typename scalar_out_t>
__device__ __forceinline__ scalar_out_t quant_one(float x, float amax,
                                                  float const* scale_ub) {
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    // Match per_token_quant_int8: round(x * (127 / absmax)).
    float capped = fmaxf(amax, 1e-10f);
    return float_to_int8_rn(x * (127.0f / capped));
  } else {
    return scaled_fp8_conversion<false, scalar_out_t>(
        x, scale_from_absmax<scalar_out_t>(amax, scale_ub));
  }
}

// Round through scalar_t the same way silu_and_mul does, so the absmax is
// taken over the values the unfused quant kernel would have reloaded.
template <typename scalar_t, int VEC>
__device__ __forceinline__ vec_n_t<scalar_t, VEC> silu_mul_vec(
    scalar_t const* __restrict__ gate, scalar_t const* __restrict__ up,
    int64_t elem, float* amax) {
  vec_n_t<scalar_t, VEC> g =
      *reinterpret_cast<vec_n_t<scalar_t, VEC> const*>(gate + elem);
  vec_n_t<scalar_t, VEC> u =
      *reinterpret_cast<vec_n_t<scalar_t, VEC> const*>(up + elem);
  vec_n_t<scalar_t, VEC> o;
  float step = 0.f;
#pragma unroll
  for (int j = 0; j < VEC; ++j) {
    float gf = static_cast<float>(g.val[j]);
    float silu = gf / (1.0f + expf(-gf));
    float y = static_cast<float>(static_cast<scalar_t>(silu)) *
              static_cast<float>(u.val[j]);
    o.val[j] = static_cast<scalar_t>(y);
    step = fmaxf(step, fabsf(static_cast<float>(o.val[j])));
  }
  *amax = fmaxf(*amax, step);
  return o;
}

template <typename scalar_t, typename scalar_out_t, int VEC>
__device__ __forceinline__ void store_quant(vec_n_t<scalar_t, VEC> v,
                                            scalar_out_t* __restrict__ dst,
                                            int64_t elem, float amax,
                                            float const* scale_ub) {
  q8_n_t<scalar_out_t, VEC> q;
#pragma unroll
  for (int j = 0; j < VEC; ++j) {
    q.val[j] =
        quant_one<scalar_out_t>(static_cast<float>(v.val[j]), amax, scale_ub);
  }
  *reinterpret_cast<q8_n_t<scalar_out_t, VEC>*>(dst + elem) = q;
}

__device__ __forceinline__ float block_reduce_max(float val, float* red) {
  int const lane = threadIdx.x % WARP_SIZE;
  int const warp = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, VLLM_SHFL_DOWN_SYNC(val, offset));
  }
  if (lane == 0) {
    red[warp] = val;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int const nwarps = blockDim.x / WARP_SIZE;
    float m = red[0];
    for (int i = 1; i < nwarps; ++i) {
      m = fmaxf(m, red[i]);
    }
    red[0] = m;
  }
  __syncthreads();
  return red[0];
}

template <typename scalar_t, typename scalar_out_t, int VEC, bool kStage>
__global__ void __launch_bounds__(512)
    silu_and_mul_per_token_quant_kernel(scalar_out_t* __restrict__ out,
                                        float* __restrict__ scales,
                                        scalar_t const* __restrict__ input,
                                        float const* __restrict__ scale_ub,
                                        int64_t hidden_size,
                                        int64_t scale_stride) {
  static_assert(VEC == 1 || VEC == 4 || VEC == 8, "unsupported vec size");

  extern __shared__ __align__(16) char smem[];
  __shared__ float red[32];
  if constexpr (!kStage) {
    // Recompute path keeps no row in shared memory.
    (void)smem;
  }

  int64_t const token_idx = static_cast<int64_t>(blockIdx.x);
  scalar_t const* gate = input + token_idx * hidden_size * 2;
  scalar_t const* up = gate + hidden_size;
  scalar_out_t* token_out = out + token_idx * hidden_size;

  int64_t const nvec = hidden_size / VEC;
  int64_t const tid = static_cast<int64_t>(threadIdx.x);
  int64_t const stride = static_cast<int64_t>(blockDim.x);

  float local = 0.f;
  for (int64_t i = tid; i < nvec; i += stride) {
    int64_t const elem = i * VEC;
    float step = 0.f;
    if constexpr (kStage) {
      vec_n_t<scalar_t, VEC> row =
          silu_mul_vec<scalar_t, VEC>(gate, up, elem, &step);
      scalar_t* staged = reinterpret_cast<scalar_t*>(smem);
      *reinterpret_cast<vec_n_t<scalar_t, VEC>*>(staged + elem) = row;
    } else {
      (void)silu_mul_vec<scalar_t, VEC>(gate, up, elem, &step);
    }
    local = fmaxf(local, step);
  }

  float const amax = block_reduce_max(local, red);
  if (threadIdx.x == 0) {
    scales[token_idx * scale_stride] =
        scale_from_absmax<scalar_out_t>(amax, scale_ub);
  }

  for (int64_t i = tid; i < nvec; i += stride) {
    int64_t const elem = i * VEC;
    if constexpr (kStage) {
      scalar_t* staged = reinterpret_cast<scalar_t*>(smem);
      vec_n_t<scalar_t, VEC> row =
          *reinterpret_cast<vec_n_t<scalar_t, VEC> const*>(staged + elem);
      store_quant<scalar_t, scalar_out_t, VEC>(row, token_out, elem, amax,
                                               scale_ub);
    } else {
      float ignored = 0.f;
      vec_n_t<scalar_t, VEC> row =
          silu_mul_vec<scalar_t, VEC>(gate, up, elem, &ignored);
      store_quant<scalar_t, scalar_out_t, VEC>(row, token_out, elem, amax,
                                               scale_ub);
    }
  }
}

}  // namespace vllm

namespace {

template <typename scalar_t, typename scalar_out_t, int VEC, bool kStage>
void launch_silu_and_mul_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales, float const* scale_ub, int64_t hidden_size,
    int64_t num_tokens, int64_t scale_stride, int threads, cudaStream_t stream,
    cudaDeviceProp const* prop) {
  auto* kernel =
      vllm::silu_and_mul_per_token_quant_kernel<scalar_t, scalar_out_t, VEC,
                                                kStage>;
  size_t smem = 0;
  if constexpr (kStage) {
    smem = static_cast<size_t>(hidden_size) * sizeof(scalar_t);
    if (smem + 1024 > static_cast<size_t>(prop->sharedMemPerBlock)) {
      cudaError_t err = VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
          reinterpret_cast<const void*>(kernel), static_cast<int>(smem));
      STD_TORCH_CHECK(err == cudaSuccess,
                      "Failed to opt in to dynamic shared memory: ",
                      cudaGetErrorString(err));
    }
  } else {
    (void)prop;
  }
  kernel<<<static_cast<unsigned int>(num_tokens), threads, smem, stream>>>(
      out.mutable_data_ptr<scalar_out_t>(), scales.mutable_data_ptr<float>(),
      input.const_data_ptr<scalar_t>(), scale_ub, hidden_size, scale_stride);
}

template <typename scalar_t, typename scalar_out_t>
void dispatch_silu_and_mul_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales, float const* scale_ub, int64_t hidden_size,
    int64_t num_tokens, int64_t scale_stride, cudaStream_t stream,
    cudaDeviceProp const* prop) {
  size_t const dyn = static_cast<size_t>(hidden_size) * sizeof(scalar_t);
  size_t const smem_limit = static_cast<size_t>(prop->sharedMemPerBlockOptin);
  // Leave headroom for the static warp-reduction buffer.
  bool const stage = dyn + 1024 <= smem_limit;
  int const threads = hidden_size >= 4096 ? 512 : 256;

  if (hidden_size % 8 == 0) {
    if (stage) {
      launch_silu_and_mul_per_token_quant<scalar_t, scalar_out_t, 8, true>(
          out, input, scales, scale_ub, hidden_size, num_tokens, scale_stride,
          threads, stream, prop);
    } else {
      launch_silu_and_mul_per_token_quant<scalar_t, scalar_out_t, 8, false>(
          out, input, scales, scale_ub, hidden_size, num_tokens, scale_stride,
          threads, stream, prop);
    }
  } else if (hidden_size % 4 == 0) {
    if (stage) {
      launch_silu_and_mul_per_token_quant<scalar_t, scalar_out_t, 4, true>(
          out, input, scales, scale_ub, hidden_size, num_tokens, scale_stride,
          threads, stream, prop);
    } else {
      launch_silu_and_mul_per_token_quant<scalar_t, scalar_out_t, 4, false>(
          out, input, scales, scale_ub, hidden_size, num_tokens, scale_stride,
          threads, stream, prop);
    }
  } else {
    if (stage) {
      launch_silu_and_mul_per_token_quant<scalar_t, scalar_out_t, 1, true>(
          out, input, scales, scale_ub, hidden_size, num_tokens, scale_stride,
          threads, stream, prop);
    } else {
      launch_silu_and_mul_per_token_quant<scalar_t, scalar_out_t, 1, false>(
          out, input, scales, scale_ub, hidden_size, num_tokens, scale_stride,
          threads, stream, prop);
    }
  }
}

}  // namespace

void silu_and_mul_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales,
    std::optional<torch::stable::Tensor> scale_ub) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  scales.is_contiguous());
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(input.dim() == 2 && out.dim() == 2,
                  "input and out must be 2D");

  int64_t const hidden_size = out.size(-1);
  int64_t const num_tokens = input.size(0);
  STD_TORCH_CHECK(hidden_size > 0, "hidden size must be positive");
  STD_TORCH_CHECK(input.size(-1) == hidden_size * 2,
                  "input last dim must be 2x output hidden size");
  STD_TORCH_CHECK(out.size(0) == num_tokens, "out token count mismatch");
  STD_TORCH_CHECK(scales.size(0) == num_tokens && scales.numel() == num_tokens,
                  "scales must be [num_tokens] or [num_tokens, 1]");
  STD_TORCH_CHECK(num_tokens <= static_cast<int64_t>(
                                    std::numeric_limits<unsigned int>::max()),
                  "num_tokens exceeds the kernel grid limit");

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type,
                    "scale_ub is only supported for FP8");
    STD_TORCH_CHECK(scale_ub->scalar_type() ==
                    torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(scale_ub->numel() == 1, "scale_ub must be a scalar");
    STD_TORCH_CHECK(scale_ub->get_device_index() == input.get_device_index(),
                    "scale_ub must be on the input device");
  }
  if (num_tokens == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  cudaDeviceProp const* prop = get_device_prop();
  int64_t const scale_stride = scales.stride(0);
  float const* scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_quant", [&] {
        using scalar_in_t = scalar_t;
        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_token_quant", [&] {
              using scalar_out_t = scalar_t;
              dispatch_silu_and_mul_per_token_quant<scalar_in_t, scalar_out_t>(
                  out, input, scales, scale_ub_ptr, hidden_size, num_tokens,
                  scale_stride, stream, prop);
            });
      });
}
