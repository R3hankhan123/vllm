#ifndef CPU_MICRO_GEMM_ZDNN_HPP
#define CPU_MICRO_GEMM_ZDNN_HPP

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

namespace cpu_micro_gemm {

// Loads 16 DLFLOAT16 (NNP-data-type-1) values as fp32.
FORCE_INLINE vec_op::FP32Vec16 load_dlf16x16(const uint16_t* __restrict__ ptr) {
  // VCLFNH / VCLFNL from NNP-data-type-1 (source format 0).
  vec_op::f32x4x4_t reg;
  for (int i = 0; i < 2; ++i) {
    const __vector unsigned short in = vec_xl(i * 16, ptr);
    reg.val[2 * i] = vec_extend_to_fp32_hi(in, 0);
    reg.val[2 * i + 1] = vec_extend_to_fp32_lo(in, 0);
  }
  return vec_op::FP32Vec16(reg);
}

// 4-4-16 pattern, [4, K] @ [K, 16]. B is a 16-column tile of an NNPA
// data-layout-format 0 (stickified) DLFLOAT16 matrix, whose rows are 64
// elements apart. An FP32Vec16 is 4 vector registers on VXE, so 4 rows of
// accumulators take 16 of the 32 registers. With accum, C is added to rather
// than overwritten.
template <typename scalar_t>
class TileGemmDLF16 {
 public:
  static constexpr int32_t MaxMSize = 4;
  static constexpr int32_t NSize = 16;
  static constexpr int64_t BRowStride = 64;

  FORCE_INLINE static void gemm(const scalar_t* __restrict__ a_ptr,
                                const uint16_t* __restrict__ b_ptr,
                                float* __restrict__ c_ptr, const int32_t m,
                                const int32_t k, const int64_t lda,
                                const int64_t ldc, const bool accum) {
    switch (m) {
      case 1:
        gemm_micro<1>(a_ptr, b_ptr, c_ptr, k, lda, ldc, accum);
        break;
      case 2:
        gemm_micro<2>(a_ptr, b_ptr, c_ptr, k, lda, ldc, accum);
        break;
      case 3:
        gemm_micro<3>(a_ptr, b_ptr, c_ptr, k, lda, ldc, accum);
        break;
      case 4:
        gemm_micro<4>(a_ptr, b_ptr, c_ptr, k, lda, ldc, accum);
        break;
    }
  }

  template <int32_t M>
  static void gemm_micro(const scalar_t* __restrict__ a_ptr,
                         const uint16_t* __restrict__ b_ptr,
                         float* __restrict__ c_ptr, const int32_t k,
                         const int64_t lda, const int64_t ldc,
                         const bool accum) {
    static_assert(0 < M && M <= MaxMSize);
    // Plain unrolled loops: lambdas capturing the accumulators make GCC keep
    // them in memory, storing all of them on every K step.
    __vector float acc[M][4];
#pragma GCC unroll 4
    for (int32_t i = 0; i < M; ++i) {
#pragma GCC unroll 4
      for (int32_t j = 0; j < 4; ++j) {
        acc[i][j] = vec_splats(0.0f);
      }
    }

    const uint16_t* __restrict__ curr_b = b_ptr;
    for (int32_t k_idx = 0; k_idx < k; ++k_idx) {
      const vec_op::FP32Vec16 b_reg = load_dlf16x16(curr_b);
#pragma GCC unroll 4
      for (int32_t i = 0; i < M; ++i) {
        const __vector float a_reg =
            vec_splats(static_cast<float>(a_ptr[i * lda]));
#pragma GCC unroll 4
        for (int32_t j = 0; j < 4; ++j) {
          acc[i][j] = vec_madd(a_reg, b_reg.reg.val[j], acc[i][j]);
        }
      }
      ++a_ptr;
      curr_b += BRowStride;
    }

#pragma GCC unroll 4
    for (int32_t i = 0; i < M; ++i) {
#pragma GCC unroll 4
      for (int32_t j = 0; j < 4; ++j) {
        auto* c = reinterpret_cast<__vector float*>(c_ptr + i * ldc + j * 4);
        *c = accum ? *c + acc[i][j] : acc[i][j];
      }
    }
  }
};

}  // namespace cpu_micro_gemm

#endif
