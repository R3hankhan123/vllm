#ifndef CPU_MICRO_GEMM_DLF16_HPP
#define CPU_MICRO_GEMM_DLF16_HPP

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
// accumulators take 16 of the 32 registers.
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
                                const int64_t ldc) {
    switch (m) {
      case 1:
        gemm_micro<1>(a_ptr, b_ptr, c_ptr, k, lda, ldc);
        break;
      case 2:
        gemm_micro<2>(a_ptr, b_ptr, c_ptr, k, lda, ldc);
        break;
      case 3:
        gemm_micro<3>(a_ptr, b_ptr, c_ptr, k, lda, ldc);
        break;
      case 4:
        gemm_micro<4>(a_ptr, b_ptr, c_ptr, k, lda, ldc);
        break;
    }
  }

  template <int32_t M>
  static void gemm_micro(const scalar_t* __restrict__ a_ptr,
                         const uint16_t* __restrict__ b_ptr,
                         float* __restrict__ c_ptr, const int32_t k,
                         const int64_t lda, const int64_t ldc) {
    static_assert(0 < M && M <= MaxMSize);
    vec_op::FP32Vec16 c_regs[M];

    const uint16_t* __restrict__ curr_b = b_ptr;
    for (int32_t k_idx = 0; k_idx < k; ++k_idx) {
      const vec_op::FP32Vec16 b_reg = load_dlf16x16(curr_b);
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        const __vector float a_reg =
            vec_splats(static_cast<float>(a_ptr[i * lda]));
        vec_op::unroll_loop<int32_t, 4>([&](int32_t j) {
          c_regs[i].reg.val[j] =
              vec_madd(a_reg, b_reg.reg.val[j], c_regs[i].reg.val[j]);
        });
      });
      ++a_ptr;
      curr_b += BRowStride;
    }

    vec_op::unroll_loop<int32_t, M>(
        [&](int32_t i) { c_regs[i].save(c_ptr + i * ldc); });
  }
};

}  // namespace cpu_micro_gemm

#endif
