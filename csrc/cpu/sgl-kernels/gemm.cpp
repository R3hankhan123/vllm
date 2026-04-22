// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

#include "common.h"
#include "vec.h"
#include "gemm.h"

// clang-format off

namespace {

// packed   layout:
//   quants {N, K}  int8_t
//   comp   {N}     int32_t
template <int BLOCK_N>
inline void s8s8_compensation(int8_t* __restrict__ packed, int K) {
#if defined(CPU_CAPABILITY_AVX512)
  constexpr int COLS = BLOCK_N / 16;
  __m512i vcomp[COLS];

  for (int col = 0; col < COLS; ++col) {
    vcomp[col] = _mm512_setzero_si512();
  }

  const int64_t offset = BLOCK_N * K;
  const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
  for (int k = 0; k < K / 4; ++k) {
    for (int col = 0; col < COLS; ++col) {
      __m512i vb = _mm512_loadu_si512((const __m512i *)(packed + k * BLOCK_N * 4 + col * 64));
      vcomp[col] = _mm512_dpbusd_epi32(vcomp[col], off, vb);
    }
  }

  for (int col = 0; col < COLS; ++col) {
    _mm512_storeu_si512((__m512i *)(packed + offset + col * 64), vcomp[col]);
  }
#else
  TORCH_CHECK(false, "s8s8_compensation not implemented!");
#endif
}

// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
template <typename packed_t>
inline void pack_vnni(packed_t* __restrict__ packed, const packed_t* __restrict__ weight, int N, int K) {
  const int VNNI_BLK = 2;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

template <>
inline void pack_vnni<int8_t>(int8_t* __restrict__ packed, const int8_t* __restrict__ weight, int N, int K) {
  constexpr int BLOCK_N = block_size_n();
  TORCH_CHECK(N == BLOCK_N);

  const int VNNI_BLK = 4;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
  s8s8_compensation<BLOCK_N>(packed, K);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_add_stub(scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) + fVec::loadu(bias + d);
    fVec data1 = fVec::loadu(input + d + fVec::size()) + fVec::loadu(bias + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
      const float* __restrict__ bias, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::BFloat16* __restrict__ B, at::BFloat16* __restrict__ C,
      const float* __restrict__ bias, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc[i] = _mm512_set1_ps(0.f);
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb; // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b_ptr = reinterpret_cast<const float*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(_mm512_loadu_si512(b_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      // for COLS = 1, 3 use 256bit store
      if constexpr (COLS % 2 == 0) {
        if constexpr (col % 2 == 0) {
          _mm512_storeu_si512(
              reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
              (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
        }
      } else {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(C + row * ldc + col * 16),
            (__m256i)(_mm512_cvtneps_pbh(vc[i])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#if defined(CPU_CAPABILITY_VXE)

// VXE micro-kernel for GEMM: C[M x 8] += A[M x K] * B[K x 8]
// B is in VNNI format [K/2, N, 2] where pairs are interleaved per column:
// Memory layout: [col0_k0, col0_k1, col1_k0, col1_k1, ..., col7_k0, col7_k1]
template <int M, typename scalar_t>
inline void gemm_micro_vxe_Mx8(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t lda, int64_t ldc, int64_t K) {
  
  static_assert(M >= 1 && M <= 4, "M must be in [1,4] for VXE micro-kernel");
  
  constexpr int PREFETCH_SIZE_K = 4;
  
  // Helper macros for code generation
  #define VXE_ROW_OP(OP) OP(0) OP(1) OP(2) OP(3)
  #define VXE_IF_M(i) if constexpr (M > (i))
  
  // Define A row pointers
  #define VXE_DECL_A(i) const scalar_t* a##i = A + (i) * lda;
  VXE_ROW_OP(VXE_DECL_A)
  #undef VXE_DECL_A
  
  // Define accumulators (2 vectors per row, each handles 4 float elements)
  #define VXE_DECL_ACC(i) __vector float acc##i##_0, acc##i##_1;
  VXE_ROW_OP(VXE_DECL_ACC)
  #undef VXE_DECL_ACC
  
  // Initialize accumulators with bias
  #define VXE_INIT_ACC(i)                                                \
    VXE_IF_M(i) {                                                        \
      if (bias != nullptr) {                                             \
        acc##i##_0 = vec_xl((long long)0, const_cast<float*>(bias + 0)); \
        acc##i##_1 = vec_xl((long long)0, const_cast<float*>(bias + 4)); \
      } else {                                                           \
        acc##i##_0 = vec_splats(0.0f);                                   \
        acc##i##_1 = vec_splats(0.0f);                                   \
      }                                                                  \
    }
  VXE_ROW_OP(VXE_INIT_ACC)
  #undef VXE_INIT_ACC
  
  // Helper: Load and deinterleave VNNI BF16 pairs, convert to FP32
  // Optimized for big-endian: BF16 at upper 16 bits of FP32
  auto load_vnni_bf16_as_fp32 = [](const at::BFloat16* p, 
                                    __vector float& b0_k0, __vector float& b1_k0,
                                    __vector float& b0_k1, __vector float& b1_k1) {
    __vector unsigned short raw0 = vec_xl((long long)0, (unsigned short*)p);
    __vector unsigned short raw1 = vec_xl((long long)16, (unsigned short*)p);

    // Deinterleave using single permute operation
    const __vector unsigned char perm_even = {
        0,1, 4,5, 8,9, 12,13, 16,17, 20,21, 24,25, 28,29};
    const __vector unsigned char perm_odd = {
        2,3, 6,7, 10,11, 14,15, 18,19, 22,23, 26,27, 30,31};
    
    __vector unsigned short k0_all = vec_perm(raw0, raw1, perm_even);
    __vector unsigned short k1_all = vec_perm(raw0, raw1, perm_odd);
    
    // Convert BF16 to FP32: BF16 at upper 16 bits on big-endian s390x
    // Use vec_mergeh/vec_mergel with swapped operands for efficiency
    __vector unsigned short zeros = vec_splat_u16(0);
    b0_k0 = (__vector float)vec_mergeh(zeros, k0_all);
    b1_k0 = (__vector float)vec_mergel(zeros, k0_all);
    b0_k1 = (__vector float)vec_mergeh(zeros, k1_all);
    b1_k1 = (__vector float)vec_mergel(zeros, k1_all);
  };
  
  // Helper: Load and deinterleave VNNI FP16 pairs, convert to FP32 (vectorized)
  auto load_vnni_fp16_as_fp32 = [](const c10::Half* p,
                                    __vector float& b0_k0, __vector float& b1_k0,
                                    __vector float& b0_k1, __vector float& b1_k1) {
    // Load 16 FP16 values (interleaved VNNI format)
    __vector unsigned short raw0 = vec_xl((long long)0, (unsigned short*)p);
    __vector unsigned short raw1 = vec_xl((long long)16, (unsigned short*)p);
    
    // Deinterleave using optimized permute
    const __vector unsigned char perm_even = {
        0,1, 4,5, 8,9, 12,13, 16,17, 20,21, 24,25, 28,29};
    const __vector unsigned char perm_odd = {
        2,3, 6,7, 10,11, 14,15, 18,19, 22,23, 26,27, 30,31};
    
    __vector unsigned short k0_all = vec_perm(raw0, raw1, perm_even);
    __vector unsigned short k1_all = vec_perm(raw0, raw1, perm_odd);
    
    // Convert FP16 to FP32 using vectorized operations
    // Split into high/low 4 elements for processing
    __vector unsigned short zeros = vec_splat_u16(0);
    
    // k0: Expand to 32-bit words
    __vector unsigned int k0_hi = (__vector unsigned int)vec_mergeh(k0_all, zeros);
    __vector unsigned int k0_lo = (__vector unsigned int)vec_mergel(k0_all, zeros);
    __vector unsigned int k1_hi = (__vector unsigned int)vec_mergeh(k1_all, zeros);
    __vector unsigned int k1_lo = (__vector unsigned int)vec_mergel(k1_all, zeros);
    
    // IEEE 754 FP16: [sign:1][exp:5][mant:10] -> FP32: [sign:1][exp:8][mant:23]
    // Strategy: shift, mask, adjust exponent, combine
    const __vector unsigned int sign_mask = vec_splats(0x8000u);
    const __vector unsigned int exp_mask = vec_splats(0x7C00u);
    const __vector unsigned int mant_mask = vec_splats(0x03FFu);
    const __vector unsigned int exp_adjust = vec_splats(112u);  // FP32_bias - FP16_bias = 127 - 15
    const __vector unsigned int fp32_inf = vec_splats(0x7F800000u);
    
    auto convert_fp16_vec = [](__vector unsigned int h16) -> __vector float {
      // Extract components
      __vector unsigned int sign = vec_sl(vec_and(h16, vec_splats(0x8000u)), vec_splats(16u));
      __vector unsigned int exp = vec_sr(vec_and(h16, vec_splats(0x7C00u)), vec_splats(10u));
      __vector unsigned int mant = vec_sl(vec_and(h16, vec_splats(0x03FFu)), vec_splats(13u));
      
      // Check for zero/subnormal (exp == 0)
      __vector bool int is_zero = vec_cmpeq(exp, vec_splats(0u));
      // Check for inf/nan (exp == 0x1F)
      __vector bool int is_special = vec_cmpeq(exp, vec_splats(0x1Fu));
      
      // Normal case: sign | ((exp + 112) << 23) | mant
      __vector unsigned int exp_shifted = vec_sl(vec_add(exp, vec_splats(112u)), vec_splats(23u));
      __vector unsigned int normal = vec_or(vec_or(sign, exp_shifted), mant);
      
      // Special case: sign | 0x7F800000 | mant (inf/nan)
      __vector unsigned int special = vec_or(vec_or(sign, vec_splats(0x7F800000u)), mant);
      
      // Select: zero -> sign, special -> special, else -> normal
      __vector unsigned int result = vec_sel(normal, sign, is_zero);
      result = vec_sel(result, special, is_special);
      
      return (__vector float)result;
    };
    
    b0_k0 = convert_fp16_vec(k0_hi);
    b1_k0 = convert_fp16_vec(k0_lo);
    b0_k1 = convert_fp16_vec(k1_hi);
    b1_k1 = convert_fp16_vec(k1_lo);
  };
  
  // VNNI format: [K/2, 8, 2] - process K in pairs
  const int64_t K2 = K / 2;
  const scalar_t* b_ptr = B;
  
  // Main loop: process K dimension in pairs (VNNI format)
  for (int64_t k = 0; k < K2; ++k) {
    // Prefetch further ahead for better memory latency hiding
    if (k + PREFETCH_SIZE_K < K2) [[likely]] {
      __builtin_prefetch(b_ptr + PREFETCH_SIZE_K * 16, 0, 3);  // Temporal locality
    }
    
    __vector float b0_k0, b1_k0, b0_k1, b1_k1;
    if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      load_vnni_bf16_as_fp32(reinterpret_cast<const at::BFloat16*>(b_ptr),
                             b0_k0, b1_k0, b0_k1, b1_k1);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      load_vnni_fp16_as_fp32(reinterpret_cast<const c10::Half*>(b_ptr),
                             b0_k0, b1_k0, b0_k1, b1_k1);
    } else {
      // FP32: already deinterleaved
      b0_k0 = vec_xl((long long)0, (float*)b_ptr);
      b1_k0 = vec_xl((long long)0, (float*)(b_ptr + 4));
      b0_k1 = vec_xl((long long)0, (float*)(b_ptr + 8));
      b1_k1 = vec_xl((long long)0, (float*)(b_ptr + 12));
    }
    b_ptr += 16;  // Advance by 16 elements (8 cols × 2 K-pairs)
    
    #define VXE_STEP_K0(i)                                               \
      VXE_IF_M(i) {                                                      \
        float a_val = static_cast<float>(a##i[k * 2]);                   \
        __vector float a_broad = vec_splats(a_val);                      \
        acc##i##_0 = vec_madd(b0_k0, a_broad, acc##i##_0);               \
        acc##i##_1 = vec_madd(b1_k0, a_broad, acc##i##_1);               \
      }
    VXE_ROW_OP(VXE_STEP_K0)
    #undef VXE_STEP_K0
    
    #define VXE_STEP_K1(i)                                               \
      VXE_IF_M(i) {                                                      \
        float a_val = static_cast<float>(a##i[k * 2 + 1]);               \
        __vector float a_broad = vec_splats(a_val);                      \
        acc##i##_0 = vec_madd(b0_k1, a_broad, acc##i##_0);               \
        acc##i##_1 = vec_madd(b1_k1, a_broad, acc##i##_1);               \
      }
    VXE_ROW_OP(VXE_STEP_K1)
    #undef VXE_STEP_K1
  }
  
  // Handle odd K tail (if K % 2 != 0)
  if (K % 2 != 0) [[unlikely]] {
    __vector float b0, b1;
    if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      __vector unsigned short raw = vec_xl((long long)0, (unsigned short*)b_ptr);
      __vector unsigned short zeros = vec_splat_u16(0);
      b0 = (__vector float)vec_mergeh(zeros, raw);
      b1 = (__vector float)vec_mergel(zeros, raw);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      // Vectorized FP16 conversion for tail
      __vector unsigned short fp16_vec = vec_xl((long long)0, (unsigned short*)b_ptr);
      __vector unsigned short zeros = vec_splat_u16(0);
      __vector unsigned int hi = (__vector unsigned int)vec_mergeh(fp16_vec, zeros);
      __vector unsigned int lo = (__vector unsigned int)vec_mergel(fp16_vec, zeros);
      
      auto convert_fp16_tail = [](__vector unsigned int h16) -> __vector float {
        __vector unsigned int sign = vec_sl(vec_and(h16, vec_splats(0x8000u)), vec_splats(16u));
        __vector unsigned int exp = vec_sr(vec_and(h16, vec_splats(0x7C00u)), vec_splats(10u));
        __vector unsigned int mant = vec_sl(vec_and(h16, vec_splats(0x03FFu)), vec_splats(13u));
        __vector bool int is_zero = vec_cmpeq(exp, vec_splats(0u));
        __vector bool int is_special = vec_cmpeq(exp, vec_splats(0x1Fu));
        __vector unsigned int exp_shifted = vec_sl(vec_add(exp, vec_splats(112u)), vec_splats(23u));
        __vector unsigned int normal = vec_or(vec_or(sign, exp_shifted), mant);
        __vector unsigned int special = vec_or(vec_or(sign, vec_splats(0x7F800000u)), mant);
        __vector unsigned int result = vec_sel(normal, sign, is_zero);
        return (__vector float)vec_sel(result, special, is_special);
      };
      
      b0 = convert_fp16_tail(hi);
      b1 = convert_fp16_tail(lo);
    } else {
      // FP32: direct load
      b0 = vec_xl((long long)0, (float*)b_ptr);
      b1 = vec_xl((long long)0, (float*)(b_ptr + 4));
    }
    
    #define VXE_TAIL(i)                                                  \
      VXE_IF_M(i) {                                                      \
        float a_val = static_cast<float>(a##i[K - 1]);                   \
        __vector float a_broad = vec_splats(a_val);                      \
        acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0);                  \
        acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1);                  \
      }
    VXE_ROW_OP(VXE_TAIL)
    #undef VXE_TAIL
  }
  
  // Vectorized store: FP32 accumulator -> BF16/FP16 output
  #define VXE_STORE_ACC(i)                                               \
    VXE_IF_M(i) {                                                        \
      if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {            \
        __vector unsigned int i0 = (__vector unsigned int)acc##i##_0;    \
        __vector unsigned int i1 = (__vector unsigned int)acc##i##_1;    \
        __vector unsigned short packed = vec_pack(i0, i1);               \
        vec_xst(packed, (long long)0, (unsigned short*)(C + (i) * ldc)); \
      } else {                                                           \
        alignas(16) float tmp[8];                                        \
        vec_xst(acc##i##_0, (long long)0, tmp);                          \
        vec_xst(acc##i##_1, (long long)0, tmp + 4);                      \
        for (int n = 0; n < 8; ++n) {                                    \
          C[(i) * ldc + n] = static_cast<scalar_t>(tmp[n]);              \
        }                                                                \
      }                                                                  \
    }
  VXE_ROW_OP(VXE_STORE_ACC)
  #undef VXE_STORE_ACC
  
  #undef VXE_IF_M
  #undef VXE_ROW_OP
}

// VXE tinygemm_kernel_nn specialization for BF16
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::BFloat16* __restrict__ B, at::BFloat16* __restrict__ C,
      const float* __restrict__ bias, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    
    static_assert(BLOCK_N % 8 == 0, "VXE kernel requires BLOCK_N to be multiple of 8");
    constexpr int ROWS = BLOCK_M;
    
    // B is in VNNI format [K/2, N, 2], so for column block:
    const int64_t K2 = K / 2;
    
    // Process in blocks of 8 columns
    for (int col_blk = 0; col_blk < BLOCK_N; col_blk += 8) {
      // B offset in VNNI format: each column block is K elements arranged as [K/2][8][2]
      const at::BFloat16* b_col = B + col_blk * K;
      at::BFloat16* c_col = C + col_blk;
      const float* bias_col = has_bias ? (bias + col_blk) : nullptr;
      
      // Direct call using ROWS template parameter
      gemm_micro_vxe_Mx8<ROWS, at::BFloat16>(
          A, b_col, c_col, bias_col, lda, ldc, K);
    }
  }
};

// VXE tinygemm_kernel_nn specialization for FP16
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::Half, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::Half* __restrict__ A, const at::Half* __restrict__ B, at::Half* __restrict__ C,
      const float* __restrict__ bias, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    
    static_assert(BLOCK_N % 8 == 0, "VXE kernel requires BLOCK_N to be multiple of 8");
    constexpr int ROWS = BLOCK_M;
    
    // Process in blocks of 8 columns
    for (int col_blk = 0; col_blk < BLOCK_N; col_blk += 8) {
      const at::Half* b_col = B + col_blk * K;
      at::Half* c_col = C + col_blk;
      const float* bias_col = has_bias ? (bias + col_blk) : nullptr;
      
      gemm_micro_vxe_Mx8<ROWS, at::Half>(
          A, b_col, c_col, bias_col, lda, ldc, K);
    }
  }
};

#endif  // CPU_CAPABILITY_VXE

#if defined(CPU_CAPABILITY_AVX512)
#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, \
        has_bias ? bias + nb_start : nullptr, K, lda, ldb, ldc);
#elif defined(CPU_CAPABILITY_VXE)
// VXE: B is in VNNI format, offset by nb_start * K
#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start * K, C + mb_start * ldc + nb_start, \
        has_bias ? bias + nb_start : nullptr, K, lda, ldb, ldc);
#else
#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start, C + mb_start * ldc + nb_start,     \
        has_bias ? bias + nb_start : nullptr, K, lda, nb_size, ldc);
#endif

template <typename scalar_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
      float* __restrict__ Ctmp, const float* __restrict__ bias,
      int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(
        M, N, K, lda, ldb, BLOCK_N, /* add_C */false,
        A, B, Ctmp);

    // copy from Ctmp to C
    for (int64_t m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {

  if (brg) {
    brgemm<scalar_t, has_bias>::apply(
        A, B, C, Ctmp, bias,
        M, N, K, lda, ldb, ldc);
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x14: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        // mb_size = 2
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x24: LAUNCH_TINYGEMM_KERNEL_NN(2, 64); break;
        // mb_size = 3
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x34: LAUNCH_TINYGEMM_KERNEL_NN(3, 64); break;
        // mb_size = 4
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        case 0x44: LAUNCH_TINYGEMM_KERNEL_NN(4, 64); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", nb_size);
      }
    }
  }
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // Use VXE kernels for BF16/FP16 when M <= 4, otherwise use brgemm
  const bool use_brgemm = (M > 4) || (!std::is_same_v<scalar_t, at::BFloat16> && !std::is_same_v<scalar_t, at::Half>);

  // l2 cache block for n
  int64_t cache_blocks_nb = get_cache_blocks<scalar_t>(BLOCK_N, K);

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t begin_mb, int64_t end_mb, int64_t begin_nb, int64_t end_nb) {

      // for brgemm, use float32 for accumulate
      alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

      for (int64_t nbb = begin_nb; nbb < end_nb; nbb += cache_blocks_nb) {
      for (int64_t mb = begin_mb; mb < end_mb; ++mb) {
      for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, end_nb); ++nb) {

        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ mat1 + mb_start * mat1_strideM,
            /*   B */ mat2 + nb_start * K /* nb * BLOCK_N * K */,
            /*   C */ out + mb_start * out_strideM + nb_start,
            /* Ctmp*/ Ctmp,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ mat1_strideM,
            /* ldb */ nb_size,
            /* ldc */ out_strideM,
            /* brg */ use_brgemm);
      }}}

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

} // anonymous namespace

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
    float* __restrict__ Ctmp, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc, bool brg) {
  tinygemm_kernel<scalar_t, false>(A, B, C, Ctmp, nullptr, M, N, K, lda, ldb, ldc, brg);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE)                                             \
    template void tinygemm_kernel<TYPE>(                                                \
        const TYPE* __restrict__ A, const TYPE* __restrict__ B, TYPE* __restrict__ C,   \
        float* __restrict__ Ctmp, int64_t M, int64_t N, int64_t K, int64_t lda,         \
        int64_t ldb, int64_t ldc, bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // for 3d moe weights
  // weight : [E, OC, IC]
  //     w1 : [E, 2N,  K]
  //     w2 : [E,  K,  N]
  CHECK_INPUT(weight);

  const int64_t ndim = weight.ndimension();
  TORCH_CHECK(ndim == 2 || ndim == 3, "expect weight to be 2d or 3d, got ", ndim, "d tensor.");
  const auto st = weight.scalar_type();
  const int64_t E = ndim == 3 ? weight.size(0) : 1;
  const int64_t OC = ndim == 3 ? weight.size(1) : weight.size(0);
  const int64_t IC = ndim == 3 ? weight.size(2) : weight.size(1);

  // we handle 2 TILE_N at a time.
  TORCH_CHECK(OC % TILE_N == 0, "invalid weight out features ", OC);
  TORCH_CHECK(IC % TILE_K == 0, "invalid weight input features ", IC);

  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t NB = div_up(OC, BLOCK_N);

  // use phony sizes here [E, OC, IC], for each [E], [OC, IC] -> [IC / 2, OC, 2]
  auto packed_weight = at::empty({}, weight.options());
  const int64_t stride = OC * IC;

  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf || st == at::kChar || st == at::kFloat8_e4m3fn,
      "expect weight to be bfloat16, float16, int8 or fp8_e4m3.");

  CPU_DISPATCH_PACKED_TYPES(st, [&] {
    // adjust most inner dimension size
    const int packed_row_size = get_row_size<packed_t>(IC);
    auto sizes = weight.sizes().vec();
    sizes[ndim - 1] = packed_row_size;
    packed_weight.resize_(sizes);

    const packed_t* w_data = weight.data_ptr<packed_t>();
    packed_t* packed_data = packed_weight.data_ptr<packed_t>();

    // parallel on {E, NB}
    at::parallel_for(0, E * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t e{0}, nb{0};
      data_index_init(begin, e, E, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);

        int64_t n = nb * BLOCK_N;
        int64_t n_size = std::min(BLOCK_N, OC - n);
        pack_vnni<packed_t>(
            packed_data + e * OC * packed_row_size + n * packed_row_size,
            w_data + e * stride + n * IC,
            n_size,
            IC);

        // move to the next index
        data_index_step(e, E, nb, NB);
      }
    });
  });
  return packed_weight;
}

// mat1 : [M, K]
// mat2 : [N, K]
// bias : [N]
// out  : [M, N]
//
at::Tensor weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2,
    const std::optional<at::Tensor>& bias, bool is_vnni) {
  RECORD_FUNCTION(
    "sgl-kernel::weight_packed_linear", std::vector<c10::IValue>({mat1, mat2, bias}));

  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1);
  CHECK_EQ(mat1.size(1), K);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  auto out = at::empty({M, N}, mat1.options());

  // strides
  int64_t mat1_strideM = mat1.stride(0);
  int64_t out_strideM = out.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(mat1.scalar_type(), "weight_packed_linear_kernel_impl", [&] {
    weight_packed_linear_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<scalar_t>(),
        bias_data,
        M,
        N,
        K,
        mat1_strideM,
        out_strideM);
  });

  return out;
}
