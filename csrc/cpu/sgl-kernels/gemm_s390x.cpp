// s390x VXE-optimized GEMM kernels for FP32 and BF16
// High-performance microkernel implementation using IBM Z Vector Extension Facility
// Optimized for maximum FMA throughput with aggressive register blocking
//
// Performance characteristics:
// - 8x16 microkernel: 128 FLOPs per K iteration, ~32 VXE registers in use
// - B is row-major [K × N]. For each kk we load B[kk, :] which is contiguous
// - Aggressive K-loop unrolling (8x) to maximize ILP
// - Prefetching tuned for L1/L2 cache hierarchy
// - Minimal memory traffic: loads A once, streams B, accumulates C in registers

#include "common.h"
#include "../cpu_types_vxe.hpp"
#include <vecintrin.h>
#include <torch/types.h>
#include <cstring>
#include <algorithm>

// clang-format off

#if defined(__s390x__)

namespace {

using namespace vec_op;

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int BLOCK_M = 8;          // M-dimension microkernel size
constexpr int BLOCK_N = 16;         // N-dimension microkernel size  
constexpr int BLOCK_K = 128;        // K-dimension blocking for cache
constexpr int K_UNROLL = 8;         // K-loop unrolling factor
constexpr int VEC_SIZE = 4;         // FP32 elements per 128-bit VXE vector

constexpr int PREFETCH_A = 256;     // Prefetch distance for A (bytes)
constexpr int PREFETCH_B = 512;     // Prefetch distance for B (bytes)

// Note: We keep B in row-major layout [K×N] for optimal memory access.
// For each K iteration, we load B[k, :] which is contiguous in memory.

// ============================================================================
// FP32 Microkernel: 8x16 (BLOCK_M=8, BLOCK_N=16)
// ============================================================================
// Computes: C[8×16] += A[8×K] × B[K×16]
// 
// Register allocation:
// - C accumulators: 32 VXE vectors (8 rows × 4 vec/row, each vec = 4 FP32)
// - A broadcasts: 8 VXE vectors (reused across K iterations)
// - B vectors: 4 VXE vectors (loaded per K iteration)
// Total: ~44 VXE registers (well within z14+ capacity)
//
// Performance: 128 FLOPs per K iteration (8×16 = 128 multiply-adds)

static inline void gemm_s390x_kernel_8x16(
    const float* A,      // [8 × K] row-major
    const float* B,      // [K × N] row-major (N >= 16)
    float* C,            // [8 × 16] row-major  
    int K,
    int lda,             // leading dimension of A (=K typically)
    int ldb,             // leading dimension of B (=N typically)
    int ldc) {           // leading dimension of C (=N typically)

  // C accumulator registers: 8 rows × 4 vectors/row = 32 vectors
  __vector float c00, c01, c02, c03;  // row 0
  __vector float c10, c11, c12, c13;  // row 1
  __vector float c20, c21, c22, c23;  // row 2
  __vector float c30, c31, c32, c33;  // row 3
  __vector float c40, c41, c42, c43;  // row 4
  __vector float c50, c51, c52, c53;  // row 5
  __vector float c60, c61, c62, c63;  // row 6
  __vector float c70, c71, c72, c73;  // row 7

  // Load existing C values
  c00 = vec_xl(0, C + 0*ldc +  0); c01 = vec_xl(0, C + 0*ldc +  4);
  c02 = vec_xl(0, C + 0*ldc +  8); c03 = vec_xl(0, C + 0*ldc + 12);
  
  c10 = vec_xl(0, C + 1*ldc +  0); c11 = vec_xl(0, C + 1*ldc +  4);
  c12 = vec_xl(0, C + 1*ldc +  8); c13 = vec_xl(0, C + 1*ldc + 12);
  
  c20 = vec_xl(0, C + 2*ldc +  0); c21 = vec_xl(0, C + 2*ldc +  4);
  c22 = vec_xl(0, C + 2*ldc +  8); c23 = vec_xl(0, C + 2*ldc + 12);
  
  c30 = vec_xl(0, C + 3*ldc +  0); c31 = vec_xl(0, C + 3*ldc +  4);
  c32 = vec_xl(0, C + 3*ldc +  8); c33 = vec_xl(0, C + 3*ldc + 12);
  
  c40 = vec_xl(0, C + 4*ldc +  0); c41 = vec_xl(0, C + 4*ldc +  4);
  c42 = vec_xl(0, C + 4*ldc +  8); c43 = vec_xl(0, C + 4*ldc + 12);
  
  c50 = vec_xl(0, C + 5*ldc +  0); c51 = vec_xl(0, C + 5*ldc +  4);
  c52 = vec_xl(0, C + 5*ldc +  8); c53 = vec_xl(0, C + 5*ldc + 12);
  
  c60 = vec_xl(0, C + 6*ldc +  0); c61 = vec_xl(0, C + 6*ldc +  4);
  c62 = vec_xl(0, C + 6*ldc +  8); c63 = vec_xl(0, C + 6*ldc + 12);
  
  c70 = vec_xl(0, C + 7*ldc +  0); c71 = vec_xl(0, C + 7*ldc +  4);
  c72 = vec_xl(0, C + 7*ldc +  8); c73 = vec_xl(0, C + 7*ldc + 12);

  // Main K loop with 8x unrolling
  int k = 0;
  for (; k + K_UNROLL <= K; k += K_UNROLL) {
    
    // Prefetch ahead for next K block
    int kk_pref = k + K_UNROLL;
    if (kk_pref < K) {
      for (int mrow = 0; mrow < 8; ++mrow) {
        __builtin_prefetch(A + mrow * lda + kk_pref, 0, 3);
      }
      __builtin_prefetch(B + kk_pref * ldb, 0, 3);
    }

    // Process K_UNROLL=8 iterations
    for (int ku = 0; ku < K_UNROLL; ku++) {
      int kk = k + ku;
      
      // Load A[0:8, kk] - 8 scalars, broadcast to vectors
      __vector float a0 = vec_splats(A[0*lda + kk]);
      __vector float a1 = vec_splats(A[1*lda + kk]);
      __vector float a2 = vec_splats(A[2*lda + kk]);
      __vector float a3 = vec_splats(A[3*lda + kk]);
      __vector float a4 = vec_splats(A[4*lda + kk]);
      __vector float a5 = vec_splats(A[5*lda + kk]);
      __vector float a6 = vec_splats(A[6*lda + kk]);
      __vector float a7 = vec_splats(A[7*lda + kk]);

      // Load B[kk, 0:16] - row kk of B (row-major, contiguous)
      const float* b_row = B + kk * ldb;
      __vector float b0 = vec_xl(0, b_row +  0);  // cols 0..3
      __vector float b1 = vec_xl(0, b_row +  4);  // cols 4..7
      __vector float b2 = vec_xl(0, b_row +  8);  // cols 8..11
      __vector float b3 = vec_xl(0, b_row + 12);  // cols 12..15

      // FMA: C[m, n] += A[m, k] * B[k, n]
      // Row 0
      c00 = vec_madd(a0, b0, c00);
      c01 = vec_madd(a0, b1, c01);
      c02 = vec_madd(a0, b2, c02);
      c03 = vec_madd(a0, b3, c03);
      
      // Row 1
      c10 = vec_madd(a1, b0, c10);
      c11 = vec_madd(a1, b1, c11);
      c12 = vec_madd(a1, b2, c12);
      c13 = vec_madd(a1, b3, c13);
      
      // Row 2
      c20 = vec_madd(a2, b0, c20);
      c21 = vec_madd(a2, b1, c21);
      c22 = vec_madd(a2, b2, c22);
      c23 = vec_madd(a2, b3, c23);
      
      // Row 3
      c30 = vec_madd(a3, b0, c30);
      c31 = vec_madd(a3, b1, c31);
      c32 = vec_madd(a3, b2, c32);
      c33 = vec_madd(a3, b3, c33);
      
      // Row 4
      c40 = vec_madd(a4, b0, c40);
      c41 = vec_madd(a4, b1, c41);
      c42 = vec_madd(a4, b2, c42);
      c43 = vec_madd(a4, b3, c43);
      
      // Row 5
      c50 = vec_madd(a5, b0, c50);
      c51 = vec_madd(a5, b1, c51);
      c52 = vec_madd(a5, b2, c52);
      c53 = vec_madd(a5, b3, c53);
      
      // Row 6
      c60 = vec_madd(a6, b0, c60);
      c61 = vec_madd(a6, b1, c61);
      c62 = vec_madd(a6, b2, c62);
      c63 = vec_madd(a6, b3, c63);
      
      // Row 7
      c70 = vec_madd(a7, b0, c70);
      c71 = vec_madd(a7, b1, c71);
      c72 = vec_madd(a7, b2, c72);
      c73 = vec_madd(a7, b3, c73);
    }
  }

  // Handle remaining K iterations (if K % K_UNROLL != 0)
  for (; k < K; k++) {
    __vector float a0 = vec_splats(A[0*lda + k]);
    __vector float a1 = vec_splats(A[1*lda + k]);
    __vector float a2 = vec_splats(A[2*lda + k]);
    __vector float a3 = vec_splats(A[3*lda + k]);
    __vector float a4 = vec_splats(A[4*lda + k]);
    __vector float a5 = vec_splats(A[5*lda + k]);
    __vector float a6 = vec_splats(A[6*lda + k]);
    __vector float a7 = vec_splats(A[7*lda + k]);

    const float* b_row = B + k * ldb;
    __vector float b0 = vec_xl(0, b_row +  0);
    __vector float b1 = vec_xl(0, b_row +  4);
    __vector float b2 = vec_xl(0, b_row +  8);
    __vector float b3 = vec_xl(0, b_row + 12);

    c00 = vec_madd(a0, b0, c00); c01 = vec_madd(a0, b1, c01);
    c02 = vec_madd(a0, b2, c02); c03 = vec_madd(a0, b3, c03);
    
    c10 = vec_madd(a1, b0, c10); c11 = vec_madd(a1, b1, c11);
    c12 = vec_madd(a1, b2, c12); c13 = vec_madd(a1, b3, c13);
    
    c20 = vec_madd(a2, b0, c20); c21 = vec_madd(a2, b1, c21);
    c22 = vec_madd(a2, b2, c22); c23 = vec_madd(a2, b3, c23);
    
    c30 = vec_madd(a3, b0, c30); c31 = vec_madd(a3, b1, c31);
    c32 = vec_madd(a3, b2, c32); c33 = vec_madd(a3, b3, c33);
    
    c40 = vec_madd(a4, b0, c40); c41 = vec_madd(a4, b1, c41);
    c42 = vec_madd(a4, b2, c42); c43 = vec_madd(a4, b3, c43);
    
    c50 = vec_madd(a5, b0, c50); c51 = vec_madd(a5, b1, c51);
    c52 = vec_madd(a5, b2, c52); c53 = vec_madd(a5, b3, c53);
    
    c60 = vec_madd(a6, b0, c60); c61 = vec_madd(a6, b1, c61);
    c62 = vec_madd(a6, b2, c62); c63 = vec_madd(a6, b3, c63);
    
    c70 = vec_madd(a7, b0, c70); c71 = vec_madd(a7, b1, c71);
    c72 = vec_madd(a7, b2, c72); c73 = vec_madd(a7, b3, c73);
  }

  // Store C back to memory
  vec_xst(c00, 0, C + 0*ldc +  0); vec_xst(c01, 0, C + 0*ldc +  4);
  vec_xst(c02, 0, C + 0*ldc +  8); vec_xst(c03, 0, C + 0*ldc + 12);
  
  vec_xst(c10, 0, C + 1*ldc +  0); vec_xst(c11, 0, C + 1*ldc +  4);
  vec_xst(c12, 0, C + 1*ldc +  8); vec_xst(c13, 0, C + 1*ldc + 12);
  
  vec_xst(c20, 0, C + 2*ldc +  0); vec_xst(c21, 0, C + 2*ldc +  4);
  vec_xst(c22, 0, C + 2*ldc +  8); vec_xst(c23, 0, C + 2*ldc + 12);
  
  vec_xst(c30, 0, C + 3*ldc +  0); vec_xst(c31, 0, C + 3*ldc +  4);
  vec_xst(c32, 0, C + 3*ldc +  8); vec_xst(c33, 0, C + 3*ldc + 12);
  
  vec_xst(c40, 0, C + 4*ldc +  0); vec_xst(c41, 0, C + 4*ldc +  4);
  vec_xst(c42, 0, C + 4*ldc +  8); vec_xst(c43, 0, C + 4*ldc + 12);
  
  vec_xst(c50, 0, C + 5*ldc +  0); vec_xst(c51, 0, C + 5*ldc +  4);
  vec_xst(c52, 0, C + 5*ldc +  8); vec_xst(c53, 0, C + 5*ldc + 12);
  
  vec_xst(c60, 0, C + 6*ldc +  0); vec_xst(c61, 0, C + 6*ldc +  4);
  vec_xst(c62, 0, C + 6*ldc +  8); vec_xst(c63, 0, C + 6*ldc + 12);
  
  vec_xst(c70, 0, C + 7*ldc +  0); vec_xst(c71, 0, C + 7*ldc +  4);
  vec_xst(c72, 0, C + 7*ldc +  8); vec_xst(c73, 0, C + 7*ldc + 12);
}

// ============================================================================
// FP32 Microkernel: 8x8 (for N tail blocks)
// ============================================================================
// Computes: C[8×8] += A[8×K] × B[K×8]

static inline void gemm_s390x_kernel_8x8(
    const float* A,
    const float* B,
    float* C,
    int K,
    int lda,
    int ldb,
    int ldc) {

  // C accumulator registers: 8 rows × 2 vectors/row = 16 vectors
  __vector float c00, c01;  // row 0
  __vector float c10, c11;  // row 1
  __vector float c20, c21;  // row 2
  __vector float c30, c31;  // row 3
  __vector float c40, c41;  // row 4
  __vector float c50, c51;  // row 5
  __vector float c60, c61;  // row 6
  __vector float c70, c71;  // row 7

  // Load existing C values
  c00 = vec_xl(0, C + 0*ldc + 0); c01 = vec_xl(0, C + 0*ldc + 4);
  c10 = vec_xl(0, C + 1*ldc + 0); c11 = vec_xl(0, C + 1*ldc + 4);
  c20 = vec_xl(0, C + 2*ldc + 0); c21 = vec_xl(0, C + 2*ldc + 4);
  c30 = vec_xl(0, C + 3*ldc + 0); c31 = vec_xl(0, C + 3*ldc + 4);
  c40 = vec_xl(0, C + 4*ldc + 0); c41 = vec_xl(0, C + 4*ldc + 4);
  c50 = vec_xl(0, C + 5*ldc + 0); c51 = vec_xl(0, C + 5*ldc + 4);
  c60 = vec_xl(0, C + 6*ldc + 0); c61 = vec_xl(0, C + 6*ldc + 4);
  c70 = vec_xl(0, C + 7*ldc + 0); c71 = vec_xl(0, C + 7*ldc + 4);

  // Main K loop
  int k = 0;
  for (; k + K_UNROLL <= K; k += K_UNROLL) {
    // Prefetch ahead for next K block
    int kk_pref = k + K_UNROLL;
    if (kk_pref < K) {
      for (int mrow = 0; mrow < 8; ++mrow) {
        __builtin_prefetch(A + mrow * lda + kk_pref, 0, 3);
      }
      __builtin_prefetch(B + kk_pref * ldb, 0, 3);
    }

    for (int ku = 0; ku < K_UNROLL; ku++) {
      int kk = k + ku;
      
      __vector float a0 = vec_splats(A[0*lda + kk]);
      __vector float a1 = vec_splats(A[1*lda + kk]);
      __vector float a2 = vec_splats(A[2*lda + kk]);
      __vector float a3 = vec_splats(A[3*lda + kk]);
      __vector float a4 = vec_splats(A[4*lda + kk]);
      __vector float a5 = vec_splats(A[5*lda + kk]);
      __vector float a6 = vec_splats(A[6*lda + kk]);
      __vector float a7 = vec_splats(A[7*lda + kk]);

      const float* b_row = B + kk * ldb;
      __vector float b0 = vec_xl(0, b_row + 0);  // cols 0..3
      __vector float b1 = vec_xl(0, b_row + 4);  // cols 4..7

      c00 = vec_madd(a0, b0, c00); c01 = vec_madd(a0, b1, c01);
      c10 = vec_madd(a1, b0, c10); c11 = vec_madd(a1, b1, c11);
      c20 = vec_madd(a2, b0, c20); c21 = vec_madd(a2, b1, c21);
      c30 = vec_madd(a3, b0, c30); c31 = vec_madd(a3, b1, c31);
      c40 = vec_madd(a4, b0, c40); c41 = vec_madd(a4, b1, c41);
      c50 = vec_madd(a5, b0, c50); c51 = vec_madd(a5, b1, c51);
      c60 = vec_madd(a6, b0, c60); c61 = vec_madd(a6, b1, c61);
      c70 = vec_madd(a7, b0, c70); c71 = vec_madd(a7, b1, c71);
    }
  }

  // Remainder K loop
  for (; k < K; k++) {
    __vector float a0 = vec_splats(A[0*lda + k]);
    __vector float a1 = vec_splats(A[1*lda + k]);
    __vector float a2 = vec_splats(A[2*lda + k]);
    __vector float a3 = vec_splats(A[3*lda + k]);
    __vector float a4 = vec_splats(A[4*lda + k]);
    __vector float a5 = vec_splats(A[5*lda + k]);
    __vector float a6 = vec_splats(A[6*lda + k]);
    __vector float a7 = vec_splats(A[7*lda + k]);

    const float* b_row = B + k * ldb;
    __vector float b0 = vec_xl(0, b_row + 0);
    __vector float b1 = vec_xl(0, b_row + 4);

    c00 = vec_madd(a0, b0, c00); c01 = vec_madd(a0, b1, c01);
    c10 = vec_madd(a1, b0, c10); c11 = vec_madd(a1, b1, c11);
    c20 = vec_madd(a2, b0, c20); c21 = vec_madd(a2, b1, c21);
    c30 = vec_madd(a3, b0, c30); c31 = vec_madd(a3, b1, c31);
    c40 = vec_madd(a4, b0, c40); c41 = vec_madd(a4, b1, c41);
    c50 = vec_madd(a5, b0, c50); c51 = vec_madd(a5, b1, c51);
    c60 = vec_madd(a6, b0, c60); c61 = vec_madd(a6, b1, c61);
    c70 = vec_madd(a7, b0, c70); c71 = vec_madd(a7, b1, c71);
  }

  // Store C
  vec_xst(c00, 0, C + 0*ldc + 0); vec_xst(c01, 0, C + 0*ldc + 4);
  vec_xst(c10, 0, C + 1*ldc + 0); vec_xst(c11, 0, C + 1*ldc + 4);
  vec_xst(c20, 0, C + 2*ldc + 0); vec_xst(c21, 0, C + 2*ldc + 4);
  vec_xst(c30, 0, C + 3*ldc + 0); vec_xst(c31, 0, C + 3*ldc + 4);
  vec_xst(c40, 0, C + 4*ldc + 0); vec_xst(c41, 0, C + 4*ldc + 4);
  vec_xst(c50, 0, C + 5*ldc + 0); vec_xst(c51, 0, C + 5*ldc + 4);
  vec_xst(c60, 0, C + 6*ldc + 0); vec_xst(c61, 0, C + 6*ldc + 4);
  vec_xst(c70, 0, C + 7*ldc + 0); vec_xst(c71, 0, C + 7*ldc + 4);
}

// ============================================================================
// Scalar tail kernel for remaining M/N dimensions
// ============================================================================

static inline void gemm_s390x_scalar_tail(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc) {
  
  // Note: B pointer is already shifted by column offset "n" in the caller,
  // so B[k * ldb + n_local] accesses the correct element where n_local < N.
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = C[m * ldc + n];
      for (int k = 0; k < K; k++) {
        sum += A[m * lda + k] * B[k * ldb + n];
      }
      C[m * ldc + n] = sum;
    }
  }
}

// ============================================================================
// High-level FP32 GEMM: C[M×N] += A[M×K] × B[K×N]
// ============================================================================

static void s390x_fp32_gemm_impl(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc) {

  // Iterate over M in blocks of BLOCK_M=8
  int m = 0;
  for (; m + BLOCK_M <= M; m += BLOCK_M) {
    
    // Iterate over N in blocks of BLOCK_N=16
    int n = 0;
    for (; n + BLOCK_N <= N; n += BLOCK_N) {
      gemm_s390x_kernel_8x16(
          A + m * lda,
          B + n,              // Start at column n of B (row-major)
          C + m * ldc + n,
          K,
          lda,
          ldb,
          ldc);
    }

    // Handle N tail with 8x8 kernel
    for (; n + 8 <= N; n += 8) {
      gemm_s390x_kernel_8x8(
          A + m * lda,
          B + n,
          C + m * ldc + n,
          K,
          lda,
          ldb,
          ldc);
    }

    // Handle remaining N with scalar code
    if (n < N) {
      gemm_s390x_scalar_tail(
          A + m * lda,
          B + n,
          C + m * ldc + n,
          BLOCK_M,
          N - n,
          K,
          lda,
          ldb,
          ldc);
    }
  }

  // Handle remaining M rows with scalar code
  if (m < M) {
    gemm_s390x_scalar_tail(
        A + m * lda,
        B,
        C + m * ldc,
        M - m,
        N,
        K,
        lda,
        ldb,
        ldc);
  }
}

// ============================================================================
// BF16 Helper Functions
// ============================================================================

// Convert BF16 (uint16_t) to FP32
// BF16 format: sign(1) | exponent(8) | mantissa(7)
// FP32 format: sign(1) | exponent(8) | mantissa(23)
// Conversion: left-shift by 16 bits and reinterpret as float
static inline __vector float bf16_to_fp32_vec(__vector unsigned short bf16_vec) {
  // Split into high and low 64-bit halves
  __vector unsigned int low_32 = vec_unpackl(bf16_vec);   // low 4 BF16 → 4 uint32
  __vector unsigned int high_32 = vec_unpackh(bf16_vec);  // high 4 BF16 → 4 uint32
  
  // Shift left by 16 to convert BF16 → FP32 bit pattern
  low_32 = vec_sl(low_32, vec_splats(16u));
  high_32 = vec_sl(high_32, vec_splats(16u));
  
  // Reinterpret as float
  __vector float low_fp32 = (__vector float)low_32;
  __vector float high_fp32 = (__vector float)high_32;
  
  return low_fp32;  // Return low half (first 4 elements)
}

static inline __vector float bf16_to_fp32_vec_high(__vector unsigned short bf16_vec) {
  __vector unsigned int high_32 = vec_unpackh(bf16_vec);
  high_32 = vec_sl(high_32, vec_splats(16u));
  return (__vector float)high_32;
}

// Convert FP32 to BF16 (uint16_t) - simple truncation (round-to-nearest-even for production)
static inline __vector unsigned short fp32_to_bf16_vec(__vector float fp32_low, __vector float fp32_high) {
  __vector unsigned int low_32 = (__vector unsigned int)fp32_low;
  __vector unsigned int high_32 = (__vector unsigned int)fp32_high;
  
  // Shift right by 16 to get BF16 bit pattern
  low_32 = (__vector unsigned int)vec_sr(low_32, vec_splats(16u));
  high_32 = (__vector unsigned int)vec_sr(high_32, vec_splats(16u));
  
  // Pack into uint16_t vector
  return vec_pack(high_32, low_32);
}

// Load 8 BF16 values and convert to 2 FP32 vectors (4 elements each)
static inline void load_bf16_to_fp32x2(const at::BFloat16* ptr, __vector float& v0, __vector float& v1) {
  // Load 8 BF16 values as uint16_t
  const uint16_t* bf16_ptr = reinterpret_cast<const uint16_t*>(ptr);
  __vector unsigned short bf16_vec = vec_xl(0, bf16_ptr);
  
  v0 = bf16_to_fp32_vec(bf16_vec);       // elements 0-3
  v1 = bf16_to_fp32_vec_high(bf16_vec);  // elements 4-7
}

// ============================================================================
// BF16 Microkernels - DEPRECATED
// ============================================================================
// These kernels had fundamental performance issues:
// 1. Manual vector init {...} spilled to stack instead of using vec_xl
// 2. Per-element vec_splat instead of true vector FMA
// 3. BF16→FP32 conversion in hot loop
// 4. No weight pre-packing
// Result: 8× slower than FP32 kernel
//
// Current approach: Convert BF16→FP32, use FP32 kernel, convert back
// Keeping these commented for future reference if direct BF16 compute is needed

#if 0  // DISABLED - BF16 direct compute approach

// ============================================================================
// BF16 Microkernel: 8x16 (BLOCK_M=8, BLOCK_N=16) - DEPRECATED
// ============================================================================
// Computes: C[8×16] += A[8×K] × B[K×16]
// 
// OPTIMIZATION: Process K in blocks of 4 to enable vectorized A loads
// - Load A as 4-element BF16 vectors → convert to FP32
// - Load B as 8-element BF16 vectors → convert to 2×FP32
// - FMA with 4 accumulator elements per iteration
// 
// This avoids scalar BF16 loads and per-element broadcasts
// Performance: ~2-3× faster than scalar-broadcast approach

static inline void gemm_s390x_kernel_bf16_8x16(
    const at::BFloat16* A,  // [8 × K] row-major
    const at::BFloat16* B,  // [K × N] row-major (N >= 16)
    at::BFloat16* C,        // [8 × 16] row-major  
    int K,
    int lda,                // leading dimension of A (=K typically)
    int ldb,                // leading dimension of B (=N typically)
    int ldc) {              // leading dimension of C (=N typically)

  // C accumulator registers: 8 rows × 4 vectors/row = 32 FP32 vectors
  __vector float c00, c01, c02, c03;  // row 0
  __vector float c10, c11, c12, c13;  // row 1
  __vector float c20, c21, c22, c23;  // row 2
  __vector float c30, c31, c32, c33;  // row 3
  __vector float c40, c41, c42, c43;  // row 4
  __vector float c50, c51, c52, c53;  // row 5
  __vector float c60, c61, c62, c63;  // row 6
  __vector float c70, c71, c72, c73;  // row 7

  // Load existing C values and convert BF16 → FP32
  __vector float c_tmp0, c_tmp1;
  load_bf16_to_fp32x2(C + 0*ldc +  0, c00, c01);
  load_bf16_to_fp32x2(C + 0*ldc +  8, c02, c03);
  
  load_bf16_to_fp32x2(C + 1*ldc +  0, c10, c11);
  load_bf16_to_fp32x2(C + 1*ldc +  8, c12, c13);
  
  load_bf16_to_fp32x2(C + 2*ldc +  0, c20, c21);
  load_bf16_to_fp32x2(C + 2*ldc +  8, c22, c23);
  
  load_bf16_to_fp32x2(C + 3*ldc +  0, c30, c31);
  load_bf16_to_fp32x2(C + 3*ldc +  8, c32, c33);
  
  load_bf16_to_fp32x2(C + 4*ldc +  0, c40, c41);
  load_bf16_to_fp32x2(C + 4*ldc +  8, c42, c43);
  
  load_bf16_to_fp32x2(C + 5*ldc +  0, c50, c51);
  load_bf16_to_fp32x2(C + 5*ldc +  8, c52, c53);
  
  load_bf16_to_fp32x2(C + 6*ldc +  0, c60, c61);
  load_bf16_to_fp32x2(C + 6*ldc +  8, c62, c63);
  
  load_bf16_to_fp32x2(C + 7*ldc +  0, c70, c71);
  load_bf16_to_fp32x2(C + 7*ldc +  8, c72, c73);

  // Main K loop: Process K in blocks of 4 for vectorized A loads
  // Each iteration processes 4 K values using FP32 vector elements
  const int K_VEC = 4;  // 4 FP32 elements per vector
  int k = 0;
  
  for (; k + K_VEC <= K; k += K_VEC) {
    // Prefetch ahead
    if (k + K_VEC + 4 < K) {
      for (int mrow = 0; mrow < 8; ++mrow) {
        __builtin_prefetch(A + mrow * lda + k + K_VEC + 4, 0, 3);
      }
      for (int brow = 0; brow < 4; ++brow) {
        __builtin_prefetch(B + (k + K_VEC + brow) * ldb, 0, 3);
      }
    }

    // Load A[0:8, k:k+4] - load 4 consecutive BF16 values per row
    // Convert to 2 FP32 vectors (low 2 elements, high 2 elements)
    __vector float a0_lo, a0_hi;  // row 0: k+0,k+1 and k+2,k+3
    __vector float a1_lo, a1_hi;  // row 1
    __vector float a2_lo, a2_hi;  // row 2
    __vector float a3_lo, a3_hi;  // row 3
    __vector float a4_lo, a4_hi;  // row 4
    __vector float a5_lo, a5_hi;  // row 5
    __vector float a6_lo, a6_hi;  // row 6
    __vector float a7_lo, a7_hi;  // row 7
    
    // Load 4 BF16 from each row and split into 2×2 FP32
    const uint16_t* a0_ptr = reinterpret_cast<const uint16_t*>(A + 0*lda + k);
    const uint16_t* a1_ptr = reinterpret_cast<const uint16_t*>(A + 1*lda + k);
    const uint16_t* a2_ptr = reinterpret_cast<const uint16_t*>(A + 2*lda + k);
    const uint16_t* a3_ptr = reinterpret_cast<const uint16_t*>(A + 3*lda + k);
    const uint16_t* a4_ptr = reinterpret_cast<const uint16_t*>(A + 4*lda + k);
    const uint16_t* a5_ptr = reinterpret_cast<const uint16_t*>(A + 5*lda + k);
    const uint16_t* a6_ptr = reinterpret_cast<const uint16_t*>(A + 6*lda + k);
    const uint16_t* a7_ptr = reinterpret_cast<const uint16_t*>(A + 7*lda + k);
    
    // Load 4×uint16 (unaligned OK, VXE handles it)
    __vector unsigned short a0_bf = {a0_ptr[0], a0_ptr[1], a0_ptr[2], a0_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a1_bf = {a1_ptr[0], a1_ptr[1], a1_ptr[2], a1_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a2_bf = {a2_ptr[0], a2_ptr[1], a2_ptr[2], a2_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a3_bf = {a3_ptr[0], a3_ptr[1], a3_ptr[2], a3_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a4_bf = {a4_ptr[0], a4_ptr[1], a4_ptr[2], a4_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a5_bf = {a5_ptr[0], a5_ptr[1], a5_ptr[2], a5_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a6_bf = {a6_ptr[0], a6_ptr[1], a6_ptr[2], a6_ptr[3], 0, 0, 0, 0};
    __vector unsigned short a7_bf = {a7_ptr[0], a7_ptr[1], a7_ptr[2], a7_ptr[3], 0, 0, 0, 0};
    
    // Convert BF16 → FP32 (first 2 elements)
    a0_lo = bf16_to_fp32_vec(a0_bf); a0_hi = bf16_to_fp32_vec_high(a0_bf);
    a1_lo = bf16_to_fp32_vec(a1_bf); a1_hi = bf16_to_fp32_vec_high(a1_bf);
    a2_lo = bf16_to_fp32_vec(a2_bf); a2_hi = bf16_to_fp32_vec_high(a2_bf);
    a3_lo = bf16_to_fp32_vec(a3_bf); a3_hi = bf16_to_fp32_vec_high(a3_bf);
    a4_lo = bf16_to_fp32_vec(a4_bf); a4_hi = bf16_to_fp32_vec_high(a4_bf);
    a5_lo = bf16_to_fp32_vec(a5_bf); a5_hi = bf16_to_fp32_vec_high(a5_bf);
    a6_lo = bf16_to_fp32_vec(a6_bf); a6_hi = bf16_to_fp32_vec_high(a6_bf);
    a7_lo = bf16_to_fp32_vec(a7_bf); a7_hi = bf16_to_fp32_vec_high(a7_bf);

    // Load B[k:k+4, 0:16] - 4 rows × 16 cols
    // Each row: 16 BF16 → 4 FP32 vectors
    __vector float b0_0, b0_1, b0_2, b0_3;  // B[k+0, :]
    __vector float b1_0, b1_1, b1_2, b1_3;  // B[k+1, :]
    __vector float b2_0, b2_1, b2_2, b2_3;  // B[k+2, :]
    __vector float b3_0, b3_1, b3_2, b3_3;  // B[k+3, :]
    
    load_bf16_to_fp32x2(B + (k+0)*ldb + 0, b0_0, b0_1);
    load_bf16_to_fp32x2(B + (k+0)*ldb + 8, b0_2, b0_3);
    
    load_bf16_to_fp32x2(B + (k+1)*ldb + 0, b1_0, b1_1);
    load_bf16_to_fp32x2(B + (k+1)*ldb + 8, b1_2, b1_3);
    
    load_bf16_to_fp32x2(B + (k+2)*ldb + 0, b2_0, b2_1);
    load_bf16_to_fp32x2(B + (k+2)*ldb + 8, b2_2, b2_3);
    
    load_bf16_to_fp32x2(B + (k+3)*ldb + 0, b3_0, b3_1);
    load_bf16_to_fp32x2(B + (k+3)*ldb + 8, b3_2, b3_3);

    // FMA with vector elements
    // C[m,n] += A[m,k+0] * B[k+0,n] + A[m,k+1] * B[k+1,n] + ...
    // Use vec_splat to broadcast individual A elements
    
    // Row 0: A[0, k:k+4] × B[k:k+4, n]
    c00 = vec_madd(vec_splat(a0_lo, 0), b0_0, c00);  // A[0,k+0] * B[k+0,0:3]
    c00 = vec_madd(vec_splat(a0_lo, 1), b1_0, c00);  // A[0,k+1] * B[k+1,0:3]
    c00 = vec_madd(vec_splat(a0_hi, 0), b2_0, c00);  // A[0,k+2] * B[k+2,0:3]
    c00 = vec_madd(vec_splat(a0_hi, 1), b3_0, c00);  // A[0,k+3] * B[k+3,0:3]
    
    c01 = vec_madd(vec_splat(a0_lo, 0), b0_1, c01);
    c01 = vec_madd(vec_splat(a0_lo, 1), b1_1, c01);
    c01 = vec_madd(vec_splat(a0_hi, 0), b2_1, c01);
    c01 = vec_madd(vec_splat(a0_hi, 1), b3_1, c01);
    
    c02 = vec_madd(vec_splat(a0_lo, 0), b0_2, c02);
    c02 = vec_madd(vec_splat(a0_lo, 1), b1_2, c02);
    c02 = vec_madd(vec_splat(a0_hi, 0), b2_2, c02);
    c02 = vec_madd(vec_splat(a0_hi, 1), b3_2, c02);
    
    c03 = vec_madd(vec_splat(a0_lo, 0), b0_3, c03);
    c03 = vec_madd(vec_splat(a0_lo, 1), b1_3, c03);
    c03 = vec_madd(vec_splat(a0_hi, 0), b2_3, c03);
    c03 = vec_madd(vec_splat(a0_hi, 1), b3_3, c03);
    
    // Row 1
    c10 = vec_madd(vec_splat(a1_lo, 0), b0_0, c10);
    c10 = vec_madd(vec_splat(a1_lo, 1), b1_0, c10);
    c10 = vec_madd(vec_splat(a1_hi, 0), b2_0, c10);
    c10 = vec_madd(vec_splat(a1_hi, 1), b3_0, c10);
    
    c11 = vec_madd(vec_splat(a1_lo, 0), b0_1, c11);
    c11 = vec_madd(vec_splat(a1_lo, 1), b1_1, c11);
    c11 = vec_madd(vec_splat(a1_hi, 0), b2_1, c11);
    c11 = vec_madd(vec_splat(a1_hi, 1), b3_1, c11);
    
    c12 = vec_madd(vec_splat(a1_lo, 0), b0_2, c12);
    c12 = vec_madd(vec_splat(a1_lo, 1), b1_2, c12);
    c12 = vec_madd(vec_splat(a1_hi, 0), b2_2, c12);
    c12 = vec_madd(vec_splat(a1_hi, 1), b3_2, c12);
    
    c13 = vec_madd(vec_splat(a1_lo, 0), b0_3, c13);
    c13 = vec_madd(vec_splat(a1_lo, 1), b1_3, c13);
    c13 = vec_madd(vec_splat(a1_hi, 0), b2_3, c13);
    c13 = vec_madd(vec_splat(a1_hi, 1), b3_3, c13);
    
    // Row 2
    c20 = vec_madd(vec_splat(a2_lo, 0), b0_0, c20);
    c20 = vec_madd(vec_splat(a2_lo, 1), b1_0, c20);
    c20 = vec_madd(vec_splat(a2_hi, 0), b2_0, c20);
    c20 = vec_madd(vec_splat(a2_hi, 1), b3_0, c20);
    
    c21 = vec_madd(vec_splat(a2_lo, 0), b0_1, c21);
    c21 = vec_madd(vec_splat(a2_lo, 1), b1_1, c21);
    c21 = vec_madd(vec_splat(a2_hi, 0), b2_1, c21);
    c21 = vec_madd(vec_splat(a2_hi, 1), b3_1, c21);
    
    c22 = vec_madd(vec_splat(a2_lo, 0), b0_2, c22);
    c22 = vec_madd(vec_splat(a2_lo, 1), b1_2, c22);
    c22 = vec_madd(vec_splat(a2_hi, 0), b2_2, c22);
    c22 = vec_madd(vec_splat(a2_hi, 1), b3_2, c22);
    
    c23 = vec_madd(vec_splat(a2_lo, 0), b0_3, c23);
    c23 = vec_madd(vec_splat(a2_lo, 1), b1_3, c23);
    c23 = vec_madd(vec_splat(a2_hi, 0), b2_3, c23);
    c23 = vec_madd(vec_splat(a2_hi, 1), b3_3, c23);
    
    // Row 3
    c30 = vec_madd(vec_splat(a3_lo, 0), b0_0, c30);
    c30 = vec_madd(vec_splat(a3_lo, 1), b1_0, c30);
    c30 = vec_madd(vec_splat(a3_hi, 0), b2_0, c30);
    c30 = vec_madd(vec_splat(a3_hi, 1), b3_0, c30);
    
    c31 = vec_madd(vec_splat(a3_lo, 0), b0_1, c31);
    c31 = vec_madd(vec_splat(a3_lo, 1), b1_1, c31);
    c31 = vec_madd(vec_splat(a3_hi, 0), b2_1, c31);
    c31 = vec_madd(vec_splat(a3_hi, 1), b3_1, c31);
    
    c32 = vec_madd(vec_splat(a3_lo, 0), b0_2, c32);
    c32 = vec_madd(vec_splat(a3_lo, 1), b1_2, c32);
    c32 = vec_madd(vec_splat(a3_hi, 0), b2_2, c32);
    c32 = vec_madd(vec_splat(a3_hi, 1), b3_2, c32);
    
    c33 = vec_madd(vec_splat(a3_lo, 0), b0_3, c33);
    c33 = vec_madd(vec_splat(a3_lo, 1), b1_3, c33);
    c33 = vec_madd(vec_splat(a3_hi, 0), b2_3, c33);
    c33 = vec_madd(vec_splat(a3_hi, 1), b3_3, c33);
    
    // Row 4
    c40 = vec_madd(vec_splat(a4_lo, 0), b0_0, c40);
    c40 = vec_madd(vec_splat(a4_lo, 1), b1_0, c40);
    c40 = vec_madd(vec_splat(a4_hi, 0), b2_0, c40);
    c40 = vec_madd(vec_splat(a4_hi, 1), b3_0, c40);
    
    c41 = vec_madd(vec_splat(a4_lo, 0), b0_1, c41);
    c41 = vec_madd(vec_splat(a4_lo, 1), b1_1, c41);
    c41 = vec_madd(vec_splat(a4_hi, 0), b2_1, c41);
    c41 = vec_madd(vec_splat(a4_hi, 1), b3_1, c41);
    
    c42 = vec_madd(vec_splat(a4_lo, 0), b0_2, c42);
    c42 = vec_madd(vec_splat(a4_lo, 1), b1_2, c42);
    c42 = vec_madd(vec_splat(a4_hi, 0), b2_2, c42);
    c42 = vec_madd(vec_splat(a4_hi, 1), b3_2, c42);
    
    c43 = vec_madd(vec_splat(a4_lo, 0), b0_3, c43);
    c43 = vec_madd(vec_splat(a4_lo, 1), b1_3, c43);
    c43 = vec_madd(vec_splat(a4_hi, 0), b2_3, c43);
    c43 = vec_madd(vec_splat(a4_hi, 1), b3_3, c43);
    
    // Row 5
    c50 = vec_madd(vec_splat(a5_lo, 0), b0_0, c50);
    c50 = vec_madd(vec_splat(a5_lo, 1), b1_0, c50);
    c50 = vec_madd(vec_splat(a5_hi, 0), b2_0, c50);
    c50 = vec_madd(vec_splat(a5_hi, 1), b3_0, c50);
    
    c51 = vec_madd(vec_splat(a5_lo, 0), b0_1, c51);
    c51 = vec_madd(vec_splat(a5_lo, 1), b1_1, c51);
    c51 = vec_madd(vec_splat(a5_hi, 0), b2_1, c51);
    c51 = vec_madd(vec_splat(a5_hi, 1), b3_1, c51);
    
    c52 = vec_madd(vec_splat(a5_lo, 0), b0_2, c52);
    c52 = vec_madd(vec_splat(a5_lo, 1), b1_2, c52);
    c52 = vec_madd(vec_splat(a5_hi, 0), b2_2, c52);
    c52 = vec_madd(vec_splat(a5_hi, 1), b3_2, c52);
    
    c53 = vec_madd(vec_splat(a5_lo, 0), b0_3, c53);
    c53 = vec_madd(vec_splat(a5_lo, 1), b1_3, c53);
    c53 = vec_madd(vec_splat(a5_hi, 0), b2_3, c53);
    c53 = vec_madd(vec_splat(a5_hi, 1), b3_3, c53);
    
    // Row 6
    c60 = vec_madd(vec_splat(a6_lo, 0), b0_0, c60);
    c60 = vec_madd(vec_splat(a6_lo, 1), b1_0, c60);
    c60 = vec_madd(vec_splat(a6_hi, 0), b2_0, c60);
    c60 = vec_madd(vec_splat(a6_hi, 1), b3_0, c60);
    
    c61 = vec_madd(vec_splat(a6_lo, 0), b0_1, c61);
    c61 = vec_madd(vec_splat(a6_lo, 1), b1_1, c61);
    c61 = vec_madd(vec_splat(a6_hi, 0), b2_1, c61);
    c61 = vec_madd(vec_splat(a6_hi, 1), b3_1, c61);
    
    c62 = vec_madd(vec_splat(a6_lo, 0), b0_2, c62);
    c62 = vec_madd(vec_splat(a6_lo, 1), b1_2, c62);
    c62 = vec_madd(vec_splat(a6_hi, 0), b2_2, c62);
    c62 = vec_madd(vec_splat(a6_hi, 1), b3_2, c62);
    
    c63 = vec_madd(vec_splat(a6_lo, 0), b0_3, c63);
    c63 = vec_madd(vec_splat(a6_lo, 1), b1_3, c63);
    c63 = vec_madd(vec_splat(a6_hi, 0), b2_3, c63);
    c63 = vec_madd(vec_splat(a6_hi, 1), b3_3, c63);
    
    // Row 7
    c70 = vec_madd(vec_splat(a7_lo, 0), b0_0, c70);
    c70 = vec_madd(vec_splat(a7_lo, 1), b1_0, c70);
    c70 = vec_madd(vec_splat(a7_hi, 0), b2_0, c70);
    c70 = vec_madd(vec_splat(a7_hi, 1), b3_0, c70);
    
    c71 = vec_madd(vec_splat(a7_lo, 0), b0_1, c71);
    c71 = vec_madd(vec_splat(a7_lo, 1), b1_1, c71);
    c71 = vec_madd(vec_splat(a7_hi, 0), b2_1, c71);
    c71 = vec_madd(vec_splat(a7_hi, 1), b3_1, c71);
    
    c72 = vec_madd(vec_splat(a7_lo, 0), b0_2, c72);
    c72 = vec_madd(vec_splat(a7_lo, 1), b1_2, c72);
    c72 = vec_madd(vec_splat(a7_hi, 0), b2_2, c72);
    c72 = vec_madd(vec_splat(a7_hi, 1), b3_2, c72);
    
    c73 = vec_madd(vec_splat(a7_lo, 0), b0_3, c73);
    c73 = vec_madd(vec_splat(a7_lo, 1), b1_3, c73);
    c73 = vec_madd(vec_splat(a7_hi, 0), b2_3, c73);
    c73 = vec_madd(vec_splat(a7_hi, 1), b3_3, c73);
  }

  // Handle remaining K iterations (scalar fallback)
  for (; k < K; k++) {
    __vector float a0 = vec_splats(static_cast<float>(A[0*lda + k]));
    __vector float a1 = vec_splats(static_cast<float>(A[1*lda + k]));
    __vector float a2 = vec_splats(static_cast<float>(A[2*lda + k]));
    __vector float a3 = vec_splats(static_cast<float>(A[3*lda + k]));
    __vector float a4 = vec_splats(static_cast<float>(A[4*lda + k]));
    __vector float a5 = vec_splats(static_cast<float>(A[5*lda + k]));
    __vector float a6 = vec_splats(static_cast<float>(A[6*lda + k]));
    __vector float a7 = vec_splats(static_cast<float>(A[7*lda + k]));

    const at::BFloat16* b_row = B + k * ldb;
    __vector float b0, b1, b2, b3;
    load_bf16_to_fp32x2(b_row + 0, b0, b1);
    load_bf16_to_fp32x2(b_row + 8, b2, b3);

    c00 = vec_madd(a0, b0, c00); c01 = vec_madd(a0, b1, c01);
    c02 = vec_madd(a0, b2, c02); c03 = vec_madd(a0, b3, c03);
    
    c10 = vec_madd(a1, b0, c10); c11 = vec_madd(a1, b1, c11);
    c12 = vec_madd(a1, b2, c12); c13 = vec_madd(a1, b3, c13);
    
    c20 = vec_madd(a2, b0, c20); c21 = vec_madd(a2, b1, c21);
    c22 = vec_madd(a2, b2, c22); c23 = vec_madd(a2, b3, c23);
    
    c30 = vec_madd(a3, b0, c30); c31 = vec_madd(a3, b1, c31);
    c32 = vec_madd(a3, b2, c32); c33 = vec_madd(a3, b3, c33);
    
    c40 = vec_madd(a4, b0, c40); c41 = vec_madd(a4, b1, c41);
    c42 = vec_madd(a4, b2, c42); c43 = vec_madd(a4, b3, c43);
    
    c50 = vec_madd(a5, b0, c50); c51 = vec_madd(a5, b1, c51);
    c52 = vec_madd(a5, b2, c52); c53 = vec_madd(a5, b3, c53);
    
    c60 = vec_madd(a6, b0, c60); c61 = vec_madd(a6, b1, c61);
    c62 = vec_madd(a6, b2, c62); c63 = vec_madd(a6, b3, c63);
    
    c70 = vec_madd(a7, b0, c70); c71 = vec_madd(a7, b1, c71);
    c72 = vec_madd(a7, b2, c72); c73 = vec_madd(a7, b3, c73);
  }

  // Convert FP32 → BF16 and store C back to memory
  __vector unsigned short c_bf16;
  
  c_bf16 = fp32_to_bf16_vec(c00, c01);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 0*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c02, c03);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 0*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c10, c11);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 1*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c12, c13);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 1*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c20, c21);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 2*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c22, c23);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 2*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c30, c31);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 3*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c32, c33);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 3*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c40, c41);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 4*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c42, c43);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 4*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c50, c51);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 5*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c52, c53);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 5*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c60, c61);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 6*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c62, c63);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 6*ldc + 8));
  
  c_bf16 = fp32_to_bf16_vec(c70, c71);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 7*ldc + 0));
  c_bf16 = fp32_to_bf16_vec(c72, c73);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 7*ldc + 8));
}

// ============================================================================
// BF16 Microkernel: 8x8 (for N tail blocks)
// ============================================================================

static inline void gemm_s390x_kernel_bf16_8x8(
    const at::BFloat16* A,
    const at::BFloat16* B,
    at::BFloat16* C,
    int K,
    int lda,
    int ldb,
    int ldc) {

  // C accumulator registers: 8 rows × 2 vectors/row = 16 FP32 vectors
  __vector float c00, c01;  // row 0
  __vector float c10, c11;  // row 1
  __vector float c20, c21;  // row 2
  __vector float c30, c31;  // row 3
  __vector float c40, c41;  // row 4
  __vector float c50, c51;  // row 5
  __vector float c60, c61;  // row 6
  __vector float c70, c71;  // row 7

  // Load existing C values
  load_bf16_to_fp32x2(C + 0*ldc, c00, c01);
  load_bf16_to_fp32x2(C + 1*ldc, c10, c11);
  load_bf16_to_fp32x2(C + 2*ldc, c20, c21);
  load_bf16_to_fp32x2(C + 3*ldc, c30, c31);
  load_bf16_to_fp32x2(C + 4*ldc, c40, c41);
  load_bf16_to_fp32x2(C + 5*ldc, c50, c51);
  load_bf16_to_fp32x2(C + 6*ldc, c60, c61);
  load_bf16_to_fp32x2(C + 7*ldc, c70, c71);

  // Main K loop
  int k = 0;
  for (; k + K_UNROLL <= K; k += K_UNROLL) {
    int kk_pref = k + K_UNROLL;
    if (kk_pref < K) {
      for (int mrow = 0; mrow < 8; ++mrow) {
        __builtin_prefetch(A + mrow * lda + kk_pref, 0, 3);
      }
      __builtin_prefetch(B + kk_pref * ldb, 0, 3);
    }

    for (int ku = 0; ku < K_UNROLL; ku++) {
      int kk = k + ku;
      
      __vector float a0 = vec_splats(static_cast<float>(A[0*lda + kk]));
      __vector float a1 = vec_splats(static_cast<float>(A[1*lda + kk]));
      __vector float a2 = vec_splats(static_cast<float>(A[2*lda + kk]));
      __vector float a3 = vec_splats(static_cast<float>(A[3*lda + kk]));
      __vector float a4 = vec_splats(static_cast<float>(A[4*lda + kk]));
      __vector float a5 = vec_splats(static_cast<float>(A[5*lda + kk]));
      __vector float a6 = vec_splats(static_cast<float>(A[6*lda + kk]));
      __vector float a7 = vec_splats(static_cast<float>(A[7*lda + kk]));

      const at::BFloat16* b_row = B + kk * ldb;
      __vector float b0, b1;
      load_bf16_to_fp32x2(b_row, b0, b1);

      c00 = vec_madd(a0, b0, c00); c01 = vec_madd(a0, b1, c01);
      c10 = vec_madd(a1, b0, c10); c11 = vec_madd(a1, b1, c11);
      c20 = vec_madd(a2, b0, c20); c21 = vec_madd(a2, b1, c21);
      c30 = vec_madd(a3, b0, c30); c31 = vec_madd(a3, b1, c31);
      c40 = vec_madd(a4, b0, c40); c41 = vec_madd(a4, b1, c41);
      c50 = vec_madd(a5, b0, c50); c51 = vec_madd(a5, b1, c51);
      c60 = vec_madd(a6, b0, c60); c61 = vec_madd(a6, b1, c61);
      c70 = vec_madd(a7, b0, c70); c71 = vec_madd(a7, b1, c71);
    }
  }

  // Remainder K loop
  for (; k < K; k++) {
    __vector float a0 = vec_splats(static_cast<float>(A[0*lda + k]));
    __vector float a1 = vec_splats(static_cast<float>(A[1*lda + k]));
    __vector float a2 = vec_splats(static_cast<float>(A[2*lda + k]));
    __vector float a3 = vec_splats(static_cast<float>(A[3*lda + k]));
    __vector float a4 = vec_splats(static_cast<float>(A[4*lda + k]));
    __vector float a5 = vec_splats(static_cast<float>(A[5*lda + k]));
    __vector float a6 = vec_splats(static_cast<float>(A[6*lda + k]));
    __vector float a7 = vec_splats(static_cast<float>(A[7*lda + k]));

    const at::BFloat16* b_row = B + k * ldb;
    __vector float b0, b1;
    load_bf16_to_fp32x2(b_row, b0, b1);

    c00 = vec_madd(a0, b0, c00); c01 = vec_madd(a0, b1, c01);
    c10 = vec_madd(a1, b0, c10); c11 = vec_madd(a1, b1, c11);
    c20 = vec_madd(a2, b0, c20); c21 = vec_madd(a2, b1, c21);
    c30 = vec_madd(a3, b0, c30); c31 = vec_madd(a3, b1, c31);
    c40 = vec_madd(a4, b0, c40); c41 = vec_madd(a4, b1, c41);
    c50 = vec_madd(a5, b0, c50); c51 = vec_madd(a5, b1, c51);
    c60 = vec_madd(a6, b0, c60); c61 = vec_madd(a6, b1, c61);
    c70 = vec_madd(a7, b0, c70); c71 = vec_madd(a7, b1, c71);
  }

  // Store C
  __vector unsigned short c_bf16;
  c_bf16 = fp32_to_bf16_vec(c00, c01);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 0*ldc));
  c_bf16 = fp32_to_bf16_vec(c10, c11);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 1*ldc));
  c_bf16 = fp32_to_bf16_vec(c20, c21);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 2*ldc));
  c_bf16 = fp32_to_bf16_vec(c30, c31);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 3*ldc));
  c_bf16 = fp32_to_bf16_vec(c40, c41);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 4*ldc));
  c_bf16 = fp32_to_bf16_vec(c50, c51);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 5*ldc));
  c_bf16 = fp32_to_bf16_vec(c60, c61);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 6*ldc));
  c_bf16 = fp32_to_bf16_vec(c70, c71);
  vec_xst(c_bf16, 0, reinterpret_cast<uint16_t*>(C + 7*ldc));
}

// ============================================================================
// BF16 Scalar tail kernel
// ============================================================================

static inline void gemm_s390x_scalar_tail_bf16(
    const at::BFloat16* A,
    const at::BFloat16* B,
    at::BFloat16* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc) {
  
  // Note: B pointer is already shifted by column offset "n" in the caller
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = static_cast<float>(C[m * ldc + n]);
      for (int k = 0; k < K; k++) {
        sum += static_cast<float>(A[m * lda + k]) * 
               static_cast<float>(B[k * ldb + n]);
      }
      C[m * ldc + n] = static_cast<at::BFloat16>(sum);
    }
  }
}

#endif  // DISABLED - BF16 direct compute

// ============================================================================
// High-level BF16 GEMM: C[M×N] = A[M×K] × B[K×N]  (on s390x)
// ============================================================================
//
// Strategy:
//   - Convert A, B from BF16 → FP32 into dense row-major buffers
//   - Call the existing s390x_fp32_gemm_impl (heavily vectorized)
//   - Convert the FP32 result back to BF16
//
// Notes:
//   - This treats GEMM as C = A * B (not C += A*B), which matches our usage:
//     the top-level s390x_gemm() zero-initializes the output tensor before GEMM.
//

static void s390x_bf16_gemm_impl(
    const at::BFloat16* A,
    const at::BFloat16* B,
    at::BFloat16* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc) {

  // Dense FP32 working buffers
  std::vector<float> A_fp32(M * K);
  std::vector<float> B_fp32(K * N);
  std::vector<float> C_fp32(M * N, 0.0f);  // zero-init: C = A * B

  // --- A: BF16 → FP32 into [M,K] with row stride K ---
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      // at::BFloat16 has a conversion operator to float
      A_fp32[i * K + k] = static_cast<float>(A[i * lda + k]);
    }
  }

  // --- B: BF16 → FP32 into [K,N] with row stride N ---
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      B_fp32[k * N + n] = static_cast<float>(B[k * ldb + n]);
    }
  }

  // --- Compute in FP32 using the optimized kernel ---
  s390x_fp32_gemm_impl(
      A_fp32.data(),
      B_fp32.data(),
      C_fp32.data(),
      M, N, K,
      /*lda=*/K,
      /*ldb=*/N,
      /*ldc=*/N);

  // --- C: FP32 → BF16 with round-to-nearest-even ---
  for (int i = 0; i < M; i++) {
    for (int n = 0; n < N; n++) {
      float val = C_fp32[i * N + n];
      uint32_t fp32_bits;
      static_assert(sizeof(fp32_bits) == sizeof(val), "size mismatch");

      // Copy bits safely to avoid strict-aliasing UB
      std::memcpy(&fp32_bits, &val, sizeof(float));

      // RNE: add 0x7FFF + lsb-of-high-part before shifting
      uint32_t rounding_bias = 0x7FFFu + ((fp32_bits >> 16) & 1u);
      uint16_t bf16_bits =
          static_cast<uint16_t>((fp32_bits + rounding_bias) >> 16);

      C[i * ldc + n] = at::BFloat16(bf16_bits);
    }
  }
}

}  // namespace

// ============================================================================
// Public API
// ============================================================================

at::Tensor s390x_gemm(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const std::optional<at::Tensor>& bias) {
  
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a 2D tensor");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a 2D tensor");
  TORCH_CHECK(mat1.size(1) == mat2.size(0), "Incompatible matrix dimensions");

  int M = mat1.size(0);
  int K = mat1.size(1);
  int N = mat2.size(1);

  auto out = at::empty({M, N}, mat1.options());

  // Zero-initialize output
  out.zero_();

  if (mat1.scalar_type() == at::kFloat) {
    s390x_fp32_gemm_impl(
        mat1.data_ptr<float>(),
        mat2.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K,
        K,  // lda
        N,  // ldb
        N   // ldc
    );
  } else {
    // BF16 support disabled due to performance issues (allocates huge temporary buffers).
    // TODO: Implement proper weight pre-packing for BF16 support.
    TORCH_CHECK(false, "s390x_gemm only supports FP32. BF16 disabled (needs pre-packing).");
  }

  // Add bias if provided
  if (bias.has_value()) {
    out.add_(*bias);
  }

  return out;
}

at::Tensor s390x_gemm_packed(
    const at::Tensor& mat1,
    const at::Tensor& mat2_packed,
    const std::optional<at::Tensor>& bias) {
  
  // For now, delegate to regular gemm (mat2_packed assumed pre-transposed)
  return s390x_gemm(mat1, mat2_packed, bias);
}

#endif  // __s390x__
