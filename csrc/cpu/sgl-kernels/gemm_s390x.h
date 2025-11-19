// s390x VXE-optimized GEMM kernels header
#pragma once

#include <torch/types.h>

// FP32 GEMM: C = A × B + bias (optional)
// A: [M, K], B: [K, N] (transposed/column-major layout), C: [M, N]
void s390x_fp32_gemm(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc);

// BF16 GEMM: C = A × B + bias (optional)
// A: [M, K], B: [K, N] (transposed/column-major layout), C: [M, N]
// Accumulates in FP32 internally for better precision
void s390x_bf16_gemm(
    const at::BFloat16* A,
    const at::BFloat16* B,
    at::BFloat16* C,
    const float* bias,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc);

// Torch wrapper for easy integration
// mat1: [M, K], mat2: [K, N] (will be transposed internally if needed)
// Returns: [M, N]
at::Tensor s390x_gemm(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const std::optional<at::Tensor>& bias = std::nullopt);

// Torch wrapper with pre-packed mat2
// mat1: [M, K], mat2_packed: [K, N] (already in column-major layout)
// Returns: [M, N]
at::Tensor s390x_gemm_packed(
    const at::Tensor& mat1,
    const at::Tensor& mat2_packed,
    const std::optional<at::Tensor>& bias = std::nullopt);
