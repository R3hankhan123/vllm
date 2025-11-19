// Test harness for s390x GEMM kernels
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#if defined(__s390x__)
#include "gemm_s390x.h"
#include <torch/torch.h>

void test_fp32_gemm() {
  std::cout << "=== Testing FP32 GEMM ===" << std::endl;
  
  const int M = 128;
  const int K = 256;
  const int N = 256;
  
  auto mat1 = torch::randn({M, K}, torch::kFloat32);
  auto mat2 = torch::randn({N, K}, torch::kFloat32);  // Will be transposed
  auto bias = torch::randn({N}, torch::kFloat32);
  
  // Reference: PyTorch
  auto mat2_t = mat2.t().contiguous();
  auto ref = torch::addmm(bias, mat1, mat2_t);
  
  // Our implementation (mat2 is already in [N, K] format, which is transposed)
  auto result = s390x_gemm(mat1, mat2, bias);
  
  // Compare
  auto diff = (result - ref).abs();
  float max_diff = diff.max().item<float>();
  float mean_diff = diff.mean().item<float>();
  
  std::cout << "  Max absolute error: " << max_diff << std::endl;
  std::cout << "  Mean absolute error: " << mean_diff << std::endl;
  
  if (max_diff < 1e-4) {
    std::cout << "  ✓ FP32 GEMM test PASSED" << std::endl;
  } else {
    std::cout << "  ✗ FP32 GEMM test FAILED" << std::endl;
  }
  std::cout << std::endl;
}

void test_bf16_gemm() {
  std::cout << "=== Testing BF16 GEMM ===" << std::endl;
  
  const int M = 128;
  const int K = 256;
  const int N = 256;
  
  auto mat1 = torch::randn({M, K}, torch::kBFloat16);
  auto mat2 = torch::randn({N, K}, torch::kBFloat16);
  auto bias = torch::randn({N}, torch::kFloat32);
  
  // Reference: PyTorch (cast to FP32 for reference)
  auto mat1_f32 = mat1.to(torch::kFloat32);
  auto mat2_f32 = mat2.to(torch::kFloat32);
  auto mat2_t = mat2_f32.t().contiguous();
  auto ref = torch::addmm(bias, mat1_f32, mat2_t).to(torch::kBFloat16);
  
  // Our implementation
  auto result = s390x_gemm(mat1, mat2, bias);
  
  // Compare (convert to FP32 for comparison)
  auto result_f32 = result.to(torch::kFloat32);
  auto ref_f32 = ref.to(torch::kFloat32);
  auto diff = (result_f32 - ref_f32).abs();
  float max_diff = diff.max().item<float>();
  float mean_diff = diff.mean().item<float>();
  
  std::cout << "  Max absolute error: " << max_diff << std::endl;
  std::cout << "  Mean absolute error: " << mean_diff << std::endl;
  
  // BF16 has lower precision, so use larger tolerance
  if (max_diff < 0.05) {
    std::cout << "  ✓ BF16 GEMM test PASSED" << std::endl;
  } else {
    std::cout << "  ✗ BF16 GEMM test FAILED" << std::endl;
  }
  std::cout << std::endl;
}

void benchmark_fp32_gemm() {
  std::cout << "=== Benchmarking FP32 GEMM ===" << std::endl;
  
  const int M = 512;
  const int K = 512;
  const int N = 512;
  const int warmup = 5;
  const int iterations = 20;
  
  auto mat1 = torch::randn({M, K}, torch::kFloat32);
  auto mat2 = torch::randn({N, K}, torch::kFloat32);
  auto bias = torch::randn({N}, torch::kFloat32);
  
  // Warmup
  for (int i = 0; i < warmup; ++i) {
    auto result = s390x_gemm(mat1, mat2, bias);
  }
  
  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto result = s390x_gemm(mat1, mat2, bias);
  }
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avg_time_ms = duration / (1000.0 * iterations);
  
  // Calculate GFLOPS
  double flops = 2.0 * M * N * K;  // Multiply-add = 2 ops
  double gflops = (flops / 1e9) / (avg_time_ms / 1000.0);
  
  std::cout << "  Matrix size: " << M << "×" << K << " × " << K << "×" << N << std::endl;
  std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
  std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "s390x VXE GEMM Kernel Tests" << std::endl;
  std::cout << "============================" << std::endl << std::endl;
  
  try {
    test_fp32_gemm();
    test_bf16_gemm();
    benchmark_fp32_gemm();
    
    std::cout << "All tests completed!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}

#else
int main() {
  std::cout << "This test requires s390x architecture" << std::endl;
  return 0;
}
#endif
