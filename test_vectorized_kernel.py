#!/usr/bin/env python3
"""
Standalone test script for the vectorized swigluoai_and_mul kernel.
Tests correctness across different data types, dimensions, and alignment scenarios.
"""

import torch
import sys

def swigluoai_and_mul_reference(x: torch.Tensor, alpha: float = 1.0, limit: float = 30.0) -> torch.Tensor:
    """Reference implementation in pure PyTorch."""
    # Split into gate and up components
    # Input shape: [..., 2*d] where data is interleaved [g0, u0, g1, u1, ...]
    gate = x[..., 0::2]  # Even indices
    up = x[..., 1::2]    # Odd indices
    
    # Clamp gate: min=None, max=limit
    clamped_gate = torch.clamp(gate, max=limit)
    
    # Clamp up: min=-limit, max=limit
    clamped_up = torch.clamp(up, min=-limit, max=limit)
    
    # GLU = gate * sigmoid(gate * alpha)
    glu = clamped_gate * torch.sigmoid(clamped_gate * alpha)
    
    # (up + 1) * glu
    return (clamped_up + 1.0) * glu


def test_vectorized_kernel():
    """Test the vectorized kernel implementation."""
    print("=" * 80)
    print("Testing Vectorized swigluoai_and_mul Kernel")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU kernels.")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"✓ Using device: {device}")
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Import vLLM ops
    try:
        import vllm._C
        swigluoai_kernel = torch.ops._C.swigluoai_and_mul
        print("✓ Successfully imported vLLM swigluoai_and_mul kernel")
    except Exception as e:
        print(f"❌ Failed to import vLLM kernel: {e}")
        print("   You may need to rebuild vLLM with: pip install -e .")
        sys.exit(1)
    
    print()
    
    # Test parameters
    test_cases = [
        # (num_tokens, d, dtype, description)
        (7, 512, torch.float32, "Small batch, FP32"),
        (83, 512, torch.float32, "Medium batch, FP32"),
        (2048, 512, torch.float32, "Large batch, FP32"),
        (7, 13824, torch.float32, "Small batch, large d, FP32"),
        (7, 511, torch.float32, "Odd dimension (511), FP32"),  # Test tail loop
        (7, 512, torch.float16, "Small batch, FP16"),
        (83, 512, torch.float16, "Medium batch, FP16"),
        (7, 512, torch.bfloat16, "Small batch, BF16"),
        (83, 512, torch.bfloat16, "Medium batch, BF16"),
    ]
    
    alpha = 1.0
    limit = 30.0
    
    passed = 0
    failed = 0
    
    for num_tokens, d, dtype, description in test_cases:
        print(f"Testing: {description}")
        print(f"  Shape: [{num_tokens}, {2*d}] -> [{num_tokens}, {d}]")
        
        # Skip BF16 on devices without support
        if dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print(f"  ⊘ SKIPPED (BF16 not supported on this GPU)")
                print()
                continue
        
        # Create test input - interleaved format [gate, up, gate, up, ...]
        torch.manual_seed(42)
        x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)
        
        # Allocate output tensor
        out = torch.empty(num_tokens, d, dtype=dtype, device=device)
        
        # Run kernel
        try:
            swigluoai_kernel(out, x, alpha, limit)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ❌ FAILED: Kernel execution error: {e}")
            failed += 1
            print()
            continue
        
        # Compute reference
        ref_out = swigluoai_and_mul_reference(x, alpha, limit)
        
        # Compare results
        # Use appropriate tolerances based on dtype
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        elif dtype == torch.float16:
            atol, rtol = 1e-3, 2e-3
        else:  # bfloat16
            atol, rtol = 1e-2, 2e-2
        
        try:
            torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
            
            # Compute statistics
            max_diff = (out - ref_out).abs().max().item()
            mean_diff = (out - ref_out).abs().mean().item()
            
            print(f"  ✓ PASSED")
            print(f"    Max diff:  {max_diff:.2e} (atol={atol:.2e})")
            print(f"    Mean diff: {mean_diff:.2e} (rtol={rtol:.2e})")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = test_vectorized_kernel()
    sys.exit(exit_code)
