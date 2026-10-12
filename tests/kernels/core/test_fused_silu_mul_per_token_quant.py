# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from tests.utils import large_gpu_test
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.platforms import current_platform

DTYPES = [torch.float16, torch.bfloat16]
QUANT_DTYPES = [current_platform.fp8_dtype(), torch.int8]
# 70 is not a multiple of 4 (scalar path). 100 is a multiple of 4 but not 8.
# 14336 is a normal MoE intermediate and stages the row in shared memory.
NUM_TOKENS_HIDDEN_SIZES = [
    (1, 64),
    (1, 70),
    (3, 100),
    (16, 128),
    (16, 2048),
    (8, 4096),
    (4, 14336),
]
SEEDS = [0]


def _reference(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unfused SiLU-and-mul, then the quant the MoE path uses."""
    hidden = x.shape[-1] // 2
    act = torch.empty(x.shape[0], hidden, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act, x)
    if quant_dtype == torch.int8:
        q, scale = per_token_quant_int8(act)
        return q, scale, act
    q, scale = ops.scaled_fp8_quant(
        act, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )
    return q, scale, act


def _assert_close(
    ref_q: torch.Tensor,
    ref_s: torch.Tensor,
    ops_q: torch.Tensor,
    ops_s: torch.Tensor,
    act: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    scale_rtol: float,
    compare_to_act: bool = True,
) -> None:
    assert ops_q.dtype == quant_dtype
    assert ops_s.dtype == torch.float32
    assert ops_s.shape == ref_s.shape
    assert not torch.isnan(ops_q.float()).any()
    assert not torch.isnan(ops_s).any()
    torch.testing.assert_close(ref_s, ops_s, rtol=scale_rtol, atol=1e-5)

    ref_deq = ref_q.to(torch.float32) * ref_s.to(torch.float32)
    ops_deq = ops_q.to(torch.float32) * ops_s.to(torch.float32)
    if quant_dtype == torch.int8:
        # One code is absmax/127, so dequant atol is the wrong check.
        delta = (ops_q.to(torch.int16) - ref_q.to(torch.int16)).abs()
        assert int(delta.max()) <= 2
        return
    # A scale upper bound saturates large values, so dequant no longer
    # matches the unclamped SiLU output.
    if compare_to_act:
        torch.testing.assert_close(ops_deq, act.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(ref_deq, ops_deq, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_silu_and_mul_per_token_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
) -> None:
    """Fused SiLU+quant matches silu_and_mul then per-token quant."""
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)

    ref_q, ref_s, act = _reference(x, quant_dtype, None)
    ops_q, ops_s = ops.silu_and_mul_per_token_quant(x, quant_dtype)

    _assert_close(ref_q, ref_s, ops_q, ops_s, act, quant_dtype, scale_rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_silu_and_mul_per_token_quant_scale_ub(
    default_vllm_config, dtype: torch.dtype
) -> None:
    """scale_ub clamps the absmax, matching dynamic_per_token_scaled_fp8_quant."""
    device = "cuda"
    # silu(1) * 1 ~= 0.73, so every row's absmax is above the bound.
    x = torch.ones(4, 256, dtype=dtype, device=device)
    quant_dtype = current_platform.fp8_dtype()
    scale_ub = torch.tensor([0.05], dtype=torch.float32, device=device)

    ref_q, ref_s, act = _reference(x, quant_dtype, None)
    ops_q, ops_s = ops.silu_and_mul_per_token_quant(x, quant_dtype)
    _assert_close(ref_q, ref_s, ops_q, ops_s, act, quant_dtype, scale_rtol=1e-5)

    ref_q, ref_s, act = _reference(x, quant_dtype, scale_ub)
    ops_q, ops_s = ops.silu_and_mul_per_token_quant(x, quant_dtype, scale_ub)
    # Both sides clamp to the same absmax, so the scale is the bound / qmax.
    _assert_close(
        ref_q,
        ref_s,
        ops_q,
        ops_s,
        act,
        quant_dtype,
        scale_rtol=1e-5,
        compare_to_act=False,
    )
    assert torch.all(ops_s > 0)
    assert torch.all(ops_s < 0.05)


@torch.inference_mode()
def test_silu_and_mul_per_token_quant_opcheck(default_vllm_config) -> None:
    device = "cuda"
    x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
    out = torch.empty(4, 64, dtype=current_platform.fp8_dtype(), device=device)
    scales = torch.empty(4, 1, dtype=torch.float32, device=device)
    opcheck(
        torch.ops._C.silu_and_mul_per_token_quant,
        (out, x, scales, None),
    )


@large_gpu_test(min_gb=24)
@torch.inference_mode()
def test_silu_and_mul_per_token_quant_int32_offset(default_vllm_config) -> None:
    """Token offsets past INT32_MAX must not wrap."""
    hidden_size = 16384
    quant_dtype = current_platform.fp8_dtype()
    num_tokens = (2**31 - 1) // (2 * hidden_size) + 2

    x = torch.randn(num_tokens, hidden_size * 2, dtype=torch.bfloat16, device="cuda")
    out, scales = ops.silu_and_mul_per_token_quant(x, quant_dtype)
    torch.accelerator.synchronize()

    expected_out, expected_scales = ops.silu_and_mul_per_token_quant(
        x[-1:].contiguous(), quant_dtype
    )
    assert torch.equal(scales[-1:], expected_scales)
    assert torch.equal(out[-1:], expected_out)
