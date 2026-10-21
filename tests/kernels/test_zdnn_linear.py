# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the zDNN (IBM Z NNPA) prepacked linear kernel."""

import pytest
import torch

from vllm import _custom_ops as ops

if not ops.zdnn_is_available():
    pytest.skip("zDNN kernels or NNPA not available", allow_module_level=True)

# Covers: the CPU decode GEMV (M <= 8) including ragged 16-column tiles and K
# split across threads with a partial last K block, N/K not multiples of the
# 64-wide stick, M not a multiple of the 32-row stick (padded last slice),
# multiple M-slices, and N above the NNPA max dimension index size (split into
# several matmul calls).
SHAPES = [
    (1, 128, 64),
    (4, 3584, 2048),
    (5, 17, 129),
    (7, 491, 1023),
    (3, 300, 1100),
    (8, 1000, 2049),
    (9, 129, 257),
    (257, 4096, 4096),
    (33, 40000, 128),
]


@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("use_bias", [True, False])
def test_zdnn_weight_packed_linear(
    m: int, n: int, k: int, dtype: torch.dtype, use_bias: bool
):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=dtype)
    weight = (torch.randn(n, k) / k**0.5).to(dtype)
    bias = torch.randn(n, dtype=dtype) if use_bias else None

    packed = ops.zdnn_convert_weight_packed(weight, bias)
    out = ops.zdnn_weight_packed_linear(x, packed, n)

    ref = torch.nn.functional.linear(
        x.float(), weight.float(), None if bias is None else bias.float()
    )
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def test_zdnn_weight_packed_linear_strided_3d_input():
    """Non-contiguous rows and leading batch dims match the dense layout."""
    torch.manual_seed(0)
    k, n = 256, 192
    x = torch.randn(2, 45, 2 * k, dtype=torch.bfloat16)[..., :k]
    weight = (torch.randn(n, k) / k**0.5).to(torch.bfloat16)

    packed = ops.zdnn_convert_weight_packed(weight, None)
    out = ops.zdnn_weight_packed_linear(x, packed, n)

    ref = torch.nn.functional.linear(x.float(), weight.float())
    assert out.shape == (2, 45, n)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


# m=4 takes the CPU decode GEMV, m=1537 the NNPA path with a partial M-slice.
@pytest.mark.parametrize("m", [4, 1537])
@torch.inference_mode()
def test_zdnn_weight_packed_linear_inference_mode(m: int):
    """Inference mode (as used by vLLM) is not inherited by OpenMP worker
    threads; both paths must still write the output without error."""
    torch.manual_seed(0)
    n, k = 3584, 2048
    x = torch.randn(m, k, dtype=torch.bfloat16)
    weight = (torch.randn(n, k) / k**0.5).to(torch.bfloat16)

    packed = ops.zdnn_convert_weight_packed(weight, None)
    out = ops.zdnn_weight_packed_linear(x, packed, n)

    ref = torch.nn.functional.linear(x.float(), weight.float())
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


# m=4 takes the CPU decode GEMV, m=64 the NNPA path.
@pytest.mark.parametrize("m", [4, 64])
def test_zdnn_weight_packed_linear_non_finite_input(m: int):
    """NaN/Inf activations (e.g. vLLM's profiling run reading uninitialized
    buffers) give non-finite rows instead of raising; other rows are exact."""
    torch.manual_seed(0)
    n, k = 300, 256
    x = torch.randn(m, k, dtype=torch.bfloat16)
    x[0, 3] = float("nan")
    x[1, 7] = float("inf")
    weight = (torch.randn(n, k) / k**0.5).to(torch.bfloat16)

    packed = ops.zdnn_convert_weight_packed(weight, None)
    out = ops.zdnn_weight_packed_linear(x, packed, n)

    ref = torch.nn.functional.linear(x.float(), weight.float())
    assert not out[:2].isfinite().all(dim=1).any()
    torch.testing.assert_close(out[2:].float(), ref[2:], atol=3e-2, rtol=3e-2)
