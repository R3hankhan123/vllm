# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NNPA matmul linear. Skipped unless libzdnn is linked and the facility is up."""

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not ops.zdnn_linear_available(),
    reason="NNPA-MATMUL-OP is not installed",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m_k_n",
    [
        (1, 8, 4),
        (3, 5, 7),
        (16, 64, 32),
        (128, 128, 64),
    ],
)
def test_zdnn_linear_matches_torch(dtype: torch.dtype, m_k_n: tuple[int, int, int]):
    m, k, n = m_k_n
    # NNP-data-type-1 is a 16-bit format, so fp32 results are not bit-exact.
    torch.manual_seed(0)
    weight_kn = torch.randn(k, n, dtype=dtype)
    bias = torch.randn(n, dtype=dtype)
    activation = torch.randn(m, k, dtype=dtype)

    handler = ops.create_zdnn_linear(weight_kn, bias)
    actual = ops.zdnn_linear(handler, activation)
    expected = torch.nn.functional.linear(activation, weight_kn.t(), bias)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    no_bias = ops.create_zdnn_linear(weight_kn, None)
    actual_no_bias = ops.zdnn_linear(no_bias, activation)
    expected_no_bias = torch.nn.functional.linear(activation, weight_kn.t(), None)
    torch.testing.assert_close(actual_no_bias, expected_no_bias, rtol=2e-2, atol=2e-2)
