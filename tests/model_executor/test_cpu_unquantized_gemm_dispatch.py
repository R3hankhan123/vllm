# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU unquantized GEMM dispatch behavior."""

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


def test_dispatch_prepacks_arm_bf16_causal_conv(monkeypatch):
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.ARM
    )
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: {"bf16": True})
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)

    packed = torch.randn(32, 4, dtype=torch.bfloat16)
    pack_inputs = []

    def pack(weight):
        pack_inputs.append(weight.clone())
        return packed

    monkeypatch.setattr(utils.ops, "causal_conv1d_weight_pack", pack)
    original = torch.randn(32, 1, 4, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(original.clone(), requires_grad=False)

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert len(pack_inputs) == 1
    # Check causal_conv1d_weight_pack received the expected weights
    torch.testing.assert_close(pack_inputs[0], original.view(32, 4))
    # Check we have correctly stashed the unpacked weights
    torch.testing.assert_close(layer._cpu_unpacked_conv_weight, original.view(32, 4))
    # Check we have correctly stored the packed weights
    assert layer.weight.data_ptr() == packed.data_ptr()


def test_dispatch_does_not_pack_3d_expert_weight(monkeypatch):
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.ARM
    )
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: {"bf16": True})
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)
    pack_calls = []
    monkeypatch.setattr(
        utils.ops,
        "causal_conv1d_weight_pack",
        lambda weight: pack_calls.append(weight),
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(8, 32, 4, dtype=torch.bfloat16), requires_grad=False
    )
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    assert pack_calls == []


@pytest.fixture(scope="module")
def _mock_zentorch_linear_unary():
    """Register a mock zentorch_linear_unary op when zentorch is not installed.

    Allows the dispatch tests to run in CI without a real zentorch build.
    Skips registration when zentorch is already available.
    """
    if hasattr(torch.ops.zentorch, "zentorch_linear_unary"):
        yield
        return

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define(
        "zentorch_linear_unary("
        "Tensor input, "
        "Tensor weight, "
        "Tensor? bias, "
        "bool is_weight_prepacked=False"
        ") -> Tensor"
    )

    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl(
        "zentorch_linear_unary",
        lambda input, weight, bias, is_weight_prepacked=False: (
            torch.nn.functional.linear(input, weight, bias)
        ),
    )

    yield

    lib_impl._destroy()
    lib_def._destroy()


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_uses_zentorch_on_zen(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    x = torch.randn(4, 16)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    output = layer.cpu_linear(x, layer.weight, layer.bias)

    torch.testing.assert_close(output, expected)


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 0


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    expected_prepacked = bool(utils.envs.VLLM_ZENTORCH_WEIGHT_PREPACK) and hasattr(
        torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
    )

    log_calls = []
    monkeypatch.setattr(
        utils.logger, "debug_once", lambda *args: log_calls.append(args)
    )

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert log_calls == [
        (
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            expected_prepacked,
        )
    ]


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dispatch_cpu_unquantized_gemm_remove_weight_keeps_dtype(
    monkeypatch, weight_dtype
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=False, dtype=weight_dtype)
    loading_dtype = (
        torch.float32 if weight_dtype is not torch.float32 else torch.bfloat16
    )
    with set_default_torch_dtype(loading_dtype):
        utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 0
    assert layer.weight.dtype is weight_dtype

    x = torch.randn(4, 16, dtype=weight_dtype)
    output = layer.cpu_linear(x, layer.weight, None)
    assert not output.isnan().any()


class _FakeZdnnHandler:
    def __init__(self, k: int, n: int) -> None:
        self.k = k
        self.n = n


def _patch_s390x_zdnn_dispatch(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(
        utils.current_platform, "get_cpu_architecture", lambda: CpuArchEnum.S390X
    )
    monkeypatch.setattr(utils.current_platform, "is_zen_cpu", lambda: False)
    monkeypatch.setattr(utils, "check_cpu_sgl_kernel", lambda *args, **kwargs: False)
    monkeypatch.setattr(utils.envs, "VLLM_ZDNN_LINEAR", True, raising=False)
    monkeypatch.setattr(utils.ops, "zdnn_linear_available", lambda: available)


def test_dispatch_s390x_uses_zdnn_linear(monkeypatch):
    _patch_s390x_zdnn_dispatch(monkeypatch, available=True)

    created = {}

    def create(weight, bias):
        created["weight"] = weight.detach().clone()
        created["bias"] = None if bias is None else bias.detach().clone()
        return _FakeZdnnHandler(weight.size(0), weight.size(1))

    ran = {}

    def run(handler, x):
        ran["k"] = handler.k
        ran["n"] = handler.n
        ran["m"] = x.shape[0]
        return torch.empty(*x.shape[:-1], handler.n, dtype=x.dtype)

    monkeypatch.setattr(utils.ops, "create_zdnn_linear", create)
    monkeypatch.setattr(utils.ops, "zdnn_linear", run)

    layer = torch.nn.Linear(4, 8, bias=True)
    weight = layer.weight.detach().clone()
    bias = layer.bias.detach().clone()
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    torch.testing.assert_close(created["weight"], weight.t().contiguous())
    torch.testing.assert_close(created["bias"], bias)
    assert layer.weight.numel() == 0
    out = layer.cpu_linear(torch.randn(3, 4), layer.weight, layer.bias)
    assert ran == {"k": 4, "n": 8, "m": 3}
    assert out.shape == (3, 8)


def test_dispatch_s390x_zdnn_disabled_keeps_weight(monkeypatch):
    _patch_s390x_zdnn_dispatch(monkeypatch, available=True)
    monkeypatch.setattr(utils.envs, "VLLM_ZDNN_LINEAR", False, raising=False)
    monkeypatch.setattr(utils.ops, "_supports_onednn", False)

    def create(weight, bias):
        raise AssertionError("zDNN linear is disabled")

    monkeypatch.setattr(utils.ops, "create_zdnn_linear", create)
    layer = torch.nn.Linear(4, 8, bias=False)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 4 * 8


def test_dispatch_s390x_zdnn_failure_keeps_weight(monkeypatch):
    _patch_s390x_zdnn_dispatch(monkeypatch, available=True)
    monkeypatch.setattr(utils.ops, "_supports_onednn", False)

    def create(weight, bias):
        raise RuntimeError("exceeds NNPA maximum tensor size")

    monkeypatch.setattr(utils.ops, "create_zdnn_linear", create)
    logs = []
    monkeypatch.setattr(utils.logger, "warning_once", lambda *args: logs.append(args))

    layer = torch.nn.Linear(4, 8, bias=False)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 4 * 8
    assert logs and "Failed to create zDNN linear" in logs[0][0]
