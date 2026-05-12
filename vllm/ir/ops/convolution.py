# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


@register_op
def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return torch.nn.functional.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@conv2d.register_input_generator
def _conv2d_input_generator(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    out_channels: int,
    kernel_size: int,
    dtype: torch.dtype,
) -> tuple:
    x = torch.randn(batch_size, in_channels, height, width, dtype=dtype)
    weight = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, dtype=dtype
    )
    bias = torch.randn(out_channels, dtype=dtype)
    return x, weight, bias, 1, 0, 1, 1


@register_op
def fused_conv2d_bias(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    out = torch.nn.functional.conv2d(
        x,
        weight,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return out + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)


@fused_conv2d_bias.register_input_generator
def _fused_conv2d_bias_input_generator(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    out_channels: int,
    kernel_size: int,
    dtype: torch.dtype,
) -> tuple:
    x = torch.randn(batch_size, in_channels, height, width, dtype=dtype)
    weight = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, dtype=dtype
    )
    bias = torch.randn(out_channels, dtype=dtype)
    return x, weight, bias, 1, 0, 1, 1
