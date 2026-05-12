# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import ir


class TestConv2d:
    """Test suite for conv2d IR operation"""

    def test_conv2d_registered(self):
        """Verify conv2d is registered in IR registry"""
        assert "conv2d" in ir.ops.conv2d.__class__.__bases__[0].registry
        assert ir.ops.conv2d is not None

    def test_conv2d_basic(self):
        """Test basic conv2d operation"""
        batch_size, in_channels, height, width = 2, 3, 8, 8
        out_channels, kernel_size = 16, 3

        x = torch.randn(batch_size, in_channels, height, width)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        bias = torch.randn(out_channels)

        # Test with explicit args
        output = ir.ops.conv2d(
            x, weight, bias, stride=1, padding=1, dilation=1, groups=1
        )

        # Reference implementation
        ref = torch.nn.functional.conv2d(
            x, weight, bias, stride=1, padding=1, dilation=1, groups=1
        )

        torch.testing.assert_close(output, ref)

    def test_conv2d_without_bias(self):
        """Test conv2d without bias"""
        batch_size, in_channels, height, width = 2, 3, 8, 8
        out_channels, kernel_size = 16, 3

        x = torch.randn(batch_size, in_channels, height, width)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

        output = ir.ops.conv2d(x, weight, None, stride=1, padding=0)
        ref = torch.nn.functional.conv2d(x, weight, None, stride=1, padding=0)

        torch.testing.assert_close(output, ref)

    def test_conv2d_with_stride(self):
        """Test conv2d with different strides"""
        batch_size, in_channels, height, width = 2, 3, 16, 16
        out_channels, kernel_size = 16, 3
        stride = 2

        x = torch.randn(batch_size, in_channels, height, width)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        bias = torch.randn(out_channels)

        output = ir.ops.conv2d(x, weight, bias, stride=stride, padding=1)
        ref = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=1)

        torch.testing.assert_close(output, ref)

    def test_conv2d_with_groups(self):
        """Test grouped convolution (depthwise)"""
        batch_size, channels, height, width = 2, 8, 8, 8
        kernel_size = 3
        groups = 2  # Depthwise separable style

        x = torch.randn(batch_size, channels, height, width)
        # For grouped conv: out_channels must be divisible by groups_size
        weight = torch.randn(channels, channels // groups, kernel_size, kernel_size)
        bias = torch.randn(channels)

        output = ir.ops.conv2d(x, weight, bias, stride=1, padding=1, groups=groups)
        ref = torch.nn.functional.conv2d(
            x, weight, bias, stride=1, padding=1, groups=groups
        )

        torch.testing.assert_close(output, ref)

    def test_fused_conv2d_bias(self):
        """Test fused conv2d with bias"""
        batch_size, in_channels, height, width = 2, 3, 8, 8
        out_channels, kernel_size = 16, 3

        x = torch.randn(batch_size, in_channels, height, width)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        bias = torch.randn(out_channels)

        # Test fused version
        output_fused = ir.ops.fused_conv2d_bias(x, weight, bias, stride=1, padding=1)

        # Reference unfused version
        ref = torch.nn.functional.conv2d(x, weight, None, stride=1, padding=1)
        ref = ref + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        torch.testing.assert_close(output_fused, ref)

    def test_conv2d_different_dtypes(self):
        """Test conv2d with different dtypes"""
        batch_size, in_channels, height, width = 2, 3, 8, 8
        out_channels, kernel_size = 16, 3

        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(batch_size, in_channels, height, width, dtype=dtype)
            weight = torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, dtype=dtype
            )
            bias = torch.randn(out_channels, dtype=dtype)

            output = ir.ops.conv2d(x, weight, bias, stride=1, padding=1)
            ref = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)

            torch.testing.assert_close(
                output, ref, rtol=1e-3, atol=1e-3 if dtype == torch.float16 else 1e-5
            )

    def test_conv2d_input_generator(self):
        """Test input generator for conv2d"""
        # Generate test inputs
        inputs = ir.ops.conv2d.generate_inputs(
            batch_size=2,
            in_channels=3,
            height=16,
            width=16,
            out_channels=32,
            kernel_size=3,
        )

        assert len(inputs) == 7  # x, weight, bias, stride, padding, dilation, groups
        x, weight, bias, stride, padding, dilation, groups = inputs

        assert x.shape == (2, 3, 16, 16)
        assert weight.shape == (32, 3, 3, 3)
        assert bias.shape == (32,)

        # Check that the generated inputs work
        output = ir.ops.conv2d(*inputs)
        assert output.shape[0] == 2  # batch size preserved

    def test_fused_conv2d_bias_input_generator(self):
        """Test input generator for fused_conv2d_bias"""
        inputs = ir.ops.fused_conv2d_bias.generate_inputs(
            batch_size=2,
            in_channels=3,
            height=16,
            width=16,
            out_channels=32,
            kernel_size=3,
        )

        assert len(inputs) == 7
        x, weight, bias, stride, padding, dilation, groups = inputs

        output = ir.ops.fused_conv2d_bias(*inputs)
        assert output.shape[0] == 2


class TestConv2dTorchOps:
    """Test torch custom op integration for conv2d"""

    def test_torch_op_registered(self):
        """Verify torch custom op is registered"""
        assert hasattr(torch.ops.vllm_ir, "conv2d")
        assert hasattr(torch.ops.vllm_ir, "fused_conv2d_bias")

    def test_torch_op_callable(self):
        """Test that torch ops are callable"""
        x = torch.randn(2, 3, 8, 8)
        weight = torch.randn(16, 3, 3, 3)
        bias = torch.randn(16)

        # Call via torch.ops
        output = torch.ops.vllm_ir.conv2d.default(x, weight, bias, 1, 0, 1, 1)

        # Reference
        ref = torch.nn.functional.conv2d(x, weight, bias, 1, 0, 1, 1)

        torch.testing.assert_close(output, ref)
