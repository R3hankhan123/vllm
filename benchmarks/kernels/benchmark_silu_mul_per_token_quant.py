# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time fused SiLU-and-mul + per-token FP8 against the two-kernel path.

The fused kernel is what MoE calls between the two expert GEMMs when
activations are dynamic per-token FP8 or INT8. INT8 uses the same kernel;
this script times FP8 because that unfused quant is a CUDA kernel, so the
comparison is kernel time only (no allocator traffic).

Run on a CUDA or ROCm GPU after building vLLM:

    .venv/bin/python benchmarks/kernels/benchmark_silu_mul_per_token_quant.py

Decode-sized rows are a few microseconds. Serving decode tokens/s will not
move, because the expert weights dominate. The number that should drop is
the large-token row (prefill, with top-k folded into the token count).

Unfused traffic per output element is 11 bytes (read gate+up, write the
bf16 SiLU, read it twice, write fp8). The fused kernel is 5 bytes (read
gate+up, write fp8) when the row fits in shared memory, which it does for
normal MoE intermediates. A bandwidth-bound kernel tops out near 2.2x.
"""

from functools import partial

import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform

# (tokens, hidden). tokens is M * top-k, the rows of the w2 activation.
CASES = [
    (32, 2048),
    (256, 2048),
    (4096, 2048),
    (8192, 7168),
    (32768, 2048),
    (32768, 14336),
]


def _unfused(
    x: torch.Tensor,
    act: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    torch.ops._C.silu_and_mul(act, x)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, act, scale, None)


def _fused(x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul_per_token_quant(out, x, scale, None)


def _capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


def _time_eager(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


_graph_failed = False


def _time_ms(fn, iters: int) -> float:
    global _graph_failed
    if _graph_failed:
        return _time_eager(fn, iters)
    try:
        return _time_graph(_capture(fn), iters)
    except Exception as exc:
        _graph_failed = True
        print(f"CUDA graph capture unavailable ({exc}); timing eagerly.")
        return _time_eager(fn, iters)


def _check(x: torch.Tensor, hidden: int, quant_dtype: torch.dtype) -> None:
    act = torch.empty(x.shape[0], hidden, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act, x)
    ref_q, ref_s = ops.scaled_fp8_quant(act, use_per_token_if_dynamic=True)
    ops_q, ops_s = ops.silu_and_mul_per_token_quant(x, quant_dtype)
    torch.testing.assert_close(ref_s, ops_s, rtol=1e-2, atol=1e-5)
    ref_deq = ref_q.to(torch.float32) * ref_s
    ops_deq = ops_q.to(torch.float32) * ops_s
    torch.testing.assert_close(ref_deq, ops_deq, rtol=5e-2, atol=5e-2)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA or ROCm GPU.")

    quant_dtype = current_platform.fp8_dtype()
    iters = 50
    print(
        f"{'tokens':>8} {'hidden':>8} {'unfused_ms':>12} {'fused_ms':>10} "
        f"{'speedup':>8} {'unfused_GB/s':>14} {'fused_GB/s':>12}"
    )
    for tokens, hidden in CASES:
        x = torch.randn(tokens, hidden * 2, dtype=torch.bfloat16, device="cuda")
        _check(x, hidden, quant_dtype)

        act = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        out_u = torch.empty(tokens, hidden, dtype=quant_dtype, device="cuda")
        out_f = torch.empty(tokens, hidden, dtype=quant_dtype, device="cuda")
        scale_u = torch.empty(tokens, 1, dtype=torch.float32, device="cuda")
        scale_f = torch.empty(tokens, 1, dtype=torch.float32, device="cuda")

        unfused_ms = _time_ms(partial(_unfused, x, act, out_u, scale_u), iters)
        fused_ms = _time_ms(partial(_fused, x, out_f, scale_f), iters)

        # Bytes moved, per the comment at the top of this file.
        unfused_bytes = tokens * 11 * hidden
        fused_bytes = tokens * 5 * hidden
        unfused_gbps = unfused_bytes / (unfused_ms * 1e-3) / 1e9
        fused_gbps = fused_bytes / (fused_ms * 1e-3) / 1e9
        speedup = unfused_ms / fused_ms
        print(
            f"{tokens:8d} {hidden:8d} {unfused_ms:12.4f} {fused_ms:10.4f} "
            f"{speedup:7.2f}x {unfused_gbps:14.1f} {fused_gbps:12.1f}"
        )

    print(
        "\nLarge-token rows are the prefill win. Decode rows (tens of tokens) "
        "are microseconds and will not change tokens/s."
    )


if __name__ == "__main__":
    main()
