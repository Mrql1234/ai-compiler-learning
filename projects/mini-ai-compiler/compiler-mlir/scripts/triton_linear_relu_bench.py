#!/usr/bin/env python3
"""Benchmark a Triton fused linear + relu kernel on a fixed case/config."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from triton_perf_common import (
    LinearReluCase,
    compare_close,
    config_tag,
    detect_device_name,
    effective_gbps,
    effective_gflops,
    load_config_payload,
    load_linear_relu_case,
    max_abs_err,
    max_rel_err,
    median_metric,
    nvtx_range,
    select_config,
    summarize_scalar,
    summarize_timings,
    validate_case,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton fused linear + relu")
    parser.add_argument(
        "--case",
        type=Path,
        required=True,
        help="Path to a Triton perf case JSON file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Triton perf config JSON file",
    )
    parser.add_argument(
        "--config-source",
        default="default",
        choices=["default", "profile_target"],
        help="Named config section used as the base before CLI overrides",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--emit-nvtx", action="store_true")
    parser.add_argument("--BLOCK_M", type=int, default=None)
    parser.add_argument("--BLOCK_N", type=int, default=None)
    parser.add_argument("--BLOCK_K", type=int, default=None)
    parser.add_argument("--GROUP_M", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    return parser.parse_args()


def require_torch_and_triton() -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("missing dependency: torch") from exc
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("missing dependency: triton") from exc
    return torch, triton, tl


def make_kernel(tl: Any) -> Any:
    # The nested function keeps Triton import local to runtime execution.
    import triton

    @triton.jit
    def kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        m,
        n,
        k,
        stride_im,
        stride_ik,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        del _constexpr_identity
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, BLOCK_M)
        num_pid_n = tl.cdiv(n, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(k, BLOCK_K)):
            current_k = k_start * BLOCK_K + offs_k
            input_ptrs = input_ptr + offs_m[:, None] * stride_im + current_k[None, :] * stride_ik
            weight_ptrs = weight_ptr + current_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
            input_mask = (offs_m[:, None] < m) & (current_k[None, :] < k)
            weight_mask = (current_k[:, None] < k) & (offs_n[None, :] < n)
            input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)
            weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
            acc += tl.dot(input_tile, weight_tile)

        bias = tl.load(bias_ptr + offs_n, mask=offs_n < n, other=0.0)
        acc += bias[None, :]
        acc = tl.maximum(acc, 0.0)

        output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        output_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        tl.store(output_ptrs, acc, mask=output_mask)

    return kernel


def make_inputs(torch: Any, case: LinearReluCase, device_index: int) -> tuple[Any, Any, Any]:
    device = f"cuda:{device_index}"
    if case.data_profile == "deterministic":
        input_tensor = torch.arange(case.m * case.k, device=device, dtype=torch.float32).reshape(
            case.m, case.k
        )
        input_tensor = (input_tensor % 17) / 17.0
        weight_tensor = torch.arange(case.n * case.k, device=device, dtype=torch.float32).reshape(
            case.n, case.k
        )
        weight_tensor = ((weight_tensor % 23) - 11) / 23.0
        bias_tensor = torch.linspace(-0.25, 0.25, case.n, device=device, dtype=torch.float32)
        return input_tensor, weight_tensor, bias_tensor
    if case.data_profile == "uniform":
        input_tensor = torch.rand((case.m, case.k), device=device, dtype=torch.float32) * 2.0 - 1.0
        weight_tensor = torch.rand((case.n, case.k), device=device, dtype=torch.float32) * 2.0 - 1.0
        bias_tensor = torch.rand((case.n,), device=device, dtype=torch.float32) * 0.5 - 0.25
        return input_tensor, weight_tensor, bias_tensor
    input_tensor = torch.randn((case.m, case.k), device=device, dtype=torch.float32)
    weight_tensor = torch.randn((case.n, case.k), device=device, dtype=torch.float32)
    bias_tensor = torch.randn((case.n,), device=device, dtype=torch.float32)
    return input_tensor, weight_tensor, bias_tensor


def reference_output(torch: Any, input_tensor: Any, weight_tensor: Any, bias_tensor: Any) -> Any:
    return torch.relu(input_tensor @ weight_tensor.t() + bias_tensor)


def run_kernel(
    torch: Any,
    triton: Any,
    tl: Any,
    case: LinearReluCase,
    config: dict[str, int],
    warmup: int,
    repeat: int,
    device_index: int,
    emit_nvtx: bool,
) -> dict[str, Any]:
    torch.cuda.set_device(device_index)
    torch.manual_seed(case.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = make_kernel(tl)
    input_tensor, weight_tensor, bias_tensor = make_inputs(torch, case, device_index)
    output_tensor = torch.empty((case.m, case.n), device=input_tensor.device, dtype=torch.float32)
    expected = reference_output(torch, input_tensor, weight_tensor, bias_tensor)

    grid = (
        triton.cdiv(case.m, config["BLOCK_M"]) * triton.cdiv(case.n, config["BLOCK_N"]),
    )

    def launch() -> None:
        kernel[grid](
            input_tensor,
            weight_tensor,
            bias_tensor,
            output_tensor,
            case.m,
            case.n,
            case.k,
            input_tensor.stride(0),
            input_tensor.stride(1),
            weight_tensor.stride(0),
            weight_tensor.stride(1),
            output_tensor.stride(0),
            output_tensor.stride(1),
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            GROUP_M=config["GROUP_M"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    with nvtx_range(torch, "triton_linear_relu/prepare", emit_nvtx):
        launch()
        torch.cuda.synchronize(device_index)
        output_tensor.zero_()

    kernel_ms: list[float] = []
    invoke_ms: list[float] = []
    with nvtx_range(torch, "triton_linear_relu/warmup", emit_nvtx):
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize(device_index)

    with nvtx_range(torch, "triton_linear_relu/benchmark", emit_nvtx):
        for _ in range(repeat):
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter()
            start_event.record()
            launch()
            stop_event.record()
            torch.cuda.synchronize(device_index)
            host_end = time.perf_counter()
            kernel_ms.append(float(start_event.elapsed_time(stop_event)))
            invoke_ms.append((host_end - host_start) * 1000.0)

    abs_err = max_abs_err(output_tensor, expected)
    rel_err = max_rel_err(output_tensor, expected)
    correct = compare_close(abs_err, rel_err, case.abs_tol, case.rel_tol)
    kernel_summary = summarize_timings(kernel_ms, "cuda_event")
    invoke_summary = summarize_timings(invoke_ms, "host_wall")
    kernel_median = median_metric({"kernel_ms": kernel_summary}, "kernel_ms")
    output_bytes = output_tensor.element_size()

    return {
        "device": detect_device_name(torch, device_index),
        "config": config,
        "metrics": {
            "kernel_ms": kernel_summary,
            "invoke_ms": invoke_summary,
            "effective_gflops": summarize_scalar(effective_gflops(case, kernel_median), "GFLOP/s"),
            "effective_gbps": summarize_scalar(
                effective_gbps(case, kernel_median, output_bytes), "GB/s"
            ),
        },
        "correctness": {
            "correct": correct,
            "max_abs_err": abs_err,
            "max_rel_err": rel_err,
            "abs_tol": case.abs_tol,
            "rel_tol": case.rel_tol,
        },
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0")

    case = load_linear_relu_case(args.case)
    validate_case(case)
    overrides = {
        "BLOCK_M": args.BLOCK_M,
        "BLOCK_N": args.BLOCK_N,
        "BLOCK_K": args.BLOCK_K,
        "GROUP_M": args.GROUP_M,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
    }
    overrides = {key: int(value) for key, value in overrides.items() if value is not None}
    config_payload = load_config_payload(args.config)
    config = select_config(config_payload, args.config_source, overrides)
    if args.seed is not None:
        case = LinearReluCase(
            name=case.name,
            description=case.description,
            operation=case.operation,
            dtype=case.dtype,
            m=case.m,
            n=case.n,
            k=case.k,
            data_profile=case.data_profile,
            seed=args.seed,
            abs_tol=case.abs_tol,
            rel_tol=case.rel_tol,
            layout=case.layout,
        )

    torch, triton, tl = require_torch_and_triton()
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise SystemExit("CUDA runtime not detected; run this script on a GPU machine")

    payload = run_kernel(
        torch=torch,
        triton=triton,
        tl=tl,
        case=case,
        config=config,
        warmup=args.warmup,
        repeat=args.repeat,
        device_index=args.device_index,
        emit_nvtx=args.emit_nvtx,
    )
    result = {
        "case": {
            "name": case.name,
            "description": case.description,
            "operation": case.operation,
            "dtype": case.dtype,
            "problem": {"m": case.m, "n": case.n, "k": case.k},
            "data_profile": case.data_profile,
            "seed": case.seed,
            "layout": case.layout,
        },
        "run": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "config_source": args.config_source,
            "config_tag": config_tag(config),
        },
        **payload,
    }
    if args.json_output is not None:
        write_json(args.json_output, result)

    kernel_median = median_metric(result["metrics"], "kernel_ms")
    correct = result["correctness"]["correct"]
    print(f"case: {case.name}")
    print(f"device: {result['device']}")
    print(f"config: {config_tag(config)}")
    print(f"correct: {'yes' if correct else 'no'}")
    print(f"kernel_ms.median: {kernel_median:.6f}" if kernel_median is not None else "kernel_ms.median: -")
    if args.json_output is not None:
        print(f"json: {args.json_output}")
    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(main())
