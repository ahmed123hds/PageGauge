#!/usr/bin/env python3
"""Measure SM120 FP16x2 arithmetic on zero, subnormal, and normal inputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=2**20)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/half2_subnormal_sm120.json"),
    )
    return parser.parse_args()


def time_case(module, source: torch.Tensor, output: torch.Tensor, iterations: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    module.half2_chain(source, output, iterations)
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.elements <= 0 or args.elements % 2:
        raise SystemExit("--elements must be positive and even")
    if args.iterations <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("invalid iterations/warmups/repeats")
    module = load(
        name="page_gauge_half2_subnormal_sm120_v2",
        sources=[str(HERE / "half2_subnormal_extension.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    output = torch.empty(args.elements, device="cuda", dtype=torch.float16)
    cases = {
        "zero": 0.0,
        "minimum_subnormal": 2.0**-24,
        "mid_subnormal": 2.0**-16,
        "minimum_normal": 2.0**-14,
        "typical_probability": 1.0 / 20480.0,
        "one": 1.0,
    }
    result = {}
    for name, value in cases.items():
        source = torch.full(
            (args.elements,), value, device="cuda", dtype=torch.float16
        )
        for _ in range(args.warmups):
            module.half2_chain(source, output, args.iterations)
        torch.cuda.synchronize()
        samples = [
            time_case(module, source, output, args.iterations)
            for _ in range(args.repeats)
        ]
        result[name] = {
            "requested_value": value,
            "stored_value": float(source[0]),
            "output_value": float(output[0]),
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
        }
    normal_median = result["minimum_normal"]["median_ms"]
    for payload in result.values():
        payload["ratio_to_minimum_normal"] = payload["median_ms"] / normal_median
    payload = {
        "experiment": "sm120_half2_subnormal_arithmetic",
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "elements": args.elements,
        "half2_pairs": args.elements // 2,
        "iterations": args.iterations,
        "half2_operations_per_sample": args.elements // 2 * args.iterations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cases": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["median_ms"] for name, value in result.items()}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
