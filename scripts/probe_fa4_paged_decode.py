#!/usr/bin/env python3
"""Probe the installed FlashAttention-4 paged-KV decode path.

The probe intentionally uses a non-identity page table.  A successful result
therefore demonstrates that the paged indirection is executed, rather than
merely accepting a page-table argument while reading contiguous storage.
Unsupported architectures are recorded as a valid probe result and do not
produce a failing exit status.  Import failures and numerical failures do.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--kv-length", type=int, default=257)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--atol", type=float, default=2.5e-3)
    return parser.parse_args()


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def classify_unsupported(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "paged kv not supported" in message or "paged-kv not supported" in message


def manual_reference(torch: Any, q: Any, k: Any, v: Any, page_table: Any, lengths: Any) -> Any:
    outputs = []
    groups = q.shape[1] // k.shape[2]
    scale = q.shape[-1] ** -0.5
    for batch_index in range(q.shape[0]):
        length = int(lengths[batch_index].item())
        logical_k = k[page_table[batch_index].long()].flatten(0, 1)[:length]
        logical_v = v[page_table[batch_index].long()].flatten(0, 1)[:length]
        logical_k = logical_k.repeat_interleave(groups, dim=1).float()
        logical_v = logical_v.repeat_interleave(groups, dim=1).float()
        query = q[batch_index].float()
        scores = torch.einsum("qhd,khd->hqk", query, logical_k) * scale
        output = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), logical_v)
        outputs.append(output)
    return torch.stack(outputs)


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "probe": "flashattention4_paged_kv_decode",
        "timestamp_unix": time.time(),
        "status": "initializing",
        "supported": False,
        "config": {
            "batch_size": args.batch_size,
            "kv_length": args.kv_length,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "atol": args.atol,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "flash_attn_4_distribution": distribution_version("flash-attn-4"),
            "flash_attn_distribution": distribution_version("flash-attn"),
        },
    }

    try:
        import torch
        import flash_attn.cute as flash_attn_cute
        from flash_attn.cute import flash_attn_varlen_func
    except Exception as exc:  # the exact optional dependency failure is useful evidence
        result.update(
            status="import_error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_result(args.output, result)
        return 1

    result["environment"].update(
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        flash_attn_module=str(Path(flash_attn_cute.__file__).resolve()),
        cuda_available=torch.cuda.is_available(),
    )
    if not torch.cuda.is_available():
        result.update(status="no_cuda", error="CUDA is not available")
        write_result(args.output, result)
        return 1

    if args.batch_size < 1 or args.kv_length < 1 or args.page_size < 1:
        raise ValueError("batch size, KV length, and page size must be positive")
    if args.cache_scrub_mib <= 0:
        raise ValueError("cache scrub size must be positive")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q-heads must be divisible by kv-heads")

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    result["environment"].update(
        gpu_name=properties.name,
        compute_capability=[properties.major, properties.minor],
        total_memory_bytes=properties.total_memory,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pages_per_sequence = math.ceil(args.kv_length / args.page_size)
    total_pages = args.batch_size * pages_per_sequence
    q = torch.randn(
        args.batch_size, 1, args.q_heads, args.head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn(
        total_pages, args.page_size, args.kv_heads, args.head_dim,
        device=device, dtype=torch.float16,
    )
    v = torch.randn_like(k)
    # Reverse all physical page IDs so an implementation that ignores the table
    # cannot accidentally pass the numerical check.
    page_table = torch.arange(
        total_pages - 1, -1, -1, device=device, dtype=torch.int32
    ).reshape(args.batch_size, pages_per_sequence)
    lengths = torch.full(
        (args.batch_size,), args.kv_length, device=device, dtype=torch.int32
    )
    cu_seqlens_q = torch.arange(
        args.batch_size + 1, device=device, dtype=torch.int32
    )
    q_varlen = q.flatten(0, 1)

    def run() -> Any:
        output = flash_attn_varlen_func(
            q_varlen,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=lengths,
            max_seqlen_q=1,
            max_seqlen_k=pages_per_sequence * args.page_size,
            page_table=page_table,
            causal=True,
        )
        return output[0] if isinstance(output, tuple) else output

    try:
        output = run()
        torch.cuda.synchronize()
    except Exception as exc:
        unsupported = classify_unsupported(exc)
        result.update(
            status="unsupported_architecture" if unsupported else "runtime_error",
            supported=False,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_result(args.output, result)
        return 0 if unsupported else 1

    reference = manual_reference(torch, q, k, v, page_table, lengths).flatten(0, 1)
    difference = (output.float() - reference).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    correct = max_abs <= args.atol

    scrub_buffer = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024 // 4,
        device=device,
        dtype=torch.int32,
    )

    def measure(mode: str) -> list[float]:
        for _ in range(args.warmup):
            if mode == "cache_neutral":
                scrub_buffer.add_(1)
            else:
                run()
            run()
        torch.cuda.synchronize()
        events = []
        for _ in range(args.repeats):
            if mode == "cache_neutral":
                scrub_buffer.add_(1)
            else:
                run()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in events]

    timing_modes = {}
    for mode in ("cache_neutral", "cache_hot"):
        timings_ms = measure(mode)
        timing_modes[mode] = {
            "median": percentile(timings_ms, 0.5),
            "p95": percentile(timings_ms, 0.95),
            "samples": timings_ms,
        }

    result.update(
        status="supported" if correct else "numerical_failure",
        supported=True,
        correctness={"passed": correct, "max_abs": max_abs, "mean_abs": mean_abs},
        timing_protocol={
            "primary_mode": "cache_neutral",
            "cache_scrub_bytes": scrub_buffer.numel() * scrub_buffer.element_size(),
        },
        timing_modes=timing_modes,
        latency_ms=timing_modes["cache_neutral"],
    )
    write_result(args.output, result)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
