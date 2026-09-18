#!/usr/bin/env python3
"""Measure PageGauge cache-finalization and small gauge-operation overheads."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[1]
PAGE = 16
DIM = 128


@triton.jit
def quantize_completed_kv_pages_kernel(
    key,
    value,
    key_center,
    value_center,
    key_codes,
    value_codes,
    key_scales,
    value_scales,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    block: tl.constexpr,
):
    page_head = tl.program_id(0)
    head = page_head % num_heads
    page = page_head // num_heads
    offsets = tl.arange(0, block)
    channel = offsets % head_dim
    token = offsets // head_dim
    input_offsets = (
        page * page_size * num_heads * head_dim
        + token * num_heads * head_dim
        + head * head_dim
        + channel
    )
    gauge_offsets = head * head_dim + channel
    centered_key = tl.load(key + input_offsets).to(tl.float32) - tl.load(
        key_center + gauge_offsets
    ).to(tl.float32)
    centered_value = tl.load(value + input_offsets).to(tl.float32) - tl.load(
        value_center + gauge_offsets
    ).to(tl.float32)
    key_scale = tl.maximum(
        tl.max(tl.abs(centered_key), axis=0) / 127.0, 2.0**-20
    )
    value_scale = tl.maximum(
        tl.max(tl.abs(centered_value), axis=0) / 127.0, 2.0**-20
    )
    # Quantize with exactly the FP16 scale that the attention kernel will read.
    stored_key_scale = key_scale.to(tl.float16).to(tl.float32)
    stored_value_scale = value_scale.to(tl.float16).to(tl.float32)
    key_quotient = centered_key / stored_key_scale
    value_quotient = centered_value / stored_value_scale
    rounded_key = tl.where(
        key_quotient >= 0.0,
        tl.floor(key_quotient + 0.5),
        -tl.floor(-key_quotient + 0.5),
    )
    rounded_value = tl.where(
        value_quotient >= 0.0,
        tl.floor(value_quotient + 0.5),
        -tl.floor(-value_quotient + 0.5),
    )
    quantized_key = tl.maximum(-127.0, tl.minimum(127.0, rounded_key))
    quantized_value = tl.maximum(-127.0, tl.minimum(127.0, rounded_value))
    tl.store(key_codes + input_offsets, quantized_key.to(tl.int8))
    tl.store(value_codes + input_offsets, quantized_value.to(tl.int8))
    tl.store(key_scales + page_head, stored_key_scale)
    tl.store(value_scales + page_head, stored_value_scale)


@triton.jit
def append_kv_kernel(
    key,
    value,
    key_pages,
    value_pages,
    page_ids,
    page_offsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
):
    request_head = tl.program_id(0)
    request = request_head // num_heads
    head = request_head % num_heads
    channel = tl.arange(0, head_dim)
    page = tl.load(page_ids + request)
    token = tl.load(page_offsets + request)
    source = (request * num_heads + head) * head_dim + channel
    destination = (
        ((page * page_size + token) * num_heads + head) * head_dim + channel
    )
    tl.store(key_pages + destination, tl.load(key + source))
    tl.store(value_pages + destination, tl.load(value + source))


@triton.jit
def append_finalize_page_gauge_kernel(
    key,
    value,
    key_center,
    value_center,
    exact_key_pages,
    exact_value_pages,
    key_codes,
    value_codes,
    key_scales,
    value_scales,
    page_ids,
    page_offsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
):
    """Append centered K/V and finalize a compressed copy in one launch."""
    request_head = tl.program_id(0)
    request = request_head // num_heads
    head = request_head % num_heads
    channel = tl.arange(0, head_dim)
    page = tl.load(page_ids + request)
    current_token = tl.load(page_offsets + request)
    source = (request * num_heads + head) * head_dim + channel
    gauge = head * head_dim + channel
    centered_key = tl.load(key + source).to(tl.float32) - tl.load(
        key_center + gauge
    ).to(tl.float32)
    centered_value = tl.load(value + source).to(tl.float32) - tl.load(
        value_center + gauge
    ).to(tl.float32)
    current_destination = (
        ((page * page_size + current_token) * num_heads + head) * head_dim
        + channel
    )
    tl.store(exact_key_pages + current_destination, centered_key)
    tl.store(exact_value_pages + current_destination, centered_value)

    closing = current_token == page_size - 1
    key_maximum = tl.zeros((head_dim,), dtype=tl.float32)
    value_maximum = tl.zeros((head_dim,), dtype=tl.float32)
    for token in tl.static_range(0, page_size):
        offset = (
            ((page * page_size + token) * num_heads + head) * head_dim + channel
        )
        page_key = tl.load(exact_key_pages + offset, mask=closing, other=0.0).to(
            tl.float32
        )
        page_value = tl.load(
            exact_value_pages + offset, mask=closing, other=0.0
        ).to(tl.float32)
        key_maximum = tl.maximum(key_maximum, tl.abs(page_key))
        value_maximum = tl.maximum(value_maximum, tl.abs(page_value))
    key_scale = tl.maximum(tl.max(key_maximum, axis=0) / 127.0, 2.0**-20)
    value_scale = tl.maximum(tl.max(value_maximum, axis=0) / 127.0, 2.0**-20)
    stored_key_scale = key_scale.to(tl.float16).to(tl.float32)
    stored_value_scale = value_scale.to(tl.float16).to(tl.float32)
    for token in tl.static_range(0, page_size):
        offset = (
            ((page * page_size + token) * num_heads + head) * head_dim + channel
        )
        page_key = tl.load(exact_key_pages + offset, mask=closing, other=0.0).to(
            tl.float32
        )
        page_value = tl.load(
            exact_value_pages + offset, mask=closing, other=0.0
        ).to(tl.float32)
        key_quotient = page_key / stored_key_scale
        value_quotient = page_value / stored_value_scale
        rounded_key = tl.where(
            key_quotient >= 0.0,
            tl.floor(key_quotient + 0.5),
            -tl.floor(-key_quotient + 0.5),
        )
        rounded_value = tl.where(
            value_quotient >= 0.0,
            tl.floor(value_quotient + 0.5),
            -tl.floor(-value_quotient + 0.5),
        )
        tl.store(
            key_codes + offset,
            tl.maximum(-127.0, tl.minimum(127.0, rounded_key)).to(tl.int8),
            mask=closing,
        )
        tl.store(
            value_codes + offset,
            tl.maximum(-127.0, tl.minimum(127.0, rounded_value)).to(tl.int8),
            mask=closing,
        )
    metadata = page * num_heads + head
    tl.store(key_scales + metadata, stored_key_scale, mask=closing)
    tl.store(value_scales + metadata, stored_value_scale, mask=closing)


def quantize_completed_kv_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    key_center: torch.Tensor,
    value_center: torch.Tensor,
    key_codes: torch.Tensor,
    value_codes: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
) -> None:
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("K and V must have identical [pages, 16, heads, 128] shapes")
    if key.shape[1] != PAGE or key.shape[-1] != DIM:
        raise ValueError("K and V must have shape [pages, 16, heads, 128]")
    if key_center.shape != (key.shape[2], DIM) or value_center.shape != key_center.shape:
        raise ValueError("centers must be [heads, 128]")
    grid = (key.shape[0] * key.shape[2],)
    quantize_completed_kv_pages_kernel[grid](
        key,
        value,
        key_center,
        value_center,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        num_heads=key.shape[2],
        head_dim=DIM,
        page_size=PAGE,
        block=PAGE * DIM,
        num_warps=8,
    )


def append_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
) -> None:
    if key.shape != value.shape or key.ndim != 3 or key.shape[-1] != DIM:
        raise ValueError("K/V inputs must be identical [batch,heads,128] tensors")
    if key_pages.shape != value_pages.shape or key_pages.ndim != 4:
        raise ValueError("page caches must be identical [pages,16,heads,128] tensors")
    if key_pages.shape[1:] != (PAGE, key.shape[1], DIM):
        raise ValueError("page cache shape does not match K/V inputs")
    if page_ids.shape != (key.shape[0],) or page_offsets.shape != page_ids.shape:
        raise ValueError("page IDs and offsets must have one entry per request")
    append_kv_kernel[(key.shape[0] * key.shape[1],)](
        key,
        value,
        key_pages,
        value_pages,
        page_ids,
        page_offsets,
        num_heads=key.shape[1],
        head_dim=DIM,
        page_size=PAGE,
        num_warps=4,
    )


def append_finalize_page_gauge(
    key: torch.Tensor,
    value: torch.Tensor,
    key_center: torch.Tensor,
    value_center: torch.Tensor,
    exact_key_pages: torch.Tensor,
    exact_value_pages: torch.Tensor,
    key_codes: torch.Tensor,
    value_codes: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
) -> None:
    if key.shape != value.shape or key.ndim != 3 or key.shape[-1] != DIM:
        raise ValueError("K/V inputs must be identical [batch,heads,128] tensors")
    expected_pages = (exact_key_pages.shape[0], PAGE, key.shape[1], DIM)
    for tensor in (exact_key_pages, exact_value_pages, key_codes, value_codes):
        if tensor.shape != expected_pages:
            raise ValueError("all exact/code caches must be [pages,16,heads,128]")
    if key_center.shape != (key.shape[1], DIM) or value_center.shape != key_center.shape:
        raise ValueError("centers must be [heads,128]")
    if key_scales.shape != (exact_key_pages.shape[0], key.shape[1]):
        raise ValueError("key scales must be [pages,heads]")
    if value_scales.shape != key_scales.shape:
        raise ValueError("value scales must match key scales")
    if page_ids.shape != (key.shape[0],) or page_offsets.shape != page_ids.shape:
        raise ValueError("page IDs and offsets must have one entry per request")
    append_finalize_page_gauge_kernel[(key.shape[0] * key.shape[1],)](
        key,
        value,
        key_center,
        value_center,
        exact_key_pages,
        exact_value_pages,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        page_ids,
        page_offsets,
        num_heads=key.shape[1],
        head_dim=DIM,
        page_size=PAGE,
        num_warps=4,
    )


def event_times(operation: Callable[[], None], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def controlled_pair_times(
    operations: dict[str, Callable[[], None]],
    cache_scrub: Callable[[], None],
    warmup: int,
    repeats: int,
    divisor: float = 1.0,
) -> dict[str, dict[str, object]]:
    names = list(operations)
    if len(names) != 2:
        raise ValueError("controlled timing requires exactly two methods")
    result: dict[str, dict[str, object]] = {}
    for mode in ("cache_neutral", "cache_hot"):
        for repeat in range(warmup):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                if mode == "cache_neutral":
                    cache_scrub()
                else:
                    operations[name]()
                operations[name]()
        torch.cuda.synchronize()
        events = {name: [] for name in names}
        for repeat in range(repeats):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                if mode == "cache_neutral":
                    cache_scrub()
                else:
                    operations[name]()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operations[name]()
                end.record()
                events[name].append((start, end))
        torch.cuda.synchronize()
        raw = {
            name: [float(start.elapsed_time(end)) / divisor for start, end in pairs]
            for name, pairs in events.items()
        }
        result[mode] = {
            "timings": {name: summarize(times) for name, times in raw.items()},
            "raw_paired_timings_ms": raw,
        }
    return result


def summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p05_ms": ordered[round(0.05 * (len(ordered) - 1))],
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "minimum_ms": min(times),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/page_gauge_int8/overheads_rtx3060.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch <= 0 or args.num_qo_heads <= 0 or args.num_kv_heads <= 0:
        raise SystemExit("batch and head counts must be positive")
    if args.num_qo_heads % args.num_kv_heads:
        raise SystemExit("query heads must be divisible by KV heads")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    key = torch.randn(
        args.batch,
        PAGE,
        args.num_kv_heads,
        DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    key_center = torch.randn(
        args.num_kv_heads,
        DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    value_center = torch.randn_like(key_center) * 0.05
    key_codes = torch.empty_like(key, dtype=torch.int8)
    value_codes = torch.empty_like(value, dtype=torch.int8)
    key_scales = torch.empty(
        args.batch, args.num_kv_heads, device="cuda", dtype=torch.float16
    )
    value_scales = torch.empty_like(key_scales)

    def quantize_count(count: int) -> None:
        quantize_completed_kv_pages(
            key[:count],
            value[:count],
            key_center,
            value_center,
            key_codes[:count],
            value_codes[:count],
            key_scales[:count],
            value_scales[:count],
        )

    quantize_count(args.batch)
    reconstructed_key = (
        key_codes.float() * key_scales[:, None, :, None].float()
        + key_center[None, None].float()
    )
    key_relative = (reconstructed_key - key.float()).norm() / key.float().norm()
    if not torch.isfinite(key_relative) or float(key_relative) >= 0.03:
        raise RuntimeError(f"quantizer correctness failure: relative error {key_relative}")

    quantization_by_pages: dict[str, dict[str, float]] = {}
    p50_by_pages: dict[int, float] = {}
    for count in range(1, args.batch + 1):
        timing = summarize(
            event_times(
                lambda count=count: quantize_count(count), args.warmup, args.repeats
            )
        )
        quantization_by_pages[str(count)] = timing
        p50_by_pages[count] = timing["p50_ms"]

    # With independent request phases, N closed pages per step is Binomial(B,1/16).
    expected_quantization_ms = 0.0
    launch_probability = 0.0
    for count in range(1, args.batch + 1):
        probability = (
            math.comb(args.batch, count)
            * (1.0 / PAGE) ** count
            * ((PAGE - 1.0) / PAGE) ** (args.batch - count)
        )
        launch_probability += probability
        expected_quantization_ms += probability * p50_by_pages[count]

    # Measure the actual serving operation: every decode step must append K/V,
    # so PageGauge finalization is fused into that existing launch.  A full
    # 16-step phase cycle closes exactly one page per request.  Requests begin
    # at evenly staggered phases to avoid a synchronized-page artifact.
    append_key = torch.randn(
        PAGE,
        args.batch,
        args.num_kv_heads,
        DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    append_value = torch.randn_like(append_key)
    page_ids = torch.arange(args.batch, device="cuda", dtype=torch.int32)
    initial_phases = torch.div(
        torch.arange(args.batch, device="cuda", dtype=torch.int32) * PAGE,
        args.batch,
        rounding_mode="floor",
    )
    phase_offsets = [
        (initial_phases + step).remainder(PAGE).contiguous() for step in range(PAGE)
    ]
    baseline_append_key_pages = torch.zeros_like(key)
    baseline_append_value_pages = torch.zeros_like(value)
    fused_exact_key_pages = torch.zeros_like(key)
    fused_exact_value_pages = torch.zeros_like(value)
    fused_key_codes = torch.empty_like(key, dtype=torch.int8)
    fused_value_codes = torch.empty_like(value, dtype=torch.int8)
    fused_key_scales = torch.empty_like(key_scales)
    fused_value_scales = torch.empty_like(value_scales)

    def append_only_cycle() -> None:
        for step in range(PAGE):
            append_kv(
                append_key[step],
                append_value[step],
                baseline_append_key_pages,
                baseline_append_value_pages,
                page_ids,
                phase_offsets[step],
            )

    def fused_append_cycle() -> None:
        for step in range(PAGE):
            append_finalize_page_gauge(
                append_key[step],
                append_value[step],
                key_center,
                value_center,
                fused_exact_key_pages,
                fused_exact_value_pages,
                fused_key_codes,
                fused_value_codes,
                fused_key_scales,
                fused_value_scales,
                page_ids,
                phase_offsets[step],
            )

    # A synchronized phase-zero cycle leaves every page at its finalized state,
    # which permits an unambiguous reconstruction check.
    correctness_offsets = [
        torch.full(
            (args.batch,), step, device="cuda", dtype=torch.int32
        )
        for step in range(PAGE)
    ]
    correctness_baseline_k = torch.zeros_like(key)
    correctness_baseline_v = torch.zeros_like(value)
    correctness_exact_k = torch.zeros_like(key)
    correctness_exact_v = torch.zeros_like(value)
    correctness_codes_k = torch.empty_like(key, dtype=torch.int8)
    correctness_codes_v = torch.empty_like(value, dtype=torch.int8)
    correctness_scales_k = torch.empty_like(key_scales)
    correctness_scales_v = torch.empty_like(value_scales)
    for step in range(PAGE):
        append_kv(
            append_key[step],
            append_value[step],
            correctness_baseline_k,
            correctness_baseline_v,
            page_ids,
            correctness_offsets[step],
        )
        append_finalize_page_gauge(
            append_key[step],
            append_value[step],
            key_center,
            value_center,
            correctness_exact_k,
            correctness_exact_v,
            correctness_codes_k,
            correctness_codes_v,
            correctness_scales_k,
            correctness_scales_v,
            page_ids,
            correctness_offsets[step],
        )
    torch.cuda.synchronize()
    exact_key_error = (
        correctness_exact_k.float()
        + key_center[None, None].float()
        - correctness_baseline_k.float()
    ).abs().max()
    reconstructed_append_key = (
        correctness_codes_k.float()
        * correctness_scales_k[:, None, :, None].float()
        + key_center[None, None].float()
    )
    append_reconstruction_relative = (
        (reconstructed_append_key - correctness_baseline_k.float()).norm()
        / correctness_baseline_k.float().norm()
    )
    if float(exact_key_error) > 2e-3 or float(append_reconstruction_relative) >= 0.03:
        raise RuntimeError(
            "fused append/finalize correctness failure: "
            f"exact={float(exact_key_error)} quantized={float(append_reconstruction_relative)}"
        )

    scrub_buffer = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024 // 4,
        device="cuda",
        dtype=torch.int32,
    )

    def cache_scrub() -> None:
        scrub_buffer.add_(1)

    append_timing_modes = controlled_pair_times(
        {
            "fp16_append_only": append_only_cycle,
            "page_gauge_fused_append_finalize": fused_append_cycle,
        },
        cache_scrub,
        args.warmup,
        args.repeats,
        divisor=PAGE,
    )
    for mode_payload in append_timing_modes.values():
        timings = mode_payload["timings"]
        baseline_append_ms = timings["fp16_append_only"]["p50_ms"]
        fused_append_ms = timings["page_gauge_fused_append_finalize"]["p50_ms"]
        mode_payload["incremental_p50_ms_per_decode_step"] = max(
            0.0, fused_append_ms - baseline_append_ms
        )

    query = torch.randn(
        args.batch,
        args.num_qo_heads,
        DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    query_normalizer = torch.rand(
        args.num_qo_heads,
        DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    ) + 0.5
    transformed_query = torch.empty_like(query)
    output = torch.randn_like(query)
    output_normalizer = torch.rand_like(query_normalizer) + 0.5
    output_center = torch.randn_like(query_normalizer) * 0.05
    restored_output = torch.empty_like(output)

    def query_gauge() -> None:
        torch.mul(query, query_normalizer[None], out=transformed_query)

    def output_gauge() -> None:
        torch.addcmul(
            output_center[None], output, output_normalizer[None], out=restored_output
        )

    query_timing = summarize(event_times(query_gauge, args.warmup, args.repeats))
    output_timing = summarize(event_times(output_gauge, args.warmup, args.repeats))

    payload = {
        "schema_version": 2,
        "experiment": "page_gauge_auxiliary_overheads",
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "workload": {
            "batch": args.batch,
            "num_qo_heads": args.num_qo_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": DIM,
            "page_size": PAGE,
        },
        "correctness": {
            "key_reconstruction_relative_l2": float(key_relative),
            "fused_append_exact_key_max_abs": float(exact_key_error),
            "fused_append_quantized_key_relative_l2": float(
                append_reconstruction_relative
            ),
        },
        "fused_append_finalize": {
            "protocol": (
                "one 16-step decode phase cycle; one page closure per request; "
                "request phases evenly staggered; timings are per decode step"
            ),
            "primary_mode": "cache_neutral",
            "cache_scrub_bytes": scrub_buffer.numel() * scrub_buffer.element_size(),
            "timing_modes": append_timing_modes,
        },
        "quantize_completed_kv_pages": {
            "fused_kv_launch_by_closed_pages": quantization_by_pages,
            "independent_phase_page_close_probability": launch_probability,
            "expected_p50_weighted_ms_per_decode_step": expected_quantization_ms,
            "synchronized_batch_amortized_p50_ms_per_step": p50_by_pages[
                args.batch
            ]
            / PAGE,
        },
        "optional_unfused_diagonal_gauge": {
            "query_transform": query_timing,
            "output_restore": output_timing,
            "p50_total_ms": query_timing["p50_ms"] + output_timing["p50_ms"],
            "note": "These pointwise launches are avoidable by fusing with RoPE/cache-write and output projection.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
