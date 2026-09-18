#!/usr/bin/env python3
"""Benchmark contraction-factorized PageGauge INT8 in FlashInfer FA2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR_INCLUDE = ROOT / "build/gauge_affine_flashinfer/include"
EXPECTED_VENDOR_HEADER_SHA256 = (
    "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"
)
PAGE = 16
DIM = 128
GROUP = 16
GROUPS = DIM // GROUP
HQ = 32
HKV = 8
DEFAULT_LENGTHS = (49152, 2048, 4096, 6144, 8192, 10240, 12288, 14336)


AFFINE_VARIANT = r"""
namespace flashinfer {

template <>
struct vec_cast<half, int8_t> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const int8_t* src) {
    static_assert(vec_size % 4 == 0);
    const half2 bias = __float2half2_rn(1152.0f);
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      uint32_t packed = reinterpret_cast<const uint32_t*>(src)[i] ^ 0x80808080u;
      uint32_t lo = 0x64006400u | (packed & 0x000000ffu) |
                    ((packed & 0x0000ff00u) << 8);
      uint32_t hi = 0x64006400u | ((packed & 0x00ff0000u) >> 16) |
                    ((packed & 0xff000000u) >> 8);
      reinterpret_cast<half2*>(dst)[2 * i] =
          __hsub2(*reinterpret_cast<half2*>(&lo), bias);
      reinterpret_cast<half2*>(dst)[2 * i + 1] =
          __hsub2(*reinterpret_cast<half2*>(&hi), bias);
    }
  }
};

}  // namespace flashinfer

struct PageAffineInt8Attention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ PageAffineInt8Attention(const Params& params,
                                               uint32_t batch_idx,
                                               uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }
};
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", default=",".join(str(value) for value in DEFAULT_LENGTHS)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument(
        "--cache-scrub-mib",
        type=int,
        default=256,
        help=(
            "GPU buffer touched before every cache-neutral sample. The primary "
            "scheduler and speed claim use this order-independent mode."
        ),
    )
    parser.add_argument(
        "--representation",
        choices=("affine_group16", "page_gauge"),
        default="affine_group16",
    )
    parser.add_argument(
        "--exact-tail",
        type=int,
        default=0,
        help="Keep this many most-recent tokens per request in exact FP16 and merge states.",
    )
    parser.add_argument(
        "--exact-sink",
        type=int,
        default=0,
        help="Keep this many first tokens per request in the same exact FP16 segment.",
    )
    parser.add_argument(
        "--fixed-split-pages",
        type=int,
        default=0,
        help="Force length-balanced FA2 KV chunks (0 keeps FlashInfer's automatic planner).",
    )
    parser.add_argument(
        "--baseline-fixed-split-pages",
        type=int,
        default=None,
        help="Override split pages for the FP16 baseline only.",
    )
    parser.add_argument(
        "--candidate-fixed-split-pages",
        type=int,
        default=None,
        help="Override split pages for INT8 and its exact segment only.",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/page_affine_int8/flashinfer_rtx3060.json",
    )
    return parser.parse_args()


def install_vendor_generator_hook() -> None:
    vendor_header = VENDOR_INCLUDE / "flashinfer/attention/prefill.cuh"
    if not vendor_header.is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/prepare_flashinfer_page_gauge.py")],
            check=True,
        )
    observed_header_sha256 = hashlib.sha256(vendor_header.read_bytes()).hexdigest()
    if observed_header_sha256 != EXPECTED_VENDOR_HEADER_SHA256:
        raise RuntimeError(
            "legacy PageGauge vendor header SHA256 mismatch: expected "
            f"{EXPECTED_VENDOR_HEADER_SHA256}, got {observed_header_sha256}"
        )
    import flashinfer.decode as decode
    from flashinfer.jit.attention import modules

    modules.dtype_map_kv[torch.int8] = "int8_t"
    original = decode.gen_customize_batch_prefill_module

    def with_vendor_include(*args, **kwargs):
        spec = original(*args, **kwargs)
        existing = list(spec.extra_include_dirs or [])
        vendor = VENDOR_INCLUDE.resolve()
        spec.extra_include_dirs = [vendor, *[path for path in existing if path != vendor]]
        return spec

    decode.gen_customize_batch_prefill_module = with_vendor_include


def make_affine_wrapper(flashinfer, workspace: torch.Tensor):
    install_vendor_generator_hook()
    jit_args = [
        "page_affine_int8_fa2_sm86_v1",
        torch.float16,
        torch.int8,
        torch.float16,
        torch.int32,
        DIM,
        DIM,
        ["k_affine_bias", "k_affine_scale", "v_affine_bias", "v_affine_scale"],
        ["half", "half", "half", "half"],
        ["sm_scale"],
        ["double"],
        "PageAffineInt8Attention",
        AFFINE_VARIANT,
    ]
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
        jit_args=jit_args,
    )


def make_page_gauge_wrapper(flashinfer, workspace: torch.Tensor, **wrapper_kwargs):
    install_vendor_generator_hook()
    header = VENDOR_INCLUDE / "flashinfer/attention/prefill.cuh"
    header_payload = header.read_bytes()
    variant_payload = AFFINE_VARIANT.encode("utf-8")
    module_source_sha256 = hashlib.sha256(
        header_payload + b"\0" + variant_payload
    ).hexdigest()
    jit_args = [
        f"page_gauge_int8_fa2_v4_{module_source_sha256[:16]}",
        torch.float16,
        torch.int8,
        torch.float16,
        torch.int32,
        DIM,
        DIM,
        ["k_page_scale", "v_page_scale"],
        ["half", "half"],
        ["sm_scale"],
        ["double"],
        "PageAffineInt8Attention",
        AFFINE_VARIANT,
    ]
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
        jit_args=jit_args,
        **wrapper_kwargs,
    )
    wrapper.page_gauge_module_uri = jit_args[0]
    wrapper.page_gauge_source_hashes = {
        "header_sha256": hashlib.sha256(header_payload).hexdigest(),
        "expected_header_sha256": EXPECTED_VENDOR_HEADER_SHA256,
        "header_matches_expected": True,
        "variant_sha256": hashlib.sha256(variant_payload).hexdigest(),
        "module_source_sha256": module_source_sha256,
    }
    return wrapper


def quantize_pages(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.ndim != 4 or values.shape[1:] != (PAGE, HKV, DIM):
        raise ValueError("values must be [pages, 16, HKV, 128]")
    pages = values.permute(0, 2, 1, 3).float().reshape(
        values.shape[0], HKV, PAGE, GROUPS, GROUP
    )
    minimum = pages.amin(dim=(2, 4), keepdim=True)
    maximum = pages.amax(dim=(2, 4), keepdim=True)
    scale = ((maximum - minimum) / 255.0).clamp_min(2.0**-20).half()
    minimum = minimum.half()
    unsigned = ((pages - minimum.float()) / scale.float()).round().clamp(0, 255)
    signed = (unsigned - 128).to(torch.int8)
    bias = (minimum + 128.0 * scale).reshape(values.shape[0], HKV, GROUPS).contiguous()
    scale = scale.reshape(values.shape[0], HKV, GROUPS).contiguous()
    reconstructed = (
        signed.float() * scale[:, :, None, :, None].float()
        + bias[:, :, None, :, None].float()
    ).reshape(values.shape[0], HKV, PAGE, DIM).permute(0, 2, 1, 3).half().contiguous()
    codes = signed.reshape(values.shape[0], HKV, PAGE, DIM).permute(0, 2, 1, 3).contiguous()
    return codes, bias, scale, reconstructed


def quantize_page_gauge(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center channels globally, then use one signed-INT8 scale per page/head."""
    if values.ndim != 4 or values.shape[1:] != (PAGE, HKV, DIM):
        raise ValueError("values must be [pages, 16, HKV, 128]")
    center = values.float().mean(dim=(0, 1)).half()
    centered = values.float() - center.float()[None, None, :, :]
    page_head = centered.permute(0, 2, 1, 3)
    scale = (page_head.abs().amax(dim=(2, 3)) / 127.0).clamp_min(2.0**-20).half()
    codes = (
        (page_head / scale.float()[:, :, None, None])
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    reconstructed_centered = (
        codes.permute(0, 2, 1, 3).float() * scale.float()[:, :, None, None]
    ).permute(0, 2, 1, 3).half().contiguous()
    centered = centered.half().contiguous()
    return codes, center, scale.contiguous(), reconstructed_centered, centered


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


def controlled_event_times(
    operations: dict[str, Callable[[], None]],
    warmup: int,
    repeats: int,
    cache_scrub: Callable[[], None],
) -> dict[str, dict[str, object]]:
    """Measure each method in explicit cache-neutral and cache-hot states.

    The previous A/B, B/A protocol placed identical candidate calls next to
    each other at every order reversal.  A cache-sized B1 workload therefore
    alternated between two timing regimes.  Here every cache-neutral sample is
    preceded by an unrelated cache scrub, while every cache-hot sample is
    preceded by an untimed invocation of the same method.
    """
    names = list(operations)
    if len(names) != 2:
        raise ValueError("paired timing requires exactly two operations")

    results: dict[str, dict[str, object]] = {}
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
        events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in names
        }
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
            name: [float(start.elapsed_time(end)) for start, end in event_pairs]
            for name, event_pairs in events.items()
        }
        results[mode] = {
            "timings": {name: summarize(times) for name, times in raw.items()},
            "raw_paired_timings_ms": raw,
        }
    return results


def summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p05_ms": ordered[round(0.05 * (len(ordered) - 1))],
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "minimum_ms": min(times),
    }


def plan(
    wrapper,
    indptr,
    indices,
    last_page_len,
    kv_dtype,
    fixed_split_pages: int = 0,
    num_qo_heads: int = HQ,
    num_kv_heads: int = HKV,
    head_dim: int = DIM,
    page_size: int = PAGE,
) -> None:
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=kv_dtype,
        o_data_type=torch.float16,
        sm_scale=1.0 / math.sqrt(head_dim),
        fixed_split_size=fixed_split_pages or None,
    )


def segment_page_table(
    pages_per_row: list[int], tail_pages: int, sink_pages: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact page tables for the INT8 middle and exact sink+tail."""
    old_counts = [max(count - tail_pages - sink_pages, 0) for count in pages_per_row]
    exact_counts = [count - old for count, old in zip(pages_per_row, old_counts)]
    old_indices: list[int] = []
    exact_indices: list[int] = []
    physical_start = 0
    for count, old_count in zip(pages_per_row, old_counts):
        middle_start = physical_start + sink_pages
        old_indices.extend(range(middle_start, middle_start + old_count))
        exact_indices.extend(range(physical_start, middle_start))
        exact_indices.extend(range(middle_start + old_count, physical_start + count))
        physical_start += count

    def make_indptr(counts: list[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )

    return (
        make_indptr(old_counts),
        torch.tensor(old_indices, device="cuda", dtype=torch.int32),
        make_indptr(exact_counts),
        torch.tensor(exact_indices, device="cuda", dtype=torch.int32),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    import flashinfer

    lengths = [int(item) for item in args.lengths.split(",") if item]
    if any(length % PAGE for length in lengths):
        raise SystemExit("all lengths must be page aligned")
    if args.exact_tail < 0 or args.exact_tail % PAGE:
        raise SystemExit("--exact-tail must be a non-negative multiple of page size")
    if args.exact_sink < 0 or args.exact_sink % PAGE:
        raise SystemExit("--exact-sink must be a non-negative multiple of page size")
    if args.fixed_split_pages < 0:
        raise SystemExit("--fixed-split-pages must be non-negative")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("--cache-scrub-mib must be positive")
    baseline_split_pages = (
        args.fixed_split_pages
        if args.baseline_fixed_split_pages is None
        else args.baseline_fixed_split_pages
    )
    candidate_split_pages = (
        args.fixed_split_pages
        if args.candidate_fixed_split_pages is None
        else args.candidate_fixed_split_pages
    )
    if baseline_split_pages < 0 or candidate_split_pages < 0:
        raise SystemExit("baseline and candidate split pages must be non-negative")
    if args.exact_tail + args.exact_sink and any(
        length <= args.exact_tail + args.exact_sink for length in lengths
    ):
        raise SystemExit("every sequence must be longer than exact sink plus tail")
    pages_per_row = [length // PAGE for length in lengths]
    total_pages = sum(pages_per_row)
    indptr = torch.tensor(
        [0, *torch.tensor(pages_per_row).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)
    last_page_len = torch.full(
        (len(lengths),), PAGE, device="cuda", dtype=torch.int32
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    query = torch.randn(
        len(lengths), HQ, DIM, generator=generator, device="cuda", dtype=torch.float16
    ) * 0.35
    dense_k = torch.randn(
        total_pages,
        PAGE,
        HKV,
        DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    ) * 0.35
    dense_v = torch.randn(
        dense_k.shape, generator=generator, device="cuda", dtype=torch.float16
    ) * 0.35
    if args.representation == "affine_group16":
        k_codes, k_bias, k_scale, reconstructed_k = quantize_pages(dense_k)
        v_codes, v_bias, v_scale, reconstructed_v = quantize_pages(dense_v)
        centered_k, centered_v = dense_k, dense_v
        output_center = None
    else:
        k_codes, k_center, k_scale, reconstructed_k, centered_k = quantize_page_gauge(
            dense_k
        )
        v_codes, v_center, v_scale, reconstructed_v, centered_v = quantize_page_gauge(
            dense_v
        )
        k_bias = v_bias = None
        output_center = (
            v_center.repeat_interleave(HQ // HKV, dim=0)
            .unsqueeze(0)
            .expand(len(lengths), -1, -1)
            .contiguous()
        )

    baseline_workspace = torch.empty(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    affine_workspace = torch.empty(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    baseline = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        baseline_workspace, "NHD", use_tensor_cores=True, backend="fa2"
    )
    affine = (
        make_affine_wrapper(flashinfer, affine_workspace)
        if args.representation == "affine_group16"
        else make_page_gauge_wrapper(flashinfer, affine_workspace)
    )
    plan(
        baseline,
        indptr,
        indices,
        last_page_len,
        torch.float16,
        baseline_split_pages,
    )
    baseline_output = torch.empty_like(query)
    affine_output = torch.empty_like(query)

    if args.exact_tail or args.exact_sink:
        tail_pages = args.exact_tail // PAGE
        sink_pages = args.exact_sink // PAGE
        old_indptr, old_indices, tail_indptr, tail_indices = segment_page_table(
            pages_per_row, tail_pages, sink_pages
        )
        segment_last_page_len = torch.full_like(last_page_len, PAGE)
        tail_workspace = torch.empty(
            128 * 1024 * 1024, device="cuda", dtype=torch.uint8
        )
        tail = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            tail_workspace, "NHD", use_tensor_cores=True, backend="fa2"
        )
        plan(
            affine,
            old_indptr,
            old_indices,
            segment_last_page_len,
            torch.int8,
            candidate_split_pages,
        )
        plan(
            tail,
            tail_indptr,
            tail_indices,
            segment_last_page_len,
            torch.float16,
            candidate_split_pages,
        )
        old_lse = torch.empty(
            len(lengths), HQ, device="cuda", dtype=torch.float32
        )
        tail_output = torch.empty_like(query)
        tail_lse = torch.empty_like(old_lse)

        def run_baseline() -> None:
            baseline.run(query, (dense_k, dense_v), out=baseline_output)

        def run_affine() -> None:
            if args.representation == "affine_group16":
                affine.run(
                    query,
                    (k_codes, v_codes),
                    k_bias,
                    k_scale,
                    v_bias,
                    v_scale,
                    1.0 / math.sqrt(DIM),
                    out=affine_output,
                    lse=old_lse,
                    return_lse=True,
                )
            else:
                affine.run(
                    query,
                    (k_codes, v_codes),
                    k_scale,
                    v_scale,
                    1.0 / math.sqrt(DIM),
                    out=affine_output,
                    lse=old_lse,
                    return_lse=True,
                )
            tail.run(
                query,
                (centered_k, centered_v),
                out=tail_output,
                lse=tail_lse,
                return_lse=True,
            )
            flashinfer.merge_state_in_place(
                affine_output, old_lse, tail_output, tail_lse
            )
            if output_center is not None:
                affine_output.add_(output_center)

        method_name = f"{args.representation}_int8_exact_segments"
        reference_description = "exact_fp16_full_cache"
    else:
        plan(
            affine,
            indptr,
            indices,
            last_page_len,
            torch.int8,
            candidate_split_pages,
        )

        def run_baseline() -> None:
            baseline.run(query, (reconstructed_k, reconstructed_v), out=baseline_output)

        def run_affine() -> None:
            if args.representation == "affine_group16":
                affine.run(
                    query,
                    (k_codes, v_codes),
                    k_bias,
                    k_scale,
                    v_bias,
                    v_scale,
                    1.0 / math.sqrt(DIM),
                    out=affine_output,
                )
            else:
                affine.run(
                    query,
                    (k_codes, v_codes),
                    k_scale,
                    v_scale,
                    1.0 / math.sqrt(DIM),
                    out=affine_output,
                )

        method_name = f"{args.representation}_int8"
        reference_description = "fp16_reconstructed_cache"

    run_baseline()
    run_affine()
    torch.cuda.synchronize()
    difference = affine_output.float() - baseline_output.float()
    relative = difference.norm(dim=-1) / baseline_output.float().norm(dim=-1).clamp_min(1e-8)

    operations = {"flashinfer_fp16": run_baseline, method_name: run_affine}
    scrub_elements = args.cache_scrub_mib * 1024 * 1024 // 4
    scrub_buffer = torch.zeros(scrub_elements, device="cuda", dtype=torch.int32)

    def cache_scrub() -> None:
        scrub_buffer.add_(1)

    timing_modes = controlled_event_times(
        operations, args.warmup, args.repeats, cache_scrub
    )
    for mode, mode_payload in timing_modes.items():
        print(mode, flush=True)
        mode_timings = mode_payload["timings"]
        for name in operations:
            print(
                f"  {name:24s} p50={mode_timings[name]['p50_ms']:.6f} ms",
                flush=True,
            )
    if args.representation == "affine_group16":
        compressed_fraction = (PAGE * GROUP + 2 * 2) / (PAGE * GROUP * 2)
    else:
        compressed_fraction = (PAGE * DIM + 2) / (PAGE * DIM * 2)
    old_tokens = sum(
        length - args.exact_tail - args.exact_sink for length in lengths
    )
    total_tokens = sum(lengths)
    traffic_fraction = (
        old_tokens * compressed_fraction
        + len(lengths) * (args.exact_tail + args.exact_sink)
    ) / total_tokens
    payload = {
        "schema_version": 3,
        "experiment": "flashinfer_factorized_int8_kv",
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "workload": {
            "lengths": lengths,
            "num_qo_heads": HQ,
            "num_kv_heads": HKV,
            "head_dim": DIM,
            "page_size": PAGE,
            "group_size": GROUP if args.representation == "affine_group16" else DIM,
            "exact_tail_tokens": args.exact_tail,
            "exact_sink_tokens": args.exact_sink,
            "representation": args.representation,
            "fixed_split_pages": args.fixed_split_pages,
            "baseline_fixed_split_pages": baseline_split_pages,
            "candidate_fixed_split_pages": candidate_split_pages,
            "seed": args.seed,
        },
        "correctness": {
            "reference": reference_description,
            "absolute_max": float(difference.abs().max()),
            "relative_l2_mean": float(relative.mean()),
            "relative_l2_max": float(relative.max()),
        },
        "timing_protocol": {
            "primary_mode": "cache_neutral",
            "cache_scrub_bytes": scrub_buffer.numel() * scrub_buffer.element_size(),
            "cache_neutral": (
                "touch an unrelated GPU buffer before each timed method; "
                "scrub execution is outside the CUDA event"
            ),
            "cache_hot": (
                "run the same method immediately before its timed invocation"
            ),
            "method_order": "alternate A/B and B/A within each explicit mode",
        },
        "timing_modes": timing_modes,
        "speedup_over_fp16_p50": {
            mode: mode_payload["timings"]["flashinfer_fp16"]["p50_ms"]
            / mode_payload["timings"][method_name]["p50_ms"]
            for mode, mode_payload in timing_modes.items()
        },
        "kv_byte_fraction": traffic_fraction,
        "optimistic_byte_speedup": 1.0 / traffic_fraction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
