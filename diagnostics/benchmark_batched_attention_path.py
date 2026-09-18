#!/usr/bin/env python3
"""Gate the final synchronized-batch append+attention integration path.

The diagnostic uses the production TransformerDecoder, request-major caches,
independently frozen split schedules, and matched per-layer CUDA graphs.  It
times the common fused RoPE/append launch plus the complete attention graph at
an ordinary token and at page close.  Their 15:1 weighted latency is the device
work paid by a steady 16-token decode page.  This is a component gate; the
publication claim still comes from the full pretrained model runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PG = load_local_module(
    "page_gauge_transformer_for_batched_attention_gate",
    ROOT / "scripts/benchmark_page_gauge_transformer.py",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "p10_ms": percentile(samples, 0.10),
        "p50_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=20480)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.context <= args.exact_tail:
        raise SystemExit("batch must be positive and context must exceed exact tail")
    if args.context % PG.PAGE or args.exact_tail % PG.PAGE:
        raise SystemExit("context and exact tail must be page aligned")
    if args.repeats <= 0 or args.repeats % 2:
        raise SystemExit("repeats must be positive and even")

    import flashinfer

    model_args = SimpleNamespace(
        model="synthetic-mistral-page-gauge-smoke",
        context=args.context,
        decode_steps=PG.PAGE,
        local_files_only=True,
    )
    model = PG.load_model(model_args)
    extension = PG.RUNTIME.load_append_extension()
    layers, hq, hkv, _ = PG.BASE_E2E.check_model(model)
    if layers != 1:
        raise RuntimeError("component gate expects the one-layer synthetic model")
    max_context = args.context + PG.PAGE + 1
    pages = math.ceil(max_context / PG.PAGE)
    initial_pages = args.context // PG.PAGE
    exact_pages = args.exact_tail // PG.PAGE
    baseline_cache, gauge_cache = PG.build_caches(
        layers,
        pages,
        initial_pages,
        exact_pages,
        hkv,
        args.seed,
        args.batch_size,
    )
    rope_cos = torch.ones(
        max_context, PG.DIM, device="cuda", dtype=torch.float16
    )
    rope_sin = torch.zeros_like(rope_cos)
    baseline = PG.TransformerDecoder(
        model,
        flashinfer,
        extension,
        "flashinfer_fp16",
        baseline_cache,
        max_context,
        args.exact_tail,
        args.baseline_split_pages,
        args.candidate_split_pages,
        rope_cos,
        rope_sin,
        "attention_add",
        "flashinfer_merge",
        args.batch_size,
    )
    candidate = PG.TransformerDecoder(
        model,
        flashinfer,
        extension,
        "page_gauge",
        gauge_cache,
        max_context,
        args.exact_tail,
        args.baseline_split_pages,
        args.candidate_split_pages,
        rope_cos,
        rope_sin,
        "attention_add",
        "flashinfer_merge",
        args.batch_size,
    )
    baseline.capture_attention_graphs(args.context + 1)
    candidate.capture_attention_graphs(args.context + 1)
    baseline.reset_attention_dispatch_counts()
    candidate.reset_attention_dispatch_counts()

    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    packed = torch.randn(
        args.batch_size,
        (hq + 2 * hkv) * PG.DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    ) * 0.35
    q_flat, k_flat, v_flat = torch.split(
        packed,
        (hq * PG.DIM, hkv * PG.DIM, hkv * PG.DIM),
        dim=-1,
    )
    # These are intentionally native row-strided views of packed QKV.  The
    # append ABI consumes their strides directly and must not insert copies.
    query = q_flat.view(args.batch_size, hq, PG.DIM)
    key = k_flat.view(args.batch_size, hkv, PG.DIM)
    value = v_flat.view(args.batch_size, hkv, PG.DIM)
    if query.is_contiguous() or key.is_contiguous() or value.is_contiguous():
        raise RuntimeError("packed-QKV diagnostic unexpectedly became contiguous")

    decoders = {"flashinfer_fp16": baseline, "page_gauge": candidate}

    def operation(decoder: PG.TransformerDecoder, position: int) -> None:
        decoder.plan(position + 1)
        decoder.append(0, query, key, value, position)
        decoder.attention(0)

    # Populate the new exact page before timing its page-close finalization.
    for offset in range(PG.PAGE - 1):
        for decoder in decoders.values():
            operation(decoder, args.context + offset)
    torch.cuda.synchronize()

    # A one-off output comparison verifies that the measured production paths
    # consume the same logical cache state.
    operation(baseline, args.context)
    operation(candidate, args.context)
    torch.cuda.synchronize()
    baseline_output = baseline.attention_output[0].float()
    candidate_output = candidate.attention_output[0].float()
    correctness = {
        "cosine_min": float(
            F.cosine_similarity(
                baseline_output.reshape(args.batch_size, -1),
                candidate_output.reshape(args.batch_size, -1),
            ).min()
        ),
        "relative_l2_max": float(
            (
                (baseline_output - candidate_output).norm(dim=(1, 2))
                / baseline_output.norm(dim=(1, 2)).clamp_min(1e-8)
            ).max()
        ),
        "absolute_max": float((baseline_output - candidate_output).abs().max()),
    }

    scrub = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024 // 4,
        device="cuda",
        dtype=torch.int32,
    )
    positions = {
        "ordinary_token": args.context,
        "page_close": args.context + PG.PAGE - 1,
    }
    result_modes: dict[str, object] = {}
    for mode in ("cache_neutral", "cache_hot"):
        phase_payload = {}
        for phase, position in positions.items():
            for _ in range(args.warmups):
                for decoder in decoders.values():
                    if mode == "cache_neutral":
                        scrub.add_(1)
                    else:
                        operation(decoder, position)
                    operation(decoder, position)
            torch.cuda.synchronize()
            events: dict[
                str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
            ] = {name: [] for name in decoders}
            names = list(decoders)
            for repeat in range(args.repeats):
                order = names if repeat % 2 == 0 else list(reversed(names))
                for name in order:
                    decoder = decoders[name]
                    if mode == "cache_neutral":
                        scrub.add_(1)
                    else:
                        operation(decoder, position)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    operation(decoder, position)
                    end.record()
                    events[name].append((start, end))
            torch.cuda.synchronize()
            samples = {
                name: [float(start.elapsed_time(end)) for start, end in pairs]
                for name, pairs in events.items()
            }
            phase_payload[phase] = {
                name: summarize(values) for name, values in samples.items()
            }
            phase_payload[phase]["paired_speedup_geomean"] = math.exp(
                statistics.mean(
                    math.log(base / pg)
                    for base, pg in zip(
                        samples["flashinfer_fp16"], samples["page_gauge"]
                    )
                )
            )
        weighted = {}
        for name in decoders:
            ordinary = phase_payload["ordinary_token"][name]["p50_ms"]
            close = phase_payload["page_close"][name]["p50_ms"]
            weighted[name] = (15 * ordinary + close) / 16
        phase_payload["weighted_p50_ms"] = weighted
        phase_payload["weighted_p50_speedup"] = (
            weighted["flashinfer_fp16"] / weighted["page_gauge"]
        )
        result_modes[mode] = phase_payload

    result = {
        "schema_version": 1,
        "experiment": "page_gauge_batched_append_attention_component_gate",
        "scope": "synchronized equal-length batch; one production decoder layer",
        "batch_size": args.batch_size,
        "context": args.context,
        "exact_tail": args.exact_tail,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "tail_attention": "flashinfer_merge",
        "center_restore": "attention_add",
        "packed_qkv_native_strides": {
            "query": list(query.stride()),
            "key": list(key.stride()),
            "value": list(value.stride()),
        },
        "correctness": correctness,
        "timing_modes": result_modes,
        "attention_dispatch": {
            name: decoder.attention_dispatch_counts()
            for name, decoder in decoders.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "scripts/benchmark_page_gauge_transformer.py",
                ROOT / "tests/page_gauge_append_extension.cu",
                ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch",
            )
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
