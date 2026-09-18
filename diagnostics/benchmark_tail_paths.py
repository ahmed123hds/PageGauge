#!/usr/bin/env python3
"""Paired same-process benchmark for PageGauge exact-tail integration paths.

This diagnostic isolates the work after the common old-INT8 attention call.  It
compares the custom exact-tail/online-merge/center kernel with the legacy
FlashInfer exact-tail decode followed by merge_state and a center add.  Both
paths are captured at the same CUDA-graph boundary and consume identical
inputs.  It is a component gate, not a full-model publication result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

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
    "page_gauge_transformer_for_tail_benchmark",
    ROOT / "scripts/benchmark_page_gauge_transformer.py",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_us": [value * 1000 for value in values],
        "p10_us": percentile(values, 0.10) * 1000,
        "p50_us": statistics.median(values) * 1000,
        "p90_us": percentile(values, 0.90) * 1000,
        "mean_us": statistics.mean(values) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tail <= 0 or args.tail % PG.PAGE:
        raise SystemExit("tail must be a positive multiple of the page size")
    if args.repeats <= 0 or args.repeats % 2:
        raise SystemExit("repeats must be positive and even")

    import flashinfer

    extension = PG.RUNTIME.load_append_extension()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260816)
    pages = args.tail // PG.PAGE
    query = torch.randn(
        1, 32, PG.DIM, generator=generator, device=device, dtype=torch.float16
    )
    key = torch.randn(
        pages,
        PG.PAGE,
        8,
        PG.DIM,
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    base_output = torch.randn_like(query)
    base_lse = torch.randn(1, 32, generator=generator, device=device)
    center = torch.randn(
        8, PG.DIM, generator=generator, device=device, dtype=torch.float16
    )
    indices = torch.arange(pages, device=device, dtype=torch.int32)
    last_len = torch.tensor([PG.PAGE], device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(PG.DIM)

    exact_wrapper = PG.GraphDecodeWrapper(
        flashinfer, pages, torch.float16, False
    )
    exact_wrapper.plan(indices, args.tail, PG.PAGE, 0)

    fused_output = base_output.clone()
    fused_lse = base_lse.clone()
    legacy_output = base_output.clone()
    legacy_lse = base_lse.clone()
    exact_output = torch.empty_like(query)
    exact_lse = torch.empty_like(base_lse)
    output_center = center.repeat_interleave(4, dim=0)[None].contiguous()

    def fused_body() -> None:
        extension.exact_tail_merge_center(
            query,
            key,
            value,
            indices,
            last_len,
            fused_output,
            fused_lse,
            center,
            scale,
        )

    def legacy_body() -> None:
        exact_wrapper.wrapper.run(
            query,
            (key, value),
            out=exact_output,
            lse=exact_lse,
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            legacy_output, legacy_lse, exact_output, exact_lse
        )
        legacy_output.add_(output_center)

    # First validate the two implementations from identical online states.
    fused_body()
    legacy_body()
    torch.cuda.synchronize()
    correctness = {
        "output_max_abs": float((fused_output - legacy_output).abs().max()),
        "output_cosine": float(
            F.cosine_similarity(
                fused_output.float().reshape(1, -1),
                legacy_output.float().reshape(1, -1),
            ).item()
        ),
        "lse_max_abs": float((fused_lse - legacy_lse).abs().max()),
    }

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(5):
            fused_output.copy_(base_output)
            fused_lse.copy_(base_lse)
            fused_body()
            legacy_output.copy_(base_output)
            legacy_lse.copy_(base_lse)
            legacy_body()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    fused_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fused_graph, stream=capture_stream):
        fused_body()
    legacy_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(legacy_graph, stream=capture_stream):
        legacy_body()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    scrub = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024 // 4,
        device=device,
        dtype=torch.int32,
    )
    graphs = {
        "fused_kernel": (fused_graph, fused_output, fused_lse),
        "flashinfer_merge": (legacy_graph, legacy_output, legacy_lse),
    }
    modes = {}
    for mode in ("cache_neutral", "cache_hot"):
        for _ in range(args.warmups):
            for graph, output, lse in graphs.values():
                if mode == "cache_neutral":
                    scrub.add_(1)
                output.copy_(base_output)
                lse.copy_(base_lse)
                graph.replay()
        torch.cuda.synchronize()

        events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in graphs
        }
        names = list(graphs)
        for repeat in range(args.repeats):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                graph, output, lse = graphs[name]
                if mode == "cache_neutral":
                    scrub.add_(1)
                output.copy_(base_output)
                lse.copy_(base_lse)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                events[name].append((start, end))
        torch.cuda.synchronize()
        samples = {
            name: [float(start.elapsed_time(end)) for start, end in pairs]
            for name, pairs in events.items()
        }
        ratios = [
            legacy / fused
            for legacy, fused in zip(
                samples["flashinfer_merge"], samples["fused_kernel"]
            )
        ]
        modes[mode] = {
            name: summarize(values) for name, values in samples.items()
        }
        modes[mode]["fused_speedup"] = {
            "paired_geomean": math.exp(
                statistics.mean(math.log(value) for value in ratios)
            ),
            "paired_median": statistics.median(ratios),
            "paired_range": [min(ratios), max(ratios)],
        }

    result = {
        "schema_version": 1,
        "experiment": "page_gauge_exact_tail_path_component_gate",
        "scope": "post-old-INT8 attention only; matched CUDA-graph boundary",
        "tail_tokens": args.tail,
        "correctness": correctness,
        "modes": modes,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
