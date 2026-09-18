#!/usr/bin/env python3
"""Aggregate paired PageGauge timing runs and include page-finalization cost."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, nargs="+", required=True)
    parser.add_argument("--overheads", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_ci(
    values: list[float], samples: int, seed: int
) -> list[float]:
    generator = random.Random(seed)
    bootstrapped = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]


def candidate_name(timings: dict[str, Any]) -> str:
    candidates = [name for name in timings if name != "flashinfer_fp16"]
    if len(candidates) != 1:
        raise ValueError(f"expected one candidate timing, found {candidates}")
    return candidates[0]


def aggregate_mode(
    records: list[dict[str, Any]], mode: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    raw_speedups = [record["timing_modes"][mode]["attention_only_speedup"] for record in records]
    inclusive_speedups = [record["timing_modes"][mode]["inclusive_speedup"] for record in records]
    return {
        "attention_only_speedup_median": statistics.median(raw_speedups),
        "attention_only_speedup_range": [min(raw_speedups), max(raw_speedups)],
        "attention_only_seed_bootstrap_median_95_ci": bootstrap_median_ci(
            raw_speedups, bootstrap_samples, seed
        ),
        "inclusive_speedup_median": statistics.median(inclusive_speedups),
        "inclusive_speedup_range": [
            min(inclusive_speedups),
            max(inclusive_speedups),
        ],
        "inclusive_seed_bootstrap_median_95_ci": bootstrap_median_ci(
            inclusive_speedups, bootstrap_samples, seed + 1
        ),
    }


def main() -> None:
    args = parse_args()
    if len(args.benchmarks) < 3:
        raise SystemExit("at least three independent benchmark runs are required")
    runs = [json.loads(path.read_text()) for path in args.benchmarks]
    overhead = json.loads(args.overheads.read_text())
    append_modes = overhead["fused_append_finalize"]["timing_modes"]
    required_modes = ("cache_neutral", "cache_hot")
    if any(mode not in append_modes for mode in required_modes):
        raise ValueError("overhead artifact is missing cache-neutral/cache-hot modes")
    reference_environment = runs[0]["environment"]
    reference_workload = {
        key: value
        for key, value in runs[0]["workload"].items()
        if key != "seed"
    }
    records: list[dict[str, Any]] = []
    for path, run in zip(args.benchmarks, runs):
        workload = {
            key: value for key, value in run["workload"].items() if key != "seed"
        }
        if run["environment"] != reference_environment or workload != reference_workload:
            raise ValueError(f"environment/workload mismatch in {path}")
        if set(run.get("timing_modes", {})) != set(required_modes):
            raise ValueError(f"timing-mode mismatch in {path}")
        mode_records: dict[str, Any] = {}
        for mode in required_modes:
            mode_payload = run["timing_modes"][mode]
            timings = mode_payload["timings"]
            method = candidate_name(timings)
            baseline_ms = timings["flashinfer_fp16"]["p50_ms"]
            candidate_ms = timings[method]["p50_ms"]
            incremental_append_ms = append_modes[mode][
                "incremental_p50_ms_per_decode_step"
            ]
            mode_records[mode] = {
                "baseline_p50_ms": baseline_ms,
                "candidate_attention_p50_ms": candidate_ms,
                "incremental_fused_append_p50_ms": incremental_append_ms,
                "candidate_inclusive_p50_ms": candidate_ms + incremental_append_ms,
                "attention_only_speedup": baseline_ms / candidate_ms,
                "inclusive_speedup": baseline_ms
                / (candidate_ms + incremental_append_ms),
                "paired_samples": len(
                    mode_payload["raw_paired_timings_ms"][method]
                ),
            }
        primary = mode_records["cache_neutral"]
        records.append(
            {
                "path": str(path.resolve()),
                "seed": run["workload"].get("seed"),
                "timing_modes": mode_records,
                **primary,
            }
        )
    mode_aggregates = {
        mode: aggregate_mode(
            records,
            mode,
            args.bootstrap_samples,
            args.seed + 100 * mode_index,
        )
        for mode_index, mode in enumerate(required_modes)
    }
    primary_aggregate = mode_aggregates["cache_neutral"]
    payload = {
        "schema_version": 2,
        "experiment": "page_gauge_cache_controlled_multi_seed_summary",
        "environment": reference_environment,
        "workload": reference_workload,
        "timing_protocol": {
            "primary_mode": "cache_neutral",
            "secondary_mode": "cache_hot",
            "scheduler_selection_mode": "cache_neutral",
        },
        "fused_append_finalize": {
            "source": str(args.overheads.resolve()),
            "cache_neutral_incremental_p50_ms_per_decode_step": append_modes[
                "cache_neutral"
            ]["incremental_p50_ms_per_decode_step"],
            "cache_hot_incremental_p50_ms_per_decode_step": append_modes[
                "cache_hot"
            ]["incremental_p50_ms_per_decode_step"],
            "implementation": "completed-page quantization fused into existing KV append launch",
        },
        "runs": records,
        "aggregate": {
            "num_runs": len(records),
            "primary_mode": "cache_neutral",
            "timing_modes": mode_aggregates,
            **primary_aggregate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
