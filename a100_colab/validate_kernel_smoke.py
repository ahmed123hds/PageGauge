#!/usr/bin/env python3
"""Validate the A100 factorized-kernel smoke against explicit reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    failures: list[str] = []
    environment = source.get("environment", {})
    workload = source.get("workload", {})
    correctness = source.get("correctness", {})
    if "A100" not in str(environment.get("gpu")):
        failures.append("kernel smoke did not run on an A100")
    if environment.get("compute_capability") != [8, 0]:
        failures.append("kernel smoke did not compile/run for sm_80")
    expected_workload = {
        "lengths": [20480],
        "representation": "page_gauge",
        "exact_tail_tokens": 0,
        "exact_sink_tokens": 0,
        "baseline_fixed_split_pages": 256,
        "candidate_fixed_split_pages": 256,
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            failures.append(
                f"workload.{key}={workload.get(key)!r}, expected {expected!r}"
            )
    if correctness.get("reference") != "fp16_reconstructed_cache":
        failures.append("smoke reference is not explicit FP16 reconstructed cache")
    absolute_max = float(correctness.get("absolute_max", math.inf))
    relative_l2_max = float(correctness.get("relative_l2_max", math.inf))
    # This smoke compares centered attention outputs.  Their norm can be close
    # to zero, so relative L2 alone is ill-conditioned (the observed A100 run
    # had only 5.01e-6 max absolute error but 4.84e-3 relative L2).  Require a
    # much tighter absolute bound and retain relative L2 as a finite sanity
    # bound.  This is a compile/factorization smoke, not a paper quality gate.
    if not math.isfinite(absolute_max) or absolute_max > 1e-5:
        failures.append(f"absolute factorization error {absolute_max} exceeds 1e-5")
    if not math.isfinite(relative_l2_max) or relative_l2_max > 1e-2:
        failures.append(
            f"relative-L2 factorization error {relative_l2_max} exceeds 1e-2"
        )
    timings = source.get("timing_modes", {})
    for mode in ("cache_neutral", "cache_hot"):
        mode_payload = timings.get(mode, {}).get("timings", {})
        for method in ("flashinfer_fp16", "page_gauge_int8"):
            p50 = float(mode_payload.get(method, {}).get("p50_ms", math.nan))
            if not math.isfinite(p50) or p50 <= 0:
                failures.append(f"invalid {mode}/{method} latency")
    result = {
        "schema_version": 2,
        "experiment": "page_gauge_a100_factorization_gate",
        "scope": "B1 compile/correctness smoke; timing is diagnostic only",
        "timing_is_acceptance_gate": False,
        "passed": not failures,
        "failures": failures,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "thresholds": {"absolute_max": 1e-5, "relative_l2_max": 1e-2},
        "observed": {
            "absolute_max": absolute_max,
            "relative_l2_max": relative_l2_max,
        },
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
