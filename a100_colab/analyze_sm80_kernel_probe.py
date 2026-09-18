#!/usr/bin/env python3
"""Fail-closed reducer for the A100 PageGauge launch-cap microprobe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_LENGTHS = [22016, 22016, 22016, 22016]
EXPECTED_REPEATS = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--cap2", type=Path, required=True)
    parser.add_argument("--split30", type=Path, required=True)
    parser.add_argument("--split30-cap2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing probe result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def method_name(payload: dict[str, Any]) -> str:
    names = set(payload["timing_modes"]["cache_neutral"]["timings"])
    names.remove("flashinfer_fp16")
    require(len(names) == 1, "expected exactly one PageGauge timing method")
    return names.pop()


def raw_times(payload: dict[str, Any], mode: str, method: str) -> list[float]:
    mode_payload = payload["timing_modes"][mode]
    values = [
        float(value)
        for value in mode_payload["raw_paired_timings_ms"][method]
    ]
    require(len(values) == EXPECTED_REPEATS, f"{mode}/{method} sample count")
    require(
        all(math.isfinite(value) and value > 0 for value in values),
        "invalid timing",
    )
    return values


def validate(
    payload: dict[str, Any], cap: int | None, exact_split_pages: int | None
) -> str:
    environment = payload["environment"]
    require("A100" in str(environment["gpu"]).upper(), "probe did not run on A100")
    require(environment["compute_capability"] == [8, 0], "probe is not SM80")
    workload = payload["workload"]
    require(workload["lengths"] == EXPECTED_LENGTHS, "workload lengths changed")
    require(workload["representation"] == "page_gauge", "representation changed")
    require(workload["baseline_fixed_split_pages"] == 256, "baseline split changed")
    require(workload["candidate_fixed_split_pages"] == 200, "candidate split changed")
    require(workload["exact_tail_tokens"] == 768, "exact tail changed")
    require(workload["exact_sink_tokens"] == 2112, "exact-region geometry changed")
    correctness = payload["correctness"]
    require(float(correctness["absolute_max"]) <= 5e-4, "absolute correctness failed")
    require(float(correctness["relative_l2_max"]) <= 1e-2, "relative correctness failed")
    override = payload.get("a100_sm80_kernel_override")
    if cap is None and exact_split_pages is None:
        require(override is None, "legacy result unexpectedly has an override")
    else:
        require(isinstance(override, dict), "SM80 override provenance missing")
        require(override["sm80_paged_num_mma_kv_cap"] == cap, "cap mismatch")
        require(
            override["exact_fp16_fixed_split_pages"] == exact_split_pages,
            "exact split mismatch",
        )
        require(override["page_gauge_math_changed"] is False, "math-change gate failed")
        if cap is not None:
            require(
                override["source_hashes"]["math_changed"] is False,
                "source math gate failed",
            )
    method = method_name(payload)
    for mode in ("cache_neutral", "cache_hot"):
        raw_times(payload, mode, "flashinfer_fp16")
        raw_times(payload, mode, method)
    return method


def p50(payload: dict[str, Any], mode: str, method: str) -> float:
    return float(payload["timing_modes"][mode]["timings"][method]["p50_ms"])


def main() -> None:
    args = parse_args()
    records = {
        "legacy": read(args.legacy),
        "cap2": read(args.cap2),
        "split30": read(args.split30),
        "split30_cap2": read(args.split30_cap2),
    }
    variants = {
        "legacy": (None, None),
        "cap2": (2, None),
        "split30": (None, 30),
        "split30_cap2": (2, 30),
    }
    methods = {
        name: validate(payload, *variants[name]) for name, payload in records.items()
    }
    rows: dict[str, Any] = {}
    for name, payload in records.items():
        method = methods[name]
        rows[name] = {}
        for mode in ("cache_neutral", "cache_hot"):
            fi = p50(payload, mode, "flashinfer_fp16")
            pg = p50(payload, mode, method)
            rows[name][mode] = {
                "flashinfer_fp16_p50_ms": fi,
                "page_gauge_p50_ms": pg,
                "flashinfer_over_page_gauge": fi / pg,
            }
    selected = min(rows, key=lambda name: rows[name]["cache_neutral"]["page_gauge_p50_ms"])
    legacy_ms = rows["legacy"]["cache_neutral"]["page_gauge_p50_ms"]
    selected_ms = rows[selected]["cache_neutral"]["page_gauge_p50_ms"]
    improved = selected != "legacy" and selected_ms < legacy_ms
    payload = {
        "schema_version": 1,
        "experiment": "page_gauge_a100_sm80_launch_cap_probe",
        "passed": True,
        "rows": rows,
        "selected_variant": selected,
        "selected_cap": variants[selected][0],
        "selected_exact_split_pages": variants[selected][1],
        "candidate_improves_legacy_neutral": improved,
        "neutral_page_gauge_improvement": legacy_ms / selected_ms,
        "full_model_run_recommended": improved,
        "scope": (
            "SM80 launch-shape selection only; this microprobe does not establish "
            "the end-to-end >=1.1 speed gate"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
