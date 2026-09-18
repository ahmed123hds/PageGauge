#!/usr/bin/env python3
"""Reduce one fresh-process A100 FI/PageGauge optimization pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DECODE_STEPS = 1536
REPEATS = 3
EXPECTED_FI_BYTES = 11_542_724_608
EXPECTED_PG_BYTES = 7_288_520_704


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer", type=Path, required=True)
    parser.add_argument("--page-gauge", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def geometric_mean(values: list[float]) -> float:
    require(len(values) == REPEATS, "wrong timing repeat count")
    require(all(math.isfinite(value) and value > 0 for value in values), "bad timing")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_execution(payload: dict[str, Any], backend: str) -> None:
    config = payload["configuration"]
    expected = {
        "backend": backend,
        "model": "mistralai/Mistral-7B-v0.3",
        "batch_size": 4,
        "context": 20480,
        "decode_steps": DECODE_STEPS,
        "exact_tail_tokens": 768,
        "exact_sink_pages": 4 if backend == "page_gauge" else 0,
        "exact_static_suffix_pages": 128 if backend == "page_gauge" else 0,
        "baseline_split_pages": 256,
        "candidate_split_pages": 200,
        "trajectory_mode": "frozen_hf_teacher_forced",
        "warmups": 1,
        "repeats": REPEATS,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"{backend}: configuration.{key}")
    environment = payload["environment"]
    require("A100" in str(environment["gpu"]).upper(), f"{backend}: not A100")
    require(environment["compute_capability"] == [8, 0], f"{backend}: not SM80")
    exclusivity = payload["exclusivity"]
    require(exclusivity["fresh_process_required"] is True, f"{backend}: not fresh")
    require(
        exclusivity["opposite_backend_full_gpu_cache_allocated"] is False,
        f"{backend}: opposite cache allocated",
    )
    correctness = payload["correctness"]
    require(
        correctness["same_backend_eager_vs_graph"]["passed"] is True,
        f"{backend}: eager/graph gate",
    )
    require(
        correctness["runtime_page_finalization_and_consumption"]["passed"] is True,
        f"{backend}: recurrence gate",
    )
    require(
        payload["scheduler_capacity"]["all_wrappers_analytically_within_capacity"]
        is True,
        f"{backend}: configured scheduler gate",
    )
    for mode in ("cache_neutral", "cache_hot"):
        rows = payload["timing_modes"][mode]["raw_samples"]
        require(len(rows) == REPEATS, f"{backend}: {mode} repeats")
        for row in rows:
            require(row["runtime_gate"]["passed"] is True, f"{backend}: runtime gate")
            require(
                row["exact_prefix_canary"]["passed"] is True,
                f"{backend}: prefix canary",
            )


def main() -> None:
    args = parse_args()
    fi = read(args.flashinfer)
    pg = read(args.page_gauge)
    selection = read(args.selection)
    validate_execution(fi, "flashinfer_fp16")
    validate_execution(pg, "page_gauge")
    require(selection["full_model_run_recommended"] is True, "probe did not improve")
    override = pg.get("a100_sm80_kernel_override")
    require(isinstance(override, dict), "PageGauge optimization provenance missing")
    require(
        override["sm80_paged_num_mma_kv_cap"] == selection["selected_cap"],
        "selected cap mismatch",
    )
    require(
        override["exact_fp16_fixed_split_pages"]
        == selection["selected_exact_split_pages"],
        "selected exact split mismatch",
    )
    capacity = override.get("effective_exact_scheduler_capacity")
    if selection["selected_exact_split_pages"] is not None:
        require(
            capacity["within_flashinfer_scheduler_capacity"] is True,
            "effective exact split exceeds capacity",
        )
    shared = (
        "teacher_inputs_sha256",
        "token_matrix_sha256",
        "token_source_provenance_sha256",
        "model_config_sha256",
        "sampled_model_parameters_sha256",
        "flashinfer_abi_source_sha256",
    )
    for key in shared:
        require(
            fi["pairing"]["configuration"][key]
            == pg["pairing"]["configuration"][key],
            f"cross-backend identity mismatch: {key}",
        )
    results: dict[str, Any] = {}
    for mode in ("cache_neutral", "cache_hot"):
        results[mode] = {}
        for metric in ("wall_ms", "cuda_ms"):
            fi_ms = geometric_mean(
                [float(row[metric]) for row in fi["timing_modes"][mode]["raw_samples"]]
            )
            pg_ms = geometric_mean(
                [float(row[metric]) for row in pg["timing_modes"][mode]["raw_samples"]]
            )
            results[mode][metric] = {
                "flashinfer_ms_per_step": fi_ms / DECODE_STEPS,
                "page_gauge_ms_per_step": pg_ms / DECODE_STEPS,
                "flashinfer_over_page_gauge": fi_ms / pg_ms,
            }
    fi_bytes = int(
        fi["cache_build"]["selected_backend_cache_served_bytes_excluding_following_canary"]
    )
    pg_bytes = int(
        pg["cache_build"]["selected_backend_cache_served_bytes_excluding_following_canary"]
    )
    require(fi_bytes == EXPECTED_FI_BYTES, "unexpected FI cache bytes")
    require(pg_bytes == EXPECTED_PG_BYTES, "unexpected PageGauge cache bytes")
    primary = results["cache_neutral"]["wall_ms"]["flashinfer_over_page_gauge"]
    payload = {
        "schema_version": 1,
        "experiment": "page_gauge_a100_sm80_optimized_fresh_pair",
        "passed": primary > 1.10,
        "execution_gates_passed": True,
        "point_speedup_gt_1_10": primary > 1.10,
        "timings": results,
        "served_cache": {
            "flashinfer_fp16_bytes": fi_bytes,
            "page_gauge_bytes": pg_bytes,
            "reduction_fraction": 1.0 - pg_bytes / fi_bytes,
        },
        "selection": {
            "path": str(args.selection),
            "sha256": sha256(args.selection),
            "variant": selection["selected_variant"],
        },
        "inputs": {
            "flashinfer_sha256": sha256(args.flashinfer),
            "page_gauge_sha256": sha256(args.page_gauge),
        },
        "scope": "adjacent fresh-process point gate; not the final hierarchical CI",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
