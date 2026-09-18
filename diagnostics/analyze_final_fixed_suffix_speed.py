#!/usr/bin/env python3
"""Fail-closed reduction of the final S4/A128/T768 fresh-process speed gate."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results/final_fixed_suffix_s4_a128_t768/performance/fresh_process_williams_split128_v2"
OUTPUT = INPUT_DIR / "final_performance_analysis.json"
QUALITY = ROOT / "results/final_fixed_suffix_s4_a128_t768/quality/aggregate.json"
DECODE_STEPS = 1536
REPEATS = 3
BOOTSTRAP_SAMPLES = 50_000
SCHEDULE = (
    (0, "flashinfer_fp16", 20260861, 0, "pair0", "AB"),
    (1, "page_gauge", 20260861, 0, "pair0", "AB"),
    (2, "page_gauge", 20260861, 0, "pair1", "BA"),
    (3, "flashinfer_fp16", 20260861, 0, "pair1", "BA"),
    (4, "page_gauge", 20260862, 94400, "pair2", "BA"),
    (5, "flashinfer_fp16", 20260862, 94400, "pair2", "BA"),
    (6, "flashinfer_fp16", 20260862, 94400, "pair3", "AB"),
    (7, "page_gauge", 20260862, 94400, "pair3", "AB"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_worker(
    path: Path, backend: str, seed: int, offset: int
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["configuration"]
    prefix = 4 if backend == "page_gauge" else 0
    suffix = 128 if backend == "page_gauge" else 0
    expected = {
        "backend": backend,
        "batch_size": 4,
        "context": 20480,
        "decode_steps": DECODE_STEPS,
        "exact_tail_tokens": 768,
        "exact_sink_pages": prefix,
        "exact_prefix_pages": prefix,
        "exact_static_suffix_pages": suffix,
        "baseline_split_pages": 256,
        "candidate_split_pages": 128,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
        "trajectory_mode": "frozen_hf_teacher_forced",
        "cuda_graph_scope": "decoder_layer_device_dynamic",
        "token_source": "wikitext2",
        "wikitext_member": "wikitext-2-raw/wiki.test.raw",
        "min_logits_cosine": 0.995,
        "min_top1_agreement": 0.99,
        "seed": seed,
        "token_offset": offset,
        "token_stride": 23600,
        "warmups": 1,
        "repeats": REPEATS,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"{path.name}: configuration.{key}")
    for source_name, expected_hash in payload["source_sha256"].items():
        source = ROOT / source_name
        require(source.is_file(), f"{path.name}: missing source {source_name}")
        require(
            sha256_file(source) == expected_hash,
            f"{path.name}: source drift {source_name}",
        )
    same = payload["correctness"]["same_backend_eager_vs_graph"]
    require(same["passed"] is True, f"{path.name}: same-backend gate")
    require(
        same["restored_graph_repeat"]["passed"] is True,
        f"{path.name}: restored repeat",
    )
    finalization = payload["correctness"][
        "runtime_page_finalization_and_consumption"
    ]
    require(finalization["passed"] is True, f"{path.name}: recurrence gate")
    if backend == "page_gauge":
        require(
            finalization["runtime_finalized_int8_pages_consumed_count"] == 48,
            f"{path.name}: generated INT8 recurrence count",
        )
        require(
            finalization["static_suffix_exclusion_gate_passed"] is True,
            f"{path.name}: suffix exclusion",
        )
    require(
        payload["scheduler_capacity"]["all_wrappers_analytically_within_capacity"]
        is True,
        f"{path.name}: scheduler capacity",
    )
    for mode in ("cache_neutral", "cache_hot"):
        timing = payload["timing_modes"][mode]
        raw = timing["raw_samples"]
        require(len(raw) == REPEATS, f"{path.name}: {mode} repeats")
        require(
            all(row["runtime_gate"]["passed"] for row in raw),
            f"{path.name}: {mode} runtime gate",
        )
        require(
            all(row["exact_prefix_canary"]["passed"] for row in raw),
            f"{path.name}: {mode} fixed-exact canary",
        )
        for row in raw:
            require(
                math.isfinite(float(row["wall_ms"]))
                and math.isfinite(float(row["cuda_ms"]))
                and row["wall_ms"] > 0
                and row["cuda_ms"] > 0,
                f"{path.name}: {mode} latency",
            )
    return payload


def bootstrap_pair_logs(
    logs: list[float], seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    draws = [
        sum(logs[rng.randrange(len(logs))] for _ in logs) / len(logs)
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return {
        "method": "paired-block percentile bootstrap",
        "pair_count": len(logs),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "speedup_95_ci": [
            math.exp(percentile(draws, 0.025)),
            math.exp(percentile(draws, 0.975)),
        ],
    }


def bootstrap_hierarchical(
    logs_by_seed: dict[int, list[float]], seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    seeds = sorted(logs_by_seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected_seed_logs: list[float] = []
        for _ in seeds:
            selected_seed = seeds[rng.randrange(len(seeds))]
            cluster = logs_by_seed[selected_seed]
            selected_seed_logs.extend(
                cluster[rng.randrange(len(cluster))] for _ in cluster
            )
        draws.append(sum(selected_seed_logs) / len(selected_seed_logs))
    return {
        "method": "hierarchical percentile bootstrap: fixture/seed then adjacent pair",
        "seed_cluster_count": len(seeds),
        "pairs_per_seed": {str(key): len(value) for key, value in logs_by_seed.items()},
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "speedup_95_ci": [
            math.exp(percentile(draws, 0.025)),
            math.exp(percentile(draws, 0.975)),
        ],
    }


def main() -> None:
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    require(quality.get("passed") is True, "frozen six-window quality aggregate failed")
    records: dict[int, dict[str, Any]] = {}
    input_manifest = []
    for index, backend, seed, offset, pair_id, order in SCHEDULE:
        path = INPUT_DIR / f"block_{index}_{backend}.json"
        require(path.is_file(), f"missing {path.name}")
        payload = validate_worker(path, backend, seed, offset)
        records[index] = payload
        input_manifest.append(
            {
                "index": index,
                "backend": backend,
                "seed": seed,
                "token_offset": offset,
                "pair_id": pair_id,
                "pair_order": order,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "worker_overall_passed": payload["passed"],
                "performance_execution_gate_passed": True,
                "worker_cross_hf_gate_passed": payload["correctness"][
                    "backend_vs_hf_sdpa_fp16"
                ]["passed"],
            }
        )
    pairs = ((0, 1, 20260861, "AB"), (3, 2, 20260861, "BA"), (5, 4, 20260862, "BA"), (6, 7, 20260862, "AB"))
    pair_records: list[dict[str, Any]] = []
    for pair_index, (fi_index, pg_index, seed, order) in enumerate(pairs):
        fi = records[fi_index]
        pg = records[pg_index]
        for key in (
            "teacher_inputs_sha256",
            "token_matrix_sha256",
            "token_source_provenance_sha256",
            "model_config_sha256",
            "sampled_model_parameters_sha256",
            "flashinfer_abi_source_sha256",
        ):
            require(
                fi["pairing"]["configuration"][key]
                == pg["pairing"]["configuration"][key],
                f"pair {pair_index}: shared identity {key}",
            )
        for mode in ("cache_neutral", "cache_hot"):
            for metric in ("wall_ms", "cuda_ms"):
                fi_values = [
                    float(row[metric]) for row in fi["timing_modes"][mode]["raw_samples"]
                ]
                pg_values = [
                    float(row[metric]) for row in pg["timing_modes"][mode]["raw_samples"]
                ]
                fi_latency = geometric_mean(fi_values)
                pg_latency = geometric_mean(pg_values)
                pair_records.append(
                    {
                        "pair_index": pair_index,
                        "seed": seed,
                        "order": order,
                        "mode": mode,
                        "metric": metric,
                        "fi_ms_per_step": fi_latency / DECODE_STEPS,
                        "page_gauge_ms_per_step": pg_latency / DECODE_STEPS,
                        "log_speedup": math.log(fi_latency / pg_latency),
                        "speedup": fi_latency / pg_latency,
                    }
                )

    aggregates: dict[str, Any] = {}
    for mode in ("cache_neutral", "cache_hot"):
        aggregates[mode] = {}
        for metric in ("wall_ms", "cuda_ms"):
            rows = [
                row
                for row in pair_records
                if row["mode"] == mode and row["metric"] == metric
            ]
            logs = [float(row["log_speedup"]) for row in rows]
            logs_by_seed: dict[int, list[float]] = {}
            for row in rows:
                logs_by_seed.setdefault(int(row["seed"]), []).append(
                    float(row["log_speedup"])
                )
            fi_ms = geometric_mean([float(row["fi_ms_per_step"]) for row in rows])
            pg_ms = geometric_mean(
                [float(row["page_gauge_ms_per_step"]) for row in rows]
            )
            paired = bootstrap_pair_logs(logs, 5090)
            hierarchical = bootstrap_hierarchical(logs_by_seed, 5140)
            aggregates[mode][metric] = {
                "estimand": "exp(mean adjacent-pair log(FI_latency/PG_latency))",
                "pair_count": len(rows),
                "seed_cluster_count": len(logs_by_seed),
                "fi_geomean_ms_per_step": fi_ms,
                "page_gauge_geomean_ms_per_step": pg_ms,
                "speedup_geomean": math.exp(sum(logs) / len(logs)),
                "paired_block_bootstrap": paired,
                "hierarchical_fixture_pair_bootstrap": hierarchical,
                "order_stratified": {
                    order: math.exp(
                        sum(row["log_speedup"] for row in rows if row["order"] == order)
                        / sum(row["order"] == order for row in rows)
                    )
                    for order in ("AB", "BA")
                },
            }

    primary = aggregates["cache_neutral"]["wall_ms"]
    lower_95 = primary["hierarchical_fixture_pair_bootstrap"]["speedup_95_ci"][0]
    fi_bytes = {records[index]["cache_build"]["selected_backend_cache_served_bytes_excluding_following_canary"] for index in (0, 3, 5, 6)}
    pg_bytes = {records[index]["cache_build"]["selected_backend_cache_served_bytes_excluding_following_canary"] for index in (1, 2, 4, 7)}
    require(len(fi_bytes) == len(pg_bytes) == 1, "served-cache bytes vary by backend block")
    fi_bytes_value = int(next(iter(fi_bytes)))
    pg_bytes_value = int(next(iter(pg_bytes)))
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_final_s4_a128_t768_fresh_process_williams",
        "passed": bool(primary["speedup_geomean"] > 1.1 and lower_95 > 1.1),
        "acceptance": {
            "point_speedup_gt_1_1": primary["speedup_geomean"] > 1.1,
            "hierarchical_lower_95_gt_1_1": lower_95 > 1.1,
            "same_backend_execution_all_blocks": True,
            "separate_frozen_quality_aggregate_passed": True,
        },
        "protocol": {
            "schedule": "ABBA then BAAB",
            "fresh_process_blocks": 8,
            "independent_fixture_seed_clusters": 2,
            "adjacent_pairs": 4,
            "within_block_repeats": REPEATS,
            "context": 20480,
            "decode_steps": DECODE_STEPS,
            "batch_size": 4,
            "exact_prefix_pages": 4,
            "exact_static_suffix_pages": 128,
            "exact_tail_tokens": 768,
            "candidate_split_pages": 128,
            "runtime_generated_int8_pages": 48,
            "worker_overall_pass_note": (
                "worker overall status also contains an HF diagnostic on timing fixtures; "
                "performance eligibility is gated by same-backend/replay/cache/recurrence "
                "checks, while fidelity is supplied by the separate frozen six-window aggregate"
            ),
        },
        "inputs": input_manifest,
        "pair_records": pair_records,
        "aggregates": aggregates,
        "served_cache": {
            "flashinfer_fp16_bytes": fi_bytes_value,
            "page_gauge_bytes": pg_bytes_value,
            "byte_reduction": fi_bytes_value - pg_bytes_value,
            "reduction_fraction": 1.0 - pg_bytes_value / fi_bytes_value,
            "effective_capacity_ratio": fi_bytes_value / pg_bytes_value,
        },
        "fidelity_evidence": {
            "path": str(QUALITY.relative_to(ROOT)),
            "sha256": sha256_file(QUALITY),
            "passed": True,
        },
        "analysis": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__)),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_used": False,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "primary": primary, "served_cache": result["served_cache"], "output": str(OUTPUT)}, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
