#!/usr/bin/env python3
"""Reduce the frozen A100 SM80-optimized S4/A128/T768 Williams run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DECODE_STEPS = 1536
REPEATS = 3
BOOTSTRAP_SAMPLES = 50_000
EXPECTED_FI_BYTES = 11_542_724_608
EXPECTED_PG_BYTES = 7_288_520_704
EXPECTED_DERIVED_HEADER_SHA256 = (
    "1f63c27992a9d1550197e915baf702619e07d8769d9c8a0234cf7c0c2aefb96f"
)
EXPECTED_BASE_HEADER_SHA256 = (
    "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"
)
EXPECTED_MODULE_SOURCE_SHA256 = (
    "9be9777252c4d4971e3b514f797608012310bb02f752940612f31ff67c0da421"
)
EXPECTED_SELECTION_SHA256 = (
    "914566c0704735d5d4dddb6df780d377785ecc306ee7f66cfe34532755bd680e"
)
EXPECTED_OVERRIDE_SOURCE_SHA256 = {
    "scripts/page_gauge_sm80_fa2.py": (
        "8ae76a8d34bf1279d1f34e9e46afce0a4de812c2b7f95428bca93563c9cede8a"
    ),
    "a100_colab/run_with_sm80_kernel.py": (
        "041aba2d17338cc25dd323419ffe69e8af5eeb0e5ae0843529405411f4996b67"
    ),
}
EXPECTED_FLASHINFER_ABI_SHA256 = {
    "decode.py": "f3e97a740cab485653614dc984979561891c06795fd2aa87aaf8f554522117dd",
    "scheduler.cuh": "09c6e89861f48f3b79fee029b6e44c8e16f41d73053fb0154f0e636498d0655f",
}
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
PAIRS = (
    (0, 1, 20260861, "AB"),
    (3, 2, 20260861, "BA"),
    (5, 4, 20260862, "BA"),
    (6, 7, 20260862, "AB"),
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
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def geometric_mean(values: list[float]) -> float:
    require(bool(values) and all(value > 0 for value in values), "invalid geomean")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_pairs(logs: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = [
        math.exp(sum(rng.choice(logs) for _ in logs) / len(logs))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def bootstrap_hierarchical(
    logs_by_seed: dict[int, list[float]], seed: int
) -> list[float]:
    rng = random.Random(seed)
    seeds = sorted(logs_by_seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected: list[float] = []
        for _ in seeds:
            selected_seed = rng.choice(seeds)
            cluster = logs_by_seed[selected_seed]
            selected.extend(rng.choice(cluster) for _ in cluster)
        draws.append(math.exp(sum(selected) / len(selected)))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def validate_worker(path: Path, backend: str, seed: int, offset: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("configuration", {})
    prefix = 4 if backend == "page_gauge" else 0
    suffix = 128 if backend == "page_gauge" else 0
    expected = {
        "backend": backend,
        "model": "mistralai/Mistral-7B-v0.3",
        "model_revision": "caa1feb0e54d415e2df31207e5f4e273e33509b1",
        "batch_size": 4,
        "context": 20480,
        "decode_steps": DECODE_STEPS,
        "exact_tail_tokens": 768,
        "exact_sink_pages": prefix,
        "exact_prefix_pages": prefix,
        "exact_static_suffix_pages": suffix,
        "prefill_chunk_tokens": 1024,
        "baseline_split_pages": 256,
        "candidate_split_pages": 200,
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
        "capture_warmups": 1,
        "maximum_graph_banks": 16,
        "warmups": 1,
        "repeats": REPEATS,
        "cache_scrub_mib": 256,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"{path.name}: configuration.{key}")

    environment = payload.get("environment", {})
    require("A100" in str(environment.get("gpu")), f"{path.name}: not A100")
    require(
        environment.get("compute_capability") == [8, 0],
        f"{path.name}: not compute capability 8.0",
    )
    require(
        int(environment.get("multiprocessor_count", 0)) == 108,
        f"{path.name}: expected full 108-SM A100",
    )
    require(environment.get("flashinfer") == "0.6.17", f"{path.name}: FlashInfer")
    require(
        environment.get("flashinfer_abi", {}).get("source_sha256")
        == EXPECTED_FLASHINFER_ABI_SHA256,
        f"{path.name}: FlashInfer planner ABI source drift",
    )

    for source_name, expected_hash in payload.get("source_sha256", {}).items():
        source = ROOT / source_name
        require(source.is_file(), f"{path.name}: missing source {source_name}")
        require(
            sha256_file(source) == expected_hash,
            f"{path.name}: source drift {source_name}",
        )

    exclusivity = payload.get("exclusivity", {})
    require(exclusivity.get("fresh_process_required") is True, f"{path.name}: fresh")
    require(
        exclusivity.get("selected_persistent_backend") == backend,
        f"{path.name}: selected backend",
    )
    require(
        exclusivity.get("opposite_backend_full_gpu_cache_allocated") is False,
        f"{path.name}: opposite full cache",
    )
    require(
        exclusivity.get("hf_dynamic_caches_released_before_decoder_construction")
        is True,
        f"{path.name}: HF cache lifetime",
    )

    same = payload.get("correctness", {}).get("same_backend_eager_vs_graph", {})
    require(same.get("passed") is True, f"{path.name}: eager/graph gate")
    require(
        same.get("restored_graph_repeat", {}).get("passed") is True,
        f"{path.name}: restored graph repeat",
    )
    finalization = payload.get("correctness", {}).get(
        "runtime_page_finalization_and_consumption", {}
    )
    require(finalization.get("passed") is True, f"{path.name}: recurrence gate")
    if backend == "page_gauge":
        require(
            finalization.get("runtime_finalized_int8_pages_consumed_count") == 48,
            f"{path.name}: generated INT8 recurrence count",
        )
        require(
            finalization.get("static_suffix_exclusion_gate_passed") is True,
            f"{path.name}: static suffix exclusion",
        )
        attention = payload.get("attention_implementation", {})
        require(
            attention.get("implementation") == "page_gauge_segmented_flashinfer_merge",
            f"{path.name}: attention implementation",
        )
        hashes = attention.get("custom_module_source_hashes", {})
        require(
            hashes.get("header_sha256") == EXPECTED_DERIVED_HEADER_SHA256
            and hashes.get("header_matches_expected") is True,
            f"{path.name}: PageGauge derived SM80 header",
        )
        require(
            hashes.get("module_source_sha256") == EXPECTED_MODULE_SOURCE_SHA256,
            f"{path.name}: PageGauge derived module source",
        )
        override = payload.get("a100_sm80_kernel_override", {})
        require(override.get("sm80_paged_num_mma_kv_cap") == 2, f"{path.name}: MMA cap")
        require(
            override.get("exact_fp16_fixed_split_pages") == 30,
            f"{path.name}: exact split",
        )
        require(override.get("page_gauge_math_changed") is False, f"{path.name}: math")
        source_hashes = override.get("source_hashes", {})
        require(
            source_hashes.get("header_sha256") == EXPECTED_DERIVED_HEADER_SHA256
            and source_hashes.get("base_header_sha256") == EXPECTED_BASE_HEADER_SHA256
            and source_hashes.get("module_source_sha256")
            == EXPECTED_MODULE_SOURCE_SHA256
            and source_hashes.get("math_changed") is False,
            f"{path.name}: SM80 source attestation",
        )
        require(
            override.get("launcher_source_sha256") == EXPECTED_OVERRIDE_SOURCE_SHA256,
            f"{path.name}: SM80 launcher source",
        )
        effective = override.get("effective_exact_scheduler_capacity", {})
        require(
            effective.get("exact_pages_per_request") == 180
            and effective.get("fixed_split_pages") == 30
            and effective.get("required_chunks_per_request") == 6
            and effective.get("required_split_tiles") == 24
            and effective.get("maximum_split_tiles") == 27
            and effective.get("within_flashinfer_scheduler_capacity") is True,
            f"{path.name}: effective exact scheduler capacity",
        )

    capacity = payload.get("scheduler_capacity", {})
    require(
        capacity.get("all_wrappers_analytically_within_capacity") is True,
        f"{path.name}: scheduler capacity",
    )
    for mode in ("cache_neutral", "cache_hot"):
        rows = payload.get("timing_modes", {}).get(mode, {}).get("raw_samples", [])
        require(len(rows) == REPEATS, f"{path.name}: {mode} repeats")
        for row in rows:
            require(
                row.get("runtime_gate", {}).get("passed") is True,
                f"{path.name}: runtime",
            )
            require(
                row.get("exact_prefix_canary", {}).get("passed") is True,
                f"{path.name}: exact canary",
            )
            for metric in ("wall_ms", "cuda_ms"):
                value = float(row.get(metric, math.nan))
                require(
                    math.isfinite(value) and value > 0, f"{path.name}: {mode}/{metric}"
                )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--kernel-gate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    kernel_gate = json.loads(args.kernel_gate.read_text(encoding="utf-8"))
    require(preflight.get("passed") is True, "A100 preflight failed")
    require(kernel_gate.get("passed") is True, "factorization kernel gate failed")
    require(
        sha256_file(args.selection) == EXPECTED_SELECTION_SHA256,
        "SM80 selection artifact hash mismatch",
    )
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    require(
        selection.get("passed") is True
        and selection.get("selected_variant") == "split30_cap2"
        and selection.get("selected_cap") == 2
        and selection.get("selected_exact_split_pages") == 30
        and selection.get("full_model_run_recommended") is True,
        "SM80 selection semantic gate failed",
    )

    records: dict[int, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for index, backend, seed, offset, pair_id, order in SCHEDULE:
        path = args.run_dir / f"block_{index}_{backend}.json"
        require(path.is_file(), f"missing {path}")
        payload = validate_worker(path, backend, seed, offset)
        records[index] = payload
        inputs.append(
            {
                "index": index,
                "backend": backend,
                "seed": seed,
                "token_offset": offset,
                "pair_id": pair_id,
                "order": order,
                "path": str(path),
                "sha256": sha256_file(path),
                "worker_overall_passed": payload.get("passed"),
                "same_backend_execution_passed": True,
            }
        )

    environment_identity_keys = (
        "gpu",
        "compute_capability",
        "multiprocessor_count",
        "torch",
        "torch_cuda",
        "flashinfer",
        "transformers",
        "python",
    )
    reference_environment = records[0]["environment"]
    for index, payload in records.items():
        environment = payload["environment"]
        for key in environment_identity_keys:
            require(
                environment.get(key) == reference_environment.get(key),
                f"block {index}: environment identity differs at {key}",
            )
        require(
            environment.get("flashinfer_abi", {}).get("source_sha256")
            == reference_environment.get("flashinfer_abi", {}).get("source_sha256"),
            f"block {index}: FlashInfer planner ABI differs",
        )

    pair_records: list[dict[str, Any]] = []
    shared_keys = (
        "teacher_inputs_sha256",
        "token_matrix_sha256",
        "token_source_provenance_sha256",
        "model_config_sha256",
        "sampled_model_parameters_sha256",
        "flashinfer_abi_source_sha256",
    )
    for pair_index, (fi_index, pg_index, seed, order) in enumerate(PAIRS):
        fi = records[fi_index]
        pg = records[pg_index]
        for key in shared_keys:
            require(
                fi["pairing"]["configuration"].get(key)
                == pg["pairing"]["configuration"].get(key),
                f"pair {pair_index}: shared identity {key}",
            )
        fi_hashes = fi.get("correctness", {}).get("hashes", {})
        pg_hashes = pg.get("correctness", {}).get("hashes", {})
        for key in ("hf_logits_sha256", "hf_generated_tokens_sha256"):
            require(
                fi_hashes.get(key) == pg_hashes.get(key),
                f"pair {pair_index}: cross-backend HF reference {key}",
            )
        for mode in ("cache_neutral", "cache_hot"):
            for metric in ("wall_ms", "cuda_ms"):
                fi_values = [
                    float(row[metric])
                    for row in fi["timing_modes"][mode]["raw_samples"]
                ]
                pg_values = [
                    float(row[metric])
                    for row in pg["timing_modes"][mode]["raw_samples"]
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
            aggregates[mode][metric] = {
                "estimand": "exp(mean adjacent-pair log(FI_latency/PG_latency))",
                "pair_count": len(rows),
                "seed_cluster_count": len(logs_by_seed),
                "fi_geomean_ms_per_step": geometric_mean(
                    [float(row["fi_ms_per_step"]) for row in rows]
                ),
                "page_gauge_geomean_ms_per_step": geometric_mean(
                    [float(row["page_gauge_ms_per_step"]) for row in rows]
                ),
                "speedup_geomean": math.exp(sum(logs) / len(logs)),
                "paired_block_bootstrap_95_ci": bootstrap_pairs(logs, 8000),
                "hierarchical_fixture_pair_bootstrap_95_ci": bootstrap_hierarchical(
                    logs_by_seed, 8050
                ),
                "order_stratified": {
                    order: math.exp(
                        sum(
                            float(row["log_speedup"])
                            for row in rows
                            if row["order"] == order
                        )
                        / sum(row["order"] == order for row in rows)
                    )
                    for order in ("AB", "BA")
                },
            }

    fi_bytes = {
        int(
            records[index]["cache_build"][
                "selected_backend_cache_served_bytes_excluding_following_canary"
            ]
        )
        for index in (0, 3, 5, 6)
    }
    pg_bytes = {
        int(
            records[index]["cache_build"][
                "selected_backend_cache_served_bytes_excluding_following_canary"
            ]
        )
        for index in (1, 2, 4, 7)
    }
    require(fi_bytes == {EXPECTED_FI_BYTES}, "unexpected FlashInfer served bytes")
    require(pg_bytes == {EXPECTED_PG_BYTES}, "unexpected PageGauge served bytes")
    primary = aggregates["cache_neutral"]["wall_ms"]
    speedup = float(primary["speedup_geomean"])
    lower_95 = float(primary["hierarchical_fixture_pair_bootstrap_95_ci"][0])
    acceptance = {
        "a100_full_device_preflight": True,
        "sm80_factorization_gate": True,
        "source_frozen_sm80_selection_gate": True,
        "all_eight_fresh_process_execution_gates": True,
        "point_speedup_gt_1_10": speedup > 1.10,
        "hierarchical_lower_95_gt_1_10": lower_95 > 1.10,
        "deterministic_cache_memory_gate": True,
    }
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_a100_sm80_optimized_s4_a128_t768_williams",
        "passed": all(acceptance.values()),
        "acceptance": acceptance,
        "protocol": {
            "gpu": "full NVIDIA A100, compute capability 8.0",
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
            "baseline_split_pages": 256,
            "candidate_split_pages": 200,
            "exact_fp16_split_pages": 30,
            "page_gauge_num_mma_kv_cap": 2,
            "candidate_split_reason": (
                "A100 SM80 capacity: 108 SMs, Hkv=8, B4 permits at most six "
                "split chunks/request; split128 would require ten, while the "
                "capacity-derived split200 requires exactly six"
            ),
            "runtime_generated_int8_pages": 48,
            "quantizer_or_attention_math_changed": False,
            "all_blocks_share_exact_software_and_planner_abi": True,
        },
        "inputs": inputs,
        "pair_records": pair_records,
        "aggregates": aggregates,
        "served_cache": {
            "flashinfer_fp16_bytes": EXPECTED_FI_BYTES,
            "page_gauge_bytes": EXPECTED_PG_BYTES,
            "byte_reduction": EXPECTED_FI_BYTES - EXPECTED_PG_BYTES,
            "reduction_fraction": 1.0 - EXPECTED_PG_BYTES / EXPECTED_FI_BYTES,
            "effective_capacity_ratio": EXPECTED_FI_BYTES / EXPECTED_PG_BYTES,
        },
        "preflight": {
            "path": str(args.preflight),
            "sha256": sha256_file(args.preflight),
        },
        "kernel_gate": {
            "path": str(args.kernel_gate),
            "sha256": sha256_file(args.kernel_gate),
        },
        "sm80_selection": {
            "path": str(args.selection),
            "sha256": sha256_file(args.selection),
            "selected_variant": "split30_cap2",
        },
        "analysis": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__)),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_used_by_reducer": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "primary": primary,
                "served_cache": result["served_cache"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
