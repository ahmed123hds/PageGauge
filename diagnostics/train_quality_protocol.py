#!/usr/bin/env python3
"""Frozen TRAIN selection/confirmation protocol for PageGauge S3/T768/D1024.

This module is deliberately CPU-only.  It contains the pre-registered cohort
coordinates, strict artifact validation, and whole-window summary/bootstrap
logic shared by the runner and aggregator.  It does not import or modify the
model implementation or benchmark worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RAW_SCHEMA_VERSION = 3
PROTOCOL_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT = "page_gauge_backend_exclusive_sustained_dynamic_graphs"

MODEL = "mistralai/Mistral-7B-v0.3"
MODEL_REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
MODEL_CONFIG_SHA256 = (
    "f223f73de240195fe40495201ecbe85c9dac820842e4fe7e50c38d576b8d22ca"
)
ARCHIVE_SHA256 = (
    "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
)
TRAIN_MEMBER = "wikitext-2-raw/wiki.train.raw"
TRAIN_MEMBER_SHA256 = (
    "6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57"
)
TRAIN_TOKEN_COUNT = 2_763_961

BATCH_SIZE = 4
CONTEXT = 20_480
DECODE_STEPS = 1_024
EXACT_TAIL_TOKENS = 768
EXACT_PREFIX_PAGES = 3
PAGE = 16
WINDOW_CORPUS_TOKENS = CONTEXT + DECODE_STEPS
ROWS_PER_WINDOW = DECODE_STEPS
ROWS_PER_SHARD = BATCH_SIZE * DECODE_STEPS
MINIMUM_LOGITS_COSINE = 0.995
MINIMUM_TOP1_AGREEMENT = 0.99
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 20_260_876

# This is the shard on which T512 failed and fixed S3/T768 was selected.  It is
# never represented as untouched confirmation evidence.
SELECTION_STARTS = (100_000, 121_984, 143_968, 165_952)
SELECTION_STRIDE = 21_984

# Twenty frozen, corpus-spanning, previously unused confirmation windows.
# Each tuple is one B4 worker process.  A 125k stride makes every interval
# disjoint while distributing the windows over most of WikiText-2 TRAIN.
CONFIRMATION_STRIDE = 125_000
CONFIRMATION_STARTS = tuple(300_000 + index * CONFIRMATION_STRIDE for index in range(20))
CONFIRMATION_GROUPS = tuple(
    tuple(CONFIRMATION_STARTS[index : index + BATCH_SIZE])
    for index in range(0, len(CONFIRMATION_STARTS), BATCH_SIZE)
)

THEORETICAL_MAX_NONOVERLAPPING_TRAIN_WINDOWS = TRAIN_TOKEN_COUNT // WINDOW_CORPUS_TOKENS

FROZEN_GATES = {
    "minimum_logits_cosine": MINIMUM_LOGITS_COSINE,
    "maximum_rows_below_minimum_logits_cosine_gate": 0,
    "minimum_top1_agreement": MINIMUM_TOP1_AGREEMENT,
    "all_b4_shards_must_pass_worker_quality_gate": True,
    "same_backend_eager_graph_repeat_and_canaries_must_pass": True,
    "runtime_finalized_int8_pages_consumed_expected": 16,
}

FROZEN_POLICY = {
    "backend": "page_gauge",
    "model": MODEL,
    "model_revision": MODEL_REVISION,
    "batch_size": BATCH_SIZE,
    "context": CONTEXT,
    "decode_steps": DECODE_STEPS,
    "exact_tail_tokens": EXACT_TAIL_TOKENS,
    "exact_prefix_pages": EXACT_PREFIX_PAGES,
    "prefill_chunk_tokens": 1_024,
    "baseline_split_pages": 256,
    "candidate_split_pages": 256,
    "tail_attention": "flashinfer_merge",
    "old_value_scale_placement": "probability",
    "trajectory_mode": "frozen_hf_teacher_forced",
    "cuda_graph_scope": "decoder_layer_device_dynamic",
    "token_source": "wikitext2",
    "wikitext_member": TRAIN_MEMBER,
    "wikitext_archive_sha256": ARCHIVE_SHA256,
    "seed": 20_260_861,
    "capture_warmups": 1,
    "maximum_graph_banks": 16,
    "warmups": 0,
    "repeats": 1,
    "cache_scrub_mib": 256,
    "quality_diagnostics_top_k": 64,
    "quality_diagnostics_top_vocab": 12,
    "min_logits_cosine": MINIMUM_LOGITS_COSINE,
    "min_top1_agreement": MINIMUM_TOP1_AGREEMENT,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}: expected a number",
    )
    converted = float(value)
    require(math.isfinite(converted), f"{label}: number is not finite")
    return converted


def percentile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "cannot take a percentile of an empty sequence")
    require(0.0 <= probability <= 1.0, "percentile is outside [0,1]")
    ordered = sorted(float(value) for value in values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    weight = coordinate - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def intervals(starts: Iterable[int]) -> list[tuple[int, int]]:
    return sorted((int(start), int(start) + WINDOW_CORPUS_TOKENS) for start in starts)


def validate_frozen_cohorts() -> None:
    require(len(SELECTION_STARTS) == BATCH_SIZE, "selection must be one B4 shard")
    require(len(CONFIRMATION_GROUPS) == 5, "confirmation must contain five shards")
    require(
        all(len(group) == BATCH_SIZE for group in CONFIRMATION_GROUPS),
        "every confirmation shard must be B4",
    )
    all_starts = SELECTION_STARTS + CONFIRMATION_STARTS
    require(len(set(all_starts)) == len(all_starts), "frozen starts contain duplicates")
    all_intervals = intervals(all_starts)
    require(
        all(
            right_start >= left_end
            for (_, left_end), (right_start, _) in zip(
                all_intervals, all_intervals[1:]
            )
        ),
        "selection and confirmation windows overlap",
    )
    require(all_intervals[0][0] >= 0, "negative corpus window")
    require(all_intervals[-1][1] <= TRAIN_TOKEN_COUNT, "window exceeds TRAIN")
    require(
        THEORETICAL_MAX_NONOVERLAPPING_TRAIN_WINDOWS == 128,
        "unexpected TRAIN feasibility calculation",
    )


validate_frozen_cohorts()


def frozen_policy_sha256() -> str:
    return canonical_sha256(FROZEN_POLICY)


def worker_arguments(starts: Sequence[int], output: Path) -> list[str]:
    """Return the exact frozen worker arguments for one arithmetic B4 group."""

    require(len(starts) == BATCH_SIZE, "worker job must contain exactly four starts")
    stride = int(starts[1]) - int(starts[0])
    require(stride >= WINDOW_CORPUS_TOKENS, "worker windows overlap")
    require(
        tuple(int(starts[0]) + request * stride for request in range(BATCH_SIZE))
        == tuple(int(value) for value in starts),
        "worker starts are not an arithmetic B4 group",
    )
    return [
        "--backend",
        "page_gauge",
        "--model",
        MODEL,
        "--batch-size",
        str(BATCH_SIZE),
        "--context",
        str(CONTEXT),
        "--decode-steps",
        str(DECODE_STEPS),
        "--exact-tail",
        str(EXACT_TAIL_TOKENS),
        "--exact-sink-pages",
        str(EXACT_PREFIX_PAGES),
        "--prefill-chunk-tokens",
        "1024",
        "--baseline-split-pages",
        "256",
        "--candidate-split-pages",
        "256",
        "--tail-attention",
        "flashinfer_merge",
        "--old-value-scale-placement",
        "probability",
        "--trajectory-mode",
        "frozen_hf_teacher_forced",
        "--seed",
        "20260861",
        "--token-source",
        "wikitext2",
        "--wikitext-zip",
        str(ROOT / "data/wikitext-2-raw-v1.zip"),
        "--wikitext-member",
        TRAIN_MEMBER,
        "--token-offset",
        str(int(starts[0])),
        "--token-stride",
        str(stride),
        "--capture-warmups",
        "1",
        "--maximum-graph-banks",
        "16",
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--cache-scrub-mib",
        "256",
        "--min-logits-cosine",
        "0.995",
        "--min-top1-agreement",
        "0.99",
        "--quality-diagnostics-top-k",
        "64",
        "--quality-diagnostics-top-vocab",
        "12",
        "--output",
        str(output),
    ]


def _expect_equal(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for name, value in expected.items():
        require(name in mapping, f"{label}: missing {name}")
        require(mapping[name] == value, f"{label}.{name}: expected {value!r}, got {mapping[name]!r}")


def _validate_window_metadata(
    payload: Mapping[str, Any], starts: tuple[int, ...], cohort: str, label: str
) -> list[dict[str, Any]]:
    windows = payload.get("quality_windows")
    require(isinstance(windows, list) and len(windows) == BATCH_SIZE, f"{label}: quality windows")
    by_request: dict[int, dict[str, Any]] = {}
    for window in windows:
        require(isinstance(window, dict), f"{label}: quality window is not an object")
        request = int(window.get("request", -1))
        require(request not in by_request and 0 <= request < BATCH_SIZE, f"{label}: request index")
        start = starts[request]
        require(window.get("dataset_split") == "train", f"{label}: TEST/non-TRAIN window")
        require(window.get("archive_member") == TRAIN_MEMBER, f"{label}: archive member")
        require(window.get("archive_member_sha256") == TRAIN_MEMBER_SHA256, f"{label}: member hash")
        require(window.get("corpus_window_start_offset") == start, f"{label}: window start")
        require(
            window.get("corpus_window_end_offset_exclusive")
            == start + WINDOW_CORPUS_TOKENS,
            f"{label}: window end",
        )
        require(
            window.get("corpus_label_start_offset") == start + CONTEXT,
            f"{label}: label start",
        )
        require(
            window.get("corpus_label_end_offset_exclusive")
            == start + CONTEXT + DECODE_STEPS,
            f"{label}: label end",
        )
        require(
            window.get("model_predicted_position_start") == CONTEXT + 1,
            f"{label}: predicted position start",
        )
        require(
            window.get("model_predicted_position_end_exclusive")
            == CONTEXT + DECODE_STEPS + 1,
            f"{label}: predicted position end",
        )
        cluster_id = window.get("cluster_unit_id")
        require(isinstance(cluster_id, str) and cluster_id, f"{label}: cluster ID")
        require(cluster_id not in {item.get("cluster_unit_id") for item in by_request.values()}, f"{label}: duplicate cluster ID")
        copied = dict(window)
        copied["cohort"] = cohort
        by_request[request] = copied
    return [by_request[index] for index in range(BATCH_SIZE)]


def _validate_source_hashes(
    payload: Mapping[str, Any],
    label: str,
    locked_source_sha256: Mapping[str, str] | None,
) -> dict[str, str]:
    source = payload.get("source_sha256")
    require(isinstance(source, dict) and source, f"{label}: source hashes")
    normalized = {str(name).replace("\\", "/"): value for name, value in source.items()}
    require(all(is_sha256(value) for value in normalized.values()), f"{label}: malformed source hash")
    if locked_source_sha256 is not None:
        locked = {str(name).replace("\\", "/"): value for name, value in locked_source_sha256.items()}
        require(normalized == locked, f"{label}: raw worker source lock mismatch")
    return normalized


def _validate_system_gates(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    correctness = payload.get("correctness")
    require(isinstance(correctness, dict), f"{label}: correctness")
    same = correctness.get("same_backend_eager_vs_graph")
    require(isinstance(same, dict) and same.get("passed") is True, f"{label}: eager/graph gate")
    same_logits = same.get("logits", {})
    require(same_logits.get("checked_steps") == DECODE_STEPS, f"{label}: eager/graph steps")
    require(same_logits.get("checked_request_steps") == ROWS_PER_SHARD, f"{label}: eager/graph rows")
    require(same_logits.get("bitwise_identical") is True, f"{label}: eager/graph logits not bitwise")
    require(same.get("full_mutated_cache_range", {}).get("passed") is True, f"{label}: cache mutation gate")
    require(same.get("every_page_close_cache_digest", {}).get("passed") is True, f"{label}: page-close digests")
    require(same.get("every_page_close_cache_digest", {}).get("checked_page_closes") == 64, f"{label}: page-close count")
    require(same.get("final_serving_metadata", {}).get("passed") is True, f"{label}: serving metadata")
    adjacent = same.get("immutable_adjacent_page_canaries", {})
    for mode in ("eager", "graph_run_1", "graph_run_2_restored"):
        require(adjacent.get(mode, {}).get("passed") is True, f"{label}: {mode} adjacent canary")
    prefix_canary = same.get("immutable_exact_prefix_timed_sample_canary", {})
    require(prefix_canary.get("passed") is True, f"{label}: prefix timed canary")
    require(prefix_canary.get("check_count") == 2, f"{label}: prefix canary count")
    for mode in ("cache_neutral", "cache_hot"):
        current = prefix_canary.get("per_mode", {}).get(mode, {})
        require(current.get("passed") is True, f"{label}: {mode} prefix canary")
        require(
            current.get("no_prefix_digest_between_hot_precondition_and_sample") is True,
            f"{label}: {mode} literal-hot protocol",
        )
    require(same.get("eager_gate_passed") is True, f"{label}: eager dispatch gate")
    graph_gate = same.get("graph_runtime_gate", {})
    require(graph_gate.get("passed") is True, f"{label}: graph runtime gate")
    dispatch = graph_gate.get("observed_dispatch", {})
    require(dispatch.get("graph_replays") == 32 * DECODE_STEPS, f"{label}: graph replay count")
    require(dispatch.get("eager_calls") == 0, f"{label}: graph eager fallback")
    require(dispatch.get("graph_bank_misses") == 0, f"{label}: graph bank miss")
    repeat = same.get("restored_graph_repeat", {})
    require(repeat.get("passed") is True, f"{label}: restored graph repeat")
    require(repeat.get("logits", {}).get("bitwise_identical") is True, f"{label}: repeat logits")
    require(repeat.get("runtime_gate", {}).get("passed") is True, f"{label}: repeat runtime gate")

    finalization = correctness.get("runtime_page_finalization_and_consumption", {})
    require(finalization.get("passed") is True, f"{label}: runtime finalization")
    require(finalization.get("first_generated_logical_page") == 1280, f"{label}: first generated page")
    require(finalization.get("last_generated_logical_page") == 1343, f"{label}: last generated page")
    require(finalization.get("generated_pages") == 64, f"{label}: generated page count")
    require(finalization.get("final_old_logical_page_end_exclusive") == 1296, f"{label}: final old-page boundary")
    require(finalization.get("final_exact_prefix_logical_pages") == [0, 1, 2], f"{label}: final exact prefix")
    require(finalization.get("final_exact_tail_logical_pages") == list(range(1296, 1344)), f"{label}: final exact tail")
    require(
        finalization.get("runtime_finalized_pages_consumed_as_int8")
        == list(range(1280, 1296)),
        f"{label}: runtime-finalized INT8 recurrence pages",
    )
    require(
        int(finalization.get("runtime_finalized_int8_pages_consumed_count", -1))
        == 16,
        f"{label}: expected exactly 16 runtime-finalized INT8 pages consumed",
    )
    for gate in (
        "final_attention_page_table_gate_passed",
        "old_attention_page_table_gate_passed",
        "exact_attention_page_table_gate_passed",
        "logical_page_sets_disjoint",
        "logical_token_coverage_exactly_once",
        "exact_prefix_excluded_from_old_segment",
        "prefix_exclusion_gate_passed",
    ):
        require(finalization.get(gate) is True, f"{label}: {gate}")

    scheduler = payload.get("scheduler_capacity", {})
    require(scheduler.get("all_wrappers_analytically_within_capacity") is True, f"{label}: scheduler capacity")
    require(scheduler.get("split_was_automatically_clamped") is False, f"{label}: split clamp")
    graph = payload.get("cuda_graph_provenance", {})
    require(graph.get("enabled") is True, f"{label}: CUDA graph disabled")
    require(graph.get("structure_gate_passed") is True, f"{label}: graph structure")
    require(graph.get("graph_bank_misses") == 0, f"{label}: provenance graph misses")
    require(graph.get("replays_per_decode_step") == 32, f"{label}: layer replay coverage")
    cache_build = payload.get("cache_build", {})
    _expect_equal(
        cache_build,
        {
            "served_pages_per_request": 1344,
            "initial_pages_per_request": 1280,
            "mutated_generated_pages_per_request": 64,
            "exact_tail_pages_per_request": 48,
            "exact_prefix_pages_per_request": 3,
            "exact_storage_pages_per_request": 51,
            "full_served_mutation_range_restored_before_every_validation_and_sample": True,
            "preceding_and_following_page_canaries_checked": True,
            "opposite_backend_full_gpu_cache_allocated": False,
        },
        f"{label}.cache_build",
    )
    return {
        "same_backend_eager_vs_graph_passed": True,
        "restored_graph_repeat_passed": True,
        "all_canaries_passed": True,
        "runtime_page_finalization_and_consumption_passed": True,
        "runtime_finalized_int8_pages_consumed_count": int(
            finalization["runtime_finalized_int8_pages_consumed_count"]
        ),
        "scheduler_capacity_passed_without_clamp": True,
        "graph_structure_passed_without_miss_or_fallback": True,
    }


def _extract_clusters(
    payload: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparison = payload["correctness"].get("backend_vs_hf_sdpa_fp16")
    require(isinstance(comparison, dict), f"{label}: backend/HF comparison")
    require(comparison.get("minimum_logits_cosine") == MINIMUM_LOGITS_COSINE, f"{label}: cosine threshold changed")
    require(comparison.get("minimum_top1_agreement") == MINIMUM_TOP1_AGREEMENT, f"{label}: top1 threshold changed")
    logits = comparison.get("logits", {})
    require(logits.get("checked_steps") == DECODE_STEPS, f"{label}: quality steps")
    require(logits.get("checked_request_steps") == ROWS_PER_SHARD, f"{label}: quality rows")
    diagnostics = comparison.get("outlier_diagnostics")
    require(isinstance(diagnostics, dict), f"{label}: quality diagnostics absent")
    require(diagnostics.get("threshold_unchanged") == MINIMUM_LOGITS_COSINE, f"{label}: diagnostic threshold")
    summary = diagnostics.get("summary", {})
    require(summary.get("checked_rows") == ROWS_PER_SHARD, f"{label}: diagnostic row count")
    matrices = diagnostics.get("row_matrices")
    require(isinstance(matrices, dict), f"{label}: row matrices")
    names = (
        "cosine_by_step_request",
        "relative_l2_by_step_request",
        "maximum_absolute_error_by_step_request",
        "top1_match_by_step_request",
    )
    for name in names:
        rows = matrices.get(name)
        require(isinstance(rows, list) and len(rows) == DECODE_STEPS, f"{label}: {name} steps")
        require(all(isinstance(row, list) and len(row) == BATCH_SIZE for row in rows), f"{label}: {name} batch")

    by_request = [
        {"cosine": [], "relative_l2": [], "maximum_absolute_error": [], "top1": []}
        for _ in range(BATCH_SIZE)
    ]
    for step in range(DECODE_STEPS):
        for request in range(BATCH_SIZE):
            cosine = finite_number(matrices["cosine_by_step_request"][step][request], f"{label}: cosine")
            relative_l2 = finite_number(matrices["relative_l2_by_step_request"][step][request], f"{label}: relative L2")
            maximum_error = finite_number(matrices["maximum_absolute_error_by_step_request"][step][request], f"{label}: maximum error")
            top1 = matrices["top1_match_by_step_request"][step][request]
            require(-1.0 <= cosine <= 1.0, f"{label}: cosine range")
            require(relative_l2 >= 0.0, f"{label}: relative L2 range")
            require(maximum_error >= 0.0, f"{label}: maximum error range")
            require(isinstance(top1, bool), f"{label}: top1 value")
            by_request[request]["cosine"].append(cosine)
            by_request[request]["relative_l2"].append(relative_l2)
            by_request[request]["maximum_absolute_error"].append(maximum_error)
            by_request[request]["top1"].append(top1)

    all_cosines = [value for request in by_request for value in request["cosine"]]
    all_top1 = [value for request in by_request for value in request["top1"]]
    observed_minimum = min(all_cosines)
    observed_mean = statistics.fmean(all_cosines)
    observed_top1 = sum(all_top1) / ROWS_PER_SHARD
    observed_rows_below = sum(
        value < MINIMUM_LOGITS_COSINE for value in all_cosines
    )
    observed_top1_mismatches = ROWS_PER_SHARD - sum(all_top1)
    require(math.isclose(observed_minimum, float(logits.get("minimum_cosine")), rel_tol=0.0, abs_tol=1e-12), f"{label}: serialized minimum mismatch")
    require(math.isclose(observed_mean, float(logits.get("mean_cosine")), rel_tol=0.0, abs_tol=2e-7), f"{label}: serialized mean mismatch")
    require(math.isclose(observed_top1, float(logits.get("top1_agreement_fraction")), rel_tol=0.0, abs_tol=1e-15), f"{label}: serialized top1 mismatch")
    require(
        summary.get("rows_below_gate") == observed_rows_below,
        f"{label}: diagnostic below-gate row count mismatch",
    )
    require(
        math.isclose(
            float(summary.get("fraction_below_gate")),
            observed_rows_below / ROWS_PER_SHARD,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{label}: diagnostic below-gate fraction mismatch",
    )
    require(
        summary.get("top1_mismatches") == observed_top1_mismatches,
        f"{label}: diagnostic top1 mismatch count",
    )
    require(
        math.isclose(
            float(summary.get("top1_agreement_fraction")),
            observed_top1,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{label}: diagnostic top1 fraction mismatch",
    )
    expected_worker_pass = bool(
        observed_minimum >= MINIMUM_LOGITS_COSINE
        and observed_top1 >= MINIMUM_TOP1_AGREEMENT
    )
    require(comparison.get("passed") is expected_worker_pass, f"{label}: inconsistent worker quality decision")

    clusters = []
    for request, window in enumerate(windows):
        current = by_request[request]
        minimum_step = min(
            range(DECODE_STEPS), key=lambda step: current["cosine"][step]
        )
        clusters.append(
            {
                "cluster_unit_id": window["cluster_unit_id"],
                "cohort": window["cohort"],
                "request": request,
                "window": dict(window),
                "row_count": ROWS_PER_WINDOW,
                "cosine": current["cosine"],
                "relative_l2": current["relative_l2"],
                "maximum_absolute_error": current["maximum_absolute_error"],
                "top1": current["top1"],
                "sums": {
                    "cosine": math.fsum(current["cosine"]),
                    "relative_l2": math.fsum(current["relative_l2"]),
                    "maximum_absolute_error": math.fsum(current["maximum_absolute_error"]),
                    "top1": sum(current["top1"]),
                },
                "minimum_cosine": min(current["cosine"]),
                "minimum_step": minimum_step,
            }
        )
    worst_cluster = min(clusters, key=lambda cluster: cluster["minimum_cosine"])
    worst_step = int(worst_cluster["minimum_step"])
    return clusters, {
        "worker_quality_gate_passed": expected_worker_pass,
        "minimum_logits_cosine": observed_minimum,
        "mean_logits_cosine": observed_mean,
        "top1_agreement_fraction": observed_top1,
        "rows_below_minimum_logits_cosine_gate": observed_rows_below,
        "top1_mismatches": observed_top1_mismatches,
        "checked_request_steps": ROWS_PER_SHARD,
        "worst_row_identity": {
            "request": int(worst_cluster["request"]),
            "step": worst_step,
            "model_input_absolute_position": CONTEXT + worst_step,
            "corpus_window_start_offset": int(
                worst_cluster["window"]["corpus_window_start_offset"]
            ),
            "corpus_label_offset": int(
                worst_cluster["window"]["corpus_label_start_offset"]
            )
            + worst_step,
            "cosine": float(worst_cluster["minimum_cosine"]),
        },
    }


def artifact_common_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    token_source = payload["token_source"]
    return {
        "policy_sha256": frozen_policy_sha256(),
        "model_revision": payload["model_provenance"]["resolved_revision"],
        "model_config_sha256": payload["model_provenance"]["config_sha256"],
        "sampled_parameters_sha256": payload["model_provenance"]["sampled_parameters"]["sha256"],
        "archive_sha256": token_source["archive_sha256"],
        "archive_member_sha256": token_source["archive_member_sha256"],
        "tokenizer_manifest_sha256": token_source["tokenizer"]["manifest_sha256"],
        "attention_custom_module_source_hashes": payload["attention_implementation"]["custom_module_source_hashes"],
        "source_sha256": payload["source_sha256"],
        "gpu": payload["environment"]["gpu"],
        "compute_capability": payload["environment"]["compute_capability"],
        "torch": payload["environment"]["torch"],
        "torch_cuda": payload["environment"]["torch_cuda"],
        "flashinfer": payload["environment"]["flashinfer"],
        "transformers": payload["environment"]["transformers"],
    }


def validate_raw_artifact(
    payload: Mapping[str, Any],
    *,
    expected_starts: Sequence[int],
    cohort: str,
    label: str = "artifact",
    locked_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed on identity, strict quality rows, recurrence, and systems gates."""

    starts = tuple(int(value) for value in expected_starts)
    require(cohort in {"selection", "confirmation"}, f"{label}: invalid cohort")
    require(len(starts) == BATCH_SIZE, f"{label}: expected B4")
    require(payload.get("schema_version") == RAW_SCHEMA_VERSION, f"{label}: raw schema")
    require(payload.get("experiment") == EXPECTED_EXPERIMENT, f"{label}: experiment")
    require(payload.get("backend") == "page_gauge", f"{label}: backend")
    require(payload.get("cuda_graph_scope") == "decoder_layer_device_dynamic", f"{label}: graph scope")
    configuration = payload.get("configuration")
    require(isinstance(configuration, dict), f"{label}: configuration")
    expected_configuration = dict(FROZEN_POLICY)
    expected_configuration.pop("exact_prefix_pages")
    expected_configuration["exact_sink_pages"] = EXACT_PREFIX_PAGES
    _expect_equal(configuration, expected_configuration, f"{label}.configuration")
    require(configuration.get("token_offset") == starts[0], f"{label}: token offset")
    stride = starts[1] - starts[0]
    require(configuration.get("token_stride") == stride, f"{label}: token stride")

    model = payload.get("model_provenance", {})
    require(model.get("requested_name_or_path") == MODEL, f"{label}: model")
    require(model.get("resolved_revision") == MODEL_REVISION, f"{label}: model revision")
    require(model.get("config_sha256") == MODEL_CONFIG_SHA256, f"{label}: model config hash")
    require(is_sha256(model.get("sampled_parameters", {}).get("sha256")), f"{label}: parameter sample hash")

    token_source = payload.get("token_source")
    require(isinstance(token_source, dict), f"{label}: token source")
    _expect_equal(
        token_source,
        {
            "kind": "wikitext2",
            "split": "train",
            "archive_sha256": ARCHIVE_SHA256,
            "archive_sha256_verified": True,
            "archive_member": TRAIN_MEMBER,
            "archive_member_sha256": TRAIN_MEMBER_SHA256,
            "corpus_window_start_offsets": list(starts),
            "corpus_window_end_offsets_exclusive": [
                start + WINDOW_CORPUS_TOKENS for start in starts
            ],
            "corpus_window_stride": stride,
            "corpus_windows_disjoint": True,
            "corpus_tokens_per_request": WINDOW_CORPUS_TOKENS,
            "available_corpus_token_count": TRAIN_TOKEN_COUNT,
            "shape": [BATCH_SIZE, WINDOW_CORPUS_TOKENS + 1],
        },
        f"{label}.token_source",
    )
    require(is_sha256(token_source.get("token_ids_sha256")), f"{label}: token content hash")
    require(is_sha256(token_source.get("tokenizer", {}).get("manifest_sha256")), f"{label}: tokenizer hash")
    windows = _validate_window_metadata(payload, starts, cohort, label)

    source_hashes = _validate_source_hashes(payload, label, locked_source_sha256)
    attention = payload.get("attention_implementation", {})
    _expect_equal(
        attention,
        {
            "implementation": "page_gauge_segmented_flashinfer_merge",
            "logical_attention_segments": 2,
            "old_int8_value_scale_placement": "probability",
            "exact_tail_pages": EXACT_TAIL_TOKENS // PAGE,
            "exact_prefix_pages": EXACT_PREFIX_PAGES,
            "exact_segment_logical_order": "[contiguous exact prefix logical pages 0..S-1, chronological recent exact tail]",
            "old_segment_logical_range": "[S, tail_start) with every exact-prefix page excluded",
            "logical_attention_segments_unchanged_by_prefix": True,
        },
        f"{label}.attention_implementation",
    )
    require(attention.get("custom_module_source_hashes", {}).get("header_matches_expected") is True, f"{label}: attention header hash")
    require(is_sha256(attention.get("custom_module_source_hashes", {}).get("module_source_sha256")), f"{label}: attention source hash")

    trajectory = payload.get("trajectory", {})
    require(trajectory.get("mode") == "frozen_hf_teacher_forced", f"{label}: trajectory")
    require(trajectory.get("frozen_identical_hf_input_chain") is True, f"{label}: frozen teacher chain")
    require(is_sha256(trajectory.get("teacher_inputs_sha256")), f"{label}: teacher hash")
    pairing_configuration = payload.get("pairing", {}).get("configuration", {})
    for name, value in configuration.items():
        if name in pairing_configuration:
            require(
                pairing_configuration[name] == value,
                f"{label}: pairing/configuration mismatch for {name}",
            )
    require(pairing_configuration.get("teacher_inputs_sha256") == trajectory["teacher_inputs_sha256"], f"{label}: teacher pairing hash")
    require(is_sha256(pairing_configuration.get("token_matrix_sha256")), f"{label}: token matrix hash")
    require(is_sha256(pairing_configuration.get("token_source_provenance_sha256")), f"{label}: token provenance hash")
    require(
        pairing_configuration["token_source_provenance_sha256"]
        == canonical_sha256(token_source),
        f"{label}: token provenance hash mismatch",
    )
    require(
        pairing_configuration.get("model_config_sha256") == MODEL_CONFIG_SHA256,
        f"{label}: paired model config hash",
    )
    require(
        pairing_configuration.get("sampled_model_parameters_sha256")
        == model["sampled_parameters"]["sha256"],
        f"{label}: paired parameter sample hash",
    )
    pairing_key = payload.get("pairing", {}).get("pairing_key_sha256")
    require(is_sha256(pairing_key), f"{label}: pairing key")
    require(
        pairing_key == canonical_sha256(pairing_configuration),
        f"{label}: pairing key hash mismatch",
    )

    system_gates = _validate_system_gates(payload, label)
    clusters, quality = _extract_clusters(payload, windows, label)
    require(payload.get("correctness", {}).get("passed") is quality["worker_quality_gate_passed"], f"{label}: correctness decision mismatch")
    require(payload.get("passed") is quality["worker_quality_gate_passed"], f"{label}: top-level decision mismatch")
    return {
        "cohort": cohort,
        "starts": list(starts),
        "ends_exclusive": [start + WINDOW_CORPUS_TOKENS for start in starts],
        "rows": ROWS_PER_SHARD,
        "clusters": clusters,
        "quality": quality,
        "system_gates": system_gates,
        "source_sha256": source_hashes,
        "policy_sha256": frozen_policy_sha256(),
        "content_hashes": {
            "token_ids_sha256": token_source["token_ids_sha256"],
            "teacher_inputs_sha256": trajectory["teacher_inputs_sha256"],
            "token_matrix_sha256": pairing_configuration["token_matrix_sha256"],
            "token_source_provenance_sha256": pairing_configuration[
                "token_source_provenance_sha256"
            ],
        },
        "common_signature": artifact_common_signature(payload),
        "invocation_utc_timestamp": payload.get("invocation", {}).get("utc_timestamp"),
    }


def cluster_public_summary(cluster: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(cluster["row_count"])
    minimum_step = int(cluster["minimum_step"])
    return {
        "cluster_unit_id": cluster["cluster_unit_id"],
        "cohort": cluster["cohort"],
        "window": cluster["window"],
        "row_count": rows,
        "minimum_logits_cosine": float(cluster["minimum_cosine"]),
        "minimum_row_identity": {
            "request_within_shard": int(cluster["request"]),
            "step": minimum_step,
            "model_input_absolute_position": CONTEXT + minimum_step,
            "corpus_label_offset": int(
                cluster["window"]["corpus_label_start_offset"]
            )
            + minimum_step,
        },
        "mean_logits_cosine": float(cluster["sums"]["cosine"]) / rows,
        "top1_agreement_fraction": float(cluster["sums"]["top1"]) / rows,
        "mean_relative_l2": float(cluster["sums"]["relative_l2"]) / rows,
        "relative_l2_p99": percentile(cluster["relative_l2"], 0.99),
        "mean_maximum_absolute_error": float(
            cluster["sums"]["maximum_absolute_error"]
        )
        / rows,
        "maximum_absolute_error": max(cluster["maximum_absolute_error"]),
    }


def cohort_point_summary(clusters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(clusters), "cohort has no clusters")
    row_count = sum(int(cluster["row_count"]) for cluster in clusters)
    cosine_values = [value for cluster in clusters for value in cluster["cosine"]]
    relative_l2_values = [
        value for cluster in clusters for value in cluster["relative_l2"]
    ]
    maximum_error_values = [
        value
        for cluster in clusters
        for value in cluster["maximum_absolute_error"]
    ]
    worst_cluster = min(clusters, key=lambda cluster: float(cluster["minimum_cosine"]))
    worst_step = int(worst_cluster["minimum_step"])
    return {
        "cluster_count": len(clusters),
        "row_count": row_count,
        "minimum_logits_cosine": min(cosine_values),
        "mean_logits_cosine": math.fsum(cosine_values) / row_count,
        "rows_below_minimum_logits_cosine_gate": sum(
            value < MINIMUM_LOGITS_COSINE for value in cosine_values
        ),
        "worst_row_identity": {
            "cluster_unit_id": worst_cluster["cluster_unit_id"],
            "request_within_shard": int(worst_cluster["request"]),
            "step": worst_step,
            "model_input_absolute_position": CONTEXT + worst_step,
            "corpus_window_start_offset": int(
                worst_cluster["window"]["corpus_window_start_offset"]
            ),
            "corpus_label_offset": int(
                worst_cluster["window"]["corpus_label_start_offset"]
            )
            + worst_step,
            "cosine": float(worst_cluster["minimum_cosine"]),
        },
        "logits_cosine_quantiles": {
            "minimum": min(cosine_values),
            "p0_1": percentile(cosine_values, 0.001),
            "p0_5": percentile(cosine_values, 0.005),
            "p1": percentile(cosine_values, 0.01),
            "p5": percentile(cosine_values, 0.05),
            "median": percentile(cosine_values, 0.5),
            "p95": percentile(cosine_values, 0.95),
            "p99": percentile(cosine_values, 0.99),
            "maximum": max(cosine_values),
        },
        "logits_cosine_p01": percentile(cosine_values, 0.01),
        "top1_agreement_fraction": sum(
            int(cluster["sums"]["top1"]) for cluster in clusters
        )
        / row_count,
        "mean_relative_l2": math.fsum(relative_l2_values) / row_count,
        "relative_l2_p99": percentile(relative_l2_values, 0.99),
        "relative_l2_maximum": max(relative_l2_values),
        "mean_maximum_absolute_error": math.fsum(maximum_error_values) / row_count,
        "maximum_absolute_error": max(maximum_error_values),
        "per_window": [cluster_public_summary(cluster) for cluster in clusters],
    }


def whole_window_bootstrap(
    clusters: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    require(samples >= 100, "bootstrap requires at least 100 samples")
    require(len(clusters) >= 4, "bootstrap requires at least four windows")
    generator = random.Random(seed)
    draws: dict[str, list[float]] = {
        "minimum_cosine": [],
        "mean_cosine": [],
        "top1": [],
        "mean_relative_l2": [],
        "mean_maximum_absolute_error": [],
    }
    for _ in range(samples):
        selected = [clusters[generator.randrange(len(clusters))] for _ in clusters]
        rows = sum(int(cluster["row_count"]) for cluster in selected)
        draws["minimum_cosine"].append(
            min(float(cluster["minimum_cosine"]) for cluster in selected)
        )
        draws["mean_cosine"].append(
            math.fsum(float(cluster["sums"]["cosine"]) for cluster in selected)
            / rows
        )
        draws["top1"].append(
            sum(int(cluster["sums"]["top1"]) for cluster in selected) / rows
        )
        draws["mean_relative_l2"].append(
            math.fsum(
                float(cluster["sums"]["relative_l2"]) for cluster in selected
            )
            / rows
        )
        draws["mean_maximum_absolute_error"].append(
            math.fsum(
                float(cluster["sums"]["maximum_absolute_error"])
                for cluster in selected
            )
            / rows
        )
    return {
        "method": "nonparametric percentile bootstrap over whole disjoint WikiText TRAIN windows",
        "within_window_tokens_resampled_independently": False,
        "cluster_count_per_draw": len(clusters),
        "samples": samples,
        "seed": seed,
        "one_sided_95": {
            "minimum_logits_cosine_lower": percentile(draws["minimum_cosine"], 0.05),
            "mean_logits_cosine_lower": percentile(draws["mean_cosine"], 0.05),
            "top1_agreement_lower": percentile(draws["top1"], 0.05),
            "mean_relative_l2_upper": percentile(draws["mean_relative_l2"], 0.95),
            "mean_maximum_absolute_error_upper": percentile(
                draws["mean_maximum_absolute_error"], 0.95
            ),
        },
        "two_sided_95": {
            name: [percentile(values, 0.025), percentile(values, 0.975)]
            for name, values in draws.items()
        },
    }


def strict_cohort_gate(
    point: Mapping[str, Any], shard_quality: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    checks = {
        "minimum_logits_cosine": {
            "value": point["minimum_logits_cosine"],
            "operator": ">=",
            "threshold": MINIMUM_LOGITS_COSINE,
            "passed": point["minimum_logits_cosine"] >= MINIMUM_LOGITS_COSINE,
        },
        "zero_rows_below_minimum_logits_cosine_gate": {
            "value": point["rows_below_minimum_logits_cosine_gate"],
            "operator": "==",
            "threshold": 0,
            "passed": point["rows_below_minimum_logits_cosine_gate"] == 0,
        },
        "top1_agreement_fraction": {
            "value": point["top1_agreement_fraction"],
            "operator": ">=",
            "threshold": MINIMUM_TOP1_AGREEMENT,
            "passed": point["top1_agreement_fraction"] >= MINIMUM_TOP1_AGREEMENT,
        },
        "all_b4_shards_passed_worker_quality_gate": {
            "value": all(
                bool(shard["worker_quality_gate_passed"])
                for shard in shard_quality
            ),
            "operator": "is",
            "threshold": True,
            "passed": all(
                bool(shard["worker_quality_gate_passed"])
                for shard in shard_quality
            ),
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def verify_manifest(payload: Mapping[str, Any]) -> None:
    require(payload.get("schema_version") == PROTOCOL_SCHEMA_VERSION, "manifest schema")
    require(payload.get("experiment") == "page_gauge_train_selection_confirmation_preregistration", "manifest experiment")
    expected = dict(payload)
    observed = expected.pop("manifest_sha256", None)
    require(is_sha256(observed), "manifest hash missing or malformed")
    require(canonical_sha256(expected) == observed, "manifest hash integrity failure")
    require(payload.get("policy_sha256") == frozen_policy_sha256(), "manifest policy hash")
    require(payload.get("frozen_policy") == FROZEN_POLICY, "manifest policy changed")
    require(payload.get("gates") == FROZEN_GATES, "manifest gates changed")
    require(payload.get("created_before_any_confirmation_job") is True, "manifest was not sealed before confirmation")
    require(payload.get("selection_was_observed_before_preregistration") is True, "manifest mislabels selection timing")
    one_shot = payload.get("one_shot_rule", {})
    require(
        one_shot.get(
            "run_all_five_preregistered_shards_even_if_an_earlier_quality_gate_fails"
        )
        is True,
        "manifest permits optional stopping",
    )
    require(one_shot.get("failed_quality_artifact_is_terminal") is True, "manifest permits quality reruns")
    require(one_shot.get("rerun_failed_quality_artifact") is False, "manifest permits quality reruns")
    require(one_shot.get("fallback_policy") is None, "manifest contains a fallback policy")
    require(one_shot.get("retune_after_confirmation") is False, "manifest permits retuning")
    cohort = payload.get("cohort_design", {})
    require(tuple(cohort.get("selection_window_starts", ())) == SELECTION_STARTS, "manifest selection starts")
    require(tuple(cohort.get("confirmation_window_starts", ())) == CONFIRMATION_STARTS, "manifest confirmation starts")
    require(cohort.get("test_split_used") is False, "manifest permits TEST")
    require(cohort.get("selection_window_count") == 4, "manifest selection count")
    require(cohort.get("confirmation_window_count") == 20, "manifest confirmation count")
    require(cohort.get("confirmation_worker_shards") == 5, "manifest shard count")
    require(cohort.get("rows_per_b4_shard") == ROWS_PER_SHARD, "manifest rows per shard")
    require(cohort.get("confirmation_total_rows") == 20 * ROWS_PER_WINDOW, "manifest confirmation rows")
    require(cohort.get("selection_confirmation_duplicate_count") == 0, "manifest duplicate windows")
    require(cohort.get("selection_confirmation_overlap_count") == 0, "manifest overlapping windows")
    selection = payload.get("selection_artifact", {})
    require(is_sha256(selection.get("sha256")), "manifest selection artifact hash")
    require(selection.get("rows") == ROWS_PER_SHARD, "manifest selection rows")
    require(tuple(selection.get("window_starts", ())) == SELECTION_STARTS, "manifest selection artifact starts")
    require(selection.get("policy_sha256") == frozen_policy_sha256(), "manifest selection policy hash")
    require(
        all(is_sha256(value) for value in selection.get("content_hashes", {}).values()),
        "manifest selection content hashes",
    )
    bootstrap = payload.get("bootstrap", {})
    require(bootstrap.get("samples") == BOOTSTRAP_SAMPLES, "manifest bootstrap samples")
    require(bootstrap.get("seed") == BOOTSTRAP_SEED, "manifest bootstrap seed")
    require(bootstrap.get("within_window_rows_resampled_independently") is False, "manifest token-row bootstrap")
    source_lock = payload.get("source_lock", {})
    raw_source = source_lock.get("raw_worker_source_sha256", {})
    tool_source = source_lock.get("protocol_tool_source_sha256", {})
    require(isinstance(raw_source, dict) and raw_source, "manifest raw source lock")
    require(isinstance(tool_source, dict) and len(tool_source) == 3, "manifest tool source lock")
    require(all(is_sha256(value) for value in raw_source.values()), "manifest raw source hash")
    require(all(is_sha256(value) for value in tool_source.values()), "manifest tool source hash")
    require(is_sha256(source_lock.get("worker_entrypoint_sha256")), "manifest worker hash")
    required_environment = payload.get("required_environment", {})
    require(
        required_environment
        == {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0",
            "fresh_process_per_shard": True,
        },
        "manifest environment changed",
    )
    jobs = payload.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 5, "manifest jobs")
    output_paths = []
    log_paths = []
    for index, (job, starts) in enumerate(zip(jobs, CONFIRMATION_GROUPS)):
        require(job.get("job_id") == f"confirmation_{index:02d}", "manifest job ID")
        require(job.get("cohort") == "confirmation", "manifest job cohort")
        require(job.get("fresh_worker_process_required") is True, "manifest reuses a worker")
        require(tuple(job.get("window_starts", ())) == starts, "manifest job starts")
        require(
            tuple(job.get("window_ends_exclusive", ()))
            == tuple(start + WINDOW_CORPUS_TOKENS for start in starts),
            "manifest job ends",
        )
        require(job.get("token_offset") == starts[0], "manifest job offset")
        require(job.get("token_stride") == CONFIRMATION_STRIDE, "manifest job stride")
        require(job.get("expected_rows") == ROWS_PER_SHARD, "manifest job rows")
        output_path = Path(str(job.get("output_path")))
        log_path = Path(str(job.get("log_path")))
        require(output_path.name == f"confirmation_{index:02d}.json", "manifest output name")
        require(log_path.name == f"confirmation_{index:02d}.log", "manifest log name")
        require(
            job.get("worker_arguments") == worker_arguments(starts, output_path),
            "manifest worker arguments",
        )
        expected_command_hash = canonical_sha256(
            [str(job.get("worker_path")), *job["worker_arguments"]]
        )
        require(job.get("command_sha256") == expected_command_hash, "manifest command hash")
        output_paths.append(str(output_path))
        log_paths.append(str(log_path))
    require(len(set(output_paths)) == 5, "manifest duplicate output paths")
    require(len(set(log_paths)) == 5, "manifest duplicate log paths")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload
