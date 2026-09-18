#!/usr/bin/env python3
"""Aggregate the frozen PageGauge WikiText-2 TEST quality protocol.

This is deliberately a CPU-only, fail-closed reducer.  It accepts the five
raw model-prefill result files for the pre-registered 14-window protocol,
recomputes quality statistics from serialized per-token scalars, and treats a
whole disjoint corpus window as the bootstrap unit.  It never resamples the
1,536 correlated tokens within a window as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
RAW_SCHEMA_VERSION = 8
EXPECTED_EXPERIMENT = "page_gauge_model_prefill_correctness"
PROTOCOL_NAME = "wikitext2_test_s3_t768_d1536_v1"
EXPECTED_SPLIT = "test"
EXPECTED_MEMBER = "wikitext-2-raw/wiki.test.raw"
EXPECTED_ARCHIVE_SHA256 = (
    "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
)
EXPECTED_TEST_TOKEN_COUNT = 328879
EXPECTED_MODEL = "mistralai/Mistral-7B-v0.3"
EXPECTED_MODEL_REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
EXPECTED_MODEL_CONFIG_SHA256 = (
    "f223f73de240195fe40495201ecbe85c9dac820842e4fe7e50c38d576b8d22ca"
)
EXPECTED_MODEL_PARAMETER_COUNT = 7_248_023_552
EXPECTED_TOKENIZER_MANIFEST_SHA256 = (
    "c97b022b02f1b0c13a96ed355471032bdbeefe28eaad8db51b488207e3cac67a"
)
CONTEXT = 20480
DECODE_STEPS = 1536
EXACT_TAIL = 768
EXACT_PREFIX_PAGES = 3
PAGE = 16
WINDOW_CORPUS_TOKENS = CONTEXT + DECODE_STEPS
WINDOW_STRIDE = 23600
EXPECTED_STARTS = tuple(index * WINDOW_STRIDE for index in range(14))
EXPECTED_GROUPS = (
    EXPECTED_STARTS[0:3],
    EXPECTED_STARTS[3:6],
    EXPECTED_STARTS[6:9],
    EXPECTED_STARTS[9:12],
    EXPECTED_STARTS[12:14],
)
EXPECTED_LAST_END = 328816
EXPECTED_PREFILL_CHUNK_TOKENS = 1024
EXPECTED_SPLIT_PAGES = 256
EXPECTED_TAIL_ATTENTION = "flashinfer_merge"
EXPECTED_OLD_VALUE_SCALE_PLACEMENT = "probability"
EXPECTED_SEED = 20_260_861
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 20_260_875
FLASHINFER_BOOTSTRAP_SEED = BOOTSTRAP_SEED + 1
MODEL_FP16_RESIDENT_BYTES = 14_496_047_616
MEMORY_RESERVE_BYTES = 2 * 1024**3
MAX_CORESIDENT_BATCH = 3

EXPECTED_SOURCE_KEYS = (
    "diagnostics/benchmark_model_prefill_correctness.py",
    "diagnostics/aggregate_heldout_quality.py",
    "diagnostics/benchmark_token_step_graphs.py",
    "diagnostics/benchmark_generated_sequence_graph.py",
    "scripts/benchmark_page_gauge_transformer.py",
    "scripts/benchmark_flashinfer_page_affine_int8.py",
    "scripts/benchmark_page_gauge_overheads.py",
    "scripts/benchmark_e2e_transformer.py",
    "scripts/page_gauge_runtime.py",
    "tests/page_gauge_append_extension.cu",
    "patches/flashinfer-0.6.17-page-gauge-int8.patch",
)

# These thresholds are intentionally constants rather than CLI knobs.  A
# publication run must not tune its acceptance rule after observing TEST.
PAGE_GAUGE_GATES = {
    "minimum_logits_cosine": 0.995,
    "minimum_top1_point": 0.99,
    "minimum_top1_cluster_bootstrap_lower_95": 0.98,
    "maximum_ppl_ratio_cluster_bootstrap_upper_95": 1.01,
    "maximum_ppl_increase_cluster_bootstrap_upper_95": 0.10,
    "maximum_single_window_ppl_ratio": 1.03,
    "maximum_mean_forward_kl_cluster_bootstrap_upper_95": 1.0e-3,
    "maximum_forward_kl_p99": 1.0e-2,
    "maximum_mean_js_cluster_bootstrap_upper_95": 2.5e-4,
    "maximum_js_p99": 2.5e-3,
}

FLASHINFER_HF_GATES = {
    "minimum_logits_cosine": 0.999,
    "minimum_top1_point": 0.995,
    "minimum_top1_cluster_bootstrap_lower_95": 0.99,
    "minimum_ppl_ratio_cluster_bootstrap_lower_90": 1.0 / 1.002,
    "maximum_ppl_ratio_cluster_bootstrap_upper_90": 1.002,
    "maximum_mean_forward_kl_cluster_bootstrap_upper_95": 1.0e-4,
    "maximum_mean_js_cluster_bootstrap_upper_95": 2.5e-5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_source_hashes(payload: dict[str, Any]) -> dict[str, str]:
    source = payload.get("source_sha256")
    require(isinstance(source, dict), "source hashes must be an object")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in source.items():
        name = str(raw_name).replace("\\", "/")
        require(name not in normalized, f"duplicate normalized source path {name}")
        require(
            isinstance(raw_digest, str)
            and len(raw_digest) == 64
            and all(character in "0123456789abcdef" for character in raw_digest),
            f"invalid source SHA256 for {name}",
        )
        normalized[name] = raw_digest
    return normalized


def current_source_hashes() -> dict[str, str]:
    result = {}
    for name in EXPECTED_SOURCE_KEYS:
        path = ROOT / Path(name)
        require(path.is_file(), f"frozen source is missing: {name}")
        result[name] = sha256_file(path)
    return result


def expected_co_resident_memory(batch_size: int) -> dict[str, int]:
    """Independent integer reproduction of the worker's B3 memory formula."""

    require(batch_size in (2, 3), "held-out shard batch must be two or three")
    layers, hq, hkv, dim = 32, 32, 8, 128
    pages = math.ceil((CONTEXT + DECODE_STEPS) / PAGE)
    tail_pages = EXACT_TAIL // PAGE
    baseline = 2 * layers * batch_size * pages * PAGE * hkv * dim * 2
    codes = 2 * layers * batch_size * pages * PAGE * hkv * dim
    scales = 2 * layers * batch_size * pages * hkv * 2
    centers = layers * batch_size * (2 * hkv * dim + hq * dim) * 2
    exact = (
        2
        * layers
        * batch_size
        * (tail_pages + EXACT_PREFIX_PAGES)
        * PAGE
        * hkv
        * dim
        * 2
    )
    page_gauge = codes + scales + centers + exact
    projected = MODEL_FP16_RESIDENT_BYTES + baseline + page_gauge
    required = projected + MEMORY_RESERVE_BYTES
    return {
        "model_fp16_resident_bytes": MODEL_FP16_RESIDENT_BYTES,
        "flashinfer_fp16_cache_bytes": baseline,
        "page_gauge_cache_bytes": page_gauge,
        "int8_codes_bytes": codes,
        "fp16_scales_bytes": scales,
        "fp16_centers_and_output_center_bytes": centers,
        "fp16_exact_prefix_and_tail_bytes": exact,
        "projected_co_resident_bytes": projected,
        "fixed_workspace_allocator_reserve_bytes": MEMORY_RESERVE_BYTES,
        "required_free_bytes": required,
        "logical_pages_per_request": pages,
        "exact_tail_pages": tail_pages,
        "exact_prefix_pages": EXACT_PREFIX_PAGES,
    }


def percentile(values: list[float], probability: float) -> float:
    require(bool(values), "cannot take a percentile of an empty sequence")
    require(0.0 <= probability <= 1.0, "percentile lies outside [0,1]")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return float(ordered[lower])
    weight = coordinate - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label)
    converted = float(value)
    require(math.isfinite(converted), f"{label} is not finite")
    return converted


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-10)


def load_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    require(len(paths) == 5, "held-out protocol requires exactly five input files")
    resolved = [path.resolve() for path in paths]
    require(len(set(resolved)) == len(resolved), "input paths must be unique")
    loaded = []
    for path in resolved:
        require(path.is_file(), f"input does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(payload, dict), f"input is not a JSON object: {path}")
        loaded.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "payload": payload,
            }
        )
    return loaded


def common_signature(payload: dict[str, Any]) -> dict[str, Any]:
    token_source = payload["token_source"]
    tokenizer = token_source["tokenizer"]
    heldout = payload["heldout_protocol"]
    return {
        "model": payload["model"],
        "model_revision": payload["model_revision"],
        "model_config_sha256": payload["model_config_sha256"],
        "parameters": payload["parameters"],
        "num_hidden_layers": payload["num_hidden_layers"],
        "num_attention_heads": payload["num_attention_heads"],
        "num_key_value_heads": payload["num_key_value_heads"],
        "head_dim": payload["head_dim"],
        "context": payload["context"],
        "decode_steps": payload["decode_steps"],
        "exact_tail_tokens": payload["exact_tail_tokens"],
        "exact_prefix_pages": payload["exact_prefix_pages"],
        "exact_sink_pages": payload["exact_sink_pages"],
        "baseline_split_pages": payload["baseline_split_pages"],
        "candidate_split_pages": payload["candidate_split_pages"],
        "prefill_chunk_tokens": payload["prefill_chunk_tokens"],
        "center_restore": payload["center_restore"],
        "tail_attention": payload["tail_attention"],
        "old_value_scale_placement": payload["old_value_scale_placement"],
        "archive_sha256": token_source["archive_sha256"],
        "archive_member_sha256": token_source["archive_member_sha256"],
        "tokenizer_manifest_sha256": tokenizer["manifest_sha256"],
        "heldout_configuration_lock": heldout["configuration_lock"],
        "heldout_model_lock": heldout["model_lock"],
        "heldout_source_content_lock": heldout["source_content_lock"],
        "heldout_worker_threshold_lock": heldout["worker_threshold_lock"],
        "source_sha256": normalized_source_hashes(payload),
    }


def validate_group(
    record: dict[str, Any], expected_starts: tuple[int, ...]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = record["payload"]
    label = str(record["path"])
    require(payload.get("schema_version") == RAW_SCHEMA_VERSION, f"{label}: raw schema")
    require(payload.get("experiment") == EXPECTED_EXPERIMENT, f"{label}: experiment")
    require(payload.get("model") == EXPECTED_MODEL, f"{label}: model")
    require(
        payload.get("model_revision") == EXPECTED_MODEL_REVISION,
        f"{label}: model revision",
    )
    require(
        payload.get("model_config_sha256") == EXPECTED_MODEL_CONFIG_SHA256,
        f"{label}: model config",
    )
    require(
        payload.get("parameters") == EXPECTED_MODEL_PARAMETER_COUNT,
        f"{label}: parameter count",
    )
    require(payload.get("num_hidden_layers") == 32, f"{label}: layer count")
    require(payload.get("num_attention_heads") == 32, f"{label}: query heads")
    require(payload.get("num_key_value_heads") == 8, f"{label}: KV heads")
    require(payload.get("head_dim") == 128, f"{label}: head dimension")
    require(payload.get("context") == CONTEXT, f"{label}: context")
    require(payload.get("decode_steps") == DECODE_STEPS, f"{label}: decode steps")
    require(payload.get("exact_tail_tokens") == EXACT_TAIL, f"{label}: exact tail")
    require(
        payload.get("exact_prefix_pages") == EXACT_PREFIX_PAGES,
        f"{label}: exact prefix",
    )
    require(
        payload.get("exact_sink_pages") == EXACT_PREFIX_PAGES, f"{label}: prefix alias"
    )
    require(
        payload.get("prefill_chunk_tokens") == EXPECTED_PREFILL_CHUNK_TOKENS,
        f"{label}: prefill chunk",
    )
    require(
        payload.get("baseline_split_pages") == EXPECTED_SPLIT_PAGES,
        f"{label}: baseline split",
    )
    require(
        payload.get("candidate_split_pages") == EXPECTED_SPLIT_PAGES,
        f"{label}: candidate split",
    )
    require(
        payload.get("center_restore") == "attention_add", f"{label}: center restore"
    )
    require(
        payload.get("tail_attention") == EXPECTED_TAIL_ATTENTION, f"{label}: tail path"
    )
    require(
        payload.get("old_value_scale_placement") == EXPECTED_OLD_VALUE_SCALE_PLACEMENT,
        f"{label}: V-scale placement",
    )
    require(payload.get("batch_size") == len(expected_starts), f"{label}: batch size")

    heldout = payload.get("heldout_protocol")
    require(isinstance(heldout, dict), f"{label}: heldout policy")
    require(heldout.get("enabled") is True, f"{label}: heldout mode")
    require(heldout.get("name") == PROTOCOL_NAME, f"{label}: protocol name")
    require(heldout.get("passed") is True, f"{label}: heldout worker gate")
    require(
        heldout.get("preregistered_before_test_access") is True,
        f"{label}: preregistration marker",
    )
    require(heldout.get("selected_on_split") == "train", f"{label}: selection split")
    require(
        heldout.get("confirmation_split") == EXPECTED_SPLIT,
        f"{label}: confirmation split",
    )
    require(
        tuple(heldout.get("all_window_start_offsets", ())) == EXPECTED_STARTS,
        f"{label}: all frozen starts",
    )
    require(
        tuple(heldout.get("all_window_end_offsets_exclusive", ()))
        == tuple(start + WINDOW_CORPUS_TOKENS for start in EXPECTED_STARTS),
        f"{label}: all frozen ends",
    )
    require(
        tuple(heldout.get("this_shard_start_offsets", ())) == expected_starts,
        f"{label}: heldout shard starts",
    )
    require(
        heldout.get("this_shard_index") == EXPECTED_GROUPS.index(expected_starts),
        f"{label}: heldout shard index",
    )
    require(
        heldout.get("shard_batch_sizes") == [3, 3, 3, 3, 2],
        f"{label}: shard batch sizes",
    )
    require(heldout.get("window_stride") == WINDOW_STRIDE, f"{label}: heldout stride")
    require(
        heldout.get("window_tokens") == WINDOW_CORPUS_TOKENS,
        f"{label}: heldout window length",
    )
    require(
        heldout.get("last_window_end_offset_exclusive") == EXPECTED_LAST_END,
        f"{label}: final TEST endpoint",
    )
    require(
        heldout.get("test_tokens_remaining_after_last_window")
        == EXPECTED_TEST_TOKEN_COUNT - EXPECTED_LAST_END,
        f"{label}: remaining TEST tokens",
    )
    require(
        heldout.get("model_lock")
        == {
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_MODEL_REVISION,
            "config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
            "parameter_count": EXPECTED_MODEL_PARAMETER_COUNT,
        },
        f"{label}: heldout model lock",
    )
    require(
        heldout.get("configuration_lock")
        == {
            "context": CONTEXT,
            "decode_steps": DECODE_STEPS,
            "exact_tail_tokens": EXACT_TAIL,
            "exact_prefix_pages": EXACT_PREFIX_PAGES,
            "baseline_split_pages": EXPECTED_SPLIT_PAGES,
            "candidate_split_pages": EXPECTED_SPLIT_PAGES,
            "tail_attention": EXPECTED_TAIL_ATTENTION,
            "old_value_scale_placement": EXPECTED_OLD_VALUE_SCALE_PLACEMENT,
            "prefill_chunk_tokens": EXPECTED_PREFILL_CHUNK_TOKENS,
            "seed": EXPECTED_SEED,
        },
        f"{label}: heldout configuration lock",
    )
    require(
        heldout.get("worker_threshold_lock")
        == {
            "minimum_logits_cosine": PAGE_GAUGE_GATES["minimum_logits_cosine"],
            "minimum_top1_agreement": PAGE_GAUGE_GATES["minimum_top1_point"],
            "minimum_flashinfer_hf_cosine": FLASHINFER_HF_GATES[
                "minimum_logits_cosine"
            ],
        },
        f"{label}: worker threshold lock",
    )
    require(
        heldout.get("source_content_lock")
        == {
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_member": EXPECTED_MEMBER,
            "available_corpus_token_count": EXPECTED_TEST_TOKEN_COUNT,
            "tokenizer_manifest_sha256": EXPECTED_TOKENIZER_MANIFEST_SHA256,
            "archive_hash_cryptographically_locks_member_bytes": True,
        },
        f"{label}: source/content lock",
    )
    require(
        tuple(
            str(value).replace("\\", "/")
            for value in heldout.get("source_closure_paths", ())
        )
        == EXPECTED_SOURCE_KEYS,
        f"{label}: source closure paths",
    )

    token_source = payload.get("token_source")
    require(isinstance(token_source, dict), f"{label}: token source")
    require(token_source.get("kind") == "wikitext2", f"{label}: token kind")
    require(token_source.get("split") == EXPECTED_SPLIT, f"{label}: dataset split")
    require(
        token_source.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256,
        f"{label}: archive hash",
    )
    require(
        token_source.get("archive_sha256_verified") is True,
        f"{label}: archive verification",
    )
    require(
        token_source.get("archive_member") == EXPECTED_MEMBER,
        f"{label}: archive member",
    )
    require(
        token_source.get("corpus_windows_disjoint") is True, f"{label}: disjoint flag"
    )
    require(
        token_source.get("corpus_window_stride") == WINDOW_STRIDE, f"{label}: stride"
    )
    require(
        token_source.get("available_corpus_token_count") == EXPECTED_TEST_TOKEN_COUNT,
        f"{label}: tokenized TEST corpus length",
    )
    starts = tuple(token_source.get("corpus_window_start_offsets", ()))
    ends = tuple(token_source.get("corpus_window_end_offsets_exclusive", ()))
    require(starts == expected_starts, f"{label}: unexpected window starts {starts}")
    require(
        ends == tuple(start + WINDOW_CORPUS_TOKENS for start in starts),
        f"{label}: unexpected window ends",
    )
    require(
        token_source.get("corpus_tokens_per_request") == WINDOW_CORPUS_TOKENS,
        f"{label}: corpus tokens per request",
    )
    tokenizer = token_source.get("tokenizer")
    require(isinstance(tokenizer, dict), f"{label}: tokenizer")
    require(
        tokenizer.get("manifest_sha256") == EXPECTED_TOKENIZER_MANIFEST_SHA256,
        f"{label}: tokenizer manifest",
    )
    require(
        tokenizer.get("resolved_snapshot_revision") == EXPECTED_MODEL_REVISION,
        f"{label}: tokenizer revision",
    )
    member_sha = token_source.get("archive_member_sha256")
    require(
        isinstance(member_sha, str)
        and len(member_sha) == 64
        and all(character in "0123456789abcdef" for character in member_sha),
        f"{label}: archive member SHA256",
    )
    token_ids_sha = token_source.get("token_ids_sha256")
    require(
        isinstance(token_ids_sha, str)
        and len(token_ids_sha) == 64
        and all(character in "0123456789abcdef" for character in token_ids_sha),
        f"{label}: token IDs SHA256",
    )

    observed_sources = normalized_source_hashes(payload)
    require(tuple(observed_sources) == EXPECTED_SOURCE_KEYS, f"{label}: source closure")
    integrity = payload.get("source_integrity_gate")
    require(
        isinstance(integrity, dict)
        and integrity.get("passed") is True
        and integrity.get("hashed_before_model_or_dataset_access") is True
        and integrity.get("unchanged_through_result_finalization") is True,
        f"{label}: source integrity gate",
    )
    require(
        tuple(str(name).replace("\\", "/") for name in integrity.get("paths", ()))
        == EXPECTED_SOURCE_KEYS,
        f"{label}: source integrity paths",
    )
    require(
        observed_sources == current_source_hashes(),
        f"{label}: frozen source hashes differ from current files",
    )

    memory = heldout.get("co_resident_memory_gate")
    require(isinstance(memory, dict), f"{label}: memory gate")
    require(
        memory.get("enabled") is True and memory.get("passed") is True,
        f"{label}: co-resident memory gate",
    )
    require(
        memory.get("max_batch_size") == MAX_CORESIDENT_BATCH,
        f"{label}: maximum co-resident batch",
    )
    expected_memory = expected_co_resident_memory(len(expected_starts))
    for key in (
        "model_fp16_resident_bytes",
        "flashinfer_fp16_cache_bytes",
        "page_gauge_cache_bytes",
        "projected_co_resident_bytes",
        "fixed_workspace_allocator_reserve_bytes",
        "required_free_bytes",
        "logical_pages_per_request",
        "exact_tail_pages",
        "exact_prefix_pages",
    ):
        require(
            memory.get(key) == expected_memory[key], f"{label}: memory formula {key}"
        )
    components = memory.get("page_gauge_components", {})
    for key in (
        "int8_codes_bytes",
        "fp16_scales_bytes",
        "fp16_centers_and_output_center_bytes",
        "fp16_exact_prefix_and_tail_bytes",
    ):
        require(
            components.get(key) == expected_memory[key],
            f"{label}: memory component {key}",
        )
    observed_free = memory.get("observed_cuda_free_bytes_before_model_load")
    require(
        isinstance(observed_free, int)
        and observed_free >= expected_memory["required_free_bytes"],
        f"{label}: observed free memory",
    )
    require(
        memory.get("headroom_after_required_bytes")
        == observed_free - expected_memory["required_free_bytes"],
        f"{label}: memory headroom",
    )

    storage = payload.get("cache_storage")
    require(isinstance(storage, dict), f"{label}: cache storage")
    require(
        storage.get("flashinfer_fp16_bytes")
        == expected_memory["flashinfer_fp16_cache_bytes"],
        f"{label}: FP16 cache bytes",
    )
    require(
        storage.get("page_gauge_bytes") == expected_memory["page_gauge_cache_bytes"],
        f"{label}: PageGauge cache bytes",
    )
    require(
        storage.get("batch_size") == len(expected_starts), f"{label}: storage batch"
    )
    require(
        storage.get("page_gauge_exact_tail_pages_per_request") == EXACT_TAIL // PAGE,
        f"{label}: storage exact tail",
    )
    require(
        storage.get("page_gauge_exact_prefix_pages_per_request") == EXACT_PREFIX_PAGES,
        f"{label}: storage exact prefix",
    )
    require(
        storage.get("page_gauge_exact_sink_pages_per_request") == EXACT_PREFIX_PAGES,
        f"{label}: storage prefix alias",
    )
    require(
        storage.get("page_gauge_exact_storage_pages_per_request")
        == EXACT_PREFIX_PAGES + EXACT_TAIL // PAGE,
        f"{label}: exact storage pages",
    )

    correctness = payload.get("correctness")
    require(isinstance(correctness, dict), f"{label}: correctness object")
    require(correctness.get("teacher_forced") is True, f"{label}: teacher forcing")
    require(correctness.get("passed") is True, f"{label}: raw correctness failed")
    require(
        correctness.get("page_gauge_threshold_passed") is True,
        f"{label}: raw PageGauge threshold",
    )
    require(
        correctness.get("baseline_hf_conversion_threshold_passed") is True,
        f"{label}: raw baseline/HF threshold",
    )
    require(
        correctness.get("base_model_prefill_correctness_passed") is True,
        f"{label}: base correctness gate",
    )
    require(
        correctness.get("graph_replay", {}).get("enabled") is False,
        f"{label}: D1536 graph mode must be disabled",
    )
    require(
        correctness.get("graph_replay", {}).get("passed") is True,
        f"{label}: graph marker",
    )
    require(
        correctness.get("thresholds")
        == {
            "page_gauge_minimum_logits_cosine": PAGE_GAUGE_GATES[
                "minimum_logits_cosine"
            ],
            "page_gauge_minimum_top1_agreement": PAGE_GAUGE_GATES["minimum_top1_point"],
            "flashinfer_minimum_hf_logits_cosine": FLASHINFER_HF_GATES[
                "minimum_logits_cosine"
            ],
        },
        f"{label}: raw threshold lock",
    )
    for diagnostic_name in (
        "static_teacher_diagnostic",
        "direct_feedback_diagnostic",
        "token_step_graph_diagnostic",
        "generated_sequence_graph_diagnostic",
    ):
        diagnostic = payload.get(diagnostic_name)
        require(
            isinstance(diagnostic, dict) and diagnostic.get("enabled") is False,
            f"{label}: optional diagnostic {diagnostic_name}",
        )
    for required_key, passed_key in (
        ("static_teacher_diagnostic_required", "static_teacher_diagnostic_passed"),
        ("direct_feedback_diagnostic_required", "direct_feedback_diagnostic_passed"),
        ("token_step_graph_diagnostic_required", "token_step_graph_diagnostic_passed"),
        (
            "generated_sequence_graph_diagnostic_required",
            "generated_sequence_graph_diagnostic_passed",
        ),
    ):
        require(
            correctness.get(required_key) is False
            and correctness.get(passed_key) is None,
            f"{label}: nested diagnostic marker {required_key}",
        )
    collection = payload.get("logit_collection")
    require(
        isinstance(collection, dict)
        and collection.get("immediate_cpu_offload") is True
        and collection.get("timing_claim") is False,
        f"{label}: CPU logit collection protocol",
    )
    finalization = payload.get("runtime_page_finalization")
    require(
        isinstance(finalization, dict) and finalization.get("passed") is True,
        f"{label}: finalization audit",
    )
    expected_completed = DECODE_STEPS // PAGE
    require(
        finalization.get("completed_pages_per_request") == expected_completed,
        f"{label}: completed pages",
    )
    require(
        finalization.get("all_completed_code_slots_overwritten") is True,
        f"{label}: code canary",
    )
    require(
        finalization.get("all_completed_scales_finite_positive") is True,
        f"{label}: scale canary",
    )
    require(finalization.get("partial_page_tokens") == 0, f"{label}: partial page")
    require(
        finalization.get("future_capacity_pages_per_request") == expected_completed,
        f"{label}: future capacity",
    )
    require(
        finalization.get("future_pages_per_request") == expected_completed,
        f"{label}: initialized future pages",
    )
    require(
        finalization.get("future_physical_pages") == len(starts) * expected_completed,
        f"{label}: initialized future physical pages",
    )
    require(
        finalization.get("decode_steps_consuming_runtime_finalized_old_pages")
        == DECODE_STEPS - EXACT_TAIL,
        f"{label}: old-page consumption",
    )
    require(
        finalization.get("request_token_outputs_consuming_runtime_finalized_old_pages")
        == len(starts) * (DECODE_STEPS - EXACT_TAIL),
        f"{label}: old-page request outputs",
    )
    require(
        finalization.get("first_decode_step_consuming_runtime_finalized_old_page")
        == EXACT_TAIL,
        f"{label}: first old-page step",
    )
    require(
        finalization.get("runtime_finalized_pages_in_old_segment_at_endpoint") == 48,
        f"{label}: generated INT8 pages",
    )
    require(
        finalization.get("runtime_generated_int8_logical_page_count") == 48,
        f"{label}: generated INT8 recurrence count",
    )
    require(
        finalization.get("runtime_generated_int8_logical_page_range") == [1280, 1328],
        f"{label}: generated INT8 recurrence range",
    )
    require(
        finalization.get("generated_completed_logical_page_range") == [1280, 1376],
        f"{label}: generated completed range",
    )
    require(
        finalization.get("endpoint_prefix_logical_page_range") == [0, 3],
        f"{label}: endpoint prefix range",
    )
    require(
        finalization.get("endpoint_old_logical_page_range") == [3, 1328],
        f"{label}: endpoint old range",
    )
    require(
        finalization.get("endpoint_tail_logical_page_range") == [1328, 1376],
        f"{label}: endpoint tail range",
    )
    require(
        finalization.get("endpoint_prefix_pages_per_request") == 3,
        f"{label}: endpoint prefix pages",
    )
    require(
        finalization.get("endpoint_old_pages_per_request") == 1325,
        f"{label}: endpoint old pages",
    )
    require(
        finalization.get("endpoint_tail_pages_per_request") == 48,
        f"{label}: endpoint tail pages",
    )
    require(
        finalization.get("endpoint_partition_coverage_disjoint") is True,
        f"{label}: endpoint partition",
    )
    require(
        finalization.get("runtime_finalized_layer_pages_total")
        == len(starts) * 32 * expected_completed,
        f"{label}: layer-page count",
    )
    nested_finalization = correctness.get("runtime_page_finalization")
    require(isinstance(nested_finalization, dict), f"{label}: nested finalization")
    require(
        all(
            finalization.get(key) == value for key, value in nested_finalization.items()
        ),
        f"{label}: finalization copies differ",
    )

    prefix = payload.get("exact_prefix_canary")
    require(
        isinstance(prefix, dict) and prefix.get("passed") is True,
        f"{label}: exact-prefix canary",
    )
    require(
        prefix.get("exact_prefix_pages") == EXACT_PREFIX_PAGES,
        f"{label}: canary prefix pages",
    )
    require(
        prefix.get("exact_tail_pages") == EXACT_TAIL // PAGE,
        f"{label}: canary tail pages",
    )
    require(
        prefix.get("storage_pages_per_request")
        == EXACT_PREFIX_PAGES + EXACT_TAIL // PAGE,
        f"{label}: canary storage pages",
    )
    expected_mapping = [
        [
            request * (EXACT_PREFIX_PAGES + EXACT_TAIL // PAGE) + logical
            for logical in range(EXACT_PREFIX_PAGES)
        ]
        for request in range(len(starts))
    ]
    population = prefix.get("population", {})
    require(
        population.get("enabled") is True and population.get("passed") is True,
        f"{label}: prefix population",
    )
    require(
        population.get("key_bitwise_identical") is True
        and population.get("value_bitwise_identical") is True,
        f"{label}: prefix population values",
    )
    observed_prefix = population.get("observed", {})
    require(
        observed_prefix.get("mapping") == expected_mapping,
        f"{label}: prefix physical mapping",
    )
    require(
        population.get("expected_combined_sha256")
        == observed_prefix.get("combined_sha256"),
        f"{label}: prefix source digest",
    )
    immutable = prefix.get("immutability", {})
    require(
        immutable.get("passed") is True
        and immutable.get("fixed_slots_unchanged") is True,
        f"{label}: prefix immutability",
    )
    before = immutable.get("before", {})
    after = immutable.get("after", {})
    require(
        before.get("mapping") == after.get("mapping") == expected_mapping,
        f"{label}: immutable prefix mapping",
    )
    require(
        before.get("combined_sha256")
        == after.get("combined_sha256")
        == observed_prefix.get("combined_sha256"),
        f"{label}: immutable prefix digest",
    )
    page_tables = prefix.get("page_tables", {})
    require(
        page_tables.get("enabled") is True and page_tables.get("passed") is True,
        f"{label}: prefix page-table canary",
    )
    require(
        page_tables.get("expected_prefix_physical_pages_by_request")
        == page_tables.get("logical_to_exact_table_prefix")
        == page_tables.get("active_exact_plan_table_prefix")
        == expected_mapping,
        f"{label}: prefix page-table mapping",
    )
    require(
        page_tables.get("exact_page_table_updates") == 96,
        f"{label}: exact page-table updates",
    )
    require(
        page_tables.get("final_layout_signature") == [1328, 1376],
        f"{label}: final exact layout signature",
    )
    require(
        correctness.get("exact_prefix_canary") == prefix,
        f"{label}: prefix canary copies differ",
    )

    metric_protocol = correctness.get("quality_metric_protocol", {})
    windows = metric_protocol.get("window_metadata")
    require(
        isinstance(windows, list) and len(windows) == len(starts),
        f"{label}: window metadata",
    )
    by_request: dict[int, dict[str, Any]] = {}
    for window in windows:
        request = int(window.get("request", -1))
        require(
            request not in by_request and 0 <= request < len(starts),
            f"{label}: request metadata",
        )
        start = starts[request]
        require(window.get("dataset_split") == EXPECTED_SPLIT, f"{label}: window split")
        require(
            window.get("corpus_window_start_offset") == start, f"{label}: window start"
        )
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
            f"{label}: predicted start",
        )
        require(
            window.get("model_predicted_position_end_exclusive")
            == CONTEXT + DECODE_STEPS + 1,
            f"{label}: predicted end",
        )
        require(
            isinstance(window.get("cluster_unit_id"), str)
            and window["cluster_unit_id"],
            f"{label}: cluster ID",
        )
        by_request[request] = window
    return by_request, common_signature(payload)


def collect_comparison_clusters(
    record: dict[str, Any],
    windows: dict[int, dict[str, Any]],
    comparison_key: str,
) -> tuple[list[dict[str, Any]], float, dict[tuple[str, int], int]]:
    payload = record["payload"]
    label = f"{record['path']}:{comparison_key}"
    comparison = payload["correctness"].get(comparison_key)
    require(isinstance(comparison, dict), f"{label}: missing comparison")
    batch_size = int(payload["batch_size"])
    require(
        comparison.get("checked_decode_steps") == DECODE_STEPS,
        f"{label}: checked steps",
    )
    require(
        comparison.get("checked_request_steps") == batch_size * DECODE_STEPS,
        f"{label}: checked request steps",
    )
    minimum_cosine = finite_number(
        comparison.get("minimum_logits_cosine"), f"{label}: cosine"
    )
    cosine_rows = comparison.get("logits_cosine_by_step_request")
    top1_rows = comparison.get("top1_agreement_by_step_request")
    require(
        isinstance(cosine_rows, list) and len(cosine_rows) == DECODE_STEPS,
        f"{label}: cosine rows",
    )
    require(
        isinstance(top1_rows, list) and len(top1_rows) == DECODE_STEPS,
        f"{label}: top1 rows",
    )
    for step in range(DECODE_STEPS):
        require(len(cosine_rows[step]) == batch_size, f"{label}: cosine batch")
        require(len(top1_rows[step]) == batch_size, f"{label}: top1 batch")
        for value in cosine_rows[step]:
            cosine_value = finite_number(value, f"{label}: cosine value")
            require(-1.0 <= cosine_value <= 1.0, f"{label}: cosine range")
        require(
            all(isinstance(value, bool) for value in top1_rows[step]),
            f"{label}: top1 value",
        )
    flattened_cosines = [
        float(value) for step_values in cosine_rows for value in step_values
    ]
    require(
        close(min(flattened_cosines), minimum_cosine),
        f"{label}: minimum cosine summary",
    )
    flattened_top1 = [bool(value) for step_values in top1_rows for value in step_values]
    reported_top1 = finite_number(
        comparison.get("top1_agreement_fraction"),
        f"{label}: top1 agreement summary",
    )
    require(
        close(reported_top1, sum(flattened_top1) / len(flattened_top1)),
        f"{label}: top1 agreement summary",
    )
    worst = comparison.get("worst_logits_cosine_location")
    require(isinstance(worst, dict), f"{label}: worst cosine location")
    worst_step = worst.get("step")
    worst_request = worst.get("request")
    require(
        isinstance(worst_step, int)
        and isinstance(worst_request, int)
        and 0 <= worst_step < DECODE_STEPS
        and 0 <= worst_request < batch_size,
        f"{label}: worst cosine coordinates",
    )
    require(
        close(float(cosine_rows[worst_step][worst_request]), minimum_cosine)
        and close(
            finite_number(worst.get("value"), f"{label}: worst cosine value"),
            minimum_cosine,
        ),
        f"{label}: worst cosine value",
    )

    quality = comparison.get("distribution_quality")
    require(isinstance(quality, dict), f"{label}: distribution quality")
    require(
        quality.get("serialized_full_vocabulary_logits_or_probabilities") is False,
        f"{label}: full logits serialized",
    )
    require(
        quality.get("label_count") == batch_size * DECODE_STEPS, f"{label}: label count"
    )
    records = quality.get("per_token_metrics_step_major")
    require(
        isinstance(records, list) and len(records) == batch_size * DECODE_STEPS,
        f"{label}: token records",
    )
    rows_by_request: dict[int, list[dict[str, Any]]] = {
        request: [] for request in range(batch_size)
    }
    label_ids: dict[tuple[str, int], int] = {}
    seen = set()
    for row in records:
        request = int(row.get("request", -1))
        step = int(row.get("step", -1))
        require(
            0 <= request < batch_size and 0 <= step < DECODE_STEPS,
            f"{label}: row coordinates",
        )
        require((request, step) not in seen, f"{label}: duplicate row")
        seen.add((request, step))
        require(row.get("input_position") == CONTEXT + step, f"{label}: input position")
        require(
            row.get("predicted_position") == CONTEXT + 1 + step,
            f"{label}: predicted position",
        )
        reference_nll = finite_number(
            row.get("reference_nll_nats"), f"{label}: reference NLL"
        )
        candidate_nll = finite_number(
            row.get("candidate_nll_nats"), f"{label}: candidate NLL"
        )
        require(reference_nll >= 0.0 and candidate_nll >= 0.0, f"{label}: NLL range")
        delta = finite_number(row.get("nll_delta_nats"), f"{label}: NLL delta")
        require(
            close(candidate_nll - reference_nll, delta),
            f"{label}: inconsistent NLL delta",
        )
        token_ppl_ratio = finite_number(
            row.get("candidate_to_reference_token_perplexity_ratio"),
            f"{label}: token PPL ratio",
        )
        require(
            token_ppl_ratio > 0.0 and close(token_ppl_ratio, math.exp(delta)),
            f"{label}: inconsistent token PPL ratio",
        )
        kl = finite_number(row.get("forward_kl_nats"), f"{label}: KL")
        js = finite_number(row.get("jensen_shannon_nats"), f"{label}: JS")
        tv = finite_number(row.get("total_variation"), f"{label}: TV")
        require(
            kl >= 0.0 and js >= 0.0 and 0.0 <= tv <= 1.0, f"{label}: distribution range"
        )
        require(isinstance(row.get("top1_agreement"), bool), f"{label}: top1 record")
        require(
            row["top1_agreement"] == top1_rows[step][request], f"{label}: top1 mismatch"
        )
        reference_rank = row.get("reference_true_token_rank")
        candidate_rank = row.get("candidate_true_token_rank")
        require(
            isinstance(reference_rank, int) and reference_rank >= 1,
            f"{label}: reference rank",
        )
        require(
            isinstance(candidate_rank, int) and candidate_rank >= 1,
            f"{label}: candidate rank",
        )
        reference_top1 = row.get("reference_top1_token_id")
        candidate_top1 = row.get("candidate_top1_token_id")
        require(
            isinstance(reference_top1, int) and reference_top1 >= 0,
            f"{label}: reference top1 ID",
        )
        require(
            isinstance(candidate_top1, int) and candidate_top1 >= 0,
            f"{label}: candidate top1 ID",
        )
        require(
            row["top1_agreement"] == (reference_top1 == candidate_top1),
            f"{label}: top1 ID agreement",
        )
        true_token = row.get("true_token_id")
        require(isinstance(true_token, int) and true_token >= 0, f"{label}: true token")
        cluster_id = windows[request]["cluster_unit_id"]
        label_ids[(cluster_id, step)] = true_token
        rows_by_request[request].append(row)

    clusters = []
    for request in range(batch_size):
        rows = sorted(rows_by_request[request], key=lambda row: int(row["step"]))
        require(
            [int(row["step"]) for row in rows] == list(range(DECODE_STEPS)),
            f"{label}: incomplete request",
        )
        reference_nll_sum = math.fsum(float(row["reference_nll_nats"]) for row in rows)
        candidate_nll_sum = math.fsum(float(row["candidate_nll_nats"]) for row in rows)
        clusters.append(
            {
                "cluster_unit_id": windows[request]["cluster_unit_id"],
                "window": windows[request],
                "token_count": DECODE_STEPS,
                "reference_nll_sum": reference_nll_sum,
                "candidate_nll_sum": candidate_nll_sum,
                "forward_kl_sum": math.fsum(
                    float(row["forward_kl_nats"]) for row in rows
                ),
                "js_sum": math.fsum(float(row["jensen_shannon_nats"]) for row in rows),
                "tv_sum": math.fsum(float(row["total_variation"]) for row in rows),
                "top1_count": sum(bool(row["top1_agreement"]) for row in rows),
                "reference_rank_sum": sum(
                    int(row["reference_true_token_rank"]) for row in rows
                ),
                "candidate_rank_sum": sum(
                    int(row["candidate_true_token_rank"]) for row in rows
                ),
                "reference_true_top1_count": sum(
                    int(row["reference_true_token_rank"]) == 1 for row in rows
                ),
                "candidate_true_top1_count": sum(
                    int(row["candidate_true_token_rank"]) == 1 for row in rows
                ),
                "forward_kl_values": [float(row["forward_kl_nats"]) for row in rows],
                "js_values": [float(row["jensen_shannon_nats"]) for row in rows],
                "tv_values": [float(row["total_variation"]) for row in rows],
                "reference_rank_values": [
                    int(row["reference_true_token_rank"]) for row in rows
                ],
                "candidate_rank_values": [
                    int(row["candidate_true_token_rank"]) for row in rows
                ],
                "ppl_ratio": math.exp(
                    (candidate_nll_sum - reference_nll_sum) / DECODE_STEPS
                ),
            }
        )
    return clusters, minimum_cosine, label_ids


def point_summary(
    clusters: list[dict[str, Any]], minimum_cosine: float
) -> dict[str, Any]:
    count = sum(int(cluster["token_count"]) for cluster in clusters)
    reference_sum = math.fsum(
        float(cluster["reference_nll_sum"]) for cluster in clusters
    )
    candidate_sum = math.fsum(
        float(cluster["candidate_nll_sum"]) for cluster in clusters
    )
    kl_values = [
        value for cluster in clusters for value in cluster["forward_kl_values"]
    ]
    js_values = [value for cluster in clusters for value in cluster["js_values"]]
    tv_values = [value for cluster in clusters for value in cluster["tv_values"]]
    reference_rank_values = [
        value for cluster in clusters for value in cluster["reference_rank_values"]
    ]
    candidate_rank_values = [
        value for cluster in clusters for value in cluster["candidate_rank_values"]
    ]
    reference_ppl = math.exp(reference_sum / count)
    candidate_ppl = math.exp(candidate_sum / count)
    return {
        "cluster_count": len(clusters),
        "token_count": count,
        "minimum_logits_cosine": minimum_cosine,
        "reference_mean_nll_nats": reference_sum / count,
        "candidate_mean_nll_nats": candidate_sum / count,
        "reference_perplexity": reference_ppl,
        "candidate_perplexity": candidate_ppl,
        "candidate_to_reference_perplexity_ratio": math.exp(
            (candidate_sum - reference_sum) / count
        ),
        "candidate_minus_reference_perplexity": candidate_ppl - reference_ppl,
        "mean_forward_kl_nats": math.fsum(kl_values) / count,
        "forward_kl_p99_nats": percentile(kl_values, 0.99),
        "mean_jensen_shannon_nats": math.fsum(js_values) / count,
        "jensen_shannon_p99_nats": percentile(js_values, 0.99),
        "mean_total_variation": math.fsum(
            float(cluster["tv_sum"]) for cluster in clusters
        )
        / count,
        "total_variation_p99": percentile(tv_values, 0.99),
        "reference_mean_true_token_rank": math.fsum(reference_rank_values) / count,
        "candidate_mean_true_token_rank": math.fsum(candidate_rank_values) / count,
        "reference_true_token_rank_p99": percentile(reference_rank_values, 0.99),
        "candidate_true_token_rank_p99": percentile(candidate_rank_values, 0.99),
        "reference_true_token_top1_fraction": sum(
            int(cluster["reference_true_top1_count"]) for cluster in clusters
        )
        / count,
        "candidate_true_token_top1_fraction": sum(
            int(cluster["candidate_true_top1_count"]) for cluster in clusters
        )
        / count,
        "top1_agreement_fraction": sum(
            int(cluster["top1_count"]) for cluster in clusters
        )
        / count,
        "maximum_single_window_perplexity_ratio": max(
            float(cluster["ppl_ratio"]) for cluster in clusters
        ),
        "per_window_perplexity_ratio": {
            str(cluster["cluster_unit_id"]): float(cluster["ppl_ratio"])
            for cluster in clusters
        },
    }


def cluster_bootstrap(
    clusters: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    require(samples >= 100, "bootstrap requires at least 100 samples")
    require(len(clusters) == 14, "bootstrap requires exactly 14 clusters")
    generator = random.Random(seed)
    draws = {
        "ppl_ratio": [],
        "ppl_difference": [],
        "mean_forward_kl": [],
        "mean_js": [],
        "mean_tv": [],
        "mean_reference_rank": [],
        "mean_candidate_rank": [],
        "top1": [],
    }
    for _ in range(samples):
        selected = [clusters[generator.randrange(len(clusters))] for _ in clusters]
        count = sum(int(cluster["token_count"]) for cluster in selected)
        reference_sum = math.fsum(
            float(cluster["reference_nll_sum"]) for cluster in selected
        )
        candidate_sum = math.fsum(
            float(cluster["candidate_nll_sum"]) for cluster in selected
        )
        reference_ppl = math.exp(reference_sum / count)
        candidate_ppl = math.exp(candidate_sum / count)
        draws["ppl_ratio"].append(math.exp((candidate_sum - reference_sum) / count))
        draws["ppl_difference"].append(candidate_ppl - reference_ppl)
        draws["mean_forward_kl"].append(
            math.fsum(float(cluster["forward_kl_sum"]) for cluster in selected) / count
        )
        draws["mean_js"].append(
            math.fsum(float(cluster["js_sum"]) for cluster in selected) / count
        )
        draws["mean_tv"].append(
            math.fsum(float(cluster["tv_sum"]) for cluster in selected) / count
        )
        draws["mean_reference_rank"].append(
            math.fsum(float(cluster["reference_rank_sum"]) for cluster in selected)
            / count
        )
        draws["mean_candidate_rank"].append(
            math.fsum(float(cluster["candidate_rank_sum"]) for cluster in selected)
            / count
        )
        draws["top1"].append(
            sum(int(cluster["top1_count"]) for cluster in selected) / count
        )
    return {
        "method": "nonparametric percentile bootstrap over whole corpus windows",
        "cluster_count_per_draw": len(clusters),
        "samples": samples,
        "seed": seed,
        "one_sided_95": {
            "ppl_ratio_upper": percentile(draws["ppl_ratio"], 0.95),
            "ppl_difference_upper": percentile(draws["ppl_difference"], 0.95),
            "mean_forward_kl_upper": percentile(draws["mean_forward_kl"], 0.95),
            "mean_js_upper": percentile(draws["mean_js"], 0.95),
            "mean_tv_upper": percentile(draws["mean_tv"], 0.95),
            "mean_reference_rank_upper": percentile(draws["mean_reference_rank"], 0.95),
            "mean_candidate_rank_upper": percentile(draws["mean_candidate_rank"], 0.95),
            "top1_lower": percentile(draws["top1"], 0.05),
        },
        "two_sided_90": {
            "ppl_ratio": [
                percentile(draws["ppl_ratio"], 0.05),
                percentile(draws["ppl_ratio"], 0.95),
            ]
        },
    }


def gate_page_gauge(point: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    upper = bootstrap["one_sided_95"]
    values = {
        "minimum_logits_cosine": point["minimum_logits_cosine"],
        "minimum_top1_point": point["top1_agreement_fraction"],
        "minimum_top1_cluster_bootstrap_lower_95": upper["top1_lower"],
        "maximum_ppl_ratio_cluster_bootstrap_upper_95": upper["ppl_ratio_upper"],
        "maximum_ppl_increase_cluster_bootstrap_upper_95": upper[
            "ppl_difference_upper"
        ],
        "maximum_single_window_ppl_ratio": point[
            "maximum_single_window_perplexity_ratio"
        ],
        "maximum_mean_forward_kl_cluster_bootstrap_upper_95": upper[
            "mean_forward_kl_upper"
        ],
        "maximum_forward_kl_p99": point["forward_kl_p99_nats"],
        "maximum_mean_js_cluster_bootstrap_upper_95": upper["mean_js_upper"],
        "maximum_js_p99": point["jensen_shannon_p99_nats"],
    }
    checks = {}
    for name, threshold in PAGE_GAUGE_GATES.items():
        minimum = name.startswith("minimum_")
        passed = values[name] >= threshold if minimum else values[name] <= threshold
        checks[name] = {
            "value": values[name],
            "operator": ">=" if minimum else "<=",
            "threshold": threshold,
            "passed": passed,
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def gate_flashinfer_hf(
    point: dict[str, Any], bootstrap: dict[str, Any]
) -> dict[str, Any]:
    one_sided = bootstrap["one_sided_95"]
    ppl_interval = bootstrap["two_sided_90"]["ppl_ratio"]
    values = {
        "minimum_logits_cosine": point["minimum_logits_cosine"],
        "minimum_top1_point": point["top1_agreement_fraction"],
        "minimum_top1_cluster_bootstrap_lower_95": one_sided["top1_lower"],
        "minimum_ppl_ratio_cluster_bootstrap_lower_90": ppl_interval[0],
        "maximum_ppl_ratio_cluster_bootstrap_upper_90": ppl_interval[1],
        "maximum_mean_forward_kl_cluster_bootstrap_upper_95": one_sided[
            "mean_forward_kl_upper"
        ],
        "maximum_mean_js_cluster_bootstrap_upper_95": one_sided["mean_js_upper"],
    }
    checks = {}
    for name, threshold in FLASHINFER_HF_GATES.items():
        minimum = name.startswith("minimum_")
        passed = values[name] >= threshold if minimum else values[name] <= threshold
        checks[name] = {
            "value": values[name],
            "operator": ">=" if minimum else "<=",
            "threshold": threshold,
            "passed": passed,
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def aggregate_records(
    input_records: list[dict[str, Any]],
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    source_sha256_at_start = current_source_hashes()
    require(
        len(input_records) == 5, "held-out protocol requires exactly five input records"
    )
    require(
        bootstrap_samples == BOOTSTRAP_SAMPLES,
        "bootstrap sample count is frozen at 50000",
    )
    require(
        bootstrap_seed == BOOTSTRAP_SEED,
        f"bootstrap seed is frozen at {BOOTSTRAP_SEED}",
    )
    record_paths = [str(record.get("path")) for record in input_records]
    require(len(set(record_paths)) == 5, "input record paths must be unique")
    keyed = []
    for record in input_records:
        starts = tuple(
            record["payload"]
            .get("token_source", {})
            .get("corpus_window_start_offsets", ())
        )
        require(bool(starts), f"{record['path']}: missing starts")
        keyed.append((starts[0], record))
    keyed.sort(key=lambda item: item[0])
    require(
        tuple(item[0] for item in keyed)
        == tuple(group[0] for group in EXPECTED_GROUPS),
        "input shard offsets do not match the frozen grouping",
    )

    signatures = []
    pg_clusters: list[dict[str, Any]] = []
    hf_clusters: list[dict[str, Any]] = []
    pg_min_cosines = []
    hf_min_cosines = []
    seen_cluster_ids = set()
    all_windows = []
    input_manifest = []
    seen_token_ids_sha256 = set()
    for (_, record), expected_starts in zip(keyed, EXPECTED_GROUPS):
        windows, signature = validate_group(record, expected_starts)
        signatures.append(signature)
        current_pg, pg_cosine, pg_labels = collect_comparison_clusters(
            record, windows, "page_gauge_vs_flashinfer_fp16"
        )
        current_hf, hf_cosine, hf_labels = collect_comparison_clusters(
            record, windows, "flashinfer_fp16_vs_hf_sdpa_fp16"
        )
        require(pg_labels == hf_labels, f"{record['path']}: comparison labels differ")
        for cluster in current_pg:
            cluster_id = cluster["cluster_unit_id"]
            require(
                cluster_id not in seen_cluster_ids, f"duplicate cluster ID {cluster_id}"
            )
            seen_cluster_ids.add(cluster_id)
            all_windows.append(cluster["window"])
        pg_clusters.extend(current_pg)
        hf_clusters.extend(current_hf)
        pg_min_cosines.append(pg_cosine)
        hf_min_cosines.append(hf_cosine)
        token_ids_sha256 = record["payload"]["token_source"]["token_ids_sha256"]
        require(
            token_ids_sha256 not in seen_token_ids_sha256,
            f"{record['path']}: duplicate token-ID content",
        )
        seen_token_ids_sha256.add(token_ids_sha256)
        input_manifest.append(
            {
                "path": str(record["path"]),
                "sha256": record["sha256"],
                "batch_size": record["payload"]["batch_size"],
                "corpus_window_start_offsets": list(expected_starts),
                "token_ids_sha256": token_ids_sha256,
            }
        )

    require(
        all(signature == signatures[0] for signature in signatures[1:]),
        "input config/source/model signatures differ",
    )
    starts = sorted(int(window["corpus_window_start_offset"]) for window in all_windows)
    require(
        starts == list(EXPECTED_STARTS),
        "combined corpus windows differ from frozen starts",
    )
    intervals = sorted(
        (
            int(window["corpus_window_start_offset"]),
            int(window["corpus_window_end_offset_exclusive"]),
        )
        for window in all_windows
    )
    require(
        all(
            right_start >= left_end
            for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])
        ),
        "combined corpus windows overlap",
    )
    require(len(seen_cluster_ids) == 14, "protocol requires 14 unique cluster IDs")
    require(len(pg_clusters) == len(hf_clusters) == 14, "protocol requires 14 clusters")

    pg_point = point_summary(pg_clusters, min(pg_min_cosines))
    hf_point = point_summary(hf_clusters, min(hf_min_cosines))
    pg_bootstrap = cluster_bootstrap(pg_clusters, bootstrap_samples, bootstrap_seed)
    hf_bootstrap = cluster_bootstrap(
        hf_clusters, bootstrap_samples, FLASHINFER_BOOTSTRAP_SEED
    )
    pg_gate = gate_page_gauge(pg_point, pg_bootstrap)
    hf_gate = gate_flashinfer_hf(hf_point, hf_bootstrap)
    passed = bool(pg_gate["passed"] and hf_gate["passed"])
    failures = [
        f"page_gauge_vs_flashinfer_fp16.{name}"
        for name, check in pg_gate["checks"].items()
        if not check["passed"]
    ] + [
        f"flashinfer_fp16_vs_hf_sdpa_fp16.{name}"
        for name, check in hf_gate["checks"].items()
        if not check["passed"]
    ]
    source_sha256_at_end = current_source_hashes()
    require(
        source_sha256_at_end == source_sha256_at_start,
        "source closure changed during aggregation",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "page_gauge_wikitext2_test_heldout_quality_aggregate",
        "passed": passed,
        "failures": failures,
        "protocol": {
            "dataset": "WikiText-2 raw",
            "split": EXPECTED_SPLIT,
            "archive_member": EXPECTED_MEMBER,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "context": CONTEXT,
            "decode_steps": DECODE_STEPS,
            "exact_tail_tokens": EXACT_TAIL,
            "exact_prefix_pages": EXACT_PREFIX_PAGES,
            "exact_sink_pages": EXACT_PREFIX_PAGES,
            "baseline_split_pages": EXPECTED_SPLIT_PAGES,
            "candidate_split_pages": EXPECTED_SPLIT_PAGES,
            "tail_attention": EXPECTED_TAIL_ATTENTION,
            "old_value_scale_placement": EXPECTED_OLD_VALUE_SCALE_PLACEMENT,
            "window_stride": WINDOW_STRIDE,
            "window_start_offsets": list(EXPECTED_STARTS),
            "window_end_offsets_exclusive": [
                start + WINDOW_CORPUS_TOKENS for start in EXPECTED_STARTS
            ],
            "last_window_end_offset_exclusive": EXPECTED_LAST_END,
            "cluster_count": 14,
            "labels_per_cluster": DECODE_STEPS,
            "total_labeled_tokens": 14 * DECODE_STEPS,
            "bootstrap_unit": "one disjoint corpus window",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "page_gauge_bootstrap_seed": BOOTSTRAP_SEED,
            "flashinfer_bootstrap_seed": FLASHINFER_BOOTSTRAP_SEED,
            "co_resident_shard_batch_sizes": [3, 3, 3, 3, 2],
            "threshold_source_hash_matched_every_raw_input": True,
        },
        "frozen_common_signature": signatures[0],
        "inputs": input_manifest,
        "page_gauge_vs_flashinfer_fp16": {
            "point": pg_point,
            "cluster_bootstrap": pg_bootstrap,
            "gate": pg_gate,
        },
        "flashinfer_fp16_vs_hf_sdpa_fp16": {
            "point": hf_point,
            "cluster_bootstrap": hf_bootstrap,
            "gate": hf_gate,
        },
        "thresholds": {
            "page_gauge_vs_flashinfer_fp16": PAGE_GAUGE_GATES,
            "flashinfer_fp16_vs_hf_sdpa_fp16": FLASHINFER_HF_GATES,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_used_by_aggregator": False,
        },
        "invocation": {
            "argv": sys.argv,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "source_integrity_gate": {
            "passed": True,
            "hashed_before_input_validation": True,
            "unchanged_through_result_finalization": True,
            "paths": list(source_sha256_at_start),
        },
        "source_sha256": source_sha256_at_start,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = aggregate_records(load_inputs(args.inputs))
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "page_gauge_wikitext2_test_heldout_quality_aggregate",
            "passed": False,
            "validation_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "inputs": [
                {
                    "path": str(path.resolve()),
                    "exists": path.is_file(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for path in args.inputs
            ],
            "thresholds": {
                "page_gauge_vs_flashinfer_fp16": PAGE_GAUGE_GATES,
                "flashinfer_fp16_vs_hf_sdpa_fp16": FLASHINFER_HF_GATES,
            },
            "invocation": {
                "argv": sys.argv,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failures": result.get("failures"),
                "validation_error": result.get("validation_error"),
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    if not result["passed"]:
        raise SystemExit("held-out quality acceptance failed")


if __name__ == "__main__":
    main()
