#!/usr/bin/env python3
"""Fail-closed paired analysis for sustained device-dynamic graph serving."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import analyze_backend_exclusive as BASE


SCHEMA_VERSION = 3
WORKER_EXPERIMENT = "page_gauge_backend_exclusive_sustained_dynamic_graphs"
MANIFEST_EXPERIMENT = "page_gauge_sustained_dynamic_graph_publication_orchestration"
ANALYSIS_EXPERIMENT = "page_gauge_sustained_dynamic_graph_paired_analysis"
DYNAMIC_GRAPH_SCOPE = "decoder_layer_device_dynamic"
BASELINE = "flashinfer_fp16"
CANDIDATE = "page_gauge"
BACKENDS = (BASELINE, CANDIDATE)
SYMBOL_TO_BACKEND = {"A": BASELINE, "B": CANDIDATE}
WILLIAMS_SEQUENCES = ("ABBA", "BAAB")
MODES = ("cache_neutral", "cache_hot")
METRICS = ("wall_ms", "cuda_ms")
PAGE = 16
POLICY_PRESET = "exact_prefix_s3_d1024_t768"
POLICY_EXACT_PREFIX_PAGES = 3
POLICY_DECODE_STEPS = 1024
POLICY_EXACT_TAIL_TOKENS = 768
POLICY_GENERATED_INT8_PAGES = 16
EXPECTED_LEGACY_VENDOR_HEADER_SHA256 = (
    "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"
)
EXPECTED_HETEROGENEOUS_VENDOR_HEADER_SHA256 = (
    "2a3f3018576d04697477e3030507379ccc2d126036255ed81a238b63e9572e56"
)
REQUIRED_WORKER_SOURCES = (
    "diagnostics/benchmark_sustained_dynamic_graphs.py",
    "diagnostics/benchmark_backend_exclusive.py",
    "diagnostics/benchmark_full_sequence_graph.py",
    "diagnostics/benchmark_model_prefill_correctness.py",
    "diagnostics/benchmark_token_step_graphs.py",
    "scripts/benchmark_page_gauge_transformer.py",
    "scripts/benchmark_flashinfer_page_affine_int8.py",
    "scripts/page_gauge_heterogeneous_fa2.py",
    "scripts/prepare_flashinfer_page_gauge.py",
    "scripts/prepare_flashinfer_page_gauge_heterogeneous.py",
    "scripts/benchmark_page_gauge_overheads.py",
    "scripts/benchmark_e2e_transformer.py",
    "scripts/page_gauge_runtime.py",
    "tests/page_gauge_append_extension.cu",
    "patches/flashinfer-0.6.17-page-gauge-int8.patch",
    "patches/flashinfer-0.6.17-page-gauge-heterogeneous.patch",
    "docs/decoder_layer_cuda_graphs.md",
)
REQUIRED_ORCHESTRATION_SOURCES = (
    "diagnostics/benchmark_sustained_dynamic_graphs.py",
    "diagnostics/analyze_sustained_dynamic_graphs.py",
    "diagnostics/orchestrate_sustained_dynamic_graphs.py",
    "diagnostics/orchestrate_backend_exclusive.py",
    "diagnostics/analyze_backend_exclusive.py",
    "docs/exact_prefix_s3_williams.md",
)
ProtocolError = BASE.ProtocolError
sha256_file = BASE.sha256_file


def build_williams_schedule(
    seeds: Sequence[int],
    pairs_per_seed: int,
    base_token_offset: int,
    seed_token_offset_stride: int,
) -> list[dict[str, Any]]:
    """Canonical adjacent-pair ABBA/BAAB schedule without import cycles."""

    if pairs_per_seed <= 0 or pairs_per_seed % 2:
        raise ProtocolError("pairs_per_seed must be positive and even")
    schedule: list[dict[str, Any]] = []
    chronological_index = 0
    for seed_index, seed in enumerate(seeds):
        token_offset = base_token_offset + seed_index * seed_token_offset_stride
        for pair_index in range(pairs_per_seed):
            quartet_index = pair_index // 2
            sequence = WILLIAMS_SEQUENCES[
                (seed_index + quartet_index) % len(WILLIAMS_SEQUENCES)
            ]
            within_quartet = pair_index % 2
            symbols = sequence[2 * within_quartet : 2 * within_quartet + 2]
            pair_id = f"seed{seed_index:03d}_pair{pair_index:03d}"
            for slot, symbol in enumerate(symbols):
                block_id = (
                    f"block{chronological_index:04d}_seed{seed_index:03d}_"
                    f"pair{pair_index:03d}_slot{slot}_{symbol}"
                )
                schedule.append(
                    {
                        "block_id": block_id,
                        "chronological_index": chronological_index,
                        "seed": int(seed),
                        "seed_index": seed_index,
                        "token_offset": token_offset,
                        "pair_id": pair_id,
                        "pair_index": pair_index,
                        "quartet_index": quartet_index,
                        "williams_sequence": sequence,
                        "pair_order": symbols,
                        "slot": slot,
                        "symbol": symbol,
                        "backend": SYMBOL_TO_BACKEND[symbol],
                    }
                )
                chronological_index += 1
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=5090)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ProtocolError(f"{label} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProtocolError(f"{label} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolError(f"{label} must include a UTC offset")
    return parsed


def validate_candidate_module_provenance(
    attention: Mapping[str, Any], tail_attention: str
) -> None:
    module_hashes = _mapping(
        attention.get("custom_module_source_hashes"),
        "candidate custom module hashes",
    )
    required = {
        "header_sha256",
        "expected_header_sha256",
        "header_matches_expected",
        "variant_sha256",
        "module_source_sha256",
    }
    if set(module_hashes) != required:
        raise ProtocolError("candidate custom module source-hash schema is incomplete")
    header = _sha256(module_hashes.get("header_sha256"), "module header")
    expected_header = _sha256(
        module_hashes.get("expected_header_sha256"), "expected module header"
    )
    variant = _sha256(module_hashes.get("variant_sha256"), "module variant")
    module_source = _sha256(
        module_hashes.get("module_source_sha256"), "module source"
    )
    del variant
    authoritative_header = (
        EXPECTED_HETEROGENEOUS_VENDOR_HEADER_SHA256
        if tail_attention == "heterogeneous_fa2"
        else EXPECTED_LEGACY_VENDOR_HEADER_SHA256
    )
    if (
        module_hashes.get("header_matches_expected") is not True
        or header != expected_header
        or expected_header != authoritative_header
    ):
        raise ProtocolError("candidate vendor header does not match its selected factory")
    uri = attention.get("custom_module_uri")
    expected_prefix = (
        "page_gauge_heterogeneous_int8_fp16_fa2_sm120_v1_"
        if tail_attention == "heterogeneous_fa2"
        else "page_gauge_int8_fa2_v4_"
    )
    if uri != expected_prefix + module_source[:16]:
        raise ProtocolError("candidate JIT URI is not content-qualified by module source")


def quality_window_identity(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = _mapping(result.get("configuration"), "worker configuration")
    source = _mapping(result.get("token_source"), "worker token source")
    windows = result.get("quality_windows")
    batch_size = config.get("batch_size")
    context = config.get("context")
    decode_steps = config.get("decode_steps")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(context, int)
        or isinstance(context, bool)
        or not isinstance(decode_steps, int)
        or isinstance(decode_steps, bool)
        or not isinstance(windows, list)
        or len(windows) != batch_size
    ):
        raise ProtocolError("quality-window dimensions are invalid")
    starts = source.get("corpus_window_start_offsets")
    ends = source.get("corpus_window_end_offsets_exclusive")
    if source.get("kind") == "wikitext2":
        fixture_length = context + decode_steps
        request_stride = int(config.get("token_stride") or fixture_length)
        expected_starts = [
            int(config["token_offset"]) + request * request_stride
            for request in range(batch_size)
        ]
        expected_ends = [start + fixture_length for start in expected_starts]
        if (
            not isinstance(starts, list)
            or not isinstance(ends, list)
            or len(starts) != batch_size
            or len(ends) != batch_size
            or starts != expected_starts
            or ends != expected_ends
            or source.get("corpus_window_stride") != request_stride
            or source.get("corpus_tokens_per_request") != fixture_length
            or source.get("corpus_windows_disjoint") is not True
        ):
            raise ProtocolError("WikiText provenance has incorrect request window bounds")
        member_sha256 = _sha256(
            source.get("archive_member_sha256"), "WikiText archive member"
        )
    else:
        if starts is not None or ends is not None:
            raise ProtocolError("non-WikiText source unexpectedly reports corpus bounds")
        member_sha256 = None

    normalized: list[dict[str, Any]] = []
    cluster_ids: list[str] = []
    for request, raw_record in enumerate(windows):
        record = _mapping(raw_record, f"quality_windows[{request}]")
        cluster_id = record.get("cluster_unit_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ProtocolError("quality window has no cluster-unit identity")
        if (
            record.get("request") != request
            or record.get("model_predicted_position_start") != context + 1
            or record.get("model_predicted_position_end_exclusive")
            != context + decode_steps + 1
        ):
            raise ProtocolError("quality window model-position bounds are inconsistent")
        if starts is not None and ends is not None:
            start = starts[request]
            end = ends[request]
            if (
                record.get("corpus_window_start_offset") != start
                or record.get("corpus_window_end_offset_exclusive") != end
                or record.get("corpus_label_start_offset") != start + context
                or record.get("corpus_label_end_offset_exclusive")
                != start + context + decode_steps
                or record.get("archive_member") != source.get("archive_member")
                or record.get("archive_member_sha256") != member_sha256
                or record.get("dataset_split") != source.get("split")
            ):
                raise ProtocolError("WikiText quality-window identity/bounds disagree")
            if end - start < context + decode_steps:
                raise ProtocolError("WikiText quality window is shorter than C+D")
        else:
            for key in (
                "corpus_window_start_offset",
                "corpus_window_end_offset_exclusive",
                "corpus_label_start_offset",
                "corpus_label_end_offset_exclusive",
            ):
                if record.get(key) is not None:
                    raise ProtocolError("random quality window has corpus offsets")
        normalized.append(dict(record))
        cluster_ids.append(cluster_id)
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ProtocolError("quality-window cluster IDs are not unique within a worker")
    return normalized


def expected_orchestration_environment(
    config_sha256: str,
    block: Mapping[str, Any],
    orchestration_session_id: str | None = None,
) -> dict[str, str]:
    expected = {
        "PAGE_GAUGE_PUBLICATION_PROTOCOL": "sustained_dynamic_graph_v2",
        "PAGE_GAUGE_PUBLICATION_CONFIG_SHA256": config_sha256,
        "PAGE_GAUGE_PUBLICATION_BLOCK_ID": str(block["block_id"]),
        "PAGE_GAUGE_PUBLICATION_PAIR_ID": str(block["pair_id"]),
        "PAGE_GAUGE_PUBLICATION_PAIR_ORDER": str(block["pair_order"]),
        "PAGE_GAUGE_PUBLICATION_SLOT": str(block["slot"]),
        "PAGE_GAUGE_PUBLICATION_BACKEND": str(block["backend"]),
    }
    if orchestration_session_id is not None:
        expected["PAGE_GAUGE_PUBLICATION_SESSION_ID"] = orchestration_session_id
    return expected


def expected_worker_configuration(
    manifest_config: Mapping[str, Any], block: Mapping[str, Any]
) -> dict[str, Any]:
    profile = _mapping(manifest_config.get("profile"), "manifest profile")
    candidate = block["backend"] == CANDIDATE
    exact_prefix_pages = (
        int(manifest_config["exact_prefix_pages"]) if candidate else 0
    )
    return {
        "backend": block["backend"],
        "model": manifest_config["model"],
        "batch_size": manifest_config["batch_size"],
        "context": manifest_config["context"],
        "decode_steps": manifest_config["decode_steps"],
        "exact_tail_tokens": manifest_config["exact_tail"],
        "exact_sink_pages": exact_prefix_pages,
        "exact_prefix_pages": exact_prefix_pages,
        "prefill_chunk_tokens": manifest_config["prefill_chunk_tokens"],
        "baseline_split_pages": manifest_config["baseline_split_pages"],
        "candidate_split_pages": manifest_config["candidate_split_pages"],
        "tail_attention": manifest_config["tail_attention"],
        "old_value_scale_placement": manifest_config[
            "old_value_scale_placement"
        ],
        "trajectory_mode": manifest_config["trajectory_mode"],
        "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
        "token_source": manifest_config["token_source"],
        "wikitext_member": manifest_config["wikitext_member"],
        "wikitext_archive_sha256": manifest_config["wikitext_archive_sha256"],
        "seed": block["seed"],
        "token_offset": block["token_offset"],
        "token_stride": manifest_config["token_stride"],
        "capture_warmups": manifest_config["capture_warmups"],
        "maximum_graph_banks": manifest_config["maximum_graph_banks"],
        "warmups": profile["warmups"],
        "repeats": profile["repeats"],
        "cache_scrub_mib": manifest_config["cache_scrub_mib"],
        "min_logits_cosine": manifest_config["min_logits_cosine"],
        "min_top1_agreement": manifest_config["min_top1_agreement"],
        "quality_diagnostics_top_k": manifest_config[
            "quality_diagnostics_top_k"
        ],
        "quality_diagnostics_top_vocab": manifest_config[
            "quality_diagnostics_top_vocab"
        ],
    }


def validate_manifest_policy(config: Mapping[str, Any]) -> None:
    expected = {
        "policy_preset": POLICY_PRESET,
        "model": "mistralai/Mistral-7B-v0.3",
        "batch_size": 4,
        "context": 20480,
        "decode_steps": POLICY_DECODE_STEPS,
        "exact_tail": POLICY_EXACT_TAIL_TOKENS,
        "exact_sink_pages": POLICY_EXACT_PREFIX_PAGES,
        "exact_prefix_pages": POLICY_EXACT_PREFIX_PAGES,
        "baseline_split_pages": 256,
        "candidate_split_pages": 256,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
        "trajectory_mode": "frozen_hf_teacher_forced",
        "token_source": "wikitext2",
        "wikitext_member": "wikitext-2-raw/wiki.train.raw",
        "base_token_offset": 100000,
        "token_stride": 21984,
        "min_logits_cosine": 0.995,
        "min_top1_agreement": 0.99,
        "quality_diagnostics_top_k": 0,
        "quality_diagnostics_top_vocab": 8,
    }
    mismatches = [
        f"{name}={config.get(name)!r} (expected {value!r})"
        for name, value in expected.items()
        if config.get(name) != value
    ]
    if mismatches:
        raise ProtocolError(
            f"manifest does not freeze {POLICY_PRESET}: " + "; ".join(mismatches)
        )
    backend_policy = _mapping(
        config.get("backend_conditioned_exact_prefix_policy"),
        "manifest backend-conditioned exact-prefix policy",
    )
    expected_backend_policy = {
        "flashinfer_fp16": 0,
        "page_gauge": POLICY_EXACT_PREFIX_PAGES,
        "candidate_exact_prefix_pages": POLICY_EXACT_PREFIX_PAGES,
        "candidate_exact_sink_pages_legacy_alias": POLICY_EXACT_PREFIX_PAGES,
    }
    for name, value in expected_backend_policy.items():
        if backend_policy.get(name) != value:
            raise ProtocolError(
                f"manifest backend-conditioned exact-prefix policy.{name} is invalid"
            )
    if not isinstance(backend_policy.get("reason"), str) or not backend_policy[
        "reason"
    ]:
        raise ProtocolError("manifest exact-prefix backend asymmetry is unexplained")
    recurrence = _mapping(
        config.get("recurrence_attestation"), "manifest recurrence attestation"
    )
    expected_recurrence = {
        "page_tokens": PAGE,
        "generated_pages": POLICY_DECODE_STEPS // PAGE,
        "exact_tail_pages": POLICY_EXACT_TAIL_TOKENS // PAGE,
        "runtime_generated_pages_consumed_as_int8": POLICY_GENERATED_INT8_PAGES,
        "required_runtime_generated_pages_consumed_as_int8": (
            POLICY_GENERATED_INT8_PAGES
        ),
        "passed": True,
    }
    if dict(recurrence) != expected_recurrence:
        raise ProtocolError("manifest recurrence attestation is not exact D1024/T768")


def worker_pair_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    pairing = _mapping(result.get("pairing"), "worker pairing")
    configuration = _mapping(
        pairing.get("configuration"), "worker pairing.configuration"
    )
    if configuration.get("cuda_graph_scope") != DYNAMIC_GRAPH_SCOPE:
        raise ProtocolError("worker pairing key is not device-dynamic decoder-layer scope")
    backend = result.get("backend")
    expected_prefix_pages = (
        POLICY_EXACT_PREFIX_PAGES if backend == CANDIDATE else 0
    )
    if (
        backend not in BACKENDS
        or configuration.get("exact_sink_pages") != expected_prefix_pages
        or configuration.get("exact_prefix_pages") != expected_prefix_pages
    ):
        raise ProtocolError(
            "worker pairing key does not attest the backend-conditioned exact-prefix policy"
        )
    if (
        pairing.get("backend_excluded_from_key") is not True
        or pairing.get("cuda_graph_scope_must_match_exactly") is not True
        or pairing.get("fixed_window_decoder_layer_scope_is_incompatible") is not True
    ):
        raise ProtocolError("worker pairing-key scope declarations are incomplete")
    expected_key = canonical_json_sha256(configuration)
    observed_key = _sha256(
        pairing.get("pairing_key_sha256"), "worker pairing key"
    )
    if observed_key != expected_key:
        raise ProtocolError("worker pairing key does not hash its configuration")
    source_hashes = _mapping(result.get("source_sha256"), "worker source_sha256")
    if not source_hashes:
        raise ProtocolError("worker source hash map is empty")
    checked_sources = {
        str(name): _sha256(digest, f"source_sha256.{name}")
        for name, digest in source_hashes.items()
    }
    missing_sources = sorted(set(REQUIRED_WORKER_SOURCES) - set(checked_sources))
    if missing_sources:
        raise ProtocolError(
            "worker source closure is incomplete: " + ", ".join(missing_sources)
        )
    environment = _mapping(result.get("environment"), "worker environment")
    gpu = environment.get("gpu")
    if not isinstance(gpu, str) or "GeForce RTX 5090" not in gpu:
        raise ProtocolError(f"publication worker is not GeForce RTX 5090: {gpu!r}")
    if environment.get("compute_capability") != [12, 0]:
        raise ProtocolError("publication worker is not SM120")
    flashinfer_abi = _mapping(
        environment.get("flashinfer_abi"), "worker FlashInfer ABI provenance"
    )
    if flashinfer_abi.get("source_sha256") != configuration.get(
        "flashinfer_abi_source_sha256"
    ):
        raise ProtocolError("environment and pairing FlashInfer ABI hashes differ")
    environment_identity = {
        key: environment.get(key)
        for key in ("gpu", "compute_capability", "torch", "torch_cuda", "flashinfer")
    }
    if any(value is None for value in environment_identity.values()):
        raise ProtocolError("worker environment identity is incomplete")
    environment_identity["flashinfer_abi_source_sha256"] = flashinfer_abi.get(
        "source_sha256"
    )
    correctness = _mapping(result.get("correctness"), "worker correctness")
    correctness_hashes = _mapping(
        correctness.get("hashes"), "worker correctness hashes"
    )
    quality_reference_sha256 = {
        "hf_logits_sha256": _sha256(
            correctness_hashes.get("hf_logits_sha256"), "worker HF logits"
        ),
        "hf_generated_tokens_sha256": _sha256(
            correctness_hashes.get("hf_generated_tokens_sha256"),
            "worker HF generated tokens",
        ),
    }
    return {
        "pairing_key_sha256": observed_key,
        "configuration": dict(configuration),
        "source_sha256": dict(sorted(checked_sources.items())),
        "environment_identity": environment_identity,
        "quality_windows": quality_window_identity(result),
        "quality_reference_sha256": quality_reference_sha256,
    }


def cross_backend_pair_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the honest FI-S0/PageGauge-S3 representation asymmetry."""

    configuration = dict(
        _mapping(identity.get("configuration"), "worker identity configuration")
    )
    configuration.pop("exact_sink_pages", None)
    configuration.pop("exact_prefix_pages", None)
    configuration["candidate_exact_sink_pages"] = POLICY_EXACT_PREFIX_PAGES
    configuration["candidate_exact_prefix_pages"] = POLICY_EXACT_PREFIX_PAGES
    return {
        "configuration": configuration,
        "source_sha256": identity.get("source_sha256"),
        "environment_identity": identity.get("environment_identity"),
        "quality_windows": identity.get("quality_windows"),
        "quality_reference_sha256": identity.get("quality_reference_sha256"),
    }


def worker_generated_token_hash(result: Mapping[str, Any]) -> str:
    correctness = _mapping(result.get("correctness"), "worker correctness")
    hashes = _mapping(correctness.get("hashes"), "worker correctness.hashes")
    return _sha256(
        hashes.get("graph_generated_tokens_sha256"),
        "worker graph generated tokens",
    )


def global_stable_worker_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    pair_identity = worker_pair_identity(result)
    configuration = pair_identity["configuration"]
    token_source = _mapping(result.get("token_source"), "worker token source")
    model = _mapping(result.get("model_provenance"), "worker model provenance")
    return {
        "model": configuration.get("model"),
        "model_revision": configuration.get("model_revision"),
        "model_config_sha256": configuration.get("model_config_sha256"),
        "sampled_model_parameters_sha256": configuration.get(
            "sampled_model_parameters_sha256"
        ),
        "flashinfer_abi_source_sha256": configuration.get(
            "flashinfer_abi_source_sha256"
        ),
        "environment_identity": pair_identity["environment_identity"],
        "model_parameter_count": model.get("parameter_count"),
        "token_artifact": {
            key: token_source.get(key)
            for key in (
                "kind",
                "dataset",
                "split",
                "archive_sha256",
                "archive_member",
                "archive_member_sha256",
                "tokenizer",
                "add_special_tokens",
                "bos_prepended_per_request",
                "bos_token_id",
            )
        },
    }


def validate_equivalence_bundle(
    bundle: Mapping[str, Any],
    *,
    label: str,
    context: int,
    decode_steps: int,
    batch_size: int,
    exact_ring_pages: int,
) -> None:
    logits = _mapping(bundle.get("logits"), f"{label}.logits")
    tokens = _mapping(bundle.get("generated_argmax_tokens"), f"{label}.tokens")
    cache = _mapping(bundle.get("full_mutated_cache_range"), f"{label}.cache")
    boundaries = _mapping(
        bundle.get("every_page_close_cache_digest"), f"{label}.boundaries"
    )
    metadata = _mapping(bundle.get("final_serving_metadata"), f"{label}.metadata")
    if (
        logits.get("bitwise_identical") is not True
        or logits.get("checked_steps") != decode_steps
        or logits.get("checked_request_steps") != decode_steps * batch_size
        or tokens.get("bitwise_identical") is not True
        or tokens.get("checked_steps") != decode_steps
        or tokens.get("checked_request_steps") != decode_steps * batch_size
        or cache.get("passed") is not True
        or cache.get("full_pages_checked")
        != batch_size * (decode_steps // PAGE)
        or cache.get("exact_ring_pages_checked")
        != batch_size * exact_ring_pages
        or cache.get("device_position_bitwise_identical") is not True
        or boundaries.get("passed") is not True
        or boundaries.get("checked_page_closes") != decode_steps // PAGE
        or metadata.get("passed") is not True
    ):
        raise ProtocolError(f"{label} nested equivalence gate failed")
    boundary_records = boundaries.get("records")
    expected_pages = list(
        range(context // PAGE, (context + decode_steps) // PAGE)
    )
    if (
        not isinstance(boundary_records, list)
        or [record.get("logical_page") for record in boundary_records]
        != expected_pages
        or any(record.get("bitwise_identical") is not True for record in boundary_records)
    ):
        raise ProtocolError(f"{label} page-close records are incomplete")
    metadata_wrappers = _mapping(metadata.get("wrappers"), f"{label}.metadata wrappers")
    if not metadata_wrappers or any(
        _mapping(record, f"{label}.metadata wrapper").get("bitwise_identical")
        is not True
        for record in metadata_wrappers.values()
    ):
        raise ProtocolError(f"{label} planner/page-table metadata differs")


def validate_adjacent_canary_gates(
    records: Any,
    *,
    context: int,
    decode_steps: int,
    exact_prefix_pages: int,
) -> None:
    gates = _mapping(records, "adjacent page canary gates")
    if set(gates) != {"eager", "graph_run_1", "graph_run_2_restored"}:
        raise ProtocolError("adjacent page canary run coverage is incomplete")
    expected_pages = {
        "preceding": context // PAGE - 1,
        "following": (context + decode_steps) // PAGE,
    }
    if exact_prefix_pages:
        expected_pages["exact_prefix"] = None
    for name, raw_gate in gates.items():
        gate = _mapping(raw_gate, f"adjacent canary {name}")
        if gate.get("passed") is not True or gate.get("logical_pages") != expected_pages:
            raise ProtocolError(f"adjacent page canary {name} failed")
        expected_hashes = _mapping(
            gate.get("expected_combined_sha256"), f"adjacent canary {name} expected"
        )
        observed_hashes = _mapping(
            gate.get("observed_combined_sha256"), f"adjacent canary {name} observed"
        )
        if set(expected_hashes) != set(expected_pages) or expected_hashes != observed_hashes:
            raise ProtocolError(f"adjacent page canary {name} digest differs")
        for page_name, digest in expected_hashes.items():
            _sha256(digest, f"adjacent canary {name}.{page_name}")


def validate_exact_prefix_canary_record(
    record: Any,
    *,
    label: str,
    batch_size: int,
    exact_tail_pages: int,
    exact_prefix_pages: int,
) -> None:
    canary = _mapping(record, label)
    enabled = exact_prefix_pages > 0
    storage_pages = exact_prefix_pages + exact_tail_pages
    expected_physical_pages = [
        request * storage_pages + page
        for request in range(batch_size)
        for page in range(exact_prefix_pages)
    ]
    if (
        canary.get("enabled") is not enabled
        or canary.get("passed") is not True
        or canary.get("exact_prefix_pages") != exact_prefix_pages
        or canary.get("logical_pages") != list(range(exact_prefix_pages))
        or canary.get("physical_pages") != expected_physical_pages
        or canary.get("physical_page_count") != len(expected_physical_pages)
    ):
        raise ProtocolError(f"{label} exact-prefix geometry/canary failed")
    expected_tensors = _mapping(
        canary.get("expected_tensor_sha256"), f"{label} expected tensor hashes"
    )
    observed_tensors = _mapping(
        canary.get("observed_tensor_sha256"), f"{label} observed tensor hashes"
    )
    if enabled:
        if (
            set(expected_tensors) != {"exact_key", "exact_value"}
            or expected_tensors != observed_tensors
            or canary.get("expected_combined_sha256")
            != canary.get("observed_combined_sha256")
        ):
            raise ProtocolError(f"{label} exact-prefix tensor digest changed")
        _sha256(
            canary.get("expected_combined_sha256"),
            f"{label} exact-prefix combined digest",
        )
        for name, digest in expected_tensors.items():
            _sha256(digest, f"{label} exact-prefix {name}")
    elif (
        expected_tensors
        or observed_tensors
        or canary.get("expected_combined_sha256") is not None
        or canary.get("observed_combined_sha256") is not None
    ):
        raise ProtocolError(f"{label} disabled exact-prefix canary is not empty")


def validate_exact_prefix_canary_aggregate(
    aggregate: Any,
    *,
    label: str,
    expected_count: int,
    batch_size: int,
    exact_tail_pages: int,
    exact_prefix_pages: int,
) -> None:
    gate = _mapping(aggregate, label)
    checks = gate.get("checks")
    if (
        gate.get("enabled") is not (exact_prefix_pages > 0)
        or gate.get("passed") is not True
        or gate.get("check_count") != expected_count
        or not isinstance(checks, list)
        or len(checks) != expected_count
        or gate.get("checked_after_terminal_synchronize_before_restore") is not True
        or gate.get("no_prefix_digest_between_hot_precondition_and_sample") is not True
        or gate.get("inside_timed_boundary") is not False
    ):
        raise ProtocolError(f"{label} exact-prefix aggregate failed")
    for index, record in enumerate(checks):
        validate_exact_prefix_canary_record(
            record,
            label=f"{label}.checks[{index}]",
            batch_size=batch_size,
            exact_tail_pages=exact_tail_pages,
            exact_prefix_pages=exact_prefix_pages,
        )


def validate_cache_tensor_manifest(
    manifest: Mapping[str, Any],
    *,
    backend: str,
    batch_size: int,
    served_pages: int,
    exact_tail_pages: int,
    exact_prefix_pages: int,
) -> dict[str, int]:
    """Verify the fixed Mistral cache geometry and derive byte accounting."""

    layers, page_tokens, kv_heads, query_heads, head_dim = 32, PAGE, 8, 32, 128
    allocated_pages = served_pages + 1
    full_physical_pages = batch_size * allocated_pages
    exact_physical_pages = batch_size * (
        exact_tail_pages + exact_prefix_pages
    )
    if backend == BASELINE:
        specifications = {
            "key": ([layers, full_physical_pages, page_tokens, kv_heads, head_dim], "torch.float16", 2),
            "value": ([layers, full_physical_pages, page_tokens, kv_heads, head_dim], "torch.float16", 2),
        }
        canary_names = ("key", "value")
    else:
        specifications = {
            "exact_key": ([layers, exact_physical_pages, page_tokens, kv_heads, head_dim], "torch.float16", 2),
            "exact_value": ([layers, exact_physical_pages, page_tokens, kv_heads, head_dim], "torch.float16", 2),
            "key_codes": ([layers, full_physical_pages, page_tokens, kv_heads, head_dim], "torch.int8", 1),
            "value_codes": ([layers, full_physical_pages, page_tokens, kv_heads, head_dim], "torch.int8", 1),
            "key_scales": ([layers, full_physical_pages, kv_heads], "torch.float16", 2),
            "value_scales": ([layers, full_physical_pages, kv_heads], "torch.float16", 2),
            "key_center": ([layers, batch_size, kv_heads, head_dim], "torch.float16", 2),
            "value_center": ([layers, batch_size, kv_heads, head_dim], "torch.float16", 2),
            "output_center": ([layers, batch_size, query_heads, head_dim], "torch.float16", 2),
        }
        canary_names = ("key_codes", "value_codes", "key_scales", "value_scales")
    tensors = _mapping(manifest.get("tensors"), "selected cache tensors")
    if set(tensors) != set(specifications):
        raise ProtocolError("selected cache tensor manifest is incomplete")
    derived_bytes: dict[str, int] = {}
    pointers = []
    for name, (shape, dtype, item_size) in specifications.items():
        record = _mapping(tensors[name], f"selected cache tensor {name}")
        expected_bytes = math.prod(shape) * item_size
        pointer = record.get("data_ptr")
        if (
            record.get("shape") != shape
            or record.get("dtype") != dtype
            or record.get("bytes") != expected_bytes
            or not isinstance(pointer, int)
            or isinstance(pointer, bool)
            or pointer <= 0
        ):
            raise ProtocolError(f"selected cache tensor {name} geometry/bytes failed")
        derived_bytes[name] = expected_bytes
        pointers.append(pointer)
    if len(set(pointers)) != len(pointers):
        raise ProtocolError("selected cache tensor pointers are not distinct")
    total_bytes = sum(derived_bytes.values())
    if manifest.get("total_bytes") != total_bytes:
        raise ProtocolError("selected cache total bytes do not sum tensor bytes")
    canary_bytes = 0
    for name in canary_names:
        shape, _dtype, item_size = specifications[name]
        one_physical_page_elements = math.prod([shape[0], *shape[2:]])
        canary_bytes += one_physical_page_elements * item_size * batch_size
    prefix_gross_bytes = (
        layers
        * batch_size
        * exact_prefix_pages
        * page_tokens
        * kv_heads
        * head_dim
        * 2
        * 2
        if backend == CANDIDATE
        else 0
    )
    return {
        "allocated_bytes": total_bytes,
        "following_canary_bytes": canary_bytes,
        "served_bytes": total_bytes - canary_bytes,
        "exact_prefix_gross_allocated_bytes": prefix_gross_bytes,
    }


def expected_wrapper_names(backend: str, tail_attention: str) -> set[str]:
    if backend == BASELINE:
        return {"baseline"}
    if tail_attention == "heterogeneous_fa2":
        return {"heterogeneous"}
    return {"old_int8", "exact_fp16"}


def validate_planner_runtime_states(
    buckets: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    tail_attention: str,
    context: int,
    decode_steps: int,
    exact_tail_pages: int,
    exact_prefix_pages: int,
) -> None:
    expected_names = expected_wrapper_names(backend, tail_attention)
    states: list[Mapping[str, Any]] = []
    for bucket_index, raw_bucket in enumerate(buckets):
        bucket = _mapping(raw_bucket, f"graph bucket {bucket_index}")
        bucket_states = bucket.get("planner_runtime_states")
        if not isinstance(bucket_states, list) or not bucket_states:
            raise ProtocolError("graph bucket has no planner runtime states")
        summary = _mapping(
            bucket.get("planner_runtime_summary"),
            f"graph bucket {bucket_index} planner summary",
        )
        if set(summary) != expected_names:
            raise ProtocolError("graph planner summary wrapper set is invalid")
        for name in expected_names:
            wrapper_states = [
                _mapping(
                    _mapping(state, "planner state").get("wrappers"),
                    "planner state wrappers",
                ).get(name)
                for state in bucket_states
            ]
            records = [
                _mapping(record, f"graph bucket {bucket_index} wrapper {name}")
                for record in wrapper_states
            ]
            active_pages = [int(record["pages_per_request"]) for record in records]
            wrapper_summary = _mapping(
                summary[name], f"graph bucket {bucket_index} summary {name}"
            )
            if (
                wrapper_summary.get("sampled_boundary_state_count")
                != len(bucket_states)
                or wrapper_summary.get("active_pages_per_request_min")
                != min(active_pages)
                or wrapper_summary.get("active_pages_per_request_max")
                != max(active_pages)
            ):
                raise ProtocolError("graph planner runtime summary is inconsistent")
        states.extend(_mapping(state, "planner runtime state") for state in bucket_states)
    expected_positions = list(range(context, context + decode_steps, PAGE))
    if [state.get("position") for state in states] != expected_positions:
        raise ProtocolError("graph planner runtime states omit a page boundary")
    exact_pages = exact_tail_pages + exact_prefix_pages
    for state in states:
        position = int(state["position"])
        current_context = position + 1
        current_pages = math.ceil(current_context / PAGE)
        wrappers = _mapping(state.get("wrappers"), "planner runtime wrappers")
        if set(wrappers) != expected_names:
            raise ProtocolError("graph planner runtime wrapper set changed")
        expected_old_pages = (
            current_pages - exact_pages if backend == CANDIDATE else 0
        )
        expected_state_exact_pages = exact_pages if backend == CANDIDATE else 1
        if (
            state.get("context") != current_context
            or state.get("current_pages") != current_pages
            or state.get("old_pages") != expected_old_pages
            or state.get("exact_pages") != expected_state_exact_pages
        ):
            raise ProtocolError("graph planner exact-prefix page state is invalid")
        expected_wrapper_pages = (
            {"baseline": current_pages}
            if backend == BASELINE
            else {
                "old_int8": expected_old_pages,
                "exact_fp16": exact_pages,
            }
        )
        for name, expected_pages in expected_wrapper_pages.items():
            record = _mapping(wrappers[name], f"planner runtime wrapper {name}")
            if (
                record.get("pages_per_request") != expected_pages
                or not isinstance(record.get("kv_chunk_size_tokens"), int)
                or record["kv_chunk_size_tokens"] <= 0
                or not isinstance(record.get("initialized_tile_count"), int)
                or record["initialized_tile_count"] <= 0
            ):
                raise ProtocolError("graph planner wrapper runtime state is invalid")
            _sha256(
                record.get("semantic_int_workspace_sha256"),
                f"planner runtime wrapper {name} workspace",
            )


def validate_runtime_gate_record(
    gate: Mapping[str, Any],
    *,
    label: str,
    backend: str,
    tail_attention: str,
    exact_prefix_pages: int,
    decode_steps: int,
) -> None:
    expected_replays = 32 * decode_steps
    expected_updates = (
        decode_steps // PAGE
        if backend == CANDIDATE and tail_attention == "heterogeneous_fa2"
        else 0
    )
    expected_exact_updates = (
        decode_steps // PAGE
        if backend == CANDIDATE and exact_prefix_pages > 0
        else 0
    )
    dispatch = _mapping(gate.get("observed_dispatch"), f"{label} dispatch")
    nested = _mapping(
        gate.get("observed_nested_attention_dispatch"), f"{label} nested dispatch"
    )
    operations = _mapping(gate.get("observed_operations"), f"{label} operations")
    declared = _mapping(gate.get("expected"), f"{label} expected counts")
    wrappers = _mapping(operations.get("wrappers"), f"{label} wrappers")
    if (
        gate.get("passed") is not True
        or gate.get("wrapper_counts_passed") is not True
        or dispatch.get("graph_replays") != expected_replays
        or dispatch.get("eager_calls") != 0
        or dispatch.get("graph_bank_misses") != 0
        or dispatch.get("total_calls") != expected_replays
        or nested.get("total_calls") != 0
        or nested.get("eager_calls") != 0
        or nested.get("graph_replays") != 0
        or operations.get("decoder_plan_calls") != decode_steps
        or operations.get("device_position_fills") != decode_steps
        or operations.get("heterogeneous_page_table_updates") != expected_updates
        or operations.get("exact_page_table_updates", 0) != expected_exact_updates
        or set(wrappers) != expected_wrapper_names(backend, tail_attention)
    ):
        raise ProtocolError(f"{label} runtime graph gate failed")
    expected_declaration = {
        "decoder_plan_calls": decode_steps,
        "device_position_fills": decode_steps,
        "wrapper_count": len(expected_wrapper_names(backend, tail_attention)),
        "plan_invocations_per_wrapper": decode_steps,
        "plan_rebuilds_per_wrapper": decode_steps // PAGE,
        "last_page_len_device_fills_per_wrapper": decode_steps,
        "decoder_layer_graph_replays": expected_replays,
        "decoder_layer_eager_calls": 0,
        "graph_bank_misses": 0,
        "nested_attention_dispatch_calls": 0,
        "heterogeneous_page_table_updates": expected_updates,
        "exact_page_table_updates": expected_exact_updates,
    }
    if dict(declared) != expected_declaration:
        raise ProtocolError(f"{label} declared raw operation counts are inconsistent")
    for name, raw_record in wrappers.items():
        record = _mapping(raw_record, f"{label} wrapper {name}")
        if (
            record.get("plan_invocations") != decode_steps
            or record.get("plan_rebuilds") != decode_steps // PAGE
            or record.get("last_page_len_device_fills") != decode_steps
        ):
            raise ProtocolError(f"{label} wrapper operation counts failed")


def validate_eager_operation_gate(
    same_backend: Mapping[str, Any],
    *,
    backend: str,
    tail_attention: str,
    exact_prefix_pages: int,
    decode_steps: int,
) -> None:
    dispatch = _mapping(same_backend.get("eager_dispatch"), "eager layer dispatch")
    attention = _mapping(
        same_backend.get("eager_attention_dispatch"), "eager attention dispatch"
    )
    operations = _mapping(same_backend.get("eager_operations"), "eager operations")
    wrappers = _mapping(operations.get("wrappers"), "eager wrappers")
    expected_calls = 32 * decode_steps
    expected_updates = (
        decode_steps // PAGE
        if backend == CANDIDATE and tail_attention == "heterogeneous_fa2"
        else 0
    )
    expected_exact_updates = (
        decode_steps // PAGE
        if backend == CANDIDATE and exact_prefix_pages > 0
        else 0
    )
    if (
        same_backend.get("eager_gate_passed") is not True
        or dispatch.get("eager_calls") != expected_calls
        or dispatch.get("graph_replays") != 0
        or dispatch.get("total_calls") != expected_calls
        or attention.get("eager_calls") != expected_calls
        or attention.get("graph_replays") != 0
        or attention.get("total_calls") != expected_calls
        or operations.get("decoder_plan_calls") != decode_steps
        or operations.get("device_position_fills") != decode_steps
        or operations.get("heterogeneous_page_table_updates") != expected_updates
        or operations.get("exact_page_table_updates", 0) != expected_exact_updates
        or set(wrappers) != expected_wrapper_names(backend, tail_attention)
    ):
        raise ProtocolError("eager dispatch/operation gate failed")
    for name, raw_record in wrappers.items():
        record = _mapping(raw_record, f"eager wrapper {name}")
        if (
            record.get("plan_invocations") != decode_steps
            or record.get("plan_rebuilds") != decode_steps // PAGE
            or record.get("last_page_len_device_fills") != decode_steps
        ):
            raise ProtocolError("eager wrapper operation counts failed")


def validate_worker_result(
    result: Mapping[str, Any],
    expected_backend: str,
    expected_seed: int,
    expected_orchestration: Mapping[str, str] | None = None,
    expected_configuration: Mapping[str, Any] | None = None,
    expected_worker_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("worker schema_version must be 3")
    if result.get("experiment") != WORKER_EXPERIMENT:
        raise ProtocolError("worker experiment is not the sustained protocol")
    if result.get("backend") != expected_backend or expected_backend not in BACKENDS:
        raise ProtocolError("worker backend does not match the scheduled backend")
    if result.get("cuda_graph_scope") != DYNAMIC_GRAPH_SCOPE:
        raise ProtocolError("worker top-level CUDA graph scope is incompatible")
    if result.get("passed") is not True:
        raise ProtocolError("worker did not pass all correctness gates")

    config = _mapping(result.get("configuration"), "worker configuration")
    if config.get("backend") != expected_backend:
        raise ProtocolError("worker configuration backend mismatch")
    if config.get("seed") != expected_seed:
        raise ProtocolError("worker seed mismatch")
    if config.get("cuda_graph_scope") != DYNAMIC_GRAPH_SCOPE:
        raise ProtocolError("worker configuration CUDA graph scope mismatch")
    if expected_configuration is not None:
        for key, expected in expected_configuration.items():
            if config.get(key) != expected:
                raise ProtocolError(
                    f"worker configuration.{key} expected {expected!r}, "
                    f"got {config.get(key)!r}"
                )
    exact_prefix_pages = config.get("exact_prefix_pages")
    expected_prefix_pages = (
        POLICY_EXACT_PREFIX_PAGES if expected_backend == CANDIDATE else 0
    )
    if (
        not isinstance(exact_prefix_pages, int)
        or isinstance(exact_prefix_pages, bool)
        or exact_prefix_pages != expected_prefix_pages
        or config.get("exact_sink_pages") != exact_prefix_pages
    ):
        raise ProtocolError("worker exact-prefix/sink policy is invalid")
    context = config.get("context")
    decode_steps = config.get("decode_steps")
    repeats = config.get("repeats")
    if (
        not isinstance(context, int)
        or isinstance(context, bool)
        or context <= 0
        or context % PAGE
    ):
        raise ProtocolError("worker context must be positive and page aligned")
    if (
        not isinstance(decode_steps, int)
        or isinstance(decode_steps, bool)
        or decode_steps < 512
        or decode_steps % PAGE
    ):
        raise ProtocolError("worker decode_steps must be page-aligned and >=512")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ProtocolError("worker repeats must be positive")
    served_pages = math.ceil((context + decode_steps) / PAGE)

    pair_identity = worker_pair_identity(result)
    if expected_worker_source_sha256 is not None:
        expected_sources = {
            str(name): _sha256(digest, f"manifest worker source {name}")
            for name, digest in expected_worker_source_sha256.items()
        }
        observed_sources = pair_identity["source_sha256"]
        if observed_sources != expected_sources:
            raise ProtocolError(
                "worker source closure differs from the orchestration manifest"
            )
    pair_config = pair_identity["configuration"]
    paired_configuration_keys = {
        "model",
        "model_revision",
        "batch_size",
        "context",
        "decode_steps",
        "exact_tail_tokens",
        "exact_sink_pages",
        "exact_prefix_pages",
        "prefill_chunk_tokens",
        "baseline_split_pages",
        "candidate_split_pages",
        "tail_attention",
        "old_value_scale_placement",
        "trajectory_mode",
        "cuda_graph_scope",
        "token_source",
        "wikitext_member",
        "wikitext_archive_sha256",
        "min_logits_cosine",
        "min_top1_agreement",
        "seed",
        "token_offset",
        "token_stride",
        "capture_warmups",
        "maximum_graph_banks",
        "warmups",
        "repeats",
        "cache_scrub_mib",
    }
    digest_names = {
        "teacher_inputs_sha256",
        "token_matrix_sha256",
        "model_config_sha256",
        "sampled_model_parameters_sha256",
        "token_source_provenance_sha256",
        "flashinfer_abi_source_sha256",
    }
    if set(pair_config) != paired_configuration_keys | digest_names:
        raise ProtocolError("worker pairing configuration schema is not exact")
    for key in paired_configuration_keys:
        if pair_config.get(key) != config.get(key):
            raise ProtocolError(f"pairing configuration disagrees on {key}")
    for digest_name in digest_names - {"flashinfer_abi_source_sha256"}:
        _sha256(pair_config.get(digest_name), f"pairing.{digest_name}")
    fi_hashes = _mapping(
        pair_config.get("flashinfer_abi_source_sha256"),
        "pairing FlashInfer ABI hashes",
    )
    if set(fi_hashes) != {"decode.py", "scheduler.cuh"}:
        raise ProtocolError("pairing key lacks exact FlashInfer ABI source hashes")
    for name, digest in fi_hashes.items():
        _sha256(digest, f"FlashInfer ABI {name}")
    trajectory = _mapping(result.get("trajectory"), "worker trajectory")
    greedy = config.get("trajectory_mode") == "greedy_feedback"
    if (
        trajectory.get("mode") != config.get("trajectory_mode")
        or trajectory.get("generated_feedback") is not greedy
        or trajectory.get("frozen_identical_hf_input_chain") is not (not greedy)
        or trajectory.get("gpu_argmax_timed_every_step") is not True
        or trajectory.get("argmax_fed_to_next_step") is not greedy
        or trajectory.get(
            "cross_backend_pairing_requires_identical_generated_trajectory_hash"
        )
        is not greedy
        or trajectory.get(
            "cross_backend_pairing_requires_identical_frozen_input_chain_hash"
        )
        is not (not greedy)
        or trajectory.get("teacher_inputs_sha256")
        != pair_config.get("teacher_inputs_sha256")
    ):
        raise ProtocolError("trajectory-mode provenance is internally inconsistent")
    worker_publication_status = _mapping(
        result.get("publication_orchestration_status"),
        "worker publication-orchestration status",
    )
    if (
        worker_publication_status.get("manual_worker_exact_prefix_supported")
        is not True
        or worker_publication_status.get("manual_worker_exact_sink_supported")
        is not True
        or worker_publication_status.get(
            "williams_orchestrator_exact_prefix_integrated"
        )
        is not False
        or worker_publication_status.get(
            "williams_analyzer_exact_prefix_attestation_integrated"
        )
        is not False
        or worker_publication_status.get(
            "williams_orchestrator_exact_sink_integrated"
        )
        is not False
        or worker_publication_status.get(
            "williams_analyzer_exact_sink_attestation_integrated"
        )
        is not False
        or worker_publication_status.get(
            "publication_pairing_deferred_until_manual_known_row_strict_gate"
        )
        is not True
    ):
        raise ProtocolError(
            "worker manual exact-prefix capability/integration declaration is invalid"
        )
    model = _mapping(result.get("model_provenance"), "worker model provenance")
    parameter_sample = _mapping(
        model.get("sampled_parameters"), "worker sampled model parameters"
    )
    if (
        model.get("requested_name_or_path") != config.get("model")
        or model.get("resolved_revision") != config.get("model_revision")
        or model.get("config_sha256") != pair_config.get("model_config_sha256")
        or parameter_sample.get("sha256")
        != pair_config.get("sampled_model_parameters_sha256")
        or not isinstance(model.get("parameter_count"), int)
        or isinstance(model.get("parameter_count"), bool)
        or model.get("parameter_count") <= 0
    ):
        raise ProtocolError("model identity/provenance is internally inconsistent")

    timed = _mapping(result.get("timed_work"), "worker timed_work")
    page_boundaries = decode_steps // PAGE
    wrapper_count = (
        1
        if expected_backend == BASELINE
        or config.get("tail_attention") == "heterogeneous_fa2"
        else 2
    )
    expected_timed = {
        "continuous_decoder_steps": decode_steps,
        "output_tokens": decode_steps * int(config["batch_size"]),
        "decoder_plan_calls": decode_steps,
        "wrapper_plan_invocations": decode_steps * wrapper_count,
        "wrapper_last_page_len_device_fills": decode_steps * wrapper_count,
        "device_position_fills": decode_steps,
        "greedy_seed_token_device_copy": (
            1 if config.get("trajectory_mode") == "greedy_feedback" else 0
        ),
        "token_embedding_calls": decode_steps,
        "embedding_to_graph_bank_input_device_copies": decode_steps,
        "complete_layer_graph_replays": 32 * decode_steps,
        "captured_persistent_layer_output_writes": 32 * decode_steps,
        "final_model_norm_calls": decode_steps,
        "lm_head_calls": decode_steps,
        "gpu_argmax_operations": decode_steps,
        "host_planner_boundary_points": page_boundaries,
        "flashinfer_full_plan_host_barriers_per_wrapper": page_boundaries,
        "flashinfer_full_plan_calls_all_wrappers": page_boundaries
        * wrapper_count,
        "blocking_d2h_metadata_copies": 2 * page_boundaries * wrapper_count,
    }
    for key, expected in expected_timed.items():
        if timed.get(key) != expected:
            raise ProtocolError(
                f"timed_work.{key} expected {expected!r}, got {timed.get(key)!r}"
            )
    for key in (
        "runtime_page_table_and_plan_updates_included",
        "prefix_plan_restored_before_timing",
        "first_generated_page_transition_included",
        "no_host_token_id_readback",
    ):
        if timed.get(key) is not True:
            raise ProtocolError(f"timed_work.{key} must be true")
    if timed.get("one_time_graph_bank_build_included") is not False:
        raise ProtocolError("one-time graph-bank build must be outside timing")
    if timed.get("host_synchronization_free_between_tokens") is not False:
        raise ProtocolError("worker incorrectly claims host-synchronization-free planning")
    if timed.get("planner_boundary_positions") != list(
        range(context, context + decode_steps, PAGE)
    ):
        raise ProtocolError("timed planner-boundary positions are incomplete")

    correctness = _mapping(result.get("correctness"), "worker correctness")
    same_backend = _mapping(
        correctness.get("same_backend_eager_vs_graph"),
        "same-backend correctness",
    )
    repeat = _mapping(
        same_backend.get("restored_graph_repeat"), "restored graph repeat"
    )
    if (
        correctness.get("passed") is not True
        or same_backend.get("passed") is not True
        or same_backend.get("eager_gate_passed") is not True
        or _mapping(same_backend.get("graph_runtime_gate"), "graph runtime gate").get(
            "passed"
        )
        is not True
        or repeat.get("passed") is not True
        or repeat.get("restore_before_repeat") is not True
        or repeat.get("no_restore_repeat_performed") is not False
        or _mapping(repeat.get("runtime_gate"), "repeat runtime gate").get(
            "passed"
        )
        is not True
    ):
        raise ProtocolError("same-backend or restored-repeat gate failed")
    validate_eager_operation_gate(
        same_backend,
        backend=expected_backend,
        tail_attention=str(config.get("tail_attention")),
        exact_prefix_pages=exact_prefix_pages,
        decode_steps=decode_steps,
    )
    validate_runtime_gate_record(
        _mapping(same_backend.get("graph_runtime_gate"), "graph runtime gate"),
        label="graph_run_1",
        backend=expected_backend,
        tail_attention=str(config.get("tail_attention")),
        exact_prefix_pages=exact_prefix_pages,
        decode_steps=decode_steps,
    )
    validate_runtime_gate_record(
        _mapping(repeat.get("runtime_gate"), "repeat runtime gate"),
        label="graph_run_2_restored",
        backend=expected_backend,
        tail_attention=str(config.get("tail_attention")),
        exact_prefix_pages=exact_prefix_pages,
        decode_steps=decode_steps,
    )
    exact_ring_pages = (
        int(config["exact_tail_tokens"]) // PAGE
        if expected_backend == CANDIDATE
        else 0
    )
    validate_equivalence_bundle(
        same_backend,
        label="eager_vs_graph",
        context=context,
        decode_steps=decode_steps,
        batch_size=int(config["batch_size"]),
        exact_ring_pages=exact_ring_pages,
    )
    validate_equivalence_bundle(
        repeat,
        label="restored_graph_repeat",
        context=context,
        decode_steps=decode_steps,
        batch_size=int(config["batch_size"]),
        exact_ring_pages=exact_ring_pages,
    )
    validate_adjacent_canary_gates(
        same_backend.get("immutable_adjacent_page_canaries"),
        context=context,
        decode_steps=decode_steps,
        exact_prefix_pages=exact_prefix_pages,
    )
    finalization = _mapping(
        correctness.get("runtime_page_finalization_and_consumption"),
        "runtime finalization",
    )
    expected_generated_pages = decode_steps // PAGE
    expected_first_page = context // PAGE
    expected_last_page = expected_first_page + expected_generated_pages - 1
    expected_consumed = (
        list(
            range(
                expected_first_page,
                expected_last_page + 1 - exact_ring_pages,
            )
        )
        if expected_backend == CANDIDATE
        else []
    )
    base_finalization_failed = (
        finalization.get("passed") is not True
        or finalization.get("first_generated_logical_page") != expected_first_page
        or finalization.get("last_generated_logical_page") != expected_last_page
        or finalization.get("generated_pages") != expected_generated_pages
        or finalization.get("runtime_finalized_pages_consumed_as_int8")
        != expected_consumed
        or finalization.get("runtime_finalized_int8_pages_consumed_count")
        != len(expected_consumed)
        or finalization.get("final_attention_page_table_gate_passed") is not True
    )
    if base_finalization_failed:
        raise ProtocolError("runtime page-finalization/consumption gate failed")
    if expected_backend == CANDIDATE:
        final_pages = served_pages
        tail_begin = final_pages - exact_ring_pages
        expected_old_pages = list(range(exact_prefix_pages, tail_begin))
        expected_tail_pages = list(range(tail_begin, final_pages))
        if (
            len(expected_consumed) != POLICY_GENERATED_INT8_PAGES
            or finalization.get("final_old_pages") != len(expected_old_pages)
            or finalization.get("final_old_logical_page_end_exclusive")
            != tail_begin
            or finalization.get("final_old_logical_pages") != expected_old_pages
            or finalization.get("final_exact_prefix_logical_pages")
            != list(range(exact_prefix_pages))
            or finalization.get("final_exact_sink_logical_pages")
            != list(range(exact_prefix_pages))
            or finalization.get("final_exact_tail_logical_pages")
            != expected_tail_pages
            or finalization.get("old_attention_page_table_gate_passed") is not True
            or finalization.get("exact_attention_page_table_gate_passed") is not True
            or finalization.get("logical_page_sets_disjoint") is not True
            or finalization.get("logical_token_coverage_exactly_once") is not True
            or finalization.get("exact_prefix_excluded_from_old_segment") is not True
            or finalization.get("page_zero_excluded_from_old_segment") is not True
            or finalization.get("prefix_exclusion_gate_passed") is not True
            or finalization.get("sink_exclusion_gate_passed") is not True
        ):
            raise ProtocolError(
                "PageGauge exact-prefix/old/tail recurrence attestation failed"
            )
    hf_gate = _mapping(
        correctness.get("backend_vs_hf_sdpa_fp16"), "backend-vs-HF correctness"
    )
    hf_logits = _mapping(hf_gate.get("logits"), "backend-vs-HF logits")
    hf_tokens = _mapping(
        hf_gate.get("generated_argmax_tokens"), "backend-vs-HF tokens"
    )
    if (
        hf_gate.get("passed") is not True
        or hf_logits.get("checked_steps") != decode_steps
        or hf_logits.get("checked_request_steps")
        != decode_steps * int(config["batch_size"])
        or hf_logits.get("minimum_cosine") < float(config["min_logits_cosine"])
        or hf_logits.get("top1_agreement_fraction")
        < float(config["min_top1_agreement"])
        or hf_tokens.get("checked_steps") != decode_steps
        or hf_tokens.get("checked_request_steps")
        != decode_steps * int(config["batch_size"])
        or (
            config.get("trajectory_mode") == "greedy_feedback"
            and hf_tokens.get("bitwise_identical") is not True
        )
    ):
        raise ProtocolError("backend-vs-HF nested correctness gate failed")

    graph = _mapping(result.get("cuda_graph_provenance"), "CUDA graph provenance")
    if (
        graph.get("enabled") is not True
        or graph.get("cuda_graph_scope") != DYNAMIC_GRAPH_SCOPE
        or graph.get("scope") != "complete_decoder_layer_device_dynamic_position"
        or graph.get("device_dynamic_position") is not True
        or graph.get("structure_gate_passed") is not True
        or graph.get("strict_missing_bucket_failure") is not True
        or graph.get("graph_bank_misses") != 0
        or graph.get("nested_attention_graphs") is not False
        or graph.get("replays_per_decode_step") != 32
        or graph.get("raw_cuda_graph_retained_after_instantiation") is not False
        or graph.get("preflight_position_count") != decode_steps
        or graph.get("preflight_range_start_inclusive_end_exclusive")
        != [context, context + decode_steps]
    ):
        raise ProtocolError("CUDA graph topology/preflight provenance failed")
    graph_banks = graph.get("graph_bank_count")
    if (
        not isinstance(graph_banks, int)
        or isinstance(graph_banks, bool)
        or graph_banks <= 0
        or graph.get("graphs_per_bank") != 32
        or graph.get("total_graphs") != graph_banks * 32
        or graph.get("graph_pools") != graph_banks
    ):
        raise ProtocolError("CUDA graph-bank count/pool ownership is inconsistent")
    buckets = graph.get("preflight_bucket_ranges")
    if (
        not isinstance(buckets, list)
        or len(buckets) != graph_banks
        or sum(int(bucket.get("position_count", -1)) for bucket in buckets)
        != decode_steps
        or any(
            bucket.get("capture_position") is None
            or not isinstance(bucket.get("planner_runtime_states"), list)
            or not bucket.get("planner_runtime_states")
            or not isinstance(bucket.get("planner_runtime_summary"), Mapping)
            for bucket in buckets
        )
    ):
        raise ProtocolError("CUDA graph preflight bucket/runtime-state coverage failed")
    validate_planner_runtime_states(
        buckets,
        backend=expected_backend,
        tail_attention=str(config.get("tail_attention")),
        context=context,
        decode_steps=decode_steps,
        exact_tail_pages=exact_ring_pages,
        exact_prefix_pages=exact_prefix_pages,
    )
    _sha256(graph.get("preflight_positions_sha256"), "graph preflight mapping")
    capture_memory = _mapping(graph.get("capture_memory"), "graph capture memory")
    integer_capture_fields = (
        "persistent_hidden_buffer_bytes",
        "torch_allocated_delta_bytes",
        "torch_reserved_delta_bytes",
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "torch_peak_allocated_increment_over_capture_start_bytes",
        "torch_peak_reserved_increment_over_capture_start_bytes",
        "cuda_free_delta_bytes",
        "cuda_consumed_delta_bytes",
    )
    for field in integer_capture_fields:
        value = capture_memory.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProtocolError(f"graph capture memory field {field} is invalid")
    for field in (
        "persistent_hidden_buffer_bytes",
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "torch_peak_allocated_increment_over_capture_start_bytes",
        "torch_peak_reserved_increment_over_capture_start_bytes",
    ):
        if capture_memory[field] < 0:
            raise ProtocolError(f"graph capture memory field {field} is negative")
    if capture_memory["cuda_free_delta_bytes"] != -capture_memory[
        "cuda_consumed_delta_bytes"
    ]:
        raise ProtocolError("graph capture free/consumed deltas disagree")
    if (
        not isinstance(
            capture_memory.get("one_time_graph_bank_capture_wall_seconds"),
            (int, float),
        )
        or capture_memory["one_time_graph_bank_capture_wall_seconds"] <= 0
        or not isinstance(
            graph.get("one_time_exhaustive_preflight_and_graph_bank_build_wall_seconds"),
            (int, float),
        )
        or graph["one_time_exhaustive_preflight_and_graph_bank_build_wall_seconds"]
        <= 0
    ):
        raise ProtocolError("graph startup timing provenance is invalid")
    capacity = _mapping(result.get("scheduler_capacity"), "scheduler capacity")
    if (
        capacity.get("all_wrappers_analytically_within_capacity") is not True
        or capacity.get("split_was_automatically_clamped") is not False
    ):
        raise ProtocolError("scheduler capacity gate failed")
    capacity_wrappers = _mapping(capacity.get("wrappers"), "scheduler wrappers")
    expected_capacity = (
        {"baseline": (served_pages, int(config["baseline_split_pages"]))}
        if expected_backend == BASELINE
        else {
            "heterogeneous": (
                served_pages,
                int(config["candidate_split_pages"]),
            )
        }
        if config.get("tail_attention") == "heterogeneous_fa2"
        else {
            "old_int8": (
                max(served_pages - exact_ring_pages - exact_prefix_pages, 1),
                int(config["candidate_split_pages"]),
            ),
            "exact_fp16": (
                exact_ring_pages + exact_prefix_pages,
                int(config["candidate_split_pages"]),
            ),
        }
    )
    if set(capacity_wrappers) != set(expected_capacity):
        raise ProtocolError("scheduler wrapper set does not match attention path")
    for name, (active_pages, fixed_split) in expected_capacity.items():
        record = _mapping(capacity_wrappers[name], f"scheduler wrapper {name}")
        if (
            record.get("maximum_active_pages_per_request") != active_pages
            or record.get("fixed_split_pages") != fixed_split
            or record.get("analytically_within_capacity") is not True
            or record.get("split_was_automatically_clamped") is not False
        ):
            raise ProtocolError(f"scheduler wrapper {name} capacity provenance failed")

    attention = _mapping(
        result.get("attention_implementation"), "attention implementation"
    )
    if expected_backend == BASELINE:
        if (
            attention.get("implementation") != "flashinfer_fp16_fa2"
            or attention.get("logical_attention_segments") != 1
            or attention.get("custom_module_uri") is not None
            or attention.get("custom_module_source_hashes") is not None
            or attention.get("old_int8_value_scale_placement") is not None
            or attention.get("exact_sink_pages") != 0
            or attention.get("exact_prefix_pages") != 0
        ):
            raise ProtocolError("baseline attention implementation is not FI FP16 FA2")
    else:
        expected_impl = (
            "page_gauge_heterogeneous_int8_fp16_fa2"
            if config.get("tail_attention") == "heterogeneous_fa2"
            else f"page_gauge_segmented_{config.get('tail_attention')}"
        )
        expected_segments = (
            1 if config.get("tail_attention") == "heterogeneous_fa2" else 2
        )
        if (
            attention.get("implementation") != expected_impl
            or attention.get("logical_attention_segments") != expected_segments
            or not isinstance(attention.get("custom_module_uri"), str)
            or not attention.get("custom_module_uri")
        ):
            raise ProtocolError("candidate attention implementation/URI is inconsistent")
        if (
            attention.get("old_int8_value_scale_placement")
            != config.get("old_value_scale_placement")
            or attention.get("exact_tail_pages") != exact_ring_pages
            or attention.get("exact_sink_pages") != exact_prefix_pages
            or attention.get("exact_prefix_pages") != exact_prefix_pages
            or attention.get("exact_segment_logical_order")
            != (
                "[contiguous exact prefix logical pages 0..S-1, chronological "
                "recent exact tail]"
            )
            or attention.get("old_segment_logical_range")
            != "[S, tail_start) with every exact-prefix page excluded"
            or attention.get("exact_physical_layout")
            != "fixed slots 0..S-1 followed by modulo tail-ring slots S..S+T-1"
            or attention.get("prefix_uses_existing_exact_wrapper") is not True
            or attention.get("logical_attention_segments_unchanged_by_prefix")
            is not True
            or attention.get("sink_uses_existing_exact_wrapper") is not True
            or attention.get("logical_attention_segments_unchanged_by_sink")
            is not True
        ):
            raise ProtocolError("candidate exact-prefix attention provenance failed")
        validate_candidate_module_provenance(
            attention, str(config.get("tail_attention"))
        )

    exclusivity = _mapping(result.get("exclusivity"), "worker exclusivity")
    required_exclusivity = {
        "fresh_process_required": True,
        "selected_persistent_backend": expected_backend,
        "opposite_backend_full_gpu_cache_allocated": False,
        "hf_dynamic_caches_released_before_decoder_construction": True,
        "cache_serialization_or_reload": False,
        "direct_destination_construction": True,
    }
    for key, expected in required_exclusivity.items():
        if exclusivity.get(key) != expected:
            raise ProtocolError(f"worker exclusivity.{key} is invalid")
    cache_build = _mapping(result.get("cache_build"), "worker cache build")
    manifest = _mapping(
        cache_build.get("selected_backend_cache"), "selected backend cache manifest"
    )
    if (
        manifest.get("backend") != expected_backend
        or manifest.get("all_storage_pointers_distinct") is not True
        or cache_build.get("opposite_backend_full_gpu_cache_allocated") is not False
        or cache_build.get(
            "full_served_mutation_range_restored_before_every_validation_and_sample"
        )
        is not True
        or cache_build.get("preceding_and_following_page_canaries_checked") is not True
        or cache_build.get("snapshot_offloaded_to_cpu_before_capture_and_timing")
        is not True
        or cache_build.get("allocated_pages_per_request")
        != cache_build.get("served_pages_per_request") + 1
        or cache_build.get("served_pages_per_request") != served_pages
        or cache_build.get("initial_pages_per_request") != context // PAGE
        or cache_build.get("mutated_generated_pages_per_request")
        != decode_steps // PAGE
        or cache_build.get("following_canary_logical_page") != served_pages
    ):
        raise ProtocolError("backend-exclusive cache/canary manifest is invalid")
    allocated_cache_bytes = manifest.get("total_bytes")
    served_cache_bytes = cache_build.get(
        "selected_backend_cache_served_bytes_excluding_following_canary"
    )
    following_canary_bytes = cache_build.get("following_canary_storage_bytes")
    expected_exact_ring_pages = (
        int(config["exact_tail_tokens"]) // PAGE
        if expected_backend == CANDIDATE
        else 0
    )
    derived_cache_bytes = validate_cache_tensor_manifest(
        manifest,
        backend=expected_backend,
        batch_size=int(config["batch_size"]),
        served_pages=served_pages,
        exact_tail_pages=expected_exact_ring_pages,
        exact_prefix_pages=exact_prefix_pages,
    )
    if (
        not isinstance(allocated_cache_bytes, int)
        or isinstance(allocated_cache_bytes, bool)
        or not isinstance(served_cache_bytes, int)
        or isinstance(served_cache_bytes, bool)
        or not isinstance(following_canary_bytes, int)
        or isinstance(following_canary_bytes, bool)
        or min(allocated_cache_bytes, served_cache_bytes, following_canary_bytes) <= 0
        or allocated_cache_bytes != served_cache_bytes + following_canary_bytes
        or allocated_cache_bytes != derived_cache_bytes["allocated_bytes"]
        or served_cache_bytes != derived_cache_bytes["served_bytes"]
        or following_canary_bytes
        != derived_cache_bytes["following_canary_bytes"]
    ):
        raise ProtocolError("served/canary cache byte accounting is invalid")
    if cache_build.get("exact_ring_pages_per_request") != expected_exact_ring_pages:
        raise ProtocolError("exact-ring page accounting is invalid")
    expected_cache_prefix_pages = (
        exact_prefix_pages if expected_backend == CANDIDATE else 0
    )
    if (
        cache_build.get("exact_tail_pages_per_request")
        != expected_exact_ring_pages
        or cache_build.get("exact_sink_pages_per_request")
        != expected_cache_prefix_pages
        or cache_build.get("exact_prefix_pages_per_request")
        != expected_cache_prefix_pages
        or cache_build.get("exact_storage_pages_per_request")
        != expected_exact_ring_pages + expected_cache_prefix_pages
    ):
        raise ProtocolError("exact-prefix/tail cache page accounting is invalid")
    if expected_backend == CANDIDATE:
        if (
            cache_build.get("exact_physical_layout")
            != "slots0..S-1=fixed logical prefix; slotsS..S+T-1=tail ring"
            or cache_build.get("prefix_storage_accounting")
            != (
                "gross FP16 allocation; prefix INT8 codes/scales remain allocated"
            )
            or cache_build.get("sink_storage_accounting")
            != (
                "legacy alias: gross FP16 prefix allocation; prefix INT8 "
                "codes/scales remain allocated"
            )
        ):
            raise ProtocolError("exact-prefix physical/storage policy is invalid")
        exact_layout = _mapping(
            manifest.get("exact_layout"), "selected cache exact layout"
        )
        prefix_gross_bytes = exact_layout.get("prefix_gross_allocated_bytes")
        if (
            exact_layout.get("tail_pages_per_request") != expected_exact_ring_pages
            or exact_layout.get("prefix_pages_per_request")
            != expected_cache_prefix_pages
            or exact_layout.get("sink_pages_per_request")
            != expected_cache_prefix_pages
            or exact_layout.get("storage_pages_per_request")
            != expected_exact_ring_pages + expected_cache_prefix_pages
            or exact_layout.get("physical_order")
            != (
                "[fixed contiguous prefix slots 0..S-1, modulo tail-ring "
                "slots S..S+T-1]"
            )
            or exact_layout.get("prefix_logical_pages")
            != list(range(expected_cache_prefix_pages))
            or exact_layout.get("sink_logical_pages")
            != list(range(expected_cache_prefix_pages))
            or exact_layout.get("prefix_int8_code_and_scale_storage_retained")
            is not True
            or exact_layout.get("all_exact_prefix_int8_storage_retained") is not True
            or exact_layout.get("page_zero_int8_storage_retained") is not True
            or not isinstance(prefix_gross_bytes, int)
            or isinstance(prefix_gross_bytes, bool)
            or prefix_gross_bytes <= 0
            or prefix_gross_bytes
            != derived_cache_bytes["exact_prefix_gross_allocated_bytes"]
            or exact_layout.get("sink_gross_allocated_bytes") != prefix_gross_bytes
            or exact_layout.get("compact_replacement_not_claimed") is not True
        ):
            raise ProtocolError("selected cache exact-prefix layout attestation failed")
    expected_attributes = (
        {"key", "value"}
        if expected_backend == BASELINE
        else {
            "exact_key",
            "exact_value",
            "key_codes",
            "value_codes",
            "key_scales",
            "value_scales",
            "key_center",
            "value_center",
            "output_center",
        }
    )
    if set(manifest.get("tensor_attributes", [])) != expected_attributes:
        raise ProtocolError("selected cache tensor set does not match backend")
    request_records = cache_build.get("request_records")
    if (
        not isinstance(request_records, list)
        or len(request_records) != int(config["batch_size"])
    ):
        raise ProtocolError("cache population records do not cover every request")
    for request, raw_request_record in enumerate(request_records):
        request_record = _mapping(
            raw_request_record, f"cache population request {request}"
        )
        prefix_population = _mapping(
            request_record.get("sampled_exact_prefix_population"),
            f"cache population request {request} exact prefix",
        )
        if (
            request_record.get("request") != request
            or request_record.get("prefix_tokens") != context
            or request_record.get("reference_greedy_steps") != decode_steps
            or request_record.get("prefill_chunks")
            != math.ceil(context / int(config["prefill_chunk_tokens"]))
            or request_record.get("chunk_tokens")
            != int(config["prefill_chunk_tokens"])
            or request_record.get("direct_selected_backend_population") is not True
            or request_record.get("opposite_backend_full_cache_allocated") is not False
            or request_record.get("sampled_population_gate_passed") is not True
            or prefix_population.get("enabled")
            is not (expected_cache_prefix_pages > 0)
            or prefix_population.get("logical_pages")
            != list(range(expected_cache_prefix_pages))
        ):
            raise ProtocolError(
                f"cache population request {request} provenance failed"
            )
        if expected_cache_prefix_pages:
            if (
                prefix_population.get("sampled_layers") != [0, 31]
                or prefix_population.get("all_prefix_pages_checked_in_every_layer")
                is not True
            ):
                raise ProtocolError(
                    f"cache population request {request} did not attest all prefix pages"
                )
            _sha256(
                prefix_population.get("sampled_destination_sha256"),
                f"cache population request {request} exact prefix",
            )
        elif (
            prefix_population.get("sampled_layers") != []
            or prefix_population.get("all_prefix_pages_checked_in_every_layer")
            is not False
            or prefix_population.get("sampled_destination_sha256") is not None
        ):
            raise ProtocolError(
                f"baseline cache population request {request} claims an exact prefix"
            )

    token_source = _mapping(result.get("token_source"), "worker token source")
    if token_source.get("kind") != config.get("token_source"):
        raise ProtocolError("token source kind disagrees with worker configuration")
    if canonical_json_sha256(token_source) != pair_config.get(
        "token_source_provenance_sha256"
    ):
        raise ProtocolError("pairing token-source provenance hash is inconsistent")
    _sha256(token_source.get("token_ids_sha256"), "token source IDs")
    if config.get("token_source") == "wikitext2":
        if (
            token_source.get("archive_sha256_verified") is not True
            or token_source.get("dataset") != "WikiText-2 raw"
            or token_source.get("split") != "train"
            or token_source.get("archive_member") != config.get("wikitext_member")
            or token_source.get("archive_sha256")
            != config.get("wikitext_archive_sha256")
            or token_source.get("corpus_windows_disjoint") is not True
        ):
            raise ProtocolError("WikiText source/artifact/window provenance failed")
    quality_windows = result.get("quality_windows")
    if not isinstance(quality_windows, list) or len(quality_windows) != int(
        config["batch_size"]
    ):
        raise ProtocolError("quality-window records do not cover every request")
    if [record.get("request") for record in quality_windows] != list(
        range(int(config["batch_size"]))
    ):
        raise ProtocolError("quality-window request indices are inconsistent")

    if expected_orchestration is not None:
        environment = _mapping(result.get("environment"), "worker environment")
        recorded = _mapping(
            environment.get("orchestration_environment"),
            "worker orchestration environment",
        )
        for key, expected in expected_orchestration.items():
            if recorded.get(key) != expected:
                raise ProtocolError(f"worker orchestration mismatch for {key}")

    samples = {mode: BASE.extract_mode_samples(result, mode) for mode in MODES}
    exact_tail_pages_policy = int(config["exact_tail_tokens"]) // PAGE
    mode_prefix_aggregates: dict[str, Mapping[str, Any]] = {}
    for mode in MODES:
        if any(len(samples[mode][metric]) != repeats for metric in METRICS):
            raise ProtocolError(f"{mode} raw sample count differs from repeats")
        payload = _mapping(result["timing_modes"][mode], f"timing_modes.{mode}")
        if (
            payload.get("restore_inside_timed_boundary") is not False
            or payload.get("precondition_inside_timed_boundary") is not False
            or payload.get("no_restore_between_hot_precondition_and_sample")
            is not True
            or payload.get("hot_precondition_steps") != PAGE
            or payload.get("planner_reset_without_cache_restore") is not True
        ):
            raise ProtocolError(f"timing_modes.{mode} boundary declaration failed")
        raw = payload.get("raw_samples")
        if not isinstance(raw, list) or len(raw) != repeats:
            raise ProtocolError(f"timing_modes.{mode}.raw_samples is incomplete")
        for index, record in enumerate(raw):
            record = _mapping(record, f"{mode}.raw_samples[{index}]")
            if record.get("sample_index") != index:
                raise ProtocolError(f"{mode} raw sample indices are not contiguous")
            gate = _mapping(
                record.get("runtime_gate"),
                f"{mode}.raw_samples[{index}].runtime_gate",
            )
            if gate.get("passed") is not True:
                raise ProtocolError(f"{mode} timed sample {index} runtime gate failed")
            validate_runtime_gate_record(
                gate,
                label=f"{mode}.raw_samples[{index}]",
                backend=expected_backend,
                tail_attention=str(config.get("tail_attention")),
                exact_prefix_pages=exact_prefix_pages,
                decode_steps=decode_steps,
            )
            validate_exact_prefix_canary_record(
                record.get("exact_prefix_canary"),
                label=f"{mode}.raw_samples[{index}].exact_prefix_canary",
                batch_size=int(config["batch_size"]),
                exact_tail_pages=exact_tail_pages_policy,
                exact_prefix_pages=exact_prefix_pages,
            )
            if record["exact_prefix_canary"].get("phase") != (
                f"{mode}.sample.{index}.sample"
            ):
                raise ProtocolError(f"{mode} raw exact-prefix phase is inconsistent")
            if record.get("exact_sink_canary") != record.get(
                "exact_prefix_canary"
            ):
                raise ProtocolError(
                    f"{mode} raw sample {index} exact-sink alias differs from prefix"
                )
            precondition = record.get("precondition")
            if mode == "cache_neutral":
                if precondition is not None:
                    raise ProtocolError("cache-neutral sample has a hot precondition")
            else:
                precondition_record = _mapping(
                    precondition, f"{mode}.raw_samples[{index}].precondition"
                )
                if (
                    precondition_record.get("precondition_steps") != PAGE
                    or precondition_record.get(
                        "planner_reset_without_cache_restore"
                    )
                    is not True
                    or precondition_record.get("no_restore_before_timed_sample")
                    is not True
                    or not isinstance(
                        precondition_record.get("exact_prefix_attestation"), str
                    )
                ):
                    raise ProtocolError(
                        f"{mode} timed sample {index} hot precondition is not literal"
                    )
                validate_runtime_gate_record(
                    _mapping(
                        precondition_record.get("runtime_gate"),
                        f"{mode}.raw_samples[{index}].precondition runtime gate",
                    ),
                    label=f"{mode}.raw_samples[{index}].precondition",
                    backend=expected_backend,
                    tail_attention=str(config.get("tail_attention")),
                    exact_prefix_pages=exact_prefix_pages,
                    decode_steps=PAGE,
                )
        warmups = payload.get("warmup_samples")
        if not isinstance(warmups, list) or len(warmups) != int(config["warmups"]):
            raise ProtocolError(f"timing_modes.{mode} warmup count is inconsistent")
        for index, raw_warmup in enumerate(warmups):
            warmup = _mapping(raw_warmup, f"{mode}.warmup_samples[{index}]")
            if warmup.get("warmup") != index:
                raise ProtocolError(f"{mode} warmup indices are not contiguous")
            validate_runtime_gate_record(
                _mapping(
                    warmup.get("runtime_gate"),
                    f"{mode}.warmup_samples[{index}].runtime gate",
                ),
                label=f"{mode}.warmup_samples[{index}]",
                backend=expected_backend,
                tail_attention=str(config.get("tail_attention")),
                exact_prefix_pages=exact_prefix_pages,
                decode_steps=decode_steps,
            )
            validate_exact_prefix_canary_record(
                warmup.get("exact_prefix_canary"),
                label=f"{mode}.warmup_samples[{index}].exact_prefix_canary",
                batch_size=int(config["batch_size"]),
                exact_tail_pages=exact_tail_pages_policy,
                exact_prefix_pages=exact_prefix_pages,
            )
            if warmup["exact_prefix_canary"].get("phase") != (
                f"{mode}.warmup.{index}.sample"
            ):
                raise ProtocolError(
                    f"{mode} warmup exact-prefix phase is inconsistent"
                )
            if warmup.get("exact_sink_canary") != warmup.get(
                "exact_prefix_canary"
            ):
                raise ProtocolError(
                    f"{mode} warmup {index} exact-sink alias differs from prefix"
                )
            precondition = warmup.get("precondition")
            if mode == "cache_neutral":
                if precondition is not None:
                    raise ProtocolError("cache-neutral warmup has a hot precondition")
            else:
                precondition_record = _mapping(
                    precondition, f"{mode}.warmup_samples[{index}].precondition"
                )
                if (
                    precondition_record.get("precondition_steps") != PAGE
                    or precondition_record.get(
                        "planner_reset_without_cache_restore"
                    )
                    is not True
                    or precondition_record.get("no_restore_before_timed_sample")
                    is not True
                    or not isinstance(
                        precondition_record.get("exact_prefix_attestation"), str
                    )
                ):
                    raise ProtocolError(
                        f"{mode} warmup {index} hot precondition is not literal"
                    )
                validate_runtime_gate_record(
                    _mapping(
                        precondition_record.get("runtime_gate"),
                        f"{mode}.warmup_samples[{index}].precondition runtime gate",
                    ),
                    label=f"{mode}.warmup_samples[{index}].precondition",
                    backend=expected_backend,
                    tail_attention=str(config.get("tail_attention")),
                    exact_prefix_pages=exact_prefix_pages,
                    decode_steps=PAGE,
                )
        prefix_aggregate = _mapping(
            payload.get("exact_prefix_canary"),
            f"timing_modes.{mode}.exact_prefix_canary",
        )
        validate_exact_prefix_canary_aggregate(
            prefix_aggregate,
            label=f"timing_modes.{mode}.exact_prefix_canary",
            expected_count=int(config["warmups"]) + repeats,
            batch_size=int(config["batch_size"]),
            exact_tail_pages=exact_tail_pages_policy,
            exact_prefix_pages=exact_prefix_pages,
        )
        expected_canary_checks = [
            warmup["exact_prefix_canary"] for warmup in warmups
        ] + [record["exact_prefix_canary"] for record in raw]
        if prefix_aggregate.get("checks") != expected_canary_checks:
            raise ProtocolError(
                f"timing_modes.{mode} exact-prefix check list differs from samples"
            )
        if payload.get("exact_sink_canary") != prefix_aggregate:
            raise ProtocolError(
                f"timing_modes.{mode} exact-sink alias differs from exact prefix"
            )
        mode_prefix_aggregates[mode] = prefix_aggregate
    timed_prefix = _mapping(
        same_backend.get("immutable_exact_prefix_timed_sample_canary"),
        "same-backend immutable exact-prefix timed canary",
    )
    expected_total_canary_checks = len(MODES) * (
        int(config["warmups"]) + repeats
    )
    if (
        timed_prefix.get("passed") is not True
        or timed_prefix.get("check_count") != expected_total_canary_checks
        or timed_prefix.get("per_mode") != mode_prefix_aggregates
        or timed_prefix.get("semantic_name") != "immutable_exact_prefix"
        or timed_prefix.get("legacy_exact_sink_alias") is not True
        or same_backend.get("immutable_exact_sink_timed_sample_canary")
        != {
            key: value
            for key, value in timed_prefix.items()
            if key not in {"semantic_name", "legacy_exact_sink_alias"}
        }
    ):
        raise ProtocolError("same-backend exact-prefix timed canary aggregate failed")
    worker_generated_token_hash(result)
    return samples


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProtocolError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON in {label}: {error}") from error
    return _mapping(value, label)


def validate_execution_integrity(
    manifest: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    run_dir: Path,
    *,
    publication: bool,
) -> dict[str, Mapping[str, Any]]:
    executions = _mapping(manifest.get("executions"), "manifest executions")
    expected_ids = {str(block["block_id"]) for block in schedule}
    if set(executions) != expected_ids:
        raise ProtocolError("manifest executions do not exactly cover the schedule")
    sessions_raw = manifest.get("orchestration_sessions")
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise ProtocolError("manifest has no orchestration-session provenance")
    session_starts: dict[str, datetime] = {}
    for index, raw_session in enumerate(sessions_raw):
        session = _mapping(raw_session, f"orchestration_sessions[{index}]")
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ProtocolError("orchestration session ID is missing")
        if session_id in session_starts:
            raise ProtocolError("orchestration session IDs are not unique")
        session_starts[session_id] = _utc_timestamp(
            session.get("started_utc"), f"orchestration session {session_id} start"
        )

    checked: dict[str, Mapping[str, Any]] = {}
    worker_pids: set[int] = set()
    prior_finished: datetime | None = None
    quartet_sessions: dict[tuple[int, int], set[str]] = defaultdict(set)
    quartet_block_counts: dict[tuple[int, int], int] = defaultdict(int)
    for chronological_index, block in enumerate(schedule):
        block_id = str(block["block_id"])
        execution = _mapping(executions.get(block_id), f"execution {block_id}")
        if (
            execution.get("status") != "completed"
            or execution.get("fresh_process_launched") is not True
            or execution.get("errors") != []
            or execution.get("return_code") != 0
            or execution.get("timeout_error") is not None
            or execution.get("idle_after_error") is not None
            or not isinstance(execution.get("worker_pid"), int)
            or isinstance(execution.get("worker_pid"), bool)
            or execution.get("worker_pid") <= 0
            or not isinstance(
                execution.get("process_wall_seconds_including_setup"), (int, float)
            )
            or execution.get("process_wall_seconds_including_setup") <= 0
        ):
            raise ProtocolError(f"execution {block_id} is not a clean fresh process")
        metadata = {
            "backend": block["backend"],
            "seed": block["seed"],
            "pair_id": block["pair_id"],
            "pair_order": block["pair_order"],
            "slot": block["slot"],
        }
        worker_pid = int(execution["worker_pid"])
        if worker_pid in worker_pids:
            raise ProtocolError("fresh-process worker PID was reused across blocks")
        worker_pids.add(worker_pid)
        for key, expected in metadata.items():
            if execution.get(key) != expected:
                raise ProtocolError(f"execution {block_id}.{key} disagrees with schedule")
        session_id = execution.get("orchestration_session_id")
        if not isinstance(session_id, str) or session_id not in session_starts:
            raise ProtocolError(f"execution {block_id} has an unknown session")
        started = _utc_timestamp(execution.get("started_utc"), f"{block_id} start")
        finished = _utc_timestamp(execution.get("finished_utc"), f"{block_id} finish")
        if started < session_starts[session_id] or finished < started:
            raise ProtocolError(f"execution {block_id} timestamp order is invalid")
        if prior_finished is not None and started < prior_finished:
            raise ProtocolError(
                "execution timestamps do not follow the canonical schedule; "
                "adjacent pairs may have been spliced across resume sessions"
            )
        if block.get("chronological_index") != chronological_index:
            raise ProtocolError("execution schedule chronology is not contiguous")
        prior_finished = finished
        for artifact in ("stdout", "stderr", "telemetry"):
            relative = execution.get(f"{artifact}_path")
            if not isinstance(relative, str):
                raise ProtocolError(f"execution {block_id} lacks {artifact}_path")
            path = (run_dir / relative).resolve()
            try:
                path.relative_to(run_dir)
            except ValueError as error:
                raise ProtocolError(
                    f"execution {block_id} {artifact} path escapes run directory"
                ) from error
            if (
                not path.is_file()
                or sha256_file(path) != execution.get(f"{artifact}_sha256")
            ):
                raise ProtocolError(
                    f"execution {block_id} {artifact} artifact hash mismatch"
                )
            if artifact == "telemetry":
                telemetry: list[Mapping[str, Any]] = []
                for line_index, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines()
                ):
                    if not line.strip():
                        continue
                    try:
                        telemetry.append(
                            _mapping(
                                json.loads(line),
                                f"{block_id} telemetry line {line_index}",
                            )
                        )
                    except json.JSONDecodeError as error:
                        raise ProtocolError(
                            f"execution {block_id} telemetry JSONL is invalid"
                        ) from error
                if execution.get("telemetry_samples") != len(telemetry):
                    raise ProtocolError(
                        f"execution {block_id} telemetry count is inconsistent"
                    )
                if publication and (
                    not telemetry
                    or not any("query_error" not in record for record in telemetry)
                ):
                    raise ProtocolError(
                        f"execution {block_id} has no successful GPU telemetry sample"
                    )
        if publication:
            for gate_name in ("idle_before", "idle_after"):
                gate = _mapping(
                    execution.get(gate_name), f"execution {block_id}.{gate_name}"
                )
                accepted = gate.get("accepted_snapshots")
                if (
                    gate.get("skipped") is True
                    or gate.get("failed") is True
                    or not isinstance(accepted, list)
                    or not accepted
                ):
                    raise ProtocolError(
                        f"execution {block_id} did not pass {gate_name}"
                    )
        quartet_key = (int(block["seed_index"]), int(block["quartet_index"]))
        quartet_sessions[quartet_key].add(session_id)
        quartet_block_counts[quartet_key] += 1
        checked[block_id] = execution
    for key, sessions in quartet_sessions.items():
        if quartet_block_counts[key] != 4 or len(sessions) != 1:
            raise ProtocolError(
                f"Williams quartet {key} was not executed atomically in one session"
            )
    return checked


def _result_path(
    run_dir: Path, manifest: Mapping[str, Any], block: Mapping[str, Any]
) -> Path:
    executions = _mapping(manifest.get("executions"), "manifest executions")
    execution = _mapping(
        executions.get(str(block["block_id"])),
        f"execution {block['block_id']}",
    )
    if execution.get("status") != "completed":
        raise ProtocolError(f"block {block['block_id']} is not completed")
    relative = execution.get("worker_result_path")
    if not isinstance(relative, str):
        raise ProtocolError("completed execution lacks worker_result_path")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as error:
        raise ProtocolError("worker result path escapes run directory") from error
    if sha256_file(path) != execution.get("worker_result_sha256"):
        raise ProtocolError(f"worker result hash mismatch for {block['block_id']}")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ProtocolError(f"refusing to write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run_directory(
    run_dir: Path,
    manifest_path: Path | None = None,
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 5090,
    write_outputs: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else run_dir / "orchestration_manifest.json"
    )
    manifest = _load_json(manifest_path, "orchestration manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("experiment") != MANIFEST_EXPERIMENT
        or manifest.get("status") != "completed"
    ):
        raise ProtocolError("manifest is not a completed sustained-v3 run")
    config = _mapping(manifest.get("config"), "manifest config")
    if config.get("cuda_graph_scope") != DYNAMIC_GRAPH_SCOPE:
        raise ProtocolError("manifest CUDA graph scope is incompatible")
    if canonical_json_sha256(config) != manifest.get("config_sha256"):
        raise ProtocolError("manifest config hash mismatch")
    validate_manifest_policy(config)
    if config.get("backend_symbols") != SYMBOL_TO_BACKEND:
        raise ProtocolError("manifest backend-symbol mapping is incompatible")
    if config.get("williams_sequences") != list(WILLIAMS_SEQUENCES):
        raise ProtocolError("manifest Williams sequence declaration is incompatible")
    orchestration_sources = _mapping(
        config.get("source_sha256"), "manifest orchestration source closure"
    )
    if set(orchestration_sources) != set(REQUIRED_ORCHESTRATION_SOURCES):
        raise ProtocolError("manifest orchestration source closure is incomplete")
    for name in REQUIRED_ORCHESTRATION_SOURCES:
        recorded_digest = _sha256(
            orchestration_sources.get(name), f"manifest source {name}"
        )
        local_path = (Path(__file__).resolve().parents[1] / name).resolve()
        if not local_path.is_file() or sha256_file(local_path) != recorded_digest:
            raise ProtocolError(
                f"local analyzer/orchestrator source differs from manifest: {name}"
            )
    expected_worker_sources = _mapping(
        config.get("worker_source_closure_sha256"),
        "manifest worker source closure",
    )
    if set(expected_worker_sources) != set(REQUIRED_WORKER_SOURCES):
        raise ProtocolError("manifest worker source closure is incomplete or ambiguous")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ProtocolError("manifest schedule is empty")
    profile = _mapping(config.get("profile"), "manifest profile")
    profile_name = profile.get("profile")
    if profile_name not in ("pilot", "publication"):
        raise ProtocolError("manifest profile must be pilot or publication")
    seeds = profile.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
    ):
        raise ProtocolError("paired sustained analysis requires at least two unique seeds")
    fixture_length = int(config["context"]) + int(config["decode_steps"])
    request_stride = int(config["token_stride"]) or fixture_length
    minimum_seed_stride = (
        (int(config["batch_size"]) - 1) * request_stride + fixture_length
    )
    if (
        config.get("token_source") == "wikitext2"
        and int(profile["seed_token_offset_stride"]) < minimum_seed_stride
    ):
        raise ProtocolError("WikiText seed fixture windows are not disjoint")
    if config.get("token_source") == "wikitext2":
        _sha256(
            config.get("wikitext_archive_sha256"),
            "manifest WikiText archive",
        )
    elif config.get("wikitext_archive_sha256") is not None:
        raise ProtocolError("random token source must not attest a WikiText archive")
    if profile_name == "publication":
        if config.get("token_source") != "wikitext2":
            raise ProtocolError(
                "publication profile requires the fixed WikiText-2 corpus artifact"
            )
        publication_minima = {
            "seeds": (len(seeds), 4),
            "pairs_per_seed": (int(profile["pairs_per_seed"]), 4),
            "warmups": (int(profile["warmups"]), 3),
            "repeats": (int(profile["repeats"]), 10),
            "bootstrap_samples": (int(profile["bootstrap_samples"]), 20_000),
        }
        for label, (observed, minimum) in publication_minima.items():
            if observed < minimum:
                raise ProtocolError(
                    f"publication {label} must be at least {minimum}, got {observed}"
                )
    if not isinstance(manifest.get("nvidia_smi_path"), str):
        raise ProtocolError("Williams manifest lacks nvidia-smi provenance")
    initial_idle = _mapping(
        manifest.get("initial_idle_baseline"), "initial GPU idle baseline"
    )
    if (
        not isinstance(initial_idle.get("accepted_snapshots"), list)
        or not initial_idle["accepted_snapshots"]
    ):
        raise ProtocolError("Williams initial GPU idle gate is incomplete")
    expected_schedule = build_williams_schedule(
        seeds,
        int(profile["pairs_per_seed"]),
        int(config["base_token_offset"]),
        int(profile["seed_token_offset_stride"]),
    )
    if schedule != expected_schedule:
        raise ProtocolError("manifest schedule is not the canonical Williams schedule")
    sequences = {str(block["williams_sequence"]) for block in schedule}
    if sequences != {"ABBA", "BAAB"}:
        raise ProtocolError("schedule does not contain both balanced Williams sequences")
    execution_records = validate_execution_integrity(
        manifest,
        schedule,
        run_dir,
        publication=True,
    )

    blocks: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    trajectories: dict[str, dict[str, str]] = defaultdict(dict)
    global_identities: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    for chronological_index, raw_block in enumerate(schedule):
        block = _mapping(raw_block, f"schedule[{chronological_index}]")
        if block.get("chronological_index") != chronological_index:
            raise ProtocolError("manifest chronological indices are not contiguous")
        backend = str(block.get("backend"))
        seed = block.get("seed")
        if backend not in BACKENDS or not isinstance(seed, int) or isinstance(seed, bool):
            raise ProtocolError("manifest block backend/seed is invalid")
        path = _result_path(run_dir, manifest, block)
        result = _load_json(path, f"worker result {block['block_id']}")
        execution = execution_records[str(block["block_id"])]
        environment = expected_orchestration_environment(
            str(manifest["config_sha256"]),
            block,
            str(execution["orchestration_session_id"]),
        )
        samples = validate_worker_result(
            result,
            backend,
            seed,
            environment,
            expected_worker_configuration(config, block),
            expected_worker_sources,
        )
        cache_build = _mapping(result.get("cache_build"), "worker cache build")
        cache_manifest = _mapping(
            cache_build.get("selected_backend_cache"),
            "selected backend cache manifest",
        )
        graph = _mapping(result.get("cuda_graph_provenance"), "graph provenance")
        graph_capture_memory = _mapping(
            graph.get("capture_memory"), "graph capture memory"
        )
        finalization = _mapping(
            _mapping(result.get("correctness"), "worker correctness").get(
                "runtime_page_finalization_and_consumption"
            ),
            "worker runtime finalization",
        )
        timed_prefix_canary = _mapping(
            _mapping(
                _mapping(result.get("correctness"), "worker correctness").get(
                    "same_backend_eager_vs_graph"
                ),
                "worker same-backend correctness",
            ).get("immutable_exact_prefix_timed_sample_canary"),
            "worker exact-prefix timed canary",
        )
        exact_layout = (
            _mapping(cache_manifest.get("exact_layout"), "candidate exact layout")
            if backend == CANDIDATE
            else {}
        )
        pair_id = str(block["pair_id"])
        identities[pair_id][backend] = worker_pair_identity(result)
        trajectories[pair_id][backend] = worker_generated_token_hash(result)
        global_identities[str(block["block_id"])] = global_stable_worker_identity(
            result
        )
        record = {
            "block_id": str(block["block_id"]),
            "chronological_index": chronological_index,
            "pair_id": pair_id,
            "pair_index": int(block["pair_index"]),
            "seed": seed,
            "seed_index": int(block["seed_index"]),
            "token_offset": int(block["token_offset"]),
            "pair_order": str(block["pair_order"]),
            "slot": int(block["slot"]),
            "backend": backend,
            "worker_pairing_key_sha256": identities[pair_id][backend][
                "pairing_key_sha256"
            ],
            "worker_exact_sink_pages": identities[pair_id][backend][
                "configuration"
            ]["exact_sink_pages"],
            "worker_exact_prefix_pages": identities[pair_id][backend][
                "configuration"
            ]["exact_prefix_pages"],
            "hf_quality_reference_sha256": identities[pair_id][backend][
                "quality_reference_sha256"
            ],
            "worker_result_path": str(path.relative_to(run_dir)),
            "worker_result_sha256": sha256_file(path),
            "served_cache_bytes": int(
                cache_build[
                    "selected_backend_cache_served_bytes_excluding_following_canary"
                ]
            ),
            "allocated_cache_bytes_including_canary": int(
                cache_manifest["total_bytes"]
            ),
            "exact_tail_pages_per_request": int(
                cache_build["exact_tail_pages_per_request"]
            ),
            "exact_prefix_pages_per_request": int(
                cache_build["exact_prefix_pages_per_request"]
            ),
            "exact_storage_pages_per_request": int(
                cache_build["exact_storage_pages_per_request"]
            ),
            "exact_prefix_gross_allocated_bytes": int(
                exact_layout.get("prefix_gross_allocated_bytes", 0)
            ),
            "runtime_generated_int8_pages_consumed_count": int(
                finalization["runtime_finalized_int8_pages_consumed_count"]
            ),
            "immutable_exact_prefix_timed_canary_check_count": int(
                timed_prefix_canary["check_count"]
            ),
            "graph_bank_count": int(graph["graph_bank_count"]),
            "graph_persistent_hidden_buffer_bytes": int(
                graph_capture_memory["persistent_hidden_buffer_bytes"]
            ),
            "graph_capture_peak_allocated_increment_bytes": int(
                graph_capture_memory[
                    "torch_peak_allocated_increment_over_capture_start_bytes"
                ]
            ),
            "samples": samples,
        }
        blocks[record["block_id"]] = record
        for mode in MODES:
            for metric in METRICS:
                for sample_index, latency in enumerate(samples[mode][metric]):
                    raw_rows.append(
                        {
                            **{
                                key: record[key]
                                for key in (
                                    "block_id",
                                    "chronological_index",
                                    "pair_id",
                                    "pair_index",
                                    "seed",
                                    "seed_index",
                                    "token_offset",
                                    "pair_order",
                                    "slot",
                                    "backend",
                                )
                            },
                            "mode": mode,
                            "metric": metric,
                            "sample_index": sample_index,
                            "latency_ms": latency,
                        }
                    )

    first_global_block = min(
        global_identities,
        key=lambda block_id: int(blocks[block_id]["chronological_index"]),
    )
    reference_global_identity = global_identities[first_global_block]
    for block_id, identity in global_identities.items():
        if identity != reference_global_identity:
            raise ProtocolError(
                "stable model/environment/token-artifact identity changed across "
                f"workers (first mismatch: {block_id})"
            )

    for pair_id, by_backend in identities.items():
        if set(by_backend) != set(BACKENDS):
            raise ProtocolError(f"pair {pair_id} does not contain both backends")
        if cross_backend_pair_identity(
            by_backend[BASELINE]
        ) != cross_backend_pair_identity(by_backend[CANDIDATE]):
            raise ProtocolError(
                f"pair {pair_id} has mismatched shared pairing/quality provenance"
            )
        if (
            by_backend[BASELINE]["configuration"].get("exact_sink_pages") != 0
            or by_backend[BASELINE]["configuration"].get("exact_prefix_pages") != 0
            or by_backend[CANDIDATE]["configuration"].get("exact_sink_pages")
            != POLICY_EXACT_PREFIX_PAGES
            or by_backend[CANDIDATE]["configuration"].get("exact_prefix_pages")
            != POLICY_EXACT_PREFIX_PAGES
        ):
            raise ProtocolError(
                f"pair {pair_id} does not attest FI-S0/PageGauge-S3 asymmetry"
            )
        trajectory_mode = by_backend[BASELINE]["configuration"]["trajectory_mode"]
        if (
            trajectory_mode == "greedy_feedback"
            and trajectories[pair_id][BASELINE] != trajectories[pair_id][CANDIDATE]
        ):
            raise ProtocolError(f"pair {pair_id} diverged on its greedy trajectory")

    quality_by_seed: dict[int, list[dict[str, Any]]] = {}
    clusters_by_seed: dict[int, set[str]] = {}
    for by_backend in identities.values():
        identity = by_backend[BASELINE]
        seed = int(identity["configuration"]["seed"])
        windows = identity["quality_windows"]
        if seed in quality_by_seed and quality_by_seed[seed] != windows:
            raise ProtocolError("pairs for one seed do not reuse one coherent fixture")
        quality_by_seed[seed] = windows
        clusters_by_seed[seed] = {
            str(record["cluster_unit_id"]) for record in windows
        }
    ordered_seeds = sorted(clusters_by_seed)
    for left_index, left_seed in enumerate(ordered_seeds):
        for right_seed in ordered_seeds[left_index + 1 :]:
            overlap = clusters_by_seed[left_seed] & clusters_by_seed[right_seed]
            if overlap:
                raise ProtocolError(
                    "seed fixtures reuse quality-window cluster units: "
                    + ", ".join(sorted(overlap))
                )
    train_cohort_records = []
    for seed in ordered_seeds:
        windows = quality_by_seed[seed]
        if any(record.get("dataset_split") != "train" for record in windows):
            raise ProtocolError("Williams cohort is not WikiText-2 TRAIN")
        cohort_start = min(
            int(record["corpus_window_start_offset"]) for record in windows
        )
        cohort_end = max(
            int(record["corpus_window_end_offset_exclusive"]) for record in windows
        )
        train_cohort_records.append(
            {
                "seed": seed,
                "dataset_split": "train",
                "corpus_start_offset": cohort_start,
                "corpus_end_offset_exclusive": cohort_end,
                "quality_window_cluster_unit_ids": sorted(clusters_by_seed[seed]),
            }
        )
    for left_index, left in enumerate(train_cohort_records):
        for right in train_cohort_records[left_index + 1 :]:
            if max(
                int(left["corpus_start_offset"]),
                int(right["corpus_start_offset"]),
            ) < min(
                int(left["corpus_end_offset_exclusive"]),
                int(right["corpus_end_offset_exclusive"]),
            ):
                raise ProtocolError("TRAIN seed cohorts overlap in corpus offsets")

    scheduled_pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw_block in schedule:
        block = _mapping(raw_block, "schedule block")
        scheduled_pairs[str(block["pair_id"])].append(block)

    pair_rows: list[dict[str, Any]] = []
    pair_memory_records: list[dict[str, Any]] = []
    for pair_id, pair_blocks in sorted(
        scheduled_pairs.items(),
        key=lambda item: min(int(block["chronological_index"]) for block in item[1]),
    ):
        if len(pair_blocks) != 2 or {str(block["backend"]) for block in pair_blocks} != set(BACKENDS):
            raise ProtocolError(f"pair {pair_id} is not one baseline/candidate pair")
        ordered = sorted(pair_blocks, key=lambda block: int(block["chronological_index"]))
        if int(ordered[1]["chronological_index"]) != int(ordered[0]["chronological_index"]) + 1:
            raise ProtocolError(f"pair {pair_id} is not temporally adjacent")
        order = "".join("A" if block["backend"] == BASELINE else "B" for block in ordered)
        if order != ordered[0]["pair_order"] or order not in ("AB", "BA"):
            raise ProtocolError(f"pair {pair_id} order metadata is inconsistent")
        by_backend = {
            str(block["backend"]): blocks[str(block["block_id"])]
            for block in pair_blocks
        }
        baseline_cache_bytes = int(by_backend[BASELINE]["served_cache_bytes"])
        candidate_cache_bytes = int(by_backend[CANDIDATE]["served_cache_bytes"])
        if candidate_cache_bytes >= baseline_cache_bytes:
            raise ProtocolError(
                f"pair {pair_id} does not demonstrate a real served-cache reduction"
            )
        pair_memory_records.append(
            {
                "pair_id": pair_id,
                "seed": int(ordered[0]["seed"]),
                "token_offset": int(ordered[0]["token_offset"]),
                "baseline_served_cache_bytes": baseline_cache_bytes,
                "candidate_served_cache_bytes": candidate_cache_bytes,
                "cache_compression_ratio": baseline_cache_bytes
                / candidate_cache_bytes,
                "cache_memory_reduction_fraction": 1.0
                - candidate_cache_bytes / baseline_cache_bytes,
                "candidate_cache_is_smaller": candidate_cache_bytes
                < baseline_cache_bytes,
                "candidate_exact_tail_pages_per_request": int(
                    by_backend[CANDIDATE]["exact_tail_pages_per_request"]
                ),
                "candidate_exact_prefix_pages_per_request": int(
                    by_backend[CANDIDATE]["exact_prefix_pages_per_request"]
                ),
                "candidate_exact_storage_pages_per_request": int(
                    by_backend[CANDIDATE]["exact_storage_pages_per_request"]
                ),
                "candidate_exact_prefix_gross_allocated_bytes": int(
                    by_backend[CANDIDATE][
                        "exact_prefix_gross_allocated_bytes"
                    ]
                ),
                "candidate_runtime_generated_int8_pages_consumed_count": int(
                    by_backend[CANDIDATE][
                        "runtime_generated_int8_pages_consumed_count"
                    ]
                ),
                "baseline_graph_bank_count": int(
                    by_backend[BASELINE]["graph_bank_count"]
                ),
                "candidate_graph_bank_count": int(
                    by_backend[CANDIDATE]["graph_bank_count"]
                ),
                "baseline_graph_persistent_hidden_buffer_bytes": int(
                    by_backend[BASELINE]["graph_persistent_hidden_buffer_bytes"]
                ),
                "candidate_graph_persistent_hidden_buffer_bytes": int(
                    by_backend[CANDIDATE]["graph_persistent_hidden_buffer_bytes"]
                ),
            }
        )
        for mode in MODES:
            for metric in METRICS:
                baseline = by_backend[BASELINE]["samples"][mode][metric]
                candidate = by_backend[CANDIDATE]["samples"][mode][metric]
                if len(baseline) != len(candidate):
                    raise ProtocolError(f"pair {pair_id} raw sample counts differ")
                baseline_log = statistics.fmean(math.log(value) for value in baseline)
                candidate_log = statistics.fmean(math.log(value) for value in candidate)
                log_speedup = baseline_log - candidate_log
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "pair_index": int(ordered[0]["pair_index"]),
                        "seed": int(ordered[0]["seed"]),
                        "seed_index": int(ordered[0]["seed_index"]),
                        "token_offset": int(ordered[0]["token_offset"]),
                        "pair_order": order,
                        "williams_sequence": str(ordered[0]["williams_sequence"]),
                        "quartet_index": int(ordered[0]["quartet_index"]),
                        "first_chronological_index": int(ordered[0]["chronological_index"]),
                        "trajectory_mode": config["trajectory_mode"],
                        "baseline_worker_pairing_key_sha256": by_backend[
                            BASELINE
                        ]["worker_pairing_key_sha256"],
                        "candidate_worker_pairing_key_sha256": by_backend[
                            CANDIDATE
                        ]["worker_pairing_key_sha256"],
                        "baseline_exact_prefix_pages": by_backend[BASELINE][
                            "worker_exact_prefix_pages"
                        ],
                        "candidate_exact_prefix_pages": by_backend[CANDIDATE][
                            "worker_exact_prefix_pages"
                        ],
                        "shared_hf_quality_reference_sha256": by_backend[
                            BASELINE
                        ]["hf_quality_reference_sha256"],
                        "baseline_generated_tokens_sha256": trajectories[pair_id][
                            BASELINE
                        ],
                        "candidate_generated_tokens_sha256": trajectories[pair_id][
                            CANDIDATE
                        ],
                        "generated_output_trajectories_identical": trajectories[
                            pair_id
                        ][BASELINE]
                        == trajectories[pair_id][CANDIDATE],
                        "mode": mode,
                        "metric": metric,
                        "baseline_sample_count": len(baseline),
                        "candidate_sample_count": len(candidate),
                        "baseline_latency_geomean_ms": math.exp(baseline_log),
                        "candidate_latency_geomean_ms": math.exp(candidate_log),
                        "log_speedup": log_speedup,
                        "speedup": math.exp(log_speedup),
                    }
                )

    chronology = sorted(
        {(row["pair_id"], row["first_chronological_index"]) for row in pair_rows},
        key=lambda item: item[1],
    )
    halves = {
        pair_id: "first_half" if index < len(chronology) / 2 else "second_half"
        for index, (pair_id, _position) in enumerate(chronology)
    }
    for row in pair_rows:
        row["chronology_half"] = halves[row["pair_id"]]

    aggregates: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        aggregates[mode] = {}
        for metric_index, metric in enumerate(METRICS):
            selected = [
                row for row in pair_rows
                if row["mode"] == mode and row["metric"] == metric
            ]
            logs = [float(row["log_speedup"]) for row in selected]
            if not logs:
                raise ProtocolError(f"no paired observations for {mode}/{metric}")
            seed_offset = mode_index * 1000 + metric_index * 100
            mean_log = statistics.fmean(logs)
            aggregates[mode][metric] = {
                "estimand": "exp(mean adjacent-pair log(FI_latency/PG_latency))",
                "pair_count": len(selected),
                "seed_cluster_count": len({row["seed"] for row in selected}),
                "mean_log_speedup": mean_log,
                "speedup_geomean": math.exp(mean_log),
                "paired_block_bootstrap": BASE._bootstrap_ci(
                    logs, bootstrap_samples, bootstrap_seed + seed_offset
                ),
                "hierarchical_seed_pair_bootstrap": BASE._hierarchical_bootstrap_ci(
                    selected, bootstrap_samples, bootstrap_seed + seed_offset + 50
                ),
                "order_stratified": BASE._stratified_geomean(selected, "pair_order"),
                "williams_sequence_stratified": BASE._stratified_geomean(
                    selected, "williams_sequence"
                ),
                "chronology_half_stratified": BASE._stratified_geomean(
                    selected, "chronology_half"
                ),
                "seed_stratified": BASE._stratified_geomean(selected, "seed"),
            }

    primary = aggregates["cache_neutral"]["wall_ms"]
    primary_ci = primary["hierarchical_seed_pair_bootstrap"]["speedup_95_ci"]
    cache_byte_pairs = {
        (
            int(record["baseline_served_cache_bytes"]),
            int(record["candidate_served_cache_bytes"]),
        )
        for record in pair_memory_records
    }
    if len(cache_byte_pairs) != 1:
        raise ProtocolError("served cache byte accounting changed across paired fixtures")
    baseline_cache_bytes, candidate_cache_bytes = next(iter(cache_byte_pairs))
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "experiment": ANALYSIS_EXPERIMENT,
        "passed": True,
        "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
        "claim_scope": (
            "fresh-process backend-exclusive continuous D>=512 decode; paired "
            "log latency ratio with Williams ABBA/BAAB order control; trajectory "
            "and token-source semantics are fixed by protocol_configuration"
        ),
        "protocol_configuration": {
            "policy_preset": config["policy_preset"],
            "profile": profile_name,
            "model": config["model"],
            "batch_size": config["batch_size"],
            "context": config["context"],
            "decode_steps": config["decode_steps"],
            "exact_tail_tokens": config["exact_tail"],
            "candidate_exact_sink_pages": config["exact_sink_pages"],
            "candidate_exact_prefix_pages": config["exact_prefix_pages"],
            "baseline_exact_prefix_pages": 0,
            "old_value_scale_placement": config[
                "old_value_scale_placement"
            ],
            "token_source": config["token_source"],
            "trajectory_mode": config["trajectory_mode"],
            "candidate_tail_attention": config["tail_attention"],
            "baseline_attention": "flashinfer_fp16_fa2",
            "candidate_attention": (
                "page_gauge_heterogeneous_int8_fp16_fa2"
                if config["tail_attention"] == "heterogeneous_fa2"
                else f"page_gauge_segmented_{config['tail_attention']}"
            ),
            "strict_min_logits_cosine": config["min_logits_cosine"],
            "strict_min_top1_agreement": config["min_top1_agreement"],
            "runtime_generated_pages_consumed_as_int8": (
                POLICY_GENERATED_INT8_PAGES
            ),
        },
        "primary_mode": "cache_neutral",
        "primary_metric": "wall_ms",
        "backend_symbols": {"A": BASELINE, "B": CANDIDATE},
        "manifest_path": str(manifest_path.relative_to(run_dir)),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_config_sha256": manifest["config_sha256"],
        "raw_sample_count": len(raw_rows),
        "paired_record_count": len(pair_rows),
        "train_seed_cohort_count": len(train_cohort_records),
        "train_seed_cohorts": train_cohort_records,
        "aggregates": aggregates,
        "target_attainment": {
            "point_estimate_at_least_1_10x": primary["speedup_geomean"] >= 1.10,
            "point_estimate_at_least_1_20x": primary["speedup_geomean"] >= 1.20,
            "hierarchical_ci_lower_at_least_1_10x": primary_ci[0] >= 1.10,
            "interpretation": (
                "target flags are descriptive gates; report the estimate and CI, "
                "including a miss, without changing the protocol"
            ),
        },
        "memory_evidence": {
            "scope": "selected backend KV-cache served pages; following canary excluded",
            "baseline_served_cache_bytes": baseline_cache_bytes,
            "candidate_served_cache_bytes": candidate_cache_bytes,
            "cache_compression_ratio": baseline_cache_bytes
            / candidate_cache_bytes,
            "cache_memory_reduction_fraction": 1.0
            - candidate_cache_bytes / baseline_cache_bytes,
            "candidate_cache_is_smaller": candidate_cache_bytes
            < baseline_cache_bytes,
            "pair_records": pair_memory_records,
        },
        "exact_prefix_publication_attestation": {
            "orchestrator_exact_prefix_integrated": True,
            "analyzer_exact_prefix_attestation_integrated": True,
            "worker_schema_version": SCHEMA_VERSION,
            "candidate_pairing_key_attests_exact_sink_pages": (
                POLICY_EXACT_PREFIX_PAGES
            ),
            "candidate_pairing_key_attests_exact_prefix_pages": (
                POLICY_EXACT_PREFIX_PAGES
            ),
            "baseline_representation_has_no_exact_prefix": True,
            "cross_backend_shared_identity_normalizes_only_s0_s3_representation": (
                True
            ),
            "exact_tail_tokens": POLICY_EXACT_TAIL_TOKENS,
            "generated_int8_recurrence_pages": POLICY_GENERATED_INT8_PAGES,
            "immutable_prefix_canaries_required_for_every_sample_and_warmup": True,
            "prefix_old_tail_disjoint_coverage_required": True,
            "gross_prefix_fp16_bytes_and_retained_int8_bytes_accounted": True,
        },
        "pair_records": pair_rows,
        "worker_result_records": [
            {key: value for key, value in record.items() if key != "samples"}
            for record in sorted(blocks.values(), key=lambda item: item["chronological_index"])
        ],
        "protocol_notes": {
            "experimental_unit": "adjacent fresh-process backend pair",
            "within_block_repeats_are_not_independent_units": True,
            "positive_speedup_direction": "greater than one favors PageGauge",
            "dynamic_and_fixed_window_graph_scopes_are_never_paired": True,
            "profile": profile_name,
            "publication_eligible": profile_name == "publication"
            and config["token_source"] == "wikitext2",
            "pilot_results_are_non_publication": profile_name == "pilot",
            "gpu_idle_and_telemetry_gates_required_for_pilot_and_publication": True,
            "distinct_nonoverlapping_train_seed_cohorts_required": True,
        },
    }
    analysis["analysis_payload_sha256"] = canonical_json_sha256(analysis)
    if write_outputs:
        output_path = output_path or run_dir / "publication_analysis.json"
        output_path.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(run_dir / "raw_latency_samples.csv", raw_rows)
        _write_csv(run_dir / "paired_log_speedups.csv", pair_rows)
    return analysis


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    try:
        analysis = analyze_run_directory(
            args.run_dir,
            args.manifest,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            output_path=args.output,
        )
    except (ProtocolError, OSError) as error:
        raise SystemExit(f"sustained paired analysis failed closed: {error}") from error
    primary = analysis["aggregates"]["cache_neutral"]["wall_ms"]
    interval = primary["hierarchical_seed_pair_bootstrap"]["speedup_95_ci"]
    print(
        f"Sustained cache-neutral wall speedup {primary['speedup_geomean']:.4f}x "
        f"(hierarchical 95% CI [{interval[0]:.4f}, {interval[1]:.4f}])."
    )


if __name__ == "__main__":
    main()
