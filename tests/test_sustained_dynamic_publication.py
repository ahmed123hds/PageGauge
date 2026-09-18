from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

import analyze_sustained_dynamic_graphs as ANALYZE  # noqa: E402
import orchestrate_sustained_dynamic_graphs as ORCHESTRATE  # noqa: E402


BATCH = 4
CONTEXT = 20480
STEPS = 1024
EXACT = 768
PREFIX = 3
TOKEN_OFFSET = 100000
TOKEN_STRIDE = 21984
FIXTURE = CONTEXT + STEPS
SEED_STRIDE = (BATCH - 1) * TOKEN_STRIDE + FIXTURE
ARCHIVE_SHA = "1" * 64
MEMBER_SHA = "2" * 64
MODEL_CONFIG_SHA = "3" * 64
MODEL_PARAMETERS_SHA = "4" * 64
FI_ABI = {"decode.py": "5" * 64, "scheduler.cuh": "6" * 64}
MODULE_SOURCE_SHA = "7" * 64


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def wrapper_names(backend: str) -> list[str]:
    return (
        ["baseline"]
        if backend == ANALYZE.BASELINE
        else ["old_int8", "exact_fp16"]
    )


def exact_prefix_pages(backend: str) -> int:
    return PREFIX if backend == ANALYZE.CANDIDATE else 0


def operation_counts(backend: str, steps: int = STEPS) -> dict[str, Any]:
    return {
        "decoder_plan_calls": steps,
        "device_position_fills": steps,
        "heterogeneous_page_table_updates": 0,
        "exact_page_table_updates": (
            steps // ANALYZE.PAGE if backend == ANALYZE.CANDIDATE else 0
        ),
        "wrappers": {
            name: {
                "plan_invocations": steps,
                "plan_rebuilds": steps // ANALYZE.PAGE,
                "last_page_len_device_fills": steps,
            }
            for name in wrapper_names(backend)
        },
    }


def runtime_gate(backend: str, steps: int = STEPS) -> dict[str, Any]:
    calls = 32 * steps
    wrappers = wrapper_names(backend)
    expected = {
        "decoder_plan_calls": steps,
        "device_position_fills": steps,
        "wrapper_count": len(wrappers),
        "plan_invocations_per_wrapper": steps,
        "plan_rebuilds_per_wrapper": steps // ANALYZE.PAGE,
        "last_page_len_device_fills_per_wrapper": steps,
        "decoder_layer_graph_replays": calls,
        "decoder_layer_eager_calls": 0,
        "graph_bank_misses": 0,
        "nested_attention_dispatch_calls": 0,
        "heterogeneous_page_table_updates": 0,
        "exact_page_table_updates": (
            steps // ANALYZE.PAGE if backend == ANALYZE.CANDIDATE else 0
        ),
    }
    return {
        "passed": True,
        "expected": expected,
        "observed_dispatch": {
            "graph_replays": calls,
            "eager_calls": 0,
            "graph_bank_misses": 0,
            "total_calls": calls,
        },
        "observed_nested_attention_dispatch": {
            "graph_replays": 0,
            "eager_calls": 0,
            "total_calls": 0,
        },
        "observed_operations": operation_counts(backend, steps),
        "wrapper_counts_passed": True,
    }


def planner_runtime_fixture(
    backend: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = []
    exact_pages = EXACT // ANALYZE.PAGE + PREFIX
    for position in range(CONTEXT, CONTEXT + STEPS, ANALYZE.PAGE):
        current_pages = (position + 1 + ANALYZE.PAGE - 1) // ANALYZE.PAGE
        old_pages = (
            current_pages - exact_pages if backend == ANALYZE.CANDIDATE else 0
        )
        wrapper_pages = (
            {"baseline": current_pages}
            if backend == ANALYZE.BASELINE
            else {"old_int8": old_pages, "exact_fp16": exact_pages}
        )
        states.append(
            {
                "position": position,
                "context": position + 1,
                "current_pages": current_pages,
                "old_pages": old_pages,
                "exact_pages": (
                    exact_pages if backend == ANALYZE.CANDIDATE else 1
                ),
                "wrappers": {
                    name: {
                        "pages_per_request": pages,
                        "kv_chunk_size_tokens": 4096,
                        "effective_valid_split_tiles": (
                            None if name == "exact_fp16" else 24
                        ),
                        "initialized_tile_count": 4
                        if name == "exact_fp16"
                        else 24,
                        "semantic_int_workspace_sha256": digest(
                            f"workspace-{name}-{pages // 128}"
                        ),
                    }
                    for name, pages in wrapper_pages.items()
                },
            }
        )
    summary = {}
    for name in wrapper_names(backend):
        active = [state["wrappers"][name]["pages_per_request"] for state in states]
        summary[name] = {
            "sampled_boundary_state_count": len(states),
            "active_pages_per_request_min": min(active),
            "active_pages_per_request_max": max(active),
            "kv_chunk_size_tokens_min": 4096,
            "kv_chunk_size_tokens_max": 4096,
            "effective_valid_split_tiles_min": None
            if name == "exact_fp16"
            else 24,
            "effective_valid_split_tiles_max": None
            if name == "exact_fp16"
            else 24,
        }
    return states, summary


def equivalence_bundle(backend: str) -> dict[str, Any]:
    boundaries = [
        {
            "logical_page": CONTEXT // ANALYZE.PAGE + index,
            "exact_ring_page": (
                (CONTEXT // ANALYZE.PAGE + index) % (EXACT // ANALYZE.PAGE)
                if backend == ANALYZE.CANDIDATE
                else None
            ),
            "bitwise_identical": True,
            "expected_combined_sha256": digest(f"boundary-{index}"),
            "observed_combined_sha256": digest(f"boundary-{index}"),
        }
        for index in range(STEPS // ANALYZE.PAGE)
    ]
    return {
        "logits": {
            "checked_steps": STEPS,
            "checked_request_steps": STEPS * BATCH,
            "bitwise_identical": True,
        },
        "generated_argmax_tokens": {
            "checked_steps": STEPS,
            "checked_request_steps": STEPS * BATCH,
            "bitwise_identical": True,
        },
        "full_mutated_cache_range": {
            "passed": True,
            "device_position_bitwise_identical": True,
            "full_pages_checked": BATCH * STEPS // ANALYZE.PAGE,
            "exact_ring_pages_checked": (
                BATCH * EXACT // ANALYZE.PAGE
                if backend == ANALYZE.CANDIDATE
                else 0
            ),
        },
        "every_page_close_cache_digest": {
            "passed": True,
            "checked_page_closes": STEPS // ANALYZE.PAGE,
            "records": boundaries,
        },
        "final_serving_metadata": {
            "passed": True,
            "wrappers": {
                name: {"bitwise_identical": True}
                for name in wrapper_names(backend)
            },
        },
    }


def canary_gates(backend: str) -> dict[str, Any]:
    expected_pages = {
        "preceding": CONTEXT // ANALYZE.PAGE - 1,
        "following": (CONTEXT + STEPS) // ANALYZE.PAGE,
    }
    hashes = {
        "preceding": digest("preceding-canary"),
        "following": digest("following-canary"),
    }
    if backend == ANALYZE.CANDIDATE:
        expected_pages["exact_prefix"] = None
        hashes["exact_prefix"] = digest("exact-prefix-canary")
    return {
        name: {
            "passed": True,
            "logical_pages": expected_pages,
            "expected_combined_sha256": hashes,
            "observed_combined_sha256": hashes,
        }
        for name in ("eager", "graph_run_1", "graph_run_2_restored")
    }


def exact_prefix_canary(backend: str, phase: str) -> dict[str, Any]:
    prefix_pages = exact_prefix_pages(backend)
    storage_pages = PREFIX + EXACT // ANALYZE.PAGE
    physical_pages = [
        request * storage_pages + page
        for request in range(BATCH)
        for page in range(prefix_pages)
    ]
    enabled = prefix_pages > 0
    tensor_hashes = (
        {
            "exact_key": digest("prefix-key"),
            "exact_value": digest("prefix-value"),
        }
        if enabled
        else {}
    )
    combined = digest("prefix-combined") if enabled else None
    return {
        "phase": phase,
        "enabled": enabled,
        "passed": True,
        "exact_prefix_pages": prefix_pages,
        "logical_pages": list(range(prefix_pages)),
        "physical_pages": physical_pages,
        "physical_page_count": len(physical_pages),
        "expected_combined_sha256": combined,
        "observed_combined_sha256": combined,
        "expected_tensor_sha256": tensor_hashes,
        "observed_tensor_sha256": tensor_hashes,
    }


def prefix_canary_aggregate(backend: str, mode: str) -> dict[str, Any]:
    check = exact_prefix_canary(backend, f"{mode}.sample.0.sample")
    return {
        "enabled": backend == ANALYZE.CANDIDATE,
        "passed": True,
        "check_count": 1,
        "checks": [check],
        "checked_after_terminal_synchronize_before_restore": True,
        "no_prefix_digest_between_hot_precondition_and_sample": True,
        "inside_timed_boundary": False,
    }


def token_provenance(seed: int, token_offset: int) -> dict[str, Any]:
    starts = [token_offset + request * TOKEN_STRIDE for request in range(BATCH)]
    return {
        "kind": "wikitext2",
        "dataset": "WikiText-2 raw",
        "split": "train",
        "archive_path": "/fixture/wikitext.zip",
        "archive_sha256": ARCHIVE_SHA,
        "archive_sha256_verified": True,
        "archive_member": "wikitext-2-raw/wiki.train.raw",
        "archive_member_sha256": MEMBER_SHA,
        "archive_member_crc32": "00000000",
        "archive_member_uncompressed_bytes": 1_000_000,
        "archive_member_compressed_bytes": 500_000,
        "tokenizer": {"class": "FixtureTokenizer", "sha256": digest("tokenizer")},
        "add_special_tokens": False,
        "bos_prepended_per_request": True,
        "bos_token_id": 1,
        "corpus_window_start_offsets": starts,
        "corpus_window_end_offsets_exclusive": [start + FIXTURE for start in starts],
        "corpus_window_stride": TOKEN_STRIDE,
        "corpus_windows_disjoint": True,
        "corpus_tokens_per_request": FIXTURE,
        "available_corpus_token_count": 10_000_000,
        "continuation_semantics": "fixture",
        "seed": seed,
        "token_ids_sha256": digest(f"tokens-{seed}-{token_offset}"),
    }


def quality_windows(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request": request,
            "cluster_unit_id": (
                f"wikitext2-train-{MEMBER_SHA[:16]}-"
                f"{start + CONTEXT}-{start + CONTEXT + STEPS}"
            ),
            "dataset_split": "train",
            "archive_member": source["archive_member"],
            "archive_member_sha256": MEMBER_SHA,
            "corpus_window_start_offset": start,
            "corpus_window_end_offset_exclusive": start + FIXTURE,
            "corpus_label_start_offset": start + CONTEXT,
            "corpus_label_end_offset_exclusive": start + CONTEXT + STEPS,
            "model_predicted_position_start": CONTEXT + 1,
            "model_predicted_position_end_exclusive": CONTEXT + STEPS + 1,
        }
        for request, start in enumerate(source["corpus_window_start_offsets"])
    ]


def worker_sources() -> dict[str, str]:
    return {name: digest(f"worker-source-{name}") for name in ANALYZE.REQUIRED_WORKER_SOURCES}


def cache_tensor_fixture(backend: str, served_pages: int) -> tuple[dict[str, Any], dict[str, int]]:
    layers, allocated_pages = 32, served_pages + 1
    full_pages = BATCH * allocated_pages
    exact_pages = BATCH * (EXACT // ANALYZE.PAGE + PREFIX)
    if backend == ANALYZE.BASELINE:
        specifications = {
            "key": ([layers, full_pages, 16, 8, 128], "torch.float16", 2),
            "value": ([layers, full_pages, 16, 8, 128], "torch.float16", 2),
        }
    else:
        specifications = {
            "exact_key": ([layers, exact_pages, 16, 8, 128], "torch.float16", 2),
            "exact_value": ([layers, exact_pages, 16, 8, 128], "torch.float16", 2),
            "key_codes": ([layers, full_pages, 16, 8, 128], "torch.int8", 1),
            "value_codes": ([layers, full_pages, 16, 8, 128], "torch.int8", 1),
            "key_scales": ([layers, full_pages, 8], "torch.float16", 2),
            "value_scales": ([layers, full_pages, 8], "torch.float16", 2),
            "key_center": ([layers, BATCH, 8, 128], "torch.float16", 2),
            "value_center": ([layers, BATCH, 8, 128], "torch.float16", 2),
            "output_center": ([layers, BATCH, 32, 128], "torch.float16", 2),
        }
    tensors = {
        name: {
            "shape": shape,
            "dtype": dtype,
            "bytes": math.prod(shape) * item_size,
            "data_ptr": 10_000 + index,
        }
        for index, (name, (shape, dtype, item_size)) in enumerate(
            specifications.items()
        )
    }
    manifest = {
        "backend": backend,
        "tensor_attributes": list(specifications),
        "tensors": tensors,
        "total_bytes": sum(record["bytes"] for record in tensors.values()),
        "all_storage_pointers_distinct": True,
    }
    derived = ANALYZE.validate_cache_tensor_manifest(
        manifest,
        backend=backend,
        batch_size=BATCH,
        served_pages=served_pages,
        exact_tail_pages=(
            EXACT // ANALYZE.PAGE if backend == ANALYZE.CANDIDATE else 0
        ),
        exact_prefix_pages=exact_prefix_pages(backend),
    )
    return manifest, derived


def worker_payload(
    backend: str,
    seed: int,
    token_offset: int,
    latency_ms: float,
    orchestration_environment: dict[str, str],
) -> dict[str, Any]:
    source = token_provenance(seed, token_offset)
    prefix_pages = exact_prefix_pages(backend)
    config = {
        "backend": backend,
        "model": "mistralai/Mistral-7B-v0.3",
        "model_revision": "fixture-revision",
        "batch_size": BATCH,
        "context": CONTEXT,
        "decode_steps": STEPS,
        "exact_tail_tokens": EXACT,
        "exact_sink_pages": prefix_pages,
        "exact_prefix_pages": prefix_pages,
        "prefill_chunk_tokens": 1024,
        "baseline_split_pages": 256,
        "candidate_split_pages": 256,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
        "trajectory_mode": "frozen_hf_teacher_forced",
        "cuda_graph_scope": ANALYZE.DYNAMIC_GRAPH_SCOPE,
        "token_source": "wikitext2",
        "wikitext_member": source["archive_member"],
        "wikitext_archive_sha256": ARCHIVE_SHA,
        "min_logits_cosine": 0.995,
        "min_top1_agreement": 0.99,
        "seed": seed,
        "token_offset": token_offset,
        "token_stride": TOKEN_STRIDE,
        "capture_warmups": 1,
        "maximum_graph_banks": 16,
        "warmups": 0,
        "repeats": 1,
        "cache_scrub_mib": 256,
        "quality_diagnostics_top_k": 0,
        "quality_diagnostics_top_vocab": 8,
    }
    pairing = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "backend",
            "quality_diagnostics_top_k",
            "quality_diagnostics_top_vocab",
        }
    }
    pairing.update(
        {
            "teacher_inputs_sha256": digest(f"teacher-{seed}-{token_offset}"),
            "token_matrix_sha256": digest(f"matrix-{seed}-{token_offset}"),
            "token_source_provenance_sha256": ANALYZE.canonical_json_sha256(source),
            "model_config_sha256": MODEL_CONFIG_SHA,
            "sampled_model_parameters_sha256": MODEL_PARAMETERS_SHA,
            "flashinfer_abi_source_sha256": FI_ABI,
        }
    )
    pages = STEPS // ANALYZE.PAGE
    first_page = CONTEXT // ANALYZE.PAGE
    exact_pages = EXACT // ANALYZE.PAGE
    consumed = (
        list(range(first_page, first_page + pages - exact_pages))
        if backend == ANALYZE.CANDIDATE
        else []
    )
    eager_dispatch = {
        "eager_calls": 32 * STEPS,
        "graph_replays": 0,
        "graph_bank_misses": 0,
        "total_calls": 32 * STEPS,
    }
    eager_attention = {
        "eager_calls": 32 * STEPS,
        "graph_replays": 0,
        "total_calls": 32 * STEPS,
    }
    graph_gate = runtime_gate(backend)
    neutral_prefix = prefix_canary_aggregate(backend, "cache_neutral")
    hot_prefix = prefix_canary_aggregate(backend, "cache_hot")
    timed_sink = {
        "passed": True,
        "check_count": 2,
        "per_mode": {
            "cache_neutral": neutral_prefix,
            "cache_hot": hot_prefix,
        },
    }
    same = {
        "passed": True,
        **equivalence_bundle(backend),
        "immutable_adjacent_page_canaries": canary_gates(backend),
        "immutable_exact_prefix_timed_sample_canary": {
            **timed_sink,
            "semantic_name": "immutable_exact_prefix",
            "legacy_exact_sink_alias": True,
        },
        "immutable_exact_sink_timed_sample_canary": timed_sink,
        "eager_dispatch": eager_dispatch,
        "eager_attention_dispatch": eager_attention,
        "eager_operations": operation_counts(backend),
        "eager_gate_passed": True,
        "graph_runtime_gate": copy.deepcopy(graph_gate),
        "restored_graph_repeat": {
            "passed": True,
            **equivalence_bundle(backend),
            "runtime_gate": copy.deepcopy(graph_gate),
            "restore_before_repeat": True,
            "no_restore_repeat_performed": False,
        },
    }
    served_pages = (CONTEXT + STEPS) // ANALYZE.PAGE
    cache_manifest, cache_bytes = cache_tensor_fixture(backend, served_pages)
    served_bytes = cache_bytes["served_bytes"]
    canary_bytes = cache_bytes["following_canary_bytes"]
    if backend == ANALYZE.CANDIDATE:
        cache_manifest["exact_layout"] = {
            "tail_pages_per_request": exact_pages,
            "prefix_pages_per_request": PREFIX,
            "sink_pages_per_request": PREFIX,
            "storage_pages_per_request": exact_pages + PREFIX,
            "physical_order": (
                "[fixed contiguous prefix slots 0..S-1, modulo tail-ring "
                "slots S..S+T-1]"
            ),
            "prefix_logical_pages": list(range(PREFIX)),
            "sink_logical_pages": list(range(PREFIX)),
            "prefix_int8_code_and_scale_storage_retained": True,
            "all_exact_prefix_int8_storage_retained": True,
            "page_zero_int8_storage_retained": True,
            "prefix_gross_allocated_bytes": cache_bytes[
                "exact_prefix_gross_allocated_bytes"
            ],
            "sink_gross_allocated_bytes": cache_bytes[
                "exact_prefix_gross_allocated_bytes"
            ],
            "compact_replacement_not_claimed": True,
        }
    if backend == ANALYZE.BASELINE:
        attention = {
            "implementation": "flashinfer_fp16_fa2",
            "logical_attention_segments": 1,
            "custom_module_uri": None,
            "custom_module_source_hashes": None,
            "old_int8_value_scale_placement": None,
            "exact_sink_pages": 0,
            "exact_prefix_pages": 0,
        }
        capacity_wrappers = {
            "baseline": {
                "maximum_active_pages_per_request": served_pages,
                "fixed_split_pages": 256,
                "analytically_within_capacity": True,
                "split_was_automatically_clamped": False,
            }
        }
    else:
        attention = {
            "implementation": "page_gauge_segmented_flashinfer_merge",
            "logical_attention_segments": 2,
            "custom_module_uri": "page_gauge_int8_fa2_v4_" + MODULE_SOURCE_SHA[:16],
            "custom_module_source_hashes": {
                "header_sha256": ANALYZE.EXPECTED_LEGACY_VENDOR_HEADER_SHA256,
                "expected_header_sha256": ANALYZE.EXPECTED_LEGACY_VENDOR_HEADER_SHA256,
                "header_matches_expected": True,
                "variant_sha256": digest("segmented-variant"),
                "module_source_sha256": MODULE_SOURCE_SHA,
            },
            "value_center_restoration": "attention_add",
            "old_int8_value_scale_placement": "probability",
            "exact_tail_pages": exact_pages,
            "exact_sink_pages": PREFIX,
            "exact_prefix_pages": PREFIX,
            "exact_segment_logical_order": (
                "[contiguous exact prefix logical pages 0..S-1, chronological "
                "recent exact tail]"
            ),
            "old_segment_logical_range": (
                "[S, tail_start) with every exact-prefix page excluded"
            ),
            "exact_physical_layout": (
                "fixed slots 0..S-1 followed by modulo tail-ring slots S..S+T-1"
            ),
            "prefix_uses_existing_exact_wrapper": True,
            "logical_attention_segments_unchanged_by_prefix": True,
            "sink_uses_existing_exact_wrapper": True,
            "logical_attention_segments_unchanged_by_sink": True,
        }
        capacity_wrappers = {
            "old_int8": {
                "maximum_active_pages_per_request": (
                    served_pages - exact_pages - PREFIX
                ),
                "fixed_split_pages": 256,
                "analytically_within_capacity": True,
                "split_was_automatically_clamped": False,
            },
            "exact_fp16": {
                "maximum_active_pages_per_request": exact_pages + PREFIX,
                "fixed_split_pages": 256,
                "analytically_within_capacity": True,
                "split_was_automatically_clamped": False,
            }
        }
    timed_work = {
        "continuous_decoder_steps": STEPS,
        "output_tokens": STEPS * BATCH,
        "decoder_plan_calls": STEPS,
        "wrapper_plan_invocations": STEPS * len(wrapper_names(backend)),
        "wrapper_last_page_len_device_fills": STEPS * len(wrapper_names(backend)),
        "device_position_fills": STEPS,
        "greedy_seed_token_device_copy": 0,
        "token_embedding_calls": STEPS,
        "embedding_to_graph_bank_input_device_copies": STEPS,
        "complete_layer_graph_replays": 32 * STEPS,
        "captured_persistent_layer_output_writes": 32 * STEPS,
        "final_model_norm_calls": STEPS,
        "lm_head_calls": STEPS,
        "gpu_argmax_operations": STEPS,
        "runtime_page_table_and_plan_updates_included": True,
        "prefix_plan_restored_before_timing": True,
        "first_generated_page_transition_included": True,
        "one_time_graph_bank_build_included": False,
        "planner_boundary_positions": list(range(CONTEXT, CONTEXT + STEPS, 16)),
        "host_planner_boundary_points": pages,
        "flashinfer_full_plan_host_barriers_per_wrapper": pages,
        "flashinfer_full_plan_calls_all_wrappers": pages
        * len(wrapper_names(backend)),
        "blocking_d2h_metadata_copies": 2
        * pages
        * len(wrapper_names(backend)),
        "host_synchronization_free_between_tokens": False,
        "no_host_token_id_readback": True,
    }
    raw_sample = {
        "sample_index": 0,
        "precondition": None,
        "exact_sink_canary": neutral_prefix["checks"][0],
        "exact_prefix_canary": neutral_prefix["checks"][0],
        "cuda_ms": latency_ms,
        "wall_ms": latency_ms,
        "runtime_gate": copy.deepcopy(graph_gate),
    }
    hot_sample = copy.deepcopy(raw_sample)
    hot_sample["exact_sink_canary"] = hot_prefix["checks"][0]
    hot_sample["exact_prefix_canary"] = hot_prefix["checks"][0]
    hot_sample["precondition"] = {
        "cuda_ms": latency_ms,
        "wall_ms": latency_ms,
        "runtime_gate": runtime_gate(backend, ANALYZE.PAGE),
        "precondition_steps": ANALYZE.PAGE,
        "planner_reset_without_cache_restore": True,
        "exact_prefix_attestation": "covered jointly by post-sample canary",
        "no_restore_before_timed_sample": True,
    }
    planner_states, planner_summary = planner_runtime_fixture(backend)
    return {
        "schema_version": 3,
        "experiment": ANALYZE.WORKER_EXPERIMENT,
        "passed": True,
        "backend": backend,
        "cuda_graph_scope": ANALYZE.DYNAMIC_GRAPH_SCOPE,
        "configuration": config,
        "pairing": {
            "configuration": pairing,
            "pairing_key_sha256": ANALYZE.canonical_json_sha256(pairing),
            "backend_excluded_from_key": True,
            "cuda_graph_scope_must_match_exactly": True,
            "fixed_window_decoder_layer_scope_is_incompatible": True,
        },
        "publication_orchestration_status": {
            "manual_worker_exact_prefix_supported": True,
            "manual_worker_exact_sink_supported": True,
            "williams_orchestrator_exact_prefix_integrated": False,
            "williams_analyzer_exact_prefix_attestation_integrated": False,
            "williams_orchestrator_exact_sink_integrated": False,
            "williams_analyzer_exact_sink_attestation_integrated": False,
            "publication_pairing_deferred_until_manual_known_row_strict_gate": True,
        },
        "trajectory": {
            "mode": "frozen_hf_teacher_forced",
            "generated_feedback": False,
            "frozen_identical_hf_input_chain": True,
            "gpu_argmax_timed_every_step": True,
            "argmax_fed_to_next_step": False,
            "cross_backend_pairing_requires_identical_generated_trajectory_hash": False,
            "cross_backend_pairing_requires_identical_frozen_input_chain_hash": True,
            "teacher_inputs_sha256": pairing["teacher_inputs_sha256"],
        },
        "timed_work": timed_work,
        "correctness": {
            "passed": True,
            "same_backend_eager_vs_graph": same,
            "backend_vs_hf_sdpa_fp16": {
                "passed": True,
                "logits": {
                    "checked_steps": STEPS,
                    "checked_request_steps": STEPS * BATCH,
                    "minimum_cosine": 0.999,
                    "top1_agreement_fraction": 0.999,
                },
                "generated_argmax_tokens": {
                    "checked_steps": STEPS,
                    "checked_request_steps": STEPS * BATCH,
                    "bitwise_identical": False,
                },
            },
            "runtime_page_finalization_and_consumption": {
                "passed": True,
                "first_generated_logical_page": first_page,
                "last_generated_logical_page": first_page + pages - 1,
                "generated_pages": pages,
                "runtime_finalized_pages_consumed_as_int8": consumed,
                "runtime_finalized_int8_pages_consumed_count": len(consumed),
                "final_attention_page_table_gate_passed": True,
                **(
                    {
                        "final_old_pages": served_pages - exact_pages - PREFIX,
                        "final_old_logical_page_end_exclusive": (
                            served_pages - exact_pages
                        ),
                        "final_old_logical_pages": list(
                            range(PREFIX, served_pages - exact_pages)
                        ),
                        "final_exact_prefix_logical_pages": list(range(PREFIX)),
                        "final_exact_sink_logical_pages": list(range(PREFIX)),
                        "final_exact_tail_logical_pages": list(
                            range(served_pages - exact_pages, served_pages)
                        ),
                        "old_attention_page_table_gate_passed": True,
                        "exact_attention_page_table_gate_passed": True,
                        "logical_page_sets_disjoint": True,
                        "logical_token_coverage_exactly_once": True,
                        "exact_prefix_excluded_from_old_segment": True,
                        "page_zero_excluded_from_old_segment": True,
                        "prefix_exclusion_gate_passed": True,
                        "sink_exclusion_gate_passed": True,
                    }
                    if backend == ANALYZE.CANDIDATE
                    else {"final_old_pages": None}
                ),
            },
            "hashes": {
                "hf_logits_sha256": digest(f"hf-logits-{seed}-{token_offset}"),
                "hf_generated_tokens_sha256": digest(
                    f"hf-tokens-{seed}-{token_offset}"
                ),
                "graph_generated_tokens_sha256": digest(
                    f"generated-{backend}-{seed}-{token_offset}"
                )
            },
        },
        "cuda_graph_provenance": {
            "enabled": True,
            "cuda_graph_scope": ANALYZE.DYNAMIC_GRAPH_SCOPE,
            "scope": "complete_decoder_layer_device_dynamic_position",
            "device_dynamic_position": True,
            "structure_gate_passed": True,
            "strict_missing_bucket_failure": True,
            "graph_bank_misses": 0,
            "nested_attention_graphs": False,
            "replays_per_decode_step": 32,
            "raw_cuda_graph_retained_after_instantiation": False,
            "preflight_position_count": STEPS,
            "preflight_range_start_inclusive_end_exclusive": [
                CONTEXT,
                CONTEXT + STEPS,
            ],
            "preflight_positions_sha256": digest("preflight"),
            "graph_bank_count": 1,
            "graphs_per_bank": 32,
            "total_graphs": 32,
            "graph_pools": 1,
            "preflight_bucket_ranges": [
                {
                    "first_position": CONTEXT,
                    "last_position": CONTEXT + STEPS - 1,
                    "position_count": STEPS,
                    "capture_position": CONTEXT,
                    "planner_runtime_states": planner_states,
                    "planner_runtime_summary": planner_summary,
                }
            ],
            "capture_memory": {
                "persistent_hidden_buffer_bytes": 1_000,
                "torch_allocated_delta_bytes": 2_000,
                "torch_reserved_delta_bytes": 3_000,
                "torch_peak_allocated_bytes": 4_000,
                "torch_peak_reserved_bytes": 5_000,
                "torch_peak_allocated_increment_over_capture_start_bytes": 2_500,
                "torch_peak_reserved_increment_over_capture_start_bytes": 3_500,
                "cuda_free_delta_bytes": -2_000,
                "cuda_consumed_delta_bytes": 2_000,
                "one_time_graph_bank_capture_wall_seconds": 0.1,
            },
            "one_time_exhaustive_preflight_and_graph_bank_build_wall_seconds": 0.2,
        },
        "scheduler_capacity": {
            "all_wrappers_analytically_within_capacity": True,
            "split_was_automatically_clamped": False,
            "wrappers": capacity_wrappers,
        },
        "attention_implementation": attention,
        "exclusivity": {
            "fresh_process_required": True,
            "selected_persistent_backend": backend,
            "opposite_backend_full_gpu_cache_allocated": False,
            "hf_dynamic_caches_released_before_decoder_construction": True,
            "cache_serialization_or_reload": False,
            "direct_destination_construction": True,
        },
        "cache_build": {
            "served_pages_per_request": served_pages,
            "allocated_pages_per_request": served_pages + 1,
            "initial_pages_per_request": CONTEXT // ANALYZE.PAGE,
            "mutated_generated_pages_per_request": pages,
            "exact_ring_pages_per_request": (
                exact_pages if backend == ANALYZE.CANDIDATE else 0
            ),
            "exact_tail_pages_per_request": (
                exact_pages if backend == ANALYZE.CANDIDATE else 0
            ),
            "exact_sink_pages_per_request": prefix_pages,
            "exact_prefix_pages_per_request": prefix_pages,
            "exact_storage_pages_per_request": (
                exact_pages + prefix_pages
                if backend == ANALYZE.CANDIDATE
                else 0
            ),
            "exact_physical_layout": (
                "slots0..S-1=fixed logical prefix; slotsS..S+T-1=tail ring"
                if backend == ANALYZE.CANDIDATE
                else "slots0..T-1=tail ring"
            ),
            "prefix_storage_accounting": (
                "gross FP16 allocation; prefix INT8 codes/scales remain allocated"
                if backend == ANALYZE.CANDIDATE
                else "no exact prefix allocation"
            ),
            "sink_storage_accounting": (
                "legacy alias: gross FP16 prefix allocation; prefix INT8 "
                "codes/scales remain allocated"
                if backend == ANALYZE.CANDIDATE
                else "legacy alias: no exact prefix allocation"
            ),
            "snapshot_offloaded_to_cpu_before_capture_and_timing": True,
            "full_served_mutation_range_restored_before_every_validation_and_sample": True,
            "preceding_and_following_page_canaries_checked": True,
            "following_canary_logical_page": served_pages,
            "selected_backend_cache": cache_manifest,
            "selected_backend_cache_served_bytes_excluding_following_canary": served_bytes,
            "following_canary_storage_bytes": canary_bytes,
            "opposite_backend_full_gpu_cache_allocated": False,
            "request_records": [
                {
                    "request": request,
                    "prefix_tokens": CONTEXT,
                    "reference_greedy_steps": STEPS,
                    "prefill_chunks": 20,
                    "chunk_tokens": 1024,
                    "direct_selected_backend_population": True,
                    "opposite_backend_full_cache_allocated": False,
                    "sampled_population_gate_passed": True,
                    "sampled_exact_prefix_population": {
                        "enabled": backend == ANALYZE.CANDIDATE,
                        "logical_pages": (
                            list(range(PREFIX))
                            if backend == ANALYZE.CANDIDATE
                            else []
                        ),
                        "sampled_layers": (
                            [0, 31] if backend == ANALYZE.CANDIDATE else []
                        ),
                        "all_prefix_pages_checked_in_every_layer": (
                            backend == ANALYZE.CANDIDATE
                        ),
                        "sampled_destination_sha256": (
                            digest(f"prefix-population-{request}")
                            if backend == ANALYZE.CANDIDATE
                            else None
                        ),
                    },
                }
                for request in range(BATCH)
            ],
        },
        "token_source": source,
        "quality_windows": quality_windows(source),
        "model_provenance": {
            "requested_name_or_path": config["model"],
            "resolved_revision": config["model_revision"],
            "config_sha256": MODEL_CONFIG_SHA,
            "sampled_parameters": {"sha256": MODEL_PARAMETERS_SHA},
            "parameter_count": 7_000_000_000,
        },
        "timing_modes": {
            "cache_neutral": {
                "raw_samples": [raw_sample],
                "warmup_samples": [],
                "restore_inside_timed_boundary": False,
                "precondition_inside_timed_boundary": False,
                "no_restore_between_hot_precondition_and_sample": True,
                "hot_precondition_steps": ANALYZE.PAGE,
                "planner_reset_without_cache_restore": True,
                "exact_sink_canary": neutral_prefix,
                "exact_prefix_canary": neutral_prefix,
            },
            "cache_hot": {
                "raw_samples": [hot_sample],
                "warmup_samples": [],
                "restore_inside_timed_boundary": False,
                "precondition_inside_timed_boundary": False,
                "no_restore_between_hot_precondition_and_sample": True,
                "hot_precondition_steps": ANALYZE.PAGE,
                "planner_reset_without_cache_restore": True,
                "exact_sink_canary": hot_prefix,
                "exact_prefix_canary": hot_prefix,
            },
        },
        "environment": {
            "gpu": "NVIDIA GeForce RTX 5090",
            "compute_capability": [12, 0],
            "torch": "2.fixture",
            "torch_cuda": "13.0",
            "flashinfer": "0.6.17",
            "flashinfer_abi": {"source_sha256": FI_ABI},
            "orchestration_environment": orchestration_environment,
        },
        "source_sha256": worker_sources(),
    }


def manifest_config() -> dict[str, Any]:
    profile = {
        "profile": "pilot",
        "seeds": [101, 102],
        "pairs_per_seed": 2,
        "warmups": 0,
        "repeats": 1,
        "bootstrap_samples": 100,
        "seed_token_offset_stride": SEED_STRIDE,
    }
    return {
        "policy_preset": ANALYZE.POLICY_PRESET,
        "profile": profile,
        "backend_symbols": dict(ANALYZE.SYMBOL_TO_BACKEND),
        "williams_sequences": list(ANALYZE.WILLIAMS_SEQUENCES),
        "fresh_process_per_backend_block": True,
        "cuda_graph_scope": ANALYZE.DYNAMIC_GRAPH_SCOPE,
        "model": "mistralai/Mistral-7B-v0.3",
        "batch_size": BATCH,
        "context": CONTEXT,
        "decode_steps": STEPS,
        "exact_tail": EXACT,
        "exact_sink_pages": PREFIX,
        "exact_prefix_pages": PREFIX,
        "prefill_chunk_tokens": 1024,
        "baseline_split_pages": 256,
        "candidate_split_pages": 256,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
        "trajectory_mode": "frozen_hf_teacher_forced",
        "capture_warmups": 1,
        "maximum_graph_banks": 16,
        "token_source": "wikitext2",
        "wikitext_zip": "/fixture/wikitext.zip",
        "wikitext_member": "wikitext-2-raw/wiki.train.raw",
        "wikitext_archive_sha256": ARCHIVE_SHA,
        "base_token_offset": TOKEN_OFFSET,
        "token_stride": TOKEN_STRIDE,
        "cache_scrub_mib": 256,
        "min_logits_cosine": 0.995,
        "min_top1_agreement": 0.99,
        "quality_diagnostics_top_k": 0,
        "quality_diagnostics_top_vocab": 8,
        "backend_conditioned_exact_prefix_policy": {
            "flashinfer_fp16": 0,
            "page_gauge": PREFIX,
            "candidate_exact_prefix_pages": PREFIX,
            "candidate_exact_sink_pages_legacy_alias": PREFIX,
            "reason": "fixture honest backend representation asymmetry",
        },
        "recurrence_attestation": {
            "page_tokens": ANALYZE.PAGE,
            "generated_pages": STEPS // ANALYZE.PAGE,
            "exact_tail_pages": EXACT // ANALYZE.PAGE,
            "runtime_generated_pages_consumed_as_int8": 16,
            "required_runtime_generated_pages_consumed_as_int8": 16,
            "passed": True,
        },
        "python_executable": sys.executable,
        "worker": str(ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py"),
        "worker_environment_overrides": {},
        "gpu_index": 0,
        "idle_gate": {},
        "telemetry_interval_ms": 500,
        "worker_timeout_s": 1_000,
        "bootstrap_seed": 17,
        "scheduler_capacity_policy": "fixture",
        "worker_source_closure_sha256": worker_sources(),
        "source_sha256": {
            name: ANALYZE.sha256_file(ROOT / name)
            for name in ANALYZE.REQUIRED_ORCHESTRATION_SOURCES
        },
    }


def write_synthetic_run(
    run_dir: Path,
    payload_mutator: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> Path:
    config = manifest_config()
    config_sha = ANALYZE.canonical_json_sha256(config)
    schedule = ANALYZE.build_williams_schedule(
        config["profile"]["seeds"],
        config["profile"]["pairs_per_seed"],
        config["base_token_offset"],
        config["profile"]["seed_token_offset_stride"],
    )
    session_ids = {0: "session-seed-0", 1: "session-seed-1"}
    sessions = [
        {
            "session_id": session_ids[index],
            "started_utc": (
                datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(minutes=index * 8)
            ).isoformat(),
        }
        for index in session_ids
    ]
    executions: dict[str, Any] = {}
    origin = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    for block in schedule:
        session_id = session_ids[int(block["seed_index"])]
        environment = ANALYZE.expected_orchestration_environment(
            config_sha, block, session_id
        )
        payload = worker_payload(
            str(block["backend"]),
            int(block["seed"]),
            int(block["token_offset"]),
            2.0 if block["backend"] == ANALYZE.BASELINE else 1.0,
            environment,
        )
        if payload_mutator is not None:
            payload_mutator(payload, block)
        block_dir = run_dir / "blocks" / str(block["block_id"])
        block_dir.mkdir(parents=True)
        result_path = block_dir / "worker_result.json"
        stdout_path = block_dir / "stdout.log"
        stderr_path = block_dir / "stderr.log"
        telemetry_path = block_dir / "telemetry.jsonl"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        stdout_path.write_text("fixture\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        telemetry_path.write_text(
            json.dumps({"gpu_utilization_percent": 1, "memory_used_mib": 1000})
            + "\n",
            encoding="utf-8",
        )
        started = origin + timedelta(minutes=int(block["chronological_index"]) * 2)
        finished = started + timedelta(minutes=1)
        relative = lambda path: str(path.relative_to(run_dir))
        executions[str(block["block_id"])] = {
            "status": "completed",
            "fresh_process_launched": True,
            "orchestration_session_id": session_id,
            "worker_pid": 1000 + int(block["chronological_index"]),
            "backend": block["backend"],
            "seed": block["seed"],
            "pair_id": block["pair_id"],
            "pair_order": block["pair_order"],
            "slot": block["slot"],
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "return_code": 0,
            "process_wall_seconds_including_setup": 60.0,
            "timeout_error": None,
            "idle_after_error": None,
            "idle_before": {"accepted_snapshots": [{"utilization_percent": 0}]},
            "idle_after": {"accepted_snapshots": [{"utilization_percent": 0}]},
            "errors": [],
            "worker_result_path": relative(result_path),
            "worker_result_sha256": ANALYZE.sha256_file(result_path),
            "stdout_path": relative(stdout_path),
            "stdout_sha256": ANALYZE.sha256_file(stdout_path),
            "stderr_path": relative(stderr_path),
            "stderr_sha256": ANALYZE.sha256_file(stderr_path),
            "telemetry_path": relative(telemetry_path),
            "telemetry_sha256": ANALYZE.sha256_file(telemetry_path),
            "telemetry_samples": 1,
        }
    manifest = {
        "schema_version": 3,
        "experiment": ANALYZE.MANIFEST_EXPERIMENT,
        "status": "completed",
        "config": config,
        "config_sha256": config_sha,
        "schedule": schedule,
        "executions": executions,
        "orchestration_sessions": sessions,
        "nvidia_smi_path": "/fixture/nvidia-smi",
        "initial_idle_baseline": {
            "accepted_snapshots": [{"utilization_percent": 0}]
        },
    }
    path = run_dir / "orchestration_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_local_williams_builder_matches_shared_orchestrator() -> None:
    arguments = ([11, 12, 13, 14], 4, TOKEN_OFFSET, SEED_STRIDE)
    assert ANALYZE.build_williams_schedule(*arguments) == ORCHESTRATE.BASE.build_schedule(
        *arguments
    )


def test_pilot_defaults_use_distinct_seed_fixtures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["orchestrator", "--output-dir", str(tmp_path), "--dry-run"]
    )
    args = ORCHESTRATE.parse_args()
    profile = ORCHESTRATE.resolved_profile(args)
    assert len(profile["seeds"]) == 2
    assert profile["seed_token_offset_stride"] == SEED_STRIDE
    assert args.decode_steps == STEPS
    assert args.exact_tail == EXACT
    assert args.exact_sink_pages == PREFIX
    assert args.tail_attention == "flashinfer_merge"
    assert args.old_value_scale_placement == "probability"
    assert args.trajectory_mode == "frozen_hf_teacher_forced"
    ORCHESTRATE.validate_args(args, profile)


def test_worker_commands_freeze_fi_s0_pg_s3_and_all_shared_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["orchestrator", "--output-dir", str(tmp_path), "--dry-run"]
    )
    args = ORCHESTRATE.parse_args()
    profile = ORCHESTRATE.resolved_profile(args)
    schedule = ANALYZE.build_williams_schedule(
        profile["seeds"],
        profile["pairs_per_seed"],
        args.token_offset,
        profile["seed_token_offset_stride"],
    )
    by_backend = {
        block["backend"]: ORCHESTRATE.worker_command(
            args, profile, block, tmp_path / f"{block['backend']}.json"
        )
        for block in schedule[:2]
    }

    def argument(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    assert argument(by_backend[ANALYZE.BASELINE], "--exact-sink-pages") == "0"
    assert argument(by_backend[ANALYZE.CANDIDATE], "--exact-sink-pages") == "3"
    for command in by_backend.values():
        assert argument(command, "--decode-steps") == "1024"
        assert argument(command, "--exact-tail") == "768"
        assert argument(command, "--tail-attention") == "flashinfer_merge"
        assert argument(command, "--old-value-scale-placement") == "probability"
        assert argument(command, "--trajectory-mode") == "frozen_hf_teacher_forced"
        assert argument(command, "--token-stride") == "21984"


def test_end_to_end_synthetic_manifest_is_exactly_two_x(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path)
    result = ANALYZE.analyze_run_directory(
        tmp_path,
        bootstrap_samples=50,
        bootstrap_seed=19,
        write_outputs=False,
    )
    assert result["aggregates"]["cache_neutral"]["wall_ms"][
        "speedup_geomean"
    ] == pytest.approx(2.0)
    _baseline_manifest, baseline_bytes = cache_tensor_fixture(
        ANALYZE.BASELINE, (CONTEXT + STEPS) // ANALYZE.PAGE
    )
    _candidate_manifest, candidate_bytes = cache_tensor_fixture(
        ANALYZE.CANDIDATE, (CONTEXT + STEPS) // ANALYZE.PAGE
    )
    assert result["memory_evidence"]["cache_compression_ratio"] == pytest.approx(
        baseline_bytes["served_bytes"] / candidate_bytes["served_bytes"]
    )
    assert result["protocol_notes"]["pilot_results_are_non_publication"] is True
    assert result["train_seed_cohort_count"] == 2
    assert result["exact_prefix_publication_attestation"][
        "generated_int8_recurrence_pages"
    ] == 16
    assert {
        row["candidate_exact_prefix_pages"] for row in result["pair_records"]
    } == {3}


def test_following_canary_tamper_fails_even_when_parent_passed() -> None:
    block = ANALYZE.build_williams_schedule([101, 102], 2, 0, BATCH * FIXTURE)[0]
    payload = worker_payload(
        ANALYZE.BASELINE,
        101,
        0,
        2.0,
        ANALYZE.expected_orchestration_environment("a" * 64, block, "session"),
    )
    payload["correctness"]["same_backend_eager_vs_graph"][
        "immutable_adjacent_page_canaries"
    ]["graph_run_1"]["passed"] = False
    with pytest.raises(ANALYZE.ProtocolError, match="adjacent page canary"):
        ANALYZE.validate_worker_result(payload, ANALYZE.BASELINE, 101)


def test_module_expected_header_tamper_fails_closed() -> None:
    block = ANALYZE.build_williams_schedule([101, 102], 2, 0, BATCH * FIXTURE)[1]
    payload = worker_payload(
        ANALYZE.CANDIDATE,
        101,
        0,
        1.0,
        ANALYZE.expected_orchestration_environment("a" * 64, block, "session"),
    )
    hashes = payload["attention_implementation"]["custom_module_source_hashes"]
    hashes["header_matches_expected"] = False
    with pytest.raises(ANALYZE.ProtocolError, match="vendor header"):
        ANALYZE.validate_worker_result(payload, ANALYZE.CANDIDATE, 101)


def test_candidate_pairing_key_must_hash_s3_aliases() -> None:
    block = ANALYZE.build_williams_schedule(
        [101, 102], 2, TOKEN_OFFSET, SEED_STRIDE
    )[1]
    payload = worker_payload(
        ANALYZE.CANDIDATE,
        101,
        TOKEN_OFFSET,
        1.0,
        ANALYZE.expected_orchestration_environment("a" * 64, block, "session"),
    )
    pairing = payload["pairing"]["configuration"]
    pairing["exact_prefix_pages"] = 2
    payload["pairing"]["pairing_key_sha256"] = ANALYZE.canonical_json_sha256(
        pairing
    )
    with pytest.raises(ANALYZE.ProtocolError, match="pairing key.*exact-prefix"):
        ANALYZE.validate_worker_result(payload, ANALYZE.CANDIDATE, 101)


def test_worker_manual_integration_status_must_match_attested_source() -> None:
    block = ANALYZE.build_williams_schedule(
        [101, 102], 2, TOKEN_OFFSET, SEED_STRIDE
    )[1]
    payload = worker_payload(
        ANALYZE.CANDIDATE,
        101,
        TOKEN_OFFSET,
        1.0,
        ANALYZE.expected_orchestration_environment("a" * 64, block, "session"),
    )
    payload["publication_orchestration_status"][
        "publication_pairing_deferred_until_manual_known_row_strict_gate"
    ] = False
    with pytest.raises(ANALYZE.ProtocolError, match="capability/integration"):
        ANALYZE.validate_worker_result(payload, ANALYZE.CANDIDATE, 101)


def test_timed_exact_prefix_canary_tamper_fails_closed() -> None:
    block = ANALYZE.build_williams_schedule(
        [101, 102], 2, TOKEN_OFFSET, SEED_STRIDE
    )[1]
    payload = worker_payload(
        ANALYZE.CANDIDATE,
        101,
        TOKEN_OFFSET,
        1.0,
        ANALYZE.expected_orchestration_environment("a" * 64, block, "session"),
    )
    payload["timing_modes"]["cache_neutral"]["raw_samples"][0][
        "exact_prefix_canary"
    ]["observed_combined_sha256"] = digest("tampered-prefix")
    with pytest.raises(ANALYZE.ProtocolError, match="exact-prefix tensor digest"):
        ANALYZE.validate_worker_result(payload, ANALYZE.CANDIDATE, 101)


def test_resume_configuration_body_and_worker_hash_fail_closed(tmp_path: Path) -> None:
    config = manifest_config()
    schedule = ANALYZE.build_williams_schedule(
        config["profile"]["seeds"],
        config["profile"]["pairs_per_seed"],
        config["base_token_offset"],
        config["profile"]["seed_token_offset_stride"],
    )
    path, _manifest = ORCHESTRATE.prepare_manifest(
        tmp_path, config, schedule, resume=False, dry_run=True
    )
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["config"]["exact_prefix_pages"] = 2
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ANALYZE.ProtocolError, match="resume configuration"):
        ORCHESTRATE.prepare_manifest(
            tmp_path, config, schedule, resume=True, dry_run=True
        )

    result_path = tmp_path / "worker_result.json"
    result_path.write_text("fixture", encoding="utf-8")
    execution = {
        "worker_result_path": "worker_result.json",
        "worker_result_sha256": ORCHESTRATE.sha256_file(result_path),
    }
    result_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ANALYZE.ProtocolError, match="resume hash check"):
        ORCHESTRATE.validate_resume_worker_result_hash(
            tmp_path, schedule[0], execution
        )


def test_global_model_identity_change_across_seed_fails_closed(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any], block: dict[str, Any]) -> None:
        if block["seed_index"] != 1:
            return
        payload["configuration"]["model_revision"] = "changed-revision"
        pairing = payload["pairing"]["configuration"]
        pairing["model_revision"] = "changed-revision"
        payload["pairing"]["pairing_key_sha256"] = ANALYZE.canonical_json_sha256(
            pairing
        )
        payload["model_provenance"]["resolved_revision"] = "changed-revision"

    write_synthetic_run(tmp_path, mutate)
    with pytest.raises(ANALYZE.ProtocolError, match="stable model/environment"):
        ANALYZE.analyze_run_directory(
            tmp_path,
            bootstrap_samples=20,
            bootstrap_seed=3,
            write_outputs=False,
        )


def test_execution_return_code_tamper_fails_closed(tmp_path: Path) -> None:
    manifest_path = write_synthetic_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["schedule"][0]["block_id"]
    manifest["executions"][first]["return_code"] = 9
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ANALYZE.ProtocolError, match="clean fresh process"):
        ANALYZE.analyze_run_directory(
            tmp_path,
            bootstrap_samples=20,
            bootstrap_seed=3,
            write_outputs=False,
        )
