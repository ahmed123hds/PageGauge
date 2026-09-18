#!/usr/bin/env python3
"""CPU-only synthetic tests for the untouched held-out quality protocol."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AGG = load_module(
    "page_gauge_heldout_quality_test_module",
    ROOT / "diagnostics/aggregate_heldout_quality.py",
)
PREFILL = load_module(
    "page_gauge_prefill_canary_test_module",
    ROOT / "diagnostics/benchmark_model_prefill_correctness.py",
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def comparison_payload(
    batch_size: int,
    *,
    nll_delta: float = 0.0,
    forward_kl: float = 0.0,
    js: float = 0.0,
    cosine: float = 1.0,
    top1: bool = True,
) -> dict:
    top1_rows = [[top1 for _ in range(batch_size)] for _ in range(AGG.DECODE_STEPS)]
    rows = []
    for step in range(AGG.DECODE_STEPS):
        for request in range(batch_size):
            reference_top1 = 10 + (step + request) % 100
            candidate_top1 = reference_top1 if top1 else reference_top1 + 1
            rows.append(
                {
                    "step": step,
                    "request": request,
                    "input_position": AGG.CONTEXT + step,
                    "predicted_position": AGG.CONTEXT + step + 1,
                    "true_token_id": reference_top1,
                    "reference_nll_nats": 1.0,
                    "candidate_nll_nats": 1.0 + nll_delta,
                    "nll_delta_nats": nll_delta,
                    "candidate_to_reference_token_perplexity_ratio": math.exp(
                        nll_delta
                    ),
                    "forward_kl_nats": forward_kl,
                    "jensen_shannon_nats": js,
                    "total_variation": 0.0,
                    "reference_true_token_rank": 1,
                    "candidate_true_token_rank": 1,
                    "reference_top1_token_id": reference_top1,
                    "candidate_top1_token_id": candidate_top1,
                    "top1_agreement": top1,
                }
            )
    return {
        "checked_decode_steps": AGG.DECODE_STEPS,
        "checked_request_steps": batch_size * AGG.DECODE_STEPS,
        "minimum_logits_cosine": cosine,
        "worst_logits_cosine_location": {
            "step": 0,
            "request": 0,
            "value": cosine,
        },
        "top1_agreement_fraction": float(top1),
        "logits_cosine_by_step_request": [
            [cosine for _ in range(batch_size)] for _ in range(AGG.DECODE_STEPS)
        ],
        "top1_agreement_by_step_request": top1_rows,
        "distribution_quality": {
            "serialized_full_vocabulary_logits_or_probabilities": False,
            "label_count": batch_size * AGG.DECODE_STEPS,
            "per_token_metrics_step_major": rows,
        },
    }


def source_hashes() -> dict[str, str]:
    return {
        name: AGG.sha256_file(ROOT / Path(name)) for name in AGG.EXPECTED_SOURCE_KEYS
    }


def memory_payload(batch_size: int) -> dict:
    expected = AGG.expected_co_resident_memory(batch_size)
    free = expected["required_free_bytes"] + 1024**3
    return {
        "enabled": True,
        "passed": True,
        "policy": "model + matched FP16 cache + PageGauge cache + fixed reserve",
        "max_batch_size": AGG.MAX_CORESIDENT_BATCH,
        "observed_cuda_free_bytes_before_model_load": free,
        "model_fp16_resident_bytes": expected["model_fp16_resident_bytes"],
        "flashinfer_fp16_cache_bytes": expected["flashinfer_fp16_cache_bytes"],
        "page_gauge_cache_bytes": expected["page_gauge_cache_bytes"],
        "page_gauge_components": {
            key: expected[key]
            for key in (
                "int8_codes_bytes",
                "fp16_scales_bytes",
                "fp16_centers_and_output_center_bytes",
                "fp16_exact_prefix_and_tail_bytes",
            )
        },
        "projected_co_resident_bytes": expected["projected_co_resident_bytes"],
        "fixed_workspace_allocator_reserve_bytes": expected[
            "fixed_workspace_allocator_reserve_bytes"
        ],
        "required_free_bytes": expected["required_free_bytes"],
        "headroom_after_required_bytes": free - expected["required_free_bytes"],
        "logical_pages_per_request": expected["logical_pages_per_request"],
        "exact_tail_pages": expected["exact_tail_pages"],
        "exact_prefix_pages": expected["exact_prefix_pages"],
    }


def prefix_canary(batch_size: int, label: str) -> dict:
    storage_pages = AGG.EXACT_PREFIX_PAGES + AGG.EXACT_TAIL // AGG.PAGE
    mapping = [
        [request * storage_pages + logical for logical in range(3)]
        for request in range(batch_size)
    ]
    combined = digest(f"prefix-{label}")
    snapshot = {
        "mapping": mapping,
        "key_sha256": digest(f"key-{label}"),
        "value_sha256": digest(f"value-{label}"),
        "combined_sha256": combined,
        "bytes_hashed": batch_size * 32 * 3 * 16 * 8 * 128 * 2 * 2,
    }
    return {
        "passed": True,
        "exact_prefix_pages": 3,
        "exact_tail_pages": 48,
        "storage_pages_per_request": 51,
        "population": {
            "enabled": True,
            "passed": True,
            "key_bitwise_identical": True,
            "value_bitwise_identical": True,
            "expected_combined_sha256": combined,
            "observed": copy.deepcopy(snapshot),
        },
        "immutability": {
            "passed": True,
            "fixed_slots_unchanged": True,
            "before": copy.deepcopy(snapshot),
            "after": copy.deepcopy(snapshot),
        },
        "page_tables": {
            "enabled": True,
            "passed": True,
            "expected_prefix_physical_pages_by_request": copy.deepcopy(mapping),
            "logical_to_exact_table_prefix": copy.deepcopy(mapping),
            "active_exact_plan_table_prefix": copy.deepcopy(mapping),
            "exact_page_table_updates": 96,
            "final_layout_signature": [1328, 1376],
        },
    }


def group_payload(
    starts: tuple[int, ...], *, page_gauge_nll_delta: float = 0.0
) -> dict:
    batch_size = len(starts)
    group_index = AGG.EXPECTED_GROUPS.index(starts)
    ends = [start + AGG.WINDOW_CORPUS_TOKENS for start in starts]
    member_sha = "b" * 64
    windows = [
        {
            "request": request,
            "cluster_unit_id": f"test-window-{start}",
            "dataset_split": AGG.EXPECTED_SPLIT,
            "archive_member": AGG.EXPECTED_MEMBER,
            "archive_member_sha256": member_sha,
            "corpus_window_start_offset": start,
            "corpus_window_end_offset_exclusive": start + AGG.WINDOW_CORPUS_TOKENS,
            "corpus_label_start_offset": start + AGG.CONTEXT,
            "corpus_label_end_offset_exclusive": start + AGG.CONTEXT + AGG.DECODE_STEPS,
            "model_predicted_position_start": AGG.CONTEXT + 1,
            "model_predicted_position_end_exclusive": AGG.CONTEXT
            + AGG.DECODE_STEPS
            + 1,
        }
        for request, start in enumerate(starts)
    ]
    memory = memory_payload(batch_size)
    exact_prefix = prefix_canary(batch_size, str(starts[0]))
    finalization = {
        "passed": True,
        "canary_initialized": True,
        "future_pages_per_request": 96,
        "future_physical_pages": batch_size * 96,
        "completed_pages_per_request": 96,
        "partial_page_tokens": 0,
        "future_capacity_pages_per_request": 96,
        "all_completed_code_slots_overwritten": True,
        "all_completed_scales_finite_positive": True,
        "partial_unfinalized_page_retained_canary": True,
        "first_decode_step_consuming_runtime_finalized_old_page": 768,
        "decode_steps_consuming_runtime_finalized_old_pages": 768,
        "request_token_outputs_consuming_runtime_finalized_old_pages": batch_size * 768,
        "runtime_finalized_pages_in_old_segment_at_endpoint": 48,
        "runtime_generated_int8_logical_page_count": 48,
        "runtime_generated_int8_logical_page_range": [1280, 1328],
        "generated_completed_logical_page_range": [1280, 1376],
        "endpoint_prefix_logical_page_range": [0, 3],
        "endpoint_old_logical_page_range": [3, 1328],
        "endpoint_tail_logical_page_range": [1328, 1376],
        "endpoint_prefix_pages_per_request": 3,
        "endpoint_old_pages_per_request": 1325,
        "endpoint_tail_pages_per_request": 48,
        "endpoint_partition_coverage_disjoint": True,
        "runtime_finalized_layer_pages_total": batch_size * 32 * 96,
    }
    thresholds = {
        "page_gauge_minimum_logits_cosine": 0.995,
        "page_gauge_minimum_top1_agreement": 0.99,
        "flashinfer_minimum_hf_logits_cosine": 0.999,
    }
    disabled = {
        "enabled": False,
        "completed": False,
        "passed": None,
        "result": None,
        "error": None,
    }
    heldout = {
        "enabled": True,
        "name": AGG.PROTOCOL_NAME,
        "passed": True,
        "preregistered_before_test_access": True,
        "selected_on_split": "train",
        "confirmation_split": "test",
        "all_window_start_offsets": list(AGG.EXPECTED_STARTS),
        "all_window_end_offsets_exclusive": [
            start + AGG.WINDOW_CORPUS_TOKENS for start in AGG.EXPECTED_STARTS
        ],
        "this_shard_start_offsets": list(starts),
        "this_shard_index": group_index,
        "shard_batch_sizes": [3, 3, 3, 3, 2],
        "window_stride": AGG.WINDOW_STRIDE,
        "window_tokens": AGG.WINDOW_CORPUS_TOKENS,
        "last_window_end_offset_exclusive": AGG.EXPECTED_LAST_END,
        "test_tokens_remaining_after_last_window": (
            AGG.EXPECTED_TEST_TOKEN_COUNT - AGG.EXPECTED_LAST_END
        ),
        "source_content_lock": {
            "archive_sha256": AGG.EXPECTED_ARCHIVE_SHA256,
            "archive_member": AGG.EXPECTED_MEMBER,
            "available_corpus_token_count": AGG.EXPECTED_TEST_TOKEN_COUNT,
            "tokenizer_manifest_sha256": AGG.EXPECTED_TOKENIZER_MANIFEST_SHA256,
            "archive_hash_cryptographically_locks_member_bytes": True,
        },
        "model_lock": {
            "model": AGG.EXPECTED_MODEL,
            "revision": AGG.EXPECTED_MODEL_REVISION,
            "config_sha256": AGG.EXPECTED_MODEL_CONFIG_SHA256,
            "parameter_count": AGG.EXPECTED_MODEL_PARAMETER_COUNT,
        },
        "configuration_lock": {
            "context": AGG.CONTEXT,
            "decode_steps": AGG.DECODE_STEPS,
            "exact_tail_tokens": AGG.EXACT_TAIL,
            "exact_prefix_pages": AGG.EXACT_PREFIX_PAGES,
            "baseline_split_pages": AGG.EXPECTED_SPLIT_PAGES,
            "candidate_split_pages": AGG.EXPECTED_SPLIT_PAGES,
            "tail_attention": AGG.EXPECTED_TAIL_ATTENTION,
            "old_value_scale_placement": AGG.EXPECTED_OLD_VALUE_SCALE_PLACEMENT,
            "prefill_chunk_tokens": AGG.EXPECTED_PREFILL_CHUNK_TOKENS,
            "seed": AGG.EXPECTED_SEED,
        },
        "worker_threshold_lock": {
            "minimum_logits_cosine": 0.995,
            "minimum_top1_agreement": 0.99,
            "minimum_flashinfer_hf_cosine": 0.999,
        },
        "co_resident_memory_gate": memory,
        "source_closure_paths": list(AGG.EXPECTED_SOURCE_KEYS),
    }
    return {
        "schema_version": AGG.RAW_SCHEMA_VERSION,
        "experiment": AGG.EXPECTED_EXPERIMENT,
        "model": AGG.EXPECTED_MODEL,
        "model_revision": AGG.EXPECTED_MODEL_REVISION,
        "model_config_sha256": AGG.EXPECTED_MODEL_CONFIG_SHA256,
        "parameters": AGG.EXPECTED_MODEL_PARAMETER_COUNT,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_size": batch_size,
        "context": AGG.CONTEXT,
        "decode_steps": AGG.DECODE_STEPS,
        "exact_tail_tokens": AGG.EXACT_TAIL,
        "exact_prefix_pages": AGG.EXACT_PREFIX_PAGES,
        "exact_sink_pages": AGG.EXACT_PREFIX_PAGES,
        "prefill_chunk_tokens": AGG.EXPECTED_PREFILL_CHUNK_TOKENS,
        "baseline_split_pages": AGG.EXPECTED_SPLIT_PAGES,
        "candidate_split_pages": AGG.EXPECTED_SPLIT_PAGES,
        "center_restore": "attention_add",
        "tail_attention": AGG.EXPECTED_TAIL_ATTENTION,
        "old_value_scale_placement": AGG.EXPECTED_OLD_VALUE_SCALE_PLACEMENT,
        "heldout_protocol": heldout,
        "token_source": {
            "kind": "wikitext2",
            "split": AGG.EXPECTED_SPLIT,
            "archive_sha256": AGG.EXPECTED_ARCHIVE_SHA256,
            "archive_sha256_verified": True,
            "archive_member": AGG.EXPECTED_MEMBER,
            "archive_member_sha256": member_sha,
            "tokenizer": {
                "manifest_sha256": AGG.EXPECTED_TOKENIZER_MANIFEST_SHA256,
                "resolved_snapshot_revision": AGG.EXPECTED_MODEL_REVISION,
            },
            "corpus_windows_disjoint": True,
            "corpus_window_stride": AGG.WINDOW_STRIDE,
            "corpus_window_start_offsets": list(starts),
            "corpus_window_end_offsets_exclusive": ends,
            "corpus_tokens_per_request": AGG.WINDOW_CORPUS_TOKENS,
            "available_corpus_token_count": AGG.EXPECTED_TEST_TOKEN_COUNT,
            "token_ids_sha256": digest(f"tokens-{starts[0]}"),
        },
        "source_sha256": source_hashes(),
        "source_integrity_gate": {
            "passed": True,
            "hashed_before_model_or_dataset_access": True,
            "unchanged_through_result_finalization": True,
            "paths": list(AGG.EXPECTED_SOURCE_KEYS),
        },
        "cache_storage": {
            "flashinfer_fp16_bytes": memory["flashinfer_fp16_cache_bytes"],
            "page_gauge_bytes": memory["page_gauge_cache_bytes"],
            "batch_size": batch_size,
            "page_gauge_exact_tail_pages_per_request": 48,
            "page_gauge_exact_prefix_pages_per_request": 3,
            "page_gauge_exact_sink_pages_per_request": 3,
            "page_gauge_exact_storage_pages_per_request": 51,
        },
        "logit_collection": {
            "immediate_cpu_offload": True,
            "timing_claim": False,
        },
        "runtime_page_finalization": finalization,
        "exact_prefix_canary": exact_prefix,
        "correctness": {
            "teacher_forced": True,
            "passed": True,
            "page_gauge_threshold_passed": True,
            "baseline_hf_conversion_threshold_passed": True,
            "base_model_prefill_correctness_passed": True,
            "graph_replay": {"enabled": False, "passed": True},
            "thresholds": thresholds,
            "runtime_page_finalization": copy.deepcopy(finalization),
            "exact_prefix_canary": copy.deepcopy(exact_prefix),
            "quality_metric_protocol": {"window_metadata": windows},
            "page_gauge_vs_flashinfer_fp16": comparison_payload(
                batch_size, nll_delta=page_gauge_nll_delta
            ),
            "flashinfer_fp16_vs_hf_sdpa_fp16": comparison_payload(batch_size),
            "static_teacher_diagnostic_required": False,
            "static_teacher_diagnostic_passed": None,
            "direct_feedback_diagnostic_required": False,
            "direct_feedback_diagnostic_passed": None,
            "token_step_graph_diagnostic_required": False,
            "token_step_graph_diagnostic_passed": None,
            "generated_sequence_graph_diagnostic_required": False,
            "generated_sequence_graph_diagnostic_passed": None,
        },
        "static_teacher_diagnostic": copy.deepcopy(disabled),
        "direct_feedback_diagnostic": copy.deepcopy(disabled),
        "token_step_graph_diagnostic": copy.deepcopy(disabled),
        "generated_sequence_graph_diagnostic": copy.deepcopy(disabled),
    }


def aggregate_input_records(*, page_gauge_nll_delta: float = 0.0) -> list[dict]:
    return [
        {
            "path": Path(f"fixture-{index}.json"),
            "sha256": f"{index + 1:064x}",
            "payload": group_payload(
                tuple(group), page_gauge_nll_delta=page_gauge_nll_delta
            ),
        }
        for index, group in enumerate(AGG.EXPECTED_GROUPS)
    ]


def heldout_args(group: tuple[int, ...] = AGG.EXPECTED_GROUPS[0]) -> SimpleNamespace:
    return SimpleNamespace(
        heldout_policy=True,
        model=AGG.EXPECTED_MODEL,
        batch_size=len(group),
        context=AGG.CONTEXT,
        decode_steps=AGG.DECODE_STEPS,
        exact_tail=AGG.EXACT_TAIL,
        exact_prefix_pages=AGG.EXACT_PREFIX_PAGES,
        prefill_chunk_tokens=AGG.EXPECTED_PREFILL_CHUNK_TOKENS,
        baseline_split_pages=AGG.EXPECTED_SPLIT_PAGES,
        candidate_split_pages=AGG.EXPECTED_SPLIT_PAGES,
        tail_attention=AGG.EXPECTED_TAIL_ATTENTION,
        old_value_scale_placement=AGG.EXPECTED_OLD_VALUE_SCALE_PLACEMENT,
        seed=AGG.EXPECTED_SEED,
        token_source="wikitext2",
        wikitext_member=AGG.EXPECTED_MEMBER,
        token_offset=group[0],
        token_stride=AGG.WINDOW_STRIDE,
        disable_attention_cuda_graphs=True,
        min_logits_cosine=0.995,
        min_top1_agreement=0.99,
        min_baseline_hf_cosine=0.999,
        run_token_step_graphs=False,
        run_direct_feedback_paths=False,
        run_static_teacher_control=False,
        run_generated_sequence_graph=False,
    )


def test_frozen_windows_are_exact_disjoint_and_fit_test() -> None:
    assert len(AGG.EXPECTED_STARTS) == 14
    assert [len(group) for group in AGG.EXPECTED_GROUPS] == [3, 3, 3, 3, 2]
    assert AGG.EXPECTED_STARTS == tuple(index * 23600 for index in range(14))
    intervals = [
        (start, start + AGG.WINDOW_CORPUS_TOKENS) for start in AGG.EXPECTED_STARTS
    ]
    assert all(right[0] >= left[1] for left, right in zip(intervals, intervals[1:]))
    assert intervals[-1][1] == 328816
    assert AGG.EXPECTED_TEST_TOKEN_COUNT - intervals[-1][1] == 63


def test_worker_policy_accepts_only_frozen_shards_without_dataset_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PREFILL.zipfile,
        "ZipFile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("TEST archive was opened")
        ),
    )
    for group in AGG.EXPECTED_GROUPS:
        assert PREFILL.validate_heldout_command(heldout_args(group)) == group
    wrong = heldout_args()
    wrong.batch_size = 4
    with pytest.raises(ValueError, match="frozen \\(offset,batch\\) group"):
        PREFILL.validate_heldout_command(wrong)


def test_b3_memory_gate_passes_and_b4_is_rejected() -> None:
    free = 32_429_309_952
    b3 = PREFILL.heldout_co_resident_memory_gate(
        batch_size=3,
        context=AGG.CONTEXT,
        decode_steps=AGG.DECODE_STEPS,
        exact_tail=AGG.EXACT_TAIL,
        exact_prefix_pages=AGG.EXACT_PREFIX_PAGES,
        free_bytes=free,
    )
    b4 = PREFILL.heldout_co_resident_memory_gate(
        batch_size=4,
        context=AGG.CONTEXT,
        decode_steps=AGG.DECODE_STEPS,
        exact_tail=AGG.EXACT_TAIL,
        exact_prefix_pages=AGG.EXACT_PREFIX_PAGES,
        free_bytes=free,
    )
    assert b3["passed"]
    assert b3["projected_co_resident_bytes"] == 27_807_883_776
    assert not b4["passed"]
    assert b4["projected_co_resident_bytes"] == 32_245_162_496


def test_heldout_aggregate_accepts_frozen_identity_fixture() -> None:
    result = AGG.aggregate_records(aggregate_input_records())
    assert result["passed"]
    assert result["protocol"]["cluster_count"] == 14
    assert result["protocol"]["total_labeled_tokens"] == 21_504
    assert result["protocol"]["bootstrap_samples"] == 50_000
    assert (
        result["page_gauge_vs_flashinfer_fp16"]["point"][
            "candidate_mean_true_token_rank"
        ]
        == 1.0
    )


def test_heldout_aggregate_rejects_bootstrap_override() -> None:
    with pytest.raises(ValueError, match="frozen at 50000"):
        AGG.aggregate_records(aggregate_input_records(), bootstrap_samples=100)


def test_heldout_aggregate_applies_preregistered_ppl_gate() -> None:
    result = AGG.aggregate_records(
        aggregate_input_records(page_gauge_nll_delta=math.log(1.02))
    )
    assert not result["passed"]
    assert (
        "page_gauge_vs_flashinfer_fp16.maximum_ppl_ratio_cluster_bootstrap_upper_95"
    ) in result["failures"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda records: records[-1]["payload"]["source_sha256"].__setitem__(
                "scripts/page_gauge_runtime.py", "f" * 64
            ),
            "frozen source hashes",
        ),
        (
            lambda records: records[-1]["payload"]["exact_prefix_canary"][
                "immutability"
            ]["after"].__setitem__("combined_sha256", "f" * 64),
            "immutable prefix digest",
        ),
        (
            lambda records: records[-1]["payload"][
                "runtime_page_finalization"
            ].__setitem__("runtime_generated_int8_logical_page_count", 47),
            "generated INT8 recurrence count",
        ),
        (
            lambda records: records[-1]["payload"].__setitem__(
                "old_value_scale_placement", "value_fragment"
            ),
            "V-scale placement",
        ),
    ],
)
def test_heldout_aggregate_fails_closed_on_gate_tampering(
    mutation, message: str
) -> None:
    records = aggregate_input_records()
    mutation(records)
    with pytest.raises(ValueError, match=message):
        AGG.aggregate_records(records)


def test_heldout_aggregate_rejects_duplicate_or_wrong_shard_counts() -> None:
    records = aggregate_input_records()
    with pytest.raises(ValueError, match="exactly five"):
        AGG.aggregate_records(records[:-1])
    duplicate = aggregate_input_records()
    duplicate[-1]["path"] = duplicate[0]["path"]
    with pytest.raises(ValueError, match="paths must be unique"):
        AGG.aggregate_records(duplicate)


def test_heldout_aggregate_rejects_mixed_content_and_duplicate_clusters() -> None:
    mixed = aggregate_input_records()
    mixed[-1]["payload"]["token_source"]["archive_member_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="signatures differ"):
        AGG.aggregate_records(mixed)

    duplicate_cluster = aggregate_input_records()
    duplicate_cluster[-1]["payload"]["correctness"]["quality_metric_protocol"][
        "window_metadata"
    ][0]["cluster_unit_id"] = "test-window-0"
    with pytest.raises(ValueError, match="duplicate cluster ID"):
        AGG.aggregate_records(duplicate_cluster)


def prefix_cache_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    layers, batch, pages, tail, prefix = 2, 2, 8, 2, 3
    baseline_key = torch.arange(
        layers * batch * pages * 16 * 1 * 2, dtype=torch.float16
    ).reshape(layers, batch * pages, 16, 1, 2)
    baseline_value = baseline_key + 7
    key_center = torch.zeros(layers, batch, 1, 2, dtype=torch.float16)
    value_center = torch.ones(layers, batch, 1, 2, dtype=torch.float16)
    exact_key = torch.zeros(
        layers, batch * (tail + prefix), 16, 1, 2, dtype=torch.float16
    )
    exact_value = torch.zeros_like(exact_key)
    for request in range(batch):
        for logical in range(prefix):
            physical = PREFILL.PG.exact_physical_page_index(
                request, logical, tail, prefix
            )
            source = request * pages + logical
            exact_key[:, physical].copy_(baseline_key[:, source])
            exact_value[:, physical].copy_(baseline_value[:, source] - 1)
    baseline = SimpleNamespace(key=baseline_key, value=baseline_value)
    cache = SimpleNamespace(
        exact_key=exact_key,
        exact_value=exact_value,
        key_center=key_center,
        value_center=value_center,
    )
    return baseline, cache


def test_generic_s3_prefix_population_mapping_and_immutability() -> None:
    baseline, cache = prefix_cache_fixture()
    population = PREFILL.audit_exact_prefix_population(
        baseline,
        cache,
        pages_per_request=8,
        exact_tail_pages=2,
        exact_prefix_pages=3,
        batch_size=2,
    )
    before = PREFILL.exact_prefix_snapshot(
        cache,
        exact_tail_pages=2,
        exact_prefix_pages=3,
        batch_size=2,
    )
    cache.exact_key[:, 3:5].add_(1)
    after = PREFILL.exact_prefix_snapshot(
        cache,
        exact_tail_pages=2,
        exact_prefix_pages=3,
        batch_size=2,
    )
    assert population["passed"]
    assert before["mapping"] == [[0, 1, 2], [5, 6, 7]]
    assert PREFILL.audit_exact_prefix_immutability(before, after)["passed"]
    logical_table = PREFILL.PG.request_major_exact_page_table(2, 8, 2, 3, device="cpu")
    decoder = SimpleNamespace(
        exact_ring_pages=logical_table,
        exact_plan_pages=logical_table[:, :5].clone(),
        exact_page_table_updates=1,
        exact_layout_signature=(6, 8),
    )
    page_tables = PREFILL.audit_exact_prefix_page_tables(
        decoder,
        exact_tail_pages=2,
        exact_prefix_pages=3,
        batch_size=2,
    )
    assert page_tables["passed"]
    assert page_tables["active_exact_plan_table_prefix"] == [[0, 1, 2], [5, 6, 7]]


def finalization_cache_fixture(batch_size: int = 2) -> SimpleNamespace:
    layers = 1
    pages = 1376
    shape = (layers, batch_size * pages, PREFILL.PG.PAGE, 1, 2)
    scale_shape = (layers, batch_size * pages, 1)
    return SimpleNamespace(
        key_codes=torch.zeros(shape, dtype=torch.int8),
        value_codes=torch.zeros(shape, dtype=torch.int8),
        key_scales=torch.ones(scale_shape, dtype=torch.float16),
        value_scales=torch.ones(scale_shape, dtype=torch.float16),
    )


def test_d1536_finalization_recurrence_has_48_generated_int8_pages() -> None:
    cache = finalization_cache_fixture()
    PREFILL.initialize_future_page_canary(
        cache,
        pages_per_request=1376,
        initial_pages_per_request=1280,
        batch_size=2,
    )
    for request in range(2):
        begin = request * 1376 + 1280
        end = request * 1376 + 1376
        cache.key_codes[:, begin:end].zero_()
        cache.value_codes[:, begin:end].zero_()
        cache.key_scales[:, begin:end].fill_(0.01)
        cache.value_scales[:, begin:end].fill_(0.02)
    audit = PREFILL.audit_runtime_page_finalization(
        cache,
        pages_per_request=1376,
        initial_pages_per_request=1280,
        batch_size=2,
        layers=1,
        context=20480,
        decode_steps=1536,
        exact_tail=768,
        exact_prefix_pages=3,
    )
    assert audit["passed"]
    assert audit["generated_completed_logical_page_range"] == [1280, 1376]
    assert audit["runtime_generated_int8_logical_page_range"] == [1280, 1328]
    assert audit["runtime_generated_int8_logical_page_count"] == 48
    assert audit["endpoint_prefix_logical_page_range"] == [0, 3]
