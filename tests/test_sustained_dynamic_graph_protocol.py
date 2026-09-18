from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "page_gauge_sustained_dynamic_graphs",
    ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hot_sample_has_no_restore_or_canary_between_precondition_and_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cache_scrub = SimpleNamespace(add_=lambda _value: events.append("scrub"))

    def timed(*_args, **_kwargs):
        events.append(f"timed{_kwargs['decode_steps']}")
        return {"cuda_ms": 1.0, "wall_ms": 1.0, "runtime_gate": {"passed": True}}

    def compare(*_args, **_kwargs):
        events.append("digest")
        return {"passed": True}

    monkeypatch.setattr(
        MODULE,
        "prepare_run",
        lambda *_args: events.append("restore"),
    )
    monkeypatch.setattr(MODULE, "timed_block", timed)
    monkeypatch.setattr(MODULE, "compare_exact_sink_canary", compare)
    monkeypatch.setattr(MODULE.torch.cuda, "synchronize", lambda: None)
    result = MODULE.measure_modes(
        SimpleNamespace(
            batch_size=1,
            plan=lambda _position: events.append("plan"),
        ),
        initial_cache={},
        initial_token=None,
        teacher_inputs=torch.empty(512, 1, dtype=torch.long),
        trajectory_mode=MODULE.FROZEN_HF_TEACHER,
        start_position=16,
        decode_steps=512,
        dynamic_token=None,
        argmax_token=None,
        cache_scrub=cache_scrub,
        warmups=0,
        repeats=1,
        expected_sink_canary={"enabled": True},
    )

    assert events == [
        "restore",
        "scrub",
        "timed512",
        "digest",
        "restore",
        "timed16",
        "plan",
        "timed512",
        "digest",
        "restore",
    ]
    hot_sample = result["cache_hot"]["raw_samples"][0]
    assert hot_sample["precondition"]["no_restore_before_timed_sample"] is True
    assert hot_sample["precondition"]["precondition_steps"] == MODULE.PG.PAGE
    assert hot_sample["precondition"]["planner_reset_without_cache_restore"] is True
    assert result["cache_hot"][
        "no_restore_between_hot_precondition_and_sample"
    ] is True


def test_sustained_defaults_are_512_steps_and_safe_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backend",
            "page_gauge",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    args = MODULE.parse_args()
    assert args.decode_steps == 512
    assert args.batch_size == 4
    assert args.context == 20480
    assert args.candidate_split_pages == 256
    assert args.trajectory_mode == MODULE.GREEDY_FEEDBACK
    assert args.old_value_scale_placement == "probability"
    assert args.exact_sink_pages == 0
    assert args.quality_diagnostics_top_k == 0
    assert args.quality_diagnostics_top_vocab == 8


def test_sustained_value_fragment_scale_placement_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backend",
            "page_gauge",
            "--old-value-scale-placement",
            "value_fragment",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert MODULE.parse_args().old_value_scale_placement == "value_fragment"


def test_sustained_exact_sink_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backend",
            "page_gauge",
            "--exact-sink-pages",
            "1",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert MODULE.parse_args().exact_sink_pages == 1


def test_sustained_exact_prefix_s3_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backend",
            "page_gauge",
            "--exact-sink-pages",
            "3",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert MODULE.parse_args().exact_sink_pages == 3


def test_scheduler_capacity_rejects_split128_without_clamping() -> None:
    unsafe = MODULE.scheduler_capacity(
        num_sms=170,
        batch_size=4,
        hkv=8,
        maximum_active_pages=1312,
        fixed_split_pages=128,
    )
    safe = MODULE.scheduler_capacity(
        num_sms=170,
        batch_size=4,
        hkv=8,
        maximum_active_pages=1312,
        fixed_split_pages=256,
    )
    assert unsafe["max_batch_size_if_split"] == 42
    assert unsafe["max_chunks_per_request"] == 10
    assert unsafe["required_chunks_per_request"] == 11
    assert unsafe["required_split_tiles"] == 44
    assert unsafe["minimum_safe_fixed_split_pages"] == 132
    assert not unsafe["analytically_within_capacity"]
    assert unsafe["split_was_automatically_clamped"] is False
    assert safe["required_chunks_per_request"] == 6
    assert safe["analytically_within_capacity"]


def test_runtime_gate_requires_every_graph_replay_and_no_fallback() -> None:
    decoder = SimpleNamespace(
        backend="page_gauge",
        tail_attention="flashinfer_merge",
        layers=32,
    )
    operations = {
        "decoder_plan_calls": 512,
        "device_position_fills": 512,
        "heterogeneous_page_table_updates": 0,
        "wrappers": {
            "old_int8": {
                "plan_invocations": 512,
                "plan_rebuilds": 32,
                "last_page_len_device_fills": 512,
            },
            "exact_fp16": {
                "plan_invocations": 512,
                "plan_rebuilds": 32,
                "last_page_len_device_fills": 512,
            },
        },
    }
    dispatch = {
        "graph_replays": 32 * 512,
        "eager_calls": 0,
        "graph_bank_misses": 0,
        "total_calls": 32 * 512,
        "graph_coverage_fraction": 1.0,
    }
    attention_dispatch = {
        "graph_replays": 0,
        "eager_calls": 0,
        "total_calls": 0,
        "graph_coverage_fraction": 0.0,
    }
    assert MODULE.runtime_count_gate(
        decoder, dispatch, attention_dispatch, operations, 512
    )["passed"]
    dispatch["eager_calls"] = 1
    assert not MODULE.runtime_count_gate(
        decoder, dispatch, attention_dispatch, operations, 512
    )["passed"]


def test_runtime_gate_rejects_nested_attention_and_missing_heterogeneous_update() -> None:
    decoder = SimpleNamespace(
        backend="page_gauge",
        tail_attention="heterogeneous_fa2",
        layers=32,
    )
    operations = {
        "decoder_plan_calls": 512,
        "device_position_fills": 512,
        "heterogeneous_page_table_updates": 32,
        "wrappers": {
            "heterogeneous": {
                "plan_invocations": 512,
                "plan_rebuilds": 32,
                "last_page_len_device_fills": 512,
            },
        },
    }
    dispatch = {
        "graph_replays": 32 * 512,
        "eager_calls": 0,
        "graph_bank_misses": 0,
        "total_calls": 32 * 512,
        "graph_coverage_fraction": 1.0,
    }
    attention_dispatch = {
        "graph_replays": 0,
        "eager_calls": 0,
        "total_calls": 0,
        "graph_coverage_fraction": 0.0,
    }
    assert MODULE.runtime_count_gate(
        decoder, dispatch, attention_dispatch, operations, 512
    )["passed"]
    operations["heterogeneous_page_table_updates"] = 31
    assert not MODULE.runtime_count_gate(
        decoder, dispatch, attention_dispatch, operations, 512
    )["passed"]
    operations["heterogeneous_page_table_updates"] = 32
    attention_dispatch["eager_calls"] = 1
    attention_dispatch["total_calls"] = 1
    assert not MODULE.runtime_count_gate(
        decoder, dispatch, attention_dispatch, operations, 512
    )["passed"]


def test_scheduler_report_uses_full_pages_for_heterogeneous_wrapper() -> None:
    report = MODULE.scheduler_capacity_report(
        backend="page_gauge",
        tail_attention="heterogeneous_fa2",
        num_sms=170,
        batch_size=4,
        hkv=8,
        maximum_pages=1312,
        exact_pages=16,
        baseline_split_pages=256,
        candidate_split_pages=256,
    )
    assert report["wrappers"]["heterogeneous"][
        "maximum_active_pages_per_request"
    ] == 1312
    assert report["all_wrappers_analytically_within_capacity"]
    assert MODULE.planner_boundary_positions(20480, 512) == list(
        range(20480, 20480 + 512, 16)
    )


def test_scheduler_report_counts_sink_inside_existing_exact_wrapper() -> None:
    report = MODULE.scheduler_capacity_report(
        backend="page_gauge",
        tail_attention="flashinfer_merge",
        num_sms=170,
        batch_size=4,
        hkv=8,
        maximum_pages=1312,
        exact_pages=16,
        baseline_split_pages=256,
        candidate_split_pages=256,
        exact_sink_pages=1,
    )
    assert set(report["wrappers"]) == {"old_int8", "exact_fp16"}
    assert report["wrappers"]["old_int8"][
        "maximum_active_pages_per_request"
    ] == 1295
    assert report["wrappers"]["exact_fp16"][
        "maximum_active_pages_per_request"
    ] == 17


def test_scheduler_report_counts_s3_inside_same_two_wrappers() -> None:
    report = MODULE.scheduler_capacity_report(
        backend="page_gauge",
        tail_attention="flashinfer_merge",
        num_sms=170,
        batch_size=4,
        hkv=8,
        maximum_pages=1312,
        exact_pages=16,
        baseline_split_pages=256,
        candidate_split_pages=256,
        exact_sink_pages=3,
    )
    assert set(report["wrappers"]) == {"old_int8", "exact_fp16"}
    assert report["wrappers"]["old_int8"][
        "maximum_active_pages_per_request"
    ] == 1293
    assert report["wrappers"]["exact_fp16"][
        "maximum_active_pages_per_request"
    ] == 19
    for unsupported in ("fused_kernel", "heterogeneous_fa2"):
        with pytest.raises(ValueError, match="segmented flashinfer_merge"):
            MODULE.scheduler_capacity_report(
                backend="page_gauge",
                tail_attention=unsupported,
                num_sms=170,
                batch_size=4,
                hkv=8,
                maximum_pages=1312,
                exact_pages=16,
                baseline_split_pages=256,
                candidate_split_pages=256,
                exact_sink_pages=3,
            )


def test_runtime_finalization_proves_sink_old_tail_disjoint_coverage() -> None:
    batch_size = 2
    max_pages = 1313
    exact_tail_pages = 16
    exact_sink_pages = 1
    final_pages = 1312
    old_logical_pages = list(range(1, final_pages - exact_tail_pages))
    table = torch.tensor(
        [
            [request * max_pages + page for page in old_logical_pages]
            for request in range(batch_size)
        ],
        dtype=torch.int32,
    )
    decoder = SimpleNamespace(
        backend="page_gauge",
        tail_attention="flashinfer_merge",
        batch_size=batch_size,
        max_pages=max_pages,
        exact_tail_pages=exact_tail_pages,
        exact_sink_pages=exact_sink_pages,
        old_pages=len(old_logical_pages),
        old_wrapper=SimpleNamespace(indices=table.reshape(-1)),
    )
    decoder.exact_physical_page = lambda request, page: MODULE.PG.exact_physical_page_index(
        request,
        page,
        exact_tail_pages,
        exact_sink_pages,
    )
    exact_logical_pages = [0, *range(final_pages - exact_tail_pages, final_pages)]
    exact_table = torch.tensor(
        [
            [decoder.exact_physical_page(request, page) for page in exact_logical_pages]
            for request in range(batch_size)
        ],
        dtype=torch.int32,
    )
    decoder.exact_wrapper = SimpleNamespace(indices=exact_table.reshape(-1))

    result = MODULE.runtime_finalization_consumption(decoder, 20480, 512)

    assert result["passed"]
    assert result["runtime_finalized_pages_consumed_as_int8"] == list(
        range(1280, 1296)
    )
    assert result["final_old_logical_pages"][0] == 1
    assert result["final_old_logical_pages"][-1] == 1295
    assert result["final_exact_sink_logical_pages"] == [0]
    assert result["final_exact_tail_logical_pages"] == list(range(1296, 1312))
    assert result["logical_page_sets_disjoint"]
    assert result["logical_token_coverage_exactly_once"]


def test_runtime_finalization_s3_covers_prefix_old_tail_exactly_once() -> None:
    batch_size = 2
    max_pages = 1313
    exact_tail_pages = 16
    exact_prefix_pages = 3
    final_pages = 1312
    old_logical_pages = list(
        range(exact_prefix_pages, final_pages - exact_tail_pages)
    )
    old_table = torch.tensor(
        [
            [request * max_pages + page for page in old_logical_pages]
            for request in range(batch_size)
        ],
        dtype=torch.int32,
    )
    decoder = SimpleNamespace(
        backend="page_gauge",
        tail_attention="flashinfer_merge",
        batch_size=batch_size,
        max_pages=max_pages,
        exact_tail_pages=exact_tail_pages,
        exact_prefix_pages=exact_prefix_pages,
        exact_sink_pages=exact_prefix_pages,
        old_pages=len(old_logical_pages),
        old_wrapper=SimpleNamespace(indices=old_table.reshape(-1)),
    )
    decoder.exact_physical_page = (
        lambda request, page: MODULE.PG.exact_physical_page_index(
            request,
            page,
            exact_tail_pages,
            exact_prefix_pages,
        )
    )
    exact_logical_pages = [
        *range(exact_prefix_pages),
        *range(final_pages - exact_tail_pages, final_pages),
    ]
    exact_table = torch.tensor(
        [
            [decoder.exact_physical_page(request, page) for page in exact_logical_pages]
            for request in range(batch_size)
        ],
        dtype=torch.int32,
    )
    decoder.exact_wrapper = SimpleNamespace(indices=exact_table.reshape(-1))

    result = MODULE.runtime_finalization_consumption(decoder, 20480, 512)

    assert result["passed"] is True
    assert result["final_exact_prefix_logical_pages"] == [0, 1, 2]
    assert result["final_old_logical_pages"][0] == 3
    assert result["final_old_logical_pages"][-1] == 1295
    assert result["final_exact_tail_logical_pages"] == list(range(1296, 1312))
    assert result["prefix_exclusion_gate_passed"] is True
    assert result["logical_page_sets_disjoint"] is True
    assert result["logical_token_coverage_exactly_once"] is True


def test_immutable_prefix_digest_covers_every_s3_slot_and_detects_mutation() -> None:
    batch_size = 2
    prefix_pages = 3
    tail_pages = 2
    storage_pages = prefix_pages + tail_pages
    exact_key = torch.arange(
        2 * batch_size * storage_pages * 16,
        dtype=torch.float32,
    ).reshape(2, batch_size * storage_pages, 16, 1, 1)
    exact_value = exact_key.add(10_000)
    decoder = SimpleNamespace(
        backend="page_gauge",
        batch_size=batch_size,
        exact_prefix_pages=prefix_pages,
        exact_sink_pages=prefix_pages,
        exact_tail_pages=tail_pages,
        cache=SimpleNamespace(exact_key=exact_key, exact_value=exact_value),
    )
    decoder.exact_physical_page = (
        lambda request, logical_page: MODULE.PG.exact_physical_page_index(
            request,
            logical_page,
            tail_pages,
            prefix_pages,
        )
    )

    expected = MODULE.immutable_exact_prefix_digest(decoder)
    assert expected["enabled"] is True
    assert expected["logical_pages"] == [0, 1, 2]
    assert expected["physical_pages"] == [0, 1, 2, 5, 6, 7]
    assert expected["physical_page_count"] == batch_size * prefix_pages

    canonical_mapping = decoder.exact_physical_page
    decoder.exact_physical_page = (
        lambda request, logical_page: request * storage_pages
        + (logical_page + 1) % storage_pages
    )
    with pytest.raises(RuntimeError, match="fixed slots"):
        MODULE.immutable_exact_prefix_digest(decoder)
    decoder.exact_physical_page = canonical_mapping

    decoder.cache.exact_key[:, 3].add_(1)
    assert MODULE.immutable_exact_prefix_digest(decoder) == expected
    decoder.cache.exact_key[:, 6].add_(1)
    comparison = MODULE.compare_exact_sink_canary(
        expected,
        decoder,
        phase="cpu_prefix_mutation",
    )
    assert comparison["passed"] is False
    assert comparison["exact_prefix_pages"] == 3
    assert comparison["physical_page_count"] == 6


def test_cpu_logit_and_token_comparisons_are_fail_closed() -> None:
    expected_logits = torch.randn(512, 2, 7)
    observed_logits = expected_logits.clone()
    expected_tokens = expected_logits.argmax(dim=-1)
    observed_tokens = expected_tokens.clone()
    assert MODULE.compare_logits(expected_logits, observed_logits)[
        "bitwise_identical"
    ]
    assert MODULE.compare_tokens(expected_tokens, observed_tokens)[
        "bitwise_identical"
    ]
    observed_logits[511, 1, 3] += 0.25
    assert not MODULE.compare_logits(expected_logits, observed_logits)[
        "bitwise_identical"
    ]


def test_quality_outlier_diagnostic_localizes_rows_without_changing_gate() -> None:
    steps, batch, vocab = 32, 2, 7
    expected = torch.randn(steps, batch, vocab, generator=torch.Generator().manual_seed(7))
    observed = expected.clone()
    # One isolated page-close failure and one two-step content cluster.
    observed[15, 1] = -expected[15, 1]
    observed[20, 0] = expected[20, 0].roll(1)
    observed[21, 0] = expected[21, 0].roll(1)
    inputs = torch.arange(steps * batch, dtype=torch.long).view(steps, batch)

    report = MODULE.diagnose_logit_outliers(
        expected,
        observed,
        input_tokens=inputs,
        start_position=20480,
        exact_tail_tokens=256,
        exact_prefix_pages=0,
        minimum_cosine=0.995,
        top_rows=8,
        top_vocab=4,
    )

    assert report["threshold_unchanged"] == 0.995
    assert report["summary"]["rows_below_gate"] == 3
    assert report["worst_rows"][0]["step"] == 15
    assert report["worst_rows"][0]["request"] == 1
    assert report["worst_rows"][0]["page_offset"] == 15
    assert report["worst_rows"][0]["is_page_close"] is True
    assert report["worst_rows"][0]["generated_int8_pages_visible"] == 0
    assert report["summary"]["failure_clusters"] == [
        {"request": 0, "start_step": 20, "end_step_inclusive": 21, "length": 2},
        {"request": 1, "start_step": 15, "end_step_inclusive": 15, "length": 1},
    ]
    offset_15 = report["per_page_offset"][15]
    assert offset_15["rows_below_gate"] == 1
    assert len(report["row_matrices"]["cosine_by_step_request"]) == steps


@pytest.mark.parametrize(
    ("exact_prefix_pages", "expected_quantized_pages"),
    ((0, 1264), (1, 1263), (3, 1261)),
)
def test_quality_outlier_page_provenance_is_generic_in_s(
    exact_prefix_pages: int, expected_quantized_pages: int
) -> None:
    logits = torch.zeros(16, 1, 2)
    logits[..., 0] = 1.0
    report = MODULE.diagnose_logit_outliers(
        logits,
        logits.clone(),
        input_tokens=torch.zeros(16, 1, dtype=torch.long),
        start_position=20480,
        exact_tail_tokens=256,
        exact_prefix_pages=exact_prefix_pages,
        minimum_cosine=0.995,
        top_rows=1,
        top_vocab=1,
    )

    configuration = report["configuration"]
    assert report["schema_version"] == 3
    assert configuration["exact_prefix_pages"] == exact_prefix_pages
    assert configuration["exact_prefix_logical_pages"] == list(
        range(exact_prefix_pages)
    )
    assert configuration["initial_quantized_old_pages"] == expected_quantized_pages
    assert configuration[
        "initial_quantized_old_logical_range_start_inclusive_end_exclusive"
    ] == [exact_prefix_pages, 1264]
    assert configuration[
        "initial_exact_tail_logical_range_start_inclusive_end_exclusive"
    ] == [1264, 1280]
    assert configuration["initial_page_partition_disjoint_and_complete"] is True
    assert report["worst_rows"][0]["prefix_quantized_pages"] == (
        expected_quantized_pages
    )
    assert report["worst_rows"][0]["exact_prefix_pages"] == exact_prefix_pages


@pytest.mark.parametrize(
    ("generated_tokens", "expected_visible_int8_pages"),
    ((256, 0), (257, 1), (271, 1), (272, 1), (273, 2)),
)
def test_generated_int8_page_visibility_uses_page_ceiling(
    generated_tokens: int, expected_visible_int8_pages: int
) -> None:
    assert MODULE.generated_int8_pages_visible_count(generated_tokens, 16) == (
        expected_visible_int8_pages
    )


@pytest.mark.parametrize(
    (
        "generated_tokens",
        "expected_initial_old",
        "expected_displaced_initial_tail",
        "expected_generated_old",
        "expected_total_old",
        "expected_tail_begin",
    ),
    (
        (1, 1246, 1, 0, 1246, 1249),
        (512, 1277, 32, 0, 1277, 1280),
        (513, 1277, 32, 1, 1278, 1281),
        (799, 1277, 32, 18, 1295, 1298),
    ),
)
def test_outlier_page_visibility_uses_planner_exact_partition(
    generated_tokens: int,
    expected_initial_old: int,
    expected_displaced_initial_tail: int,
    expected_generated_old: int,
    expected_total_old: int,
    expected_tail_begin: int,
) -> None:
    counts = MODULE.outlier_page_visibility_counts(
        start_position=20480,
        generated_tokens_including_current=generated_tokens,
        exact_tail_pages=32,
        exact_prefix_pages=3,
    )

    assert counts["initial_quantized_old_pages_at_prefix"] == 1245
    assert counts["initial_prefix_pages_currently_old_int8"] == expected_initial_old
    assert (
        counts["displaced_initial_exact_tail_pages_now_old_int8"]
        == expected_displaced_initial_tail
    )
    assert counts["generated_int8_pages_visible"] == expected_generated_old
    assert counts["total_quantized_pages_visible"] == expected_total_old
    assert counts["planner_tail_logical_begin"] == expected_tail_begin
    assert counts["planner_total_pages_visible"] == (
        1280 + (generated_tokens + 15) // 16
    )


def test_short_sustained_protocol_is_rejected_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--backend",
            "flashinfer_fp16",
            "--decode-steps",
            "16",
            "--output",
            str(tmp_path / "invalid.json"),
        ],
    )
    with pytest.raises(SystemExit, match=">=512"):
        MODULE.validate_args(MODULE.parse_args())


@pytest.mark.parametrize("backend", ["flashinfer_fp16", "page_gauge"])
def test_following_page_canary_initializes_real_storage_not_advanced_index_copy(
    backend: str,
) -> None:
    batch_size = 2
    max_pages = 5
    physical_pages = batch_size * max_pages
    if backend == "flashinfer_fp16":
        cache = SimpleNamespace(
            key=torch.zeros(1, physical_pages, 2),
            value=torch.zeros(1, physical_pages, 2),
        )
    else:
        cache = SimpleNamespace(
            key_codes=torch.zeros(1, physical_pages, 2, dtype=torch.int8),
            value_codes=torch.zeros(1, physical_pages, 2, dtype=torch.int8),
            key_scales=torch.zeros(1, physical_pages, 2),
            value_scales=torch.zeros(1, physical_pages, 2),
        )
    decoder = SimpleNamespace(
        backend=backend,
        cache=cache,
        max_pages=max_pages,
        batch_size=batch_size,
    )

    logical_page = MODULE.initialize_following_page_canary(
        decoder, start_position=16, decode_steps=16
    )
    assert logical_page == 2
    selected = [2, max_pages + 2]
    untouched = [index for index in range(physical_pages) if index not in selected]
    if backend == "flashinfer_fp16":
        assert torch.all(cache.key[:, selected] == 0.125)
        assert torch.all(cache.value[:, selected] == -0.25)
        assert torch.count_nonzero(cache.key[:, untouched]) == 0
        assert torch.count_nonzero(cache.value[:, untouched]) == 0
    else:
        assert torch.all(cache.key_codes[:, selected] == -113)
        assert torch.all(cache.value_codes[:, selected] == 107)
        assert torch.all(cache.key_scales[:, selected] == 0.75)
        assert torch.all(cache.value_scales[:, selected] == 1.25)
        assert torch.count_nonzero(cache.key_codes[:, untouched]) == 0
        assert torch.count_nonzero(cache.value_codes[:, untouched]) == 0
        assert torch.count_nonzero(cache.key_scales[:, untouched]) == 0
        assert torch.count_nonzero(cache.value_scales[:, untouched]) == 0
