from __future__ import annotations

import importlib.util
import sys
from collections import UserList
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "page_gauge_transformer", ROOT / "scripts/benchmark_page_gauge_transformer.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_transformer_statistics_are_deterministic() -> None:
    values = [1.1, 1.2, 1.3, 1.4]
    assert 1.1 < MODULE.geometric_mean(values) < 1.4
    first = MODULE.bootstrap_geomean(values, 91, samples=200)
    second = MODULE.bootstrap_geomean(values, 91, samples=200)
    assert first == second
    assert first[0] <= MODULE.geometric_mean(values) <= first[1]

    groups = [[1.01, 1.02, 0.99], [1.03, 1.00, 1.01]]
    clustered_first = MODULE.hierarchical_bootstrap_geomean(
        groups, 92, samples=200
    )
    clustered_second = MODULE.hierarchical_bootstrap_geomean(
        groups, 92, samples=200
    )
    assert clustered_first == clustered_second
    flattened_geomean = MODULE.geometric_mean([value for group in groups for value in group])
    assert clustered_first[0] <= flattened_geomean <= clustered_first[1]


def test_percentile_endpoints() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert MODULE.percentile(values, 0.0) == 1.0
    assert MODULE.percentile(values, 1.0) == 4.0


def test_logit_sequence_comparison_reports_offsets_and_exactness() -> None:
    expected = [MODULE.torch.tensor([float(index)]) for index in range(17)]
    identical = MODULE.compare_logit_sequences(expected, [value.clone() for value in expected])
    assert identical["bitwise_identical"]
    assert identical["per_offset_maximum_absolute_error"] == {
        "0": 0.0,
        "14": 0.0,
        "15": 0.0,
        "16": 0.0,
    }

    changed = [value.clone() for value in expected]
    changed[15] += 0.5
    comparison = MODULE.compare_logit_sequences(expected, changed)
    assert not comparison["bitwise_identical"]
    assert comparison["maximum_absolute_error"] == 0.5


def test_value_center_folds_into_output_projection_bias() -> None:
    generator = MODULE.torch.Generator().manual_seed(7)
    weight = MODULE.torch.randn(9, 12, generator=generator, dtype=MODULE.torch.float64)
    original_bias = MODULE.torch.randn(9, generator=generator, dtype=MODULE.torch.float64)
    center = MODULE.torch.randn(3, 4, generator=generator, dtype=MODULE.torch.float64)
    attended = MODULE.torch.randn(2, 12, generator=generator, dtype=MODULE.torch.float64)
    folded_bias = MODULE.folded_output_projection_bias(
        weight, original_bias, center
    )
    legacy = MODULE.F.linear(attended + center.reshape(1, -1), weight, original_bias)
    folded = MODULE.F.linear(attended, weight, folded_bias)
    MODULE.torch.testing.assert_close(legacy, folded, rtol=1e-12, atol=1e-12)


def test_batched_value_centers_produce_per_request_projection_corrections() -> None:
    generator = MODULE.torch.Generator().manual_seed(8)
    weight = MODULE.torch.randn(9, 12, generator=generator, dtype=MODULE.torch.float64)
    original_bias = MODULE.torch.randn(9, generator=generator, dtype=MODULE.torch.float64)
    centers = MODULE.torch.randn(
        2, 3, 4, generator=generator, dtype=MODULE.torch.float64
    )
    attended = MODULE.torch.randn(2, 12, generator=generator, dtype=MODULE.torch.float64)
    corrections = MODULE.folded_output_projection_bias(
        weight, original_bias, centers
    )
    legacy = MODULE.F.linear(
        attended + centers.reshape(2, -1), weight, original_bias
    )
    explicit_per_request = MODULE.F.linear(attended, weight, None) + corrections
    MODULE.torch.testing.assert_close(
        legacy, explicit_per_request, rtol=1e-12, atol=1e-12
    )


def test_request_major_page_tables_cover_full_and_ring_storage() -> None:
    full = MODULE.request_major_page_table(2, 4, device="cpu")
    ring = MODULE.request_major_page_table(2, 4, 2, device="cpu")
    assert full.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert ring.tolist() == [[0, 1, 0, 1], [2, 3, 2, 3]]


def test_exact_sink_partition_is_disjoint_and_covers_context_once() -> None:
    partition = MODULE.page_gauge_logical_partition(
        total_pages=1283,
        exact_tail_pages=16,
        exact_sink_pages=1,
        last_page_len=7,
    )

    assert partition["sink_logical_pages"] == (0,)
    assert partition["old_logical_pages"] == tuple(range(1, 1267))
    assert partition["tail_logical_pages"] == tuple(range(1267, 1283))
    assert partition["old_page_count"] == 1266
    assert partition["exact_page_count"] == 17
    assert partition["old_token_count"] == 1266 * MODULE.PAGE
    assert partition["exact_token_count"] == MODULE.PAGE + 15 * MODULE.PAGE + 7
    assert partition["context_token_count"] == 1282 * MODULE.PAGE + 7
    assert partition["coverage_disjoint"] is True


def test_exact_sink_page_table_uses_slot_zero_then_chronological_tail() -> None:
    physical = MODULE.request_major_exact_page_table(
        batch_size=2,
        logical_page_capacity=8,
        exact_tail_pages=2,
        exact_sink_pages=1,
        device="cpu",
    )
    destination = MODULE.torch.full((2, 3), -1, dtype=MODULE.torch.int32)

    active = MODULE.write_exact_sink_tail_page_table(
        destination,
        physical,
        tail_logical_begin=6,
        total_page_count=8,
        exact_sink_pages=1,
    )

    assert physical.tolist() == [
        [0, 2, 1, 2, 1, 2, 1, 2],
        [3, 5, 4, 5, 4, 5, 4, 5],
    ]
    assert active.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert active.data_ptr() == destination.data_ptr()
    assert len(set(active[0].tolist())) == 3
    assert len(set(active[1].tolist())) == 3
    assert all(0 <= value < 3 for value in active[0].tolist())
    assert all(3 <= value < 6 for value in active[1].tolist())


def test_exact_prefix_s3_partition_and_page_table_are_disjoint_and_chronological() -> None:
    partition = MODULE.page_gauge_logical_partition(8, 2, 3, MODULE.PAGE)
    assert partition["prefix_logical_pages"] == (0, 1, 2)
    assert partition["sink_logical_pages"] == (0, 1, 2)
    assert partition["old_logical_pages"] == (3, 4, 5)
    assert partition["tail_logical_pages"] == (6, 7)
    assert partition["coverage_disjoint"] is True
    assert partition["old_token_count"] + partition["exact_token_count"] == (
        partition["context_token_count"]
    )

    physical = MODULE.request_major_exact_page_table(2, 8, 2, 3, device="cpu")
    destination = MODULE.torch.full((2, 5), -1, dtype=MODULE.torch.int32)
    active = MODULE.write_exact_prefix_tail_page_table(
        destination,
        physical,
        tail_logical_begin=6,
        total_page_count=8,
        exact_sink_pages=3,
    )
    assert physical.tolist() == [
        [0, 1, 2, 4, 3, 4, 3, 4],
        [5, 6, 7, 9, 8, 9, 8, 9],
    ]
    assert active.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert len(set(active[0].tolist())) == 5
    assert len(set(active[1].tolist())) == 5


def test_exact_prefix_partition_fails_closed_for_negative_or_no_old_pages() -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        MODULE.page_gauge_logical_partition(8, 2, -1, MODULE.PAGE)
    with pytest.raises(ValueError, match="quantized old page"):
        MODULE.page_gauge_logical_partition(3, 2, 1, MODULE.PAGE)
    with pytest.raises(ValueError, match="quantized old page"):
        MODULE.page_gauge_logical_partition(5, 2, 3, MODULE.PAGE)
    with pytest.raises(ValueError, match="segmented flashinfer_merge"):
        MODULE.validate_exact_prefix_attention_path(3, "fused_kernel")
    with pytest.raises(ValueError, match="segmented flashinfer_merge"):
        MODULE.validate_exact_prefix_attention_path(3, "heterogeneous_fa2")
    assert MODULE.validate_exact_prefix_attention_path(3, "flashinfer_merge") == 3


def test_heterogeneous_page_table_uses_old_storage_then_exact_ring() -> None:
    full = MODULE.request_major_page_table(2, 5, device="cpu")
    ring = MODULE.request_major_page_table(2, 5, 2, device="cpu")
    destination = MODULE.torch.full((2, 5), -1, dtype=MODULE.torch.int32)

    active = MODULE.write_heterogeneous_page_table(
        destination,
        full,
        ring,
        old_page_count=3,
        total_page_count=5,
    )

    assert active.tolist() == [[0, 1, 2, 1, 0], [5, 6, 7, 3, 2]]
    assert active.data_ptr() == destination.data_ptr()
    with pytest.raises(ValueError, match="outside table capacity"):
        MODULE.write_heterogeneous_page_table(destination, full, ring, 4, 6)


def test_graph_decode_wrapper_plans_request_major_batched_metadata() -> None:
    class FakeWrapper:
        def __init__(self) -> None:
            self.calls = []
            self._plan_info = [8, 2, 64, 16, 0, 16, 32, 48, 64, 80, 0, 0, 96, 1, 1]
            self._backend = "fa2"

        def plan(self, indptr, indices, last_len, *args, **kwargs) -> None:
            self.calls.append(
                (indptr.clone(), indices.clone(), last_len.clone(), args, kwargs)
            )

    graph = MODULE.GraphDecodeWrapper.__new__(MODULE.GraphDecodeWrapper)
    graph.batch_size = 2
    graph.indptr = MODULE.torch.empty(3, dtype=MODULE.torch.int32)
    graph.indices = MODULE.torch.empty(8, dtype=MODULE.torch.int32)
    graph.last_len = MODULE.torch.empty(2, dtype=MODULE.torch.int32)
    graph.wrapper = FakeWrapper()
    graph.kv_dtype = MODULE.torch.float16
    graph.run_semantics = ("NHD", "fake")
    graph.signature = None
    graph.structural_signature = None
    graph.planned_split_pages = None
    graph.plan_invocations = 0
    graph.plan_rebuilds = 0
    graph.last_len_fills = 0
    indices = MODULE.torch.tensor([[0, 1], [4, 5]], dtype=MODULE.torch.int32)

    graph.plan(indices, logical_tokens=19, last_page_len=3, split_pages=0)
    assert len(graph.wrapper.calls) == 1
    indptr, flat_indices, last_len, _, _ = graph.wrapper.calls[0]
    assert indptr.tolist() == [0, 2, 4]
    assert flat_indices.tolist() == [0, 1, 4, 5]
    assert last_len.tolist() == [3, 3]

    graph.plan(indices, logical_tokens=20, last_page_len=4, split_pages=0)
    assert len(graph.wrapper.calls) == 1
    assert graph.last_len.tolist() == [4, 4]
    assert graph.structural_signature is not None
    assert graph.operation_counts() == {
        "plan_invocations": 2,
        "plan_rebuilds": 1,
        "last_page_len_device_fills": 2,
    }

    indices[0, 0] = 7
    graph.plan(
        indices,
        logical_tokens=20,
        last_page_len=4,
        split_pages=0,
        page_table_epoch=2,
    )
    assert len(graph.wrapper.calls) == 2
    assert graph.indices[:4].tolist() == [7, 1, 4, 5]
    assert graph.operation_counts()["plan_rebuilds"] == 2


def test_decoder_plan_excludes_sink_from_old_and_prepends_it_to_exact() -> None:
    class FakePlan:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, indices, tokens, last_len, split, **kwargs) -> None:
            self.calls.append(
                (indices.clone(), tokens, last_len, split, kwargs)
            )

    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.decoder_plan_calls = 0
    decoder.max_pages = 8
    decoder.batch_size = 2
    decoder.backend = "page_gauge"
    decoder.exact_tail_pages = 2
    decoder.exact_sink_pages = 1
    decoder.candidate_split_pages = 32
    decoder.tail_attention = "flashinfer_merge"
    decoder.all_pages = MODULE.request_major_page_table(2, 8, device="cpu")
    decoder.exact_ring_pages = MODULE.request_major_exact_page_table(
        2, 8, 2, 1, device="cpu"
    )
    decoder.exact_plan_pages = MODULE.torch.empty(2, 3, dtype=MODULE.torch.int32)
    decoder.exact_layout_signature = None
    decoder.exact_page_table_updates = 0
    decoder.old_wrapper = FakePlan()
    decoder.exact_wrapper = FakePlan()
    decoder.heterogeneous_wrapper = None
    decoder.heterogeneous_pages = None
    decoder.heterogeneous_old_kv_len = None

    decoder.plan(8 * MODULE.PAGE)

    old_indices, old_tokens, old_last, _, _ = decoder.old_wrapper.calls[-1]
    exact_indices, exact_tokens, exact_last, _, exact_kwargs = (
        decoder.exact_wrapper.calls[-1]
    )
    assert old_indices.tolist() == [
        [1, 2, 3, 4, 5],
        [9, 10, 11, 12, 13],
    ]
    assert exact_indices.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert old_tokens == 5 * MODULE.PAGE
    assert exact_tokens == 3 * MODULE.PAGE
    assert old_last == exact_last == MODULE.PAGE
    assert exact_kwargs["page_table_epoch"] == 8
    assert decoder.exact_page_table_updates == 1
    assert decoder.old_logical_begin == 1
    assert decoder.old_logical_end == decoder.tail_logical_begin == 6

    decoder.plan(8 * MODULE.PAGE - 1)
    assert decoder.exact_page_table_updates == 1
    assert decoder.exact_wrapper.calls[-1][1] == 3 * MODULE.PAGE - 1


def test_decoder_plan_s3_uses_one_old_and_one_exact_wrapper_without_duplication() -> None:
    class FakePlan:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, indices, tokens, last_len, split, **kwargs) -> None:
            self.calls.append((indices.clone(), tokens, last_len, split, kwargs))

    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.decoder_plan_calls = 0
    decoder.max_pages = 8
    decoder.batch_size = 2
    decoder.backend = "page_gauge"
    decoder.exact_tail_pages = 2
    decoder.exact_prefix_pages = 3
    decoder.exact_sink_pages = 3
    decoder.candidate_split_pages = 32
    decoder.tail_attention = "flashinfer_merge"
    decoder.all_pages = MODULE.request_major_page_table(2, 8, device="cpu")
    decoder.exact_ring_pages = MODULE.request_major_exact_page_table(
        2, 8, 2, 3, device="cpu"
    )
    decoder.exact_plan_pages = MODULE.torch.empty(2, 5, dtype=MODULE.torch.int32)
    decoder.exact_layout_signature = None
    decoder.exact_page_table_updates = 0
    decoder.old_wrapper = FakePlan()
    decoder.exact_wrapper = FakePlan()
    decoder.heterogeneous_wrapper = None
    decoder.heterogeneous_pages = None
    decoder.heterogeneous_old_kv_len = None

    decoder.plan(8 * MODULE.PAGE)

    old_indices, old_tokens, _, _, _ = decoder.old_wrapper.calls[-1]
    exact_indices, exact_tokens, _, _, exact_kwargs = decoder.exact_wrapper.calls[-1]
    assert old_indices.tolist() == [[3, 4, 5], [11, 12, 13]]
    assert exact_indices.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert old_tokens == 3 * MODULE.PAGE
    assert exact_tokens == 5 * MODULE.PAGE
    assert exact_kwargs["page_table_epoch"] == 8
    assert decoder.old_logical_begin == 3
    assert decoder.old_logical_end == decoder.tail_logical_begin == 6
    assert decoder.old_pages + decoder.exact_pages == 8


def test_decoder_plan_fixed_suffix_uses_two_wrappers_and_exact_once_tables() -> None:
    class FakePlan:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, indices, tokens, last_len, split, **kwargs) -> None:
            self.calls.append((indices.clone(), tokens, last_len, split, kwargs))

    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.decoder_plan_calls = 0
    decoder.max_pages = 10
    decoder.batch_size = 1
    decoder.backend = "page_gauge"
    decoder.exact_tail_pages = 2
    decoder.exact_prefix_pages = 1
    decoder.exact_sink_pages = 1
    decoder.exact_static_suffix_pages = 2
    decoder.initial_context_pages = 8
    decoder.exact_fixed_pages = 3
    decoder.candidate_split_pages = 32
    decoder.tail_attention = "flashinfer_merge"
    decoder.all_pages = MODULE.request_major_page_table(1, 10, device="cpu")
    decoder.exact_ring_pages = MODULE.request_major_exact_page_table(
        1, 10, 2, 1, 2, 8, device="cpu"
    )
    decoder.exact_plan_pages = MODULE.torch.empty(1, 5, dtype=MODULE.torch.int32)
    decoder.old_plan_pages = MODULE.torch.empty(1, 10, dtype=MODULE.torch.int32)
    decoder.exact_layout_signature = None
    decoder.old_layout_signature = None
    decoder.exact_page_table_updates = 0
    decoder.old_wrapper = FakePlan()
    decoder.exact_wrapper = FakePlan()
    decoder.heterogeneous_wrapper = None
    decoder.heterogeneous_pages = None
    decoder.heterogeneous_old_kv_len = None

    decoder.plan(8 * MODULE.PAGE)
    old_indices, old_tokens, _, _, _ = decoder.old_wrapper.calls[-1]
    exact_indices, exact_tokens, _, _, _ = decoder.exact_wrapper.calls[-1]
    assert old_indices.tolist() == [[1, 2, 3, 4, 5]]
    assert exact_indices.tolist() == [[0, 1, 2]]
    assert old_tokens == 5 * MODULE.PAGE
    assert exact_tokens == 3 * MODULE.PAGE

    decoder.plan(8 * MODULE.PAGE + 1)
    old_indices = decoder.old_wrapper.calls[-1][0]
    exact_indices = decoder.exact_wrapper.calls[-1][0]
    assert old_indices.tolist() == [[1, 2, 3, 4, 5]]
    assert exact_indices.tolist() == [[0, 1, 2, 3]]
    assert len(set(exact_indices[0].tolist())) == 4
    assert decoder.old_pages + decoder.exact_pages == 9


def test_mutated_cache_range_covers_all_generated_pages_and_ring_wraps() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.max_pages = 1312
    decoder.batch_size = 4
    decoder.backend = "page_gauge"
    decoder.exact_tail_pages = 16
    decoder.exact_sink_pages = 0

    pages = MODULE.mutated_cache_page_indices(decoder, 20480, 512)

    assert len(pages["full"]) == 4 * 32
    assert pages["full"][:32] == tuple(range(1280, 1312))
    assert pages["full"][32:64] == tuple(range(2592, 2624))
    assert len(pages["exact_ring"]) == 4 * 16
    assert pages["exact_ring"][:16] == tuple(range(16))
    assert pages["exact_ring"][16:32] == tuple(range(16, 32))


def test_mutated_cache_snapshot_excludes_immutable_sink_slot() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.max_pages = 1312
    decoder.batch_size = 2
    decoder.backend = "page_gauge"
    decoder.exact_tail_pages = 16
    decoder.exact_sink_pages = 1

    pages = MODULE.mutated_cache_page_indices(decoder, 20480, 512)

    assert len(pages["exact_ring"]) == 2 * 16
    assert pages["exact_ring"][:16] == tuple(range(1, 17))
    assert pages["exact_ring"][16:] == tuple(range(18, 34))
    assert 0 not in pages["exact_ring"]
    assert 17 not in pages["exact_ring"]


def test_mutated_cache_snapshot_s3_excludes_every_immutable_prefix_slot() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.max_pages = 1312
    decoder.batch_size = 2
    decoder.backend = "page_gauge"
    decoder.exact_tail_pages = 16
    decoder.exact_prefix_pages = 3
    decoder.exact_sink_pages = 3

    pages = MODULE.mutated_cache_page_indices(decoder, 20480, 512)

    assert len(pages["exact_ring"]) == 2 * 16
    assert pages["exact_ring"][:16] == tuple(range(3, 19))
    assert pages["exact_ring"][16:] == tuple(range(22, 38))
    assert set(pages["exact_ring"]).isdisjoint({0, 1, 2, 19, 20, 21})
    with pytest.raises(ValueError, match="overlaps immutable exact-prefix"):
        MODULE.mutated_cache_page_indices(decoder, 0, MODULE.PAGE)

    snapshot = {
        "backend": "page_gauge",
        "start_position": 20480,
        "decode_steps": 512,
        "mapping_policy": MODULE.cache_range_mapping_policy(decoder),
        "page_indices": pages,
    }
    assert MODULE.validated_cache_range_snapshot_pages(decoder, snapshot) == pages
    stale_indices = dict(pages)
    stale_indices["exact_ring"] = (0, *pages["exact_ring"][1:])
    stale_snapshot = {**snapshot, "page_indices": stale_indices}
    with pytest.raises(ValueError, match="stale or unsafe"):
        MODULE.validated_cache_range_snapshot_pages(decoder, stale_snapshot)
    stale_policy = dict(snapshot["mapping_policy"])
    stale_policy["exact_prefix_pages_per_request"] = 1
    with pytest.raises(ValueError, match="mapping policy is stale"):
        MODULE.validated_cache_range_snapshot_pages(
            decoder, {**snapshot, "mapping_policy": stale_policy}
        )


def test_dynamic_graph_preflight_attests_every_position_and_bucket() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.device_dynamic_decoder_layer_graphs = True
    decoder.rope_cos = MODULE.torch.empty(1024, 128)
    decoder.decoder_layer_graph_preflight_positions = {}
    decoder.current_pages = 1
    decoder.old_pages = 0
    decoder.exact_pages = 1
    current = {"context": 0}

    def plan(context: int) -> None:
        current["context"] = context
        decoder.current_pages = (context + MODULE.PAGE - 1) // MODULE.PAGE

    decoder.plan = plan
    decoder.structural_plan_signature = lambda: (
        "bucket",
        (current["context"] - 1) // 128,
    )
    decoder.structural_plan_info = lambda: {
        "fake": {"plan_info": [current["context"] // 128]}
    }
    decoder.plan_signature = lambda: (
        "runtime",
        (current["context"] - 1) // 128,
    )
    decoder.serving_metadata_snapshot = lambda: {
        "wrappers": {
            "fake": {
                "pages_per_request": current["context"],
                "kv_chunk_size_tokens": 256,
                "effective_valid_split_tiles": 4,
                "initialized_tile_count": 4,
                "semantic_int_workspace_sha256": "0" * 64,
            }
        }
    }

    representatives = decoder.preflight_dynamic_decoder_layer_graph_banks(
        256, 512, maximum_graph_banks=8
    )

    assert len(representatives) == 4
    assert decoder.decoder_layer_graph_preflight_position_count == 512
    assert decoder.decoder_layer_graph_preflight_range == (256, 768)
    assert len(decoder.decoder_layer_graph_preflight_positions_sha256) == 64
    assert sum(decoder.decoder_layer_graph_preflight_counts.values()) == 512


def test_freeze_structural_value_rejects_unknown_plan_abi() -> None:
    with pytest.raises(TypeError, match="unsupported FlashInfer structural plan value"):
        MODULE.freeze_structural_value(object())


def test_freeze_structural_value_canonicalizes_installed_tvm_ffi_array() -> None:
    container = pytest.importorskip("tvm_ffi.container")
    plan_info = container.Array(list(range(15)))
    frozen = MODULE.freeze_structural_value(plan_info)
    assert frozen == (
        "tvm_ffi.container.Array",
        tuple(range(15)),
    )
    assert frozen == MODULE.freeze_structural_value(container.Array(range(15)))
    assert hash(frozen) == hash(MODULE.freeze_structural_value(plan_info))
    changed = container.Array([*range(14), 99])
    assert MODULE.freeze_structural_value(changed) != frozen
    nested = container.Array([container.Array([1, 2]), 3])
    assert MODULE.freeze_structural_value(nested) == (
        "tvm_ffi.container.Array",
        (("tvm_ffi.container.Array", (1, 2)), 3),
    )
    with pytest.raises(TypeError, match="unsupported FlashInfer structural plan value"):
        MODULE.freeze_structural_value(UserList([1, 2, 3]))


def test_serving_metadata_rejects_wrong_length_tvm_ffi_plan() -> None:
    container = pytest.importorskip("tvm_ffi.container")
    wrapper = MODULE.GraphDecodeWrapper.__new__(MODULE.GraphDecodeWrapper)
    wrapper.signature = (1, 1, 1, 0, 0)
    wrapper.wrapper = SimpleNamespace(_plan_info=container.Array(range(14)))
    with pytest.raises(RuntimeError, match="15 fields"):
        wrapper.serving_metadata_snapshot()


def test_step_major_token_conversion_preserves_batch_one_compatibility() -> None:
    legacy = MODULE.token_tensor_from_sequence([2, 3, 4], 1, device="cpu")
    batched = MODULE.token_tensor_from_sequence(
        [[2, 20], [3, 30], [4, 40]], 2, device="cpu"
    )
    assert legacy.shape == (3, 1)
    assert legacy[:, 0].tolist() == [2, 3, 4]
    assert batched.shape == (3, 2)
    assert batched.tolist() == [[2, 20], [3, 30], [4, 40]]
    with pytest.raises(ValueError, match="step-major"):
        MODULE.token_tensor_from_sequence([2, 3], 2, device="cpu")


def test_batched_sequence_summary_distinguishes_steps_and_output_tokens() -> None:
    summary = MODULE.summarize_sequence([8.0, 8.0], [10.0, 10.0], 4, 2)
    assert summary["decode_steps"] == 4
    assert summary["output_tokens_per_sequence"] == 8
    assert summary["wall_p50_ms_per_decode_step"] == 2.5
    assert summary["wall_p50_ms_per_token"] == 1.25
    assert summary["wall_tokens_per_second"] == 800.0
    assert summary["wall_decode_steps_per_second"] == 400.0


def test_production_defaults_use_validated_tail_and_center_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--output", str(tmp_path / "result.json")],
    )
    args = MODULE.parse_args()
    assert args.batch_size == 1
    assert args.tail_attention == "flashinfer_merge"
    assert args.center_restore == "attention_add"
    assert args.old_value_scale_placement == "probability"
    assert args.exact_sink_pages == 0


def test_exact_prefix_s3_is_explicitly_selectable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--exact-sink-pages",
            "3",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert MODULE.parse_args().exact_sink_pages == 3


def test_value_fragment_scale_placement_is_explicitly_selectable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--old-value-scale-placement",
            "value_fragment",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert MODULE.parse_args().old_value_scale_placement == "value_fragment"


def test_heterogeneous_fa2_is_explicitly_selectable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--tail-attention",
            "heterogeneous_fa2",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert MODULE.parse_args().tail_attention == "heterogeneous_fa2"


def test_decoder_layer_graph_scope_is_opt_in_and_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--output", str(tmp_path / "default.json")],
    )
    assert MODULE.selected_cuda_graph_scope(MODULE.parse_args()) == "attention"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--decoder-layer-cuda-graphs",
            "--output",
            str(tmp_path / "layer.json"),
        ],
    )
    assert MODULE.selected_cuda_graph_scope(MODULE.parse_args()) == "decoder_layer"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--decoder-layer-cuda-graphs",
            "--disable-attention-cuda-graphs",
            "--output",
            str(tmp_path / "invalid.json"),
        ],
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE.selected_cuda_graph_scope(MODULE.parse_args())


def test_decoder_layer_graph_guard_binds_exact_logical_page_and_plan() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.decoder_layer_graphs = [[object()]] * MODULE.PAGE
    decoder.decoder_layer_graph_start_position = 20480
    decoder.decoder_layer_graph_plan_signature = ("bucket", 641)
    decoder.plan_signature = lambda: ("bucket", 641)

    assert decoder._decoder_layer_graph_offset(20480) == 0
    assert decoder._decoder_layer_graph_offset(20495) == 15
    assert decoder._decoder_layer_graph_offset(20479) is None
    assert decoder._decoder_layer_graph_offset(20496) is None
    assert decoder._decoder_layer_graph_offset(20464) is None

    decoder.plan_signature = lambda: ("different", 642)
    assert decoder._decoder_layer_graph_offset(20480) is None


def test_decoder_layer_dispatch_is_not_counted_as_nested_attention_graphs() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.attention_graph_replays = 0
    decoder.attention_eager_calls = 0
    decoder.decoder_layer_graph_replays = MODULE.PAGE * 32
    decoder.decoder_layer_eager_calls = 0

    layer = decoder.decoder_layer_dispatch_counts()
    attention = decoder.attention_dispatch_counts()
    assert layer["graph_replays"] == 512
    assert layer["graph_coverage_fraction"] == 1.0
    assert attention["total_calls"] == 0
    assert MODULE.selected_cuda_graph_dispatch_counts(
        decoder, "decoder_layer"
    ) == layer
    assert MODULE.selected_cuda_graph_dispatch_counts(decoder, "attention") == attention


def test_decoder_layer_graph_provenance_reports_complete_boundary() -> None:
    decoder = MODULE.TransformerDecoder.__new__(MODULE.TransformerDecoder)
    decoder.layers = 32
    decoder.decoder_layer_graphs = [
        [object() for _ in range(decoder.layers)] for _ in range(MODULE.PAGE)
    ]
    decoder.decoder_layer_graph_buffers = MODULE.torch.empty(
        MODULE.PAGE, decoder.layers + 1, 1, 2
    )
    decoder.decoder_layer_graph_start_position = 20480

    provenance = decoder.decoder_layer_graph_provenance()
    assert provenance["enabled"]
    assert provenance["total_graphs"] == 512
    assert provenance["graphs_per_offset"] == 32
    assert provenance["captured_page_offsets"] == list(range(16))
    assert provenance["nested_attention_graphs"] is False
    assert "persistent_layer_output_write" in provenance["included_operations"]
    assert "lm_head" in provenance["excluded_operations"]


def test_mistral_none_head_dim_uses_hidden_size_fallback() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=None,
            model_type="mistral",
        )
    )
    assert MODULE.BASE_E2E.check_model(model) == (32, 32, 8, 4096)


def test_fixed_suffix_mapping_and_partition_are_exact_once() -> None:
    prefix = 4
    suffix = 64
    tail = 48
    initial = 1280
    storage = prefix + suffix + tail
    assert MODULE.exact_physical_page_index(
        0, 0, tail, prefix, suffix, initial
    ) == 0
    assert MODULE.exact_physical_page_index(
        0, initial - suffix, tail, prefix, suffix, initial
    ) == prefix
    assert MODULE.exact_physical_page_index(
        0, initial - 1, tail, prefix, suffix, initial
    ) == prefix + suffix - 1
    generated = initial
    assert MODULE.exact_physical_page_index(
        0, generated, tail, prefix, suffix, initial
    ) == prefix + suffix + generated % tail
    assert MODULE.exact_physical_page_index(
        1, generated, tail, prefix, suffix, initial
    ) == storage + prefix + suffix + generated % tail

    initial_partition = MODULE.page_gauge_logical_partition(
        initial, tail, prefix, MODULE.PAGE, suffix, initial
    )
    assert initial_partition["exact_page_count"] == prefix + suffix
    assert initial_partition["old_logical_pages"] == tuple(
        range(prefix, initial - suffix)
    )
    assert initial_partition["coverage_disjoint"]

    decode_partition = MODULE.page_gauge_logical_partition(
        initial + tail, tail, prefix, 13, suffix, initial
    )
    assert decode_partition["exact_page_count"] == prefix + suffix + tail
    assert set(decode_partition["old_logical_pages"]).isdisjoint(
        decode_partition["static_suffix_logical_pages"]
    )
    assert decode_partition["coverage_disjoint"]


def test_fixed_suffix_active_tables_are_unique_and_keep_current_page_last() -> None:
    prefix = 4
    suffix = 64
    tail = 48
    initial = 1280
    capacity = initial + 96
    logical = MODULE.request_major_exact_page_table(
        1,
        capacity,
        tail,
        prefix,
        suffix,
        initial,
        device="cpu",
    )
    destination = MODULE.torch.empty(
        1, prefix + suffix + tail, dtype=MODULE.torch.int32
    )
    initial_active = MODULE.write_exact_prefix_tail_page_table(
        destination,
        logical,
        initial - tail,
        initial,
        prefix,
        suffix,
        initial,
    )
    assert initial_active.shape[1] == prefix + suffix
    assert len(set(initial_active[0].tolist())) == prefix + suffix

    next_active = MODULE.write_exact_prefix_tail_page_table(
        destination,
        logical,
        initial - tail + 1,
        initial + 1,
        prefix,
        suffix,
        initial,
    )
    assert next_active.shape[1] == prefix + suffix + 1
    assert len(set(next_active[0].tolist())) == prefix + suffix + 1
    assert next_active[0, -1].item() == MODULE.exact_physical_page_index(
        0, initial, tail, prefix, suffix, initial
    )
