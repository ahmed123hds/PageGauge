from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "page_gauge_quality_layer_trace",
    ROOT / "diagnostics/trace_quality_outlier_layers.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_result() -> dict:
    return {
        "configuration": {
            "context": 20480,
            "decode_steps": 512,
            "batch_size": 4,
            "exact_tail_tokens": 512,
            "exact_sink_pages": 1,
            "exact_prefix_pages": 1,
            "tail_attention": "flashinfer_merge",
        },
        "correctness": {
            "backend_vs_hf_sdpa_fp16": {
                "outlier_diagnostics": {
                    "worst_rows": [
                        {
                            "rank": 1,
                            "step": 364,
                            "request": 0,
                            "absolute_position": 20844,
                            "below_strict_gate": True,
                        },
                        {
                            "rank": 2,
                            "step": 415,
                            "request": 0,
                            "absolute_position": 20895,
                            "below_strict_gate": True,
                        },
                        {
                            "rank": 3,
                            "step": 10,
                            "request": 2,
                            "absolute_position": 20490,
                            "below_strict_gate": False,
                        },
                    ]
                }
            }
        },
    }


def test_target_selection_uses_only_saved_below_gate_rows() -> None:
    targets = MODULE.select_trace_targets(source_result(), [], maximum_targets=4)
    assert [(row["step"], row["request"]) for row in targets] == [
        (364, 0),
        (415, 0),
    ]
    assert targets[0]["source_outlier_row"]["rank"] == 1


def test_explicit_target_is_validated_and_deduplicated() -> None:
    targets = MODULE.select_trace_targets(
        source_result(), ["37:2", "37:2"], maximum_targets=1
    )
    assert [(row["step"], row["request"]) for row in targets] == [(37, 2)]
    with pytest.raises(ValueError, match="outside D"):
        MODULE.select_trace_targets(source_result(), ["512:0"], maximum_targets=2)
    with pytest.raises(ValueError, match="STEP:REQUEST"):
        MODULE.select_trace_targets(source_result(), ["37"], maximum_targets=2)


def test_source_exact_cache_policy_reproduces_sink_and_tail() -> None:
    policy = MODULE.source_exact_cache_policy(source_result()["configuration"])
    assert policy == {
        "exact_tail_tokens": 512,
        "exact_tail_pages": 32,
        "exact_prefix_pages": 1,
        "exact_sink_pages": 1,
        "exact_storage_pages": 33,
    }

    generic_s3 = dict(
        source_result()["configuration"],
        exact_sink_pages=3,
        exact_prefix_pages=3,
    )
    assert MODULE.source_exact_cache_policy(generic_s3) == {
        "exact_tail_tokens": 512,
        "exact_tail_pages": 32,
        "exact_prefix_pages": 3,
        "exact_sink_pages": 3,
        "exact_storage_pages": 35,
    }

    invalid_prefix = dict(generic_s3, exact_prefix_pages=-1, exact_sink_pages=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        MODULE.source_exact_cache_policy(invalid_prefix)

    inconsistent_alias = dict(generic_s3, exact_sink_pages=1)
    with pytest.raises(ValueError, match="fields disagree"):
        MODULE.source_exact_cache_policy(inconsistent_alias)

    invalid_backend = dict(
        generic_s3, tail_attention="fused_kernel"
    )
    with pytest.raises(ValueError, match="exact prefix requires segmented"):
        MODULE.source_exact_cache_policy(invalid_backend)


def test_planner_exact_page_age_mapping_corrects_partial_page_floor() -> None:
    before_generated_aging = MODULE.page_age_mapping(
        context=20480, step=37, exact_tail_tokens=256
    )
    assert before_generated_aging["total_pages"] == 1283
    assert before_generated_aging["old_int8_pages"] == 1267
    assert before_generated_aging["exact_fp16_pages"] == 16
    assert before_generated_aging["generated_pages_currently_old_int8"] == 0
    assert before_generated_aging["initial_prefix_pages_currently_old_int8"] == 1267

    partial_page = MODULE.page_age_mapping(
        context=20480, step=364, exact_tail_tokens=256
    )
    assert partial_page["sequence_length_after_append"] == 20845
    assert partial_page["total_pages"] == 1303
    assert partial_page["old_int8_pages"] == 1287
    assert partial_page["generated_pages_currently_old_int8"] == 7
    assert partial_page["legacy_floor_formula_generated_old_pages"] == 6
    assert partial_page["legacy_floor_formula_matches_planner"] is False
    assert partial_page["exact_ring_mapping"][0]["logical_page"] == 1287
    assert partial_page["exact_ring_mapping"][-1]["logical_page"] == 1302

    aligned = MODULE.page_age_mapping(
        context=20480, step=415, exact_tail_tokens=256
    )
    assert aligned["generated_pages_currently_old_int8"] == 10
    assert aligned["legacy_floor_formula_matches_planner"] is True


def test_sink_aware_page_mapping_is_exact_disjoint_and_request_local() -> None:
    mapping = MODULE.page_age_mapping(
        context=20480,
        step=594,
        exact_tail_tokens=512,
        exact_sink_pages=1,
    )
    assert mapping["sequence_length_after_append"] == 21075
    assert mapping["total_pages"] == 1318
    assert mapping["old_logical_range_start_inclusive_end_exclusive"] == [1, 1286]
    assert mapping["old_int8_pages"] == 1285
    assert mapping["exact_fp16_pages"] == 33
    assert mapping["exact_sink_logical_pages"] == [0]
    assert mapping["exact_tail_logical_range_start_inclusive_end_exclusive"] == [
        1286,
        1318,
    ]
    assert mapping["logical_page_sets_disjoint"] is True
    assert mapping["logical_token_coverage_exactly_once"] is True

    sink = mapping["exact_sink_mapping"]
    tail = mapping["exact_ring_mapping"]
    assert len(sink) == 1
    assert len(tail) == 32
    assert len(mapping["exact_page_mapping"]) == 33
    assert sink[0]["logical_page"] == 0
    assert sink[0]["segment"] == "exact_fp16_sink"
    assert sink[0]["is_exact_attention_sink"] is True
    assert sink[0]["physical_exact_storage_page_within_request"] == 0
    assert sink[0]["physical_exact_ring_page_within_request"] is None
    assert tail[0]["logical_page"] == 1286
    assert tail[0]["segment"] == "exact_fp16_tail"
    assert tail[0]["physical_exact_storage_page_within_request"] == 7
    assert tail[0]["physical_exact_ring_page_within_request"] == 6
    assert sorted(
        row["physical_exact_storage_page_within_request"] for row in tail
    ) == list(range(1, 33))
    assert sorted(
        row["physical_exact_ring_page_within_request"] for row in tail
    ) == list(range(32))


def test_s3_page_mapping_uses_three_fixed_prefix_slots_and_disjoint_tail() -> None:
    mapping = MODULE.page_age_mapping(
        context=20480,
        step=798,
        exact_tail_tokens=512,
        exact_sink_pages=3,
    )
    assert mapping["sequence_length_after_append"] == 21279
    assert mapping["total_pages"] == 1330
    assert mapping["old_logical_range_start_inclusive_end_exclusive"] == [3, 1298]
    assert mapping["old_int8_pages"] == 1295
    assert mapping["exact_fp16_pages"] == 35
    assert mapping["exact_prefix_pages"] == 3
    assert mapping["exact_prefix_logical_pages"] == [0, 1, 2]
    assert mapping["exact_prefix_fixed_slot_gate_passed"] is True
    assert mapping["exact_prefix_tail_physical_pages_unique"] is True
    prefix = mapping["exact_prefix_mapping"]
    assert [row["logical_page"] for row in prefix] == [0, 1, 2]
    assert [row["segment"] for row in prefix] == [
        "exact_fp16_prefix",
        "exact_fp16_prefix",
        "exact_fp16_prefix",
    ]
    assert [
        row["physical_exact_storage_page_within_request"] for row in prefix
    ] == [0, 1, 2]
    assert all(row["is_exact_attention_prefix"] for row in prefix)
    assert [row["is_exact_attention_sink"] for row in prefix] == [True, False, False]
    tail = mapping["exact_ring_mapping"]
    assert len(tail) == 32
    assert tail[0]["logical_page"] == 1298
    assert tail[0]["physical_exact_storage_page_within_request"] == 21
    assert sorted(
        row["physical_exact_storage_page_within_request"] for row in tail
    ) == list(range(3, 35))
    assert len(mapping["exact_page_mapping"]) == 35


def test_pre_tail_band_source_gate_is_strict_s3_t512_train_row() -> None:
    configuration = dict(
        source_result()["configuration"],
        decode_steps=1024,
        exact_sink_pages=3,
        exact_prefix_pages=3,
        trajectory_mode=MODULE.SUSTAINED.FROZEN_HF_TEACHER,
        token_source="wikitext2",
        wikitext_member="wikitext-2-raw/wiki.train.raw",
        seed=20260861,
        token_offset=100000,
        token_stride=21984,
        min_logits_cosine=0.995,
    )
    source_row = {
        "rank": 1,
        "step": 798,
        "request": 3,
        "absolute_position": 21278,
        "below_strict_gate": True,
    }
    source = {
        "configuration": configuration,
        "correctness": {
            "backend_vs_hf_sdpa_fp16": {
                "outlier_diagnostics": {
                    "threshold_unchanged": 0.995,
                    "summary": {"checked_rows": 4096, "rows_below_gate": 1},
                    "worst_rows": [source_row],
                }
            }
        },
    }
    target = {
        "step": 798,
        "request": 3,
        "source_outlier_row": source_row,
    }
    gate = MODULE.pre_tail_band_ablation_source_gate(source, [target])
    assert gate["enabled"] is True
    assert gate["gate_passed"] is True
    assert gate["required_target_step_request"] == [798, 3]
    assert gate["source_exact_tail_tokens"] == 512
    assert gate["checked_source_rows"] == 4096
    assert gate["source_rows_below_gate"] == 1
    assert all(gate["frozen_train_configuration_gates"].values())
    assert "local representation counterfactual" in gate["scope"]

    legacy = MODULE.pre_tail_band_ablation_source_gate(
        source_result(),
        [target],
    )
    assert legacy["enabled"] is False
    assert legacy["gate_passed"] is True

    with pytest.raises(ValueError, match="exactly frozen TRAIN target 798:3"):
        MODULE.pre_tail_band_ablation_source_gate(
            source,
            [dict(target, step=797)],
        )
    with pytest.raises(ValueError, match="retained below-gate"):
        MODULE.pre_tail_band_ablation_source_gate(
            source,
            [dict(target, source_outlier_row={"below_strict_gate": False})],
        )
    invalid_tail = {
        **source,
        "configuration": dict(configuration, exact_tail_tokens=256),
    }
    with pytest.raises(ValueError, match="S3/T512"):
        MODULE.pre_tail_band_ablation_source_gate(invalid_tail, [target])

    non_train = {
        **source,
        "configuration": dict(
            configuration, wikitext_member="wikitext-2-raw/wiki.test.raw"
        ),
    }
    with pytest.raises(ValueError, match="not the frozen TRAIN configuration"):
        MODULE.pre_tail_band_ablation_source_gate(non_train, [target])

    second_row = {
        "rank": 2,
        "step": 799,
        "request": 2,
        "absolute_position": 21279,
        "below_strict_gate": True,
    }
    additional_below_gate = {
        **source,
        "correctness": {
            "backend_vs_hf_sdpa_fp16": {
                "outlier_diagnostics": {
                    "threshold_unchanged": 0.995,
                    "summary": {"checked_rows": 4096, "rows_below_gate": 2},
                    "worst_rows": [source_row, second_row],
                }
            }
        },
    }
    with pytest.raises(ValueError, match="exactly one below-gate row"):
        MODULE.pre_tail_band_ablation_source_gate(
            additional_below_gate,
            [target],
        )


def test_pre_tail_band_definition_is_nested_equal_byte_and_fail_closed() -> None:
    mapping = MODULE.page_age_mapping(
        context=20480,
        step=798,
        exact_tail_tokens=512,
        exact_sink_pages=3,
    )
    definition = MODULE.pre_tail_band_definition(mapping, step=798, request=3)
    assert definition["tail_start_logical_page_E"] == 1298
    assert definition["page_sets"] == {
        "B16": list(range(1282, 1298)),
        "B32": list(range(1266, 1298)),
        "C16": list(range(1234, 1250)),
        "C32": list(range(1234, 1266)),
    }
    assert definition["logical_ranges_start_inclusive_end_exclusive"] == {
        "B16": [1282, 1298],
        "B32": [1266, 1298],
        "C16": [1234, 1250],
        "C32": [1234, 1266],
    }
    assert definition["tail_token_equivalents_if_restored"] == {
        "B16": 768,
        "B32": 1024,
    }
    assert all(definition["nested_band_gates"].values())
    assert definition["local_counterfactual_only"] is True
    assert "T768-equivalent" in definition["labels"]["B16"]
    assert "T1024-equivalent" in definition["labels"]["B32"]

    storage = MODULE.pre_tail_band_storage_bytes(
        kv_heads=8,
        head_dim=128,
        definition=definition,
    )
    assert storage["fp16_kv_bytes_per_page_per_layer_request"] == 65_536
    assert storage["bytes_by_page_set_per_layer_request"] == {
        "B16": 1_048_576,
        "B32": 2_097_152,
        "C16": 1_048_576,
        "C32": 2_097_152,
    }
    assert all(storage["equal_byte_gates"].values())

    with pytest.raises(ValueError, match="step/request 798:3"):
        MODULE.pre_tail_band_definition(mapping, step=798, request=2)
    s1_mapping = MODULE.page_age_mapping(
        context=20480,
        step=798,
        exact_tail_tokens=512,
        exact_sink_pages=1,
    )
    with pytest.raises(ValueError, match="exact prefix S3"):
        MODULE.pre_tail_band_definition(s1_mapping, step=798, request=3)
    t256_mapping = MODULE.page_age_mapping(
        context=20480,
        step=798,
        exact_tail_tokens=256,
        exact_sink_pages=3,
    )
    with pytest.raises(ValueError, match="exact tail T512"):
        MODULE.pre_tail_band_definition(t256_mapping, step=798, request=3)
    insufficient_old_range = MODULE.page_age_mapping(
        context=512,
        step=798,
        exact_tail_tokens=512,
        exact_sink_pages=3,
    )
    with pytest.raises(ValueError, match="lies outside source old segment"):
        MODULE.pre_tail_band_definition(
            insufficient_old_range,
            step=798,
            request=3,
        )
    invalid_partition = dict(mapping, logical_page_sets_disjoint=False)
    with pytest.raises(ValueError, match="lacks exact partition gates"):
        MODULE.pre_tail_band_definition(
            invalid_partition,
            step=798,
            request=3,
        )


def test_scalar_quantizer_reconstruction_keeps_exact_segment_unchanged() -> None:
    generator = torch.Generator().manual_seed(3)
    source = torch.randn(32, 2, 4, generator=generator)
    center = source[:16].float().mean(dim=0).half()
    reconstructed, codes, scale = MODULE.reconstruct_old_pages(
        source, center, old_pages=1
    )
    centered = source.float() - center.float()[None]
    assert reconstructed.shape == source.shape
    assert codes.dtype == torch.int8
    assert tuple(scale.shape) == (1, 2)
    assert torch.equal(reconstructed[16:], centered[16:].half().float())
    assert int(codes.min()) >= -127
    assert int(codes.max()) <= 127


def test_scalar_quantizer_preserves_sink_and_tail_around_old_middle() -> None:
    generator = torch.Generator().manual_seed(7)
    # S=1, one old middle page, and the source run's T=512 exact tail.
    source = torch.randn(544, 2, 4, generator=generator)
    center = source[:16].float().mean(dim=0).half()
    reconstructed, codes, scale = MODULE.reconstruct_old_pages(
        source,
        center,
        old_pages=1,
        old_logical_begin=1,
    )
    centered_exact = (source.float() - center.float()[None]).half().float()
    assert tuple(codes.shape) == (1, 2, 16, 4)
    assert tuple(scale.shape) == (1, 2)
    assert torch.equal(reconstructed[:16], centered_exact[:16])
    assert torch.equal(reconstructed[32:], centered_exact[32:])
    assert not torch.equal(reconstructed[16:32], centered_exact[16:32])


def test_scalar_quantizer_preserves_s3_prefix_and_tail_around_old_middle() -> None:
    generator = torch.Generator().manual_seed(71)
    source = torch.randn(96, 2, 4, generator=generator)
    center = source.float().mean(dim=0).half()
    reconstructed, codes, scale = MODULE.reconstruct_old_pages(
        source,
        center,
        old_pages=1,
        old_logical_begin=3,
    )
    centered_exact = (source.float() - center.float()[None]).half().float()
    assert tuple(codes.shape) == (1, 2, 16, 4)
    assert tuple(scale.shape) == (1, 2)
    assert torch.equal(reconstructed[:48], centered_exact[:48])
    assert not torch.equal(reconstructed[48:64], centered_exact[48:64])
    assert torch.equal(reconstructed[64:], centered_exact[64:])


def test_page_error_localization_labels_exact_sink_separately() -> None:
    mapping = MODULE.page_age_mapping(
        context=64,
        step=0,
        exact_tail_tokens=32,
        exact_sink_pages=1,
    )
    reference = torch.zeros(65, 2, 4)
    reconstructed = reference.clone()
    reconstructed[:16] = 1.0
    probability = torch.zeros(2, 2, 65)
    probability[:, :, 0] = 1.0
    localization = MODULE._page_error_rows(
        reference,
        reconstructed,
        probability,
        mapping=mapping,
        page_top_k=1,
    )
    assert len(localization["retained_pages"]) == 1
    sink = localization["retained_pages"][0]
    assert sink["logical_page"] == 0
    assert sink["segment"] == "exact_fp16_sink"
    assert sink["is_exact_attention_sink"] is True
    assert sink["physical_exact_storage_page_within_request"] == 0
    assert sink["physical_exact_ring_page_within_request"] is None


def test_page_ablation_sets_are_equal_byte_and_fail_closed_to_old_segment() -> None:
    mapping = MODULE.page_age_mapping(
        context=20480,
        step=594,
        exact_tail_tokens=512,
        exact_sink_pages=1,
    )
    page_sets = MODULE.page_ablation_page_sets(mapping)
    assert page_sets == {
        "page2_semantic_anchor": [2],
        "contiguous_added_prefix_pages1_2": [1, 2],
        "equal_byte_fixed_control_pages3_4": [3, 4],
    }
    assert len(page_sets["contiguous_added_prefix_pages1_2"]) == len(
        page_sets["equal_byte_fixed_control_pages3_4"]
    )

    page2_not_old = dict(mapping)
    page2_not_old["old_logical_range_start_inclusive_end_exclusive"] = [1, 2]
    with pytest.raises(ValueError, match="page2_semantic_anchor.*not inside"):
        MODULE.page_ablation_page_sets(page2_not_old)

    control_not_old = dict(mapping)
    control_not_old["old_logical_range_start_inclusive_end_exclusive"] = [1, 4]
    with pytest.raises(ValueError, match="equal_byte_fixed_control.*not inside"):
        MODULE.page_ablation_page_sets(control_not_old)

    no_sink = MODULE.page_age_mapping(
        context=20480,
        step=594,
        exact_tail_tokens=512,
        exact_sink_pages=0,
    )
    with pytest.raises(ValueError, match="exact_sink_pages=1"):
        MODULE.page_ablation_page_sets(no_sink)

    s3 = MODULE.page_age_mapping(
        context=20480,
        step=798,
        exact_tail_tokens=512,
        exact_sink_pages=3,
    )
    applicability = MODULE.page_ablation_applicability(s3)
    assert applicability["enabled"] is False
    assert applicability["source_exact_prefix_pages"] == 3
    assert "already contains logical page2" in applicability["reason"]


def test_replace_logical_pages_changes_only_selected_complete_pages() -> None:
    base = torch.zeros(80, 2, 4)
    donor = torch.arange(base.numel(), dtype=torch.float32).reshape_as(base)
    replaced = MODULE.replace_logical_pages(base, donor, [1, 3])
    assert torch.equal(replaced[:16], base[:16])
    assert torch.equal(replaced[16:32], donor[16:32])
    assert torch.equal(replaced[32:48], base[32:48])
    assert torch.equal(replaced[48:64], donor[48:64])
    assert torch.equal(replaced[64:], base[64:])
    with pytest.raises(ValueError, match="outside the active sequence"):
        MODULE.replace_logical_pages(base, donor, [5])


def _small_page_ablation() -> dict:
    generator = torch.Generator().manual_seed(19)
    source_key = torch.randn(129, 2, 4, generator=generator)
    source_value = torch.randn(129, 2, 4, generator=generator)
    center = torch.zeros(2, 4, dtype=torch.float16)
    reference_key = source_key.half().float()
    reference_value = source_value.half().float()
    reconstructed_key, _, _ = MODULE.reconstruct_old_pages(
        source_key, center, old_pages=6, old_logical_begin=1
    )
    reconstructed_value, _, _ = MODULE.reconstruct_old_pages(
        source_value, center, old_pages=6, old_logical_begin=1
    )
    query = torch.randn(4, 4, generator=generator)
    exact_output, exact_probability = MODULE.grouped_query_attention(
        query, reference_key, reference_value
    )
    k_only_output, quantized_probability = MODULE.grouped_query_attention(
        query, reconstructed_key, reference_value
    )
    v_only_output = torch.einsum(
        "hgt,thd->hgd", exact_probability, reconstructed_value
    ).reshape_as(exact_output)
    both_output = torch.einsum(
        "hgt,thd->hgd", quantized_probability, reconstructed_value
    ).reshape_as(exact_output)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            o_proj=SimpleNamespace(weight=torch.eye(16, dtype=torch.float16))
        )
    )
    value_center_gqa = torch.zeros_like(exact_output)
    projected = {
        name: MODULE._project_attention_output(layer, output, value_center_gqa)
        for name, output in {
            "reference_centered_form": exact_output,
            "old_k_quantized_only": k_only_output,
            "old_v_quantized_only": v_only_output,
            "old_k_and_v_quantized": both_output,
        }.items()
    }
    mapping = MODULE.page_age_mapping(
        context=128,
        step=0,
        exact_tail_tokens=32,
        exact_sink_pages=1,
    )
    return MODULE.page_ablation_counterfactual(
        layer=layer,
        query=query,
        reference_key=reference_key,
        reference_value=reference_value,
        reconstructed_key=reconstructed_key,
        reconstructed_value=reconstructed_value,
        reference_probability=exact_probability,
        quantized_key_probability=quantized_probability,
        value_center_gqa=value_center_gqa,
        mapping=mapping,
        existing_projected=projected,
    )


def _small_pre_tail_band_ablation() -> tuple[dict, dict]:
    generator = torch.Generator().manual_seed(29)
    mapping = MODULE.page_age_mapping(
        context=1536,
        step=798,
        exact_tail_tokens=512,
        exact_sink_pages=3,
    )
    tokens = mapping["sequence_length_after_append"]
    source_key = torch.randn(tokens, 1, 2, generator=generator)
    source_value = torch.randn(tokens, 1, 2, generator=generator)
    center = torch.zeros(1, 2, dtype=torch.float16)
    reference_key = source_key.half().float()
    reference_value = source_value.half().float()
    old_begin, old_end = mapping[
        "old_logical_range_start_inclusive_end_exclusive"
    ]
    reconstructed_key, _, _ = MODULE.reconstruct_old_pages(
        source_key,
        center,
        old_pages=old_end - old_begin,
        old_logical_begin=old_begin,
    )
    reconstructed_value, _, _ = MODULE.reconstruct_old_pages(
        source_value,
        center,
        old_pages=old_end - old_begin,
        old_logical_begin=old_begin,
    )
    query = torch.randn(2, 2, generator=generator)
    exact_output, exact_probability = MODULE.grouped_query_attention(
        query, reference_key, reference_value
    )
    k_only_output, quantized_probability = MODULE.grouped_query_attention(
        query, reconstructed_key, reference_value
    )
    v_only_output = torch.einsum(
        "hgt,thd->hgd", exact_probability, reconstructed_value
    ).reshape_as(exact_output)
    both_output = torch.einsum(
        "hgt,thd->hgd", quantized_probability, reconstructed_value
    ).reshape_as(exact_output)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            o_proj=SimpleNamespace(weight=torch.eye(4, dtype=torch.float16))
        )
    )
    value_center_gqa = torch.zeros_like(exact_output)
    projected = {
        name: MODULE._project_attention_output(layer, output, value_center_gqa)
        for name, output in {
            "reference_centered_form": exact_output,
            "old_k_quantized_only": k_only_output,
            "old_v_quantized_only": v_only_output,
            "old_k_and_v_quantized": both_output,
        }.items()
    }
    return (
        MODULE.pre_tail_band_ablation_counterfactual(
            layer=layer,
            query=query,
            reference_key=reference_key,
            reference_value=reference_value,
            reconstructed_key=reconstructed_key,
            reconstructed_value=reconstructed_value,
            reference_probability=exact_probability,
            value_center_gqa=value_center_gqa,
            mapping=mapping,
            existing_projected=projected,
            step=798,
            request=3,
        ),
        mapping,
    )


def test_pre_tail_band_ablation_has_component_matched_rescue_and_recreation() -> None:
    ablation, _ = _small_pre_tail_band_ablation()
    assert ablation["enabled"] is True
    assert ablation["component_matched_denominators"] is True
    assert ablation["page_sets_validated_inside_source_old_segment"] is True
    assert ablation["attention_quantizer_and_threshold_math_unchanged"] is True
    assert "one full-length K or V" in ablation["materialization_schedule"]
    variants = ablation["variants"]
    assert list(variants) == [
        "all_old_quantized_baseline",
        "b16_t768_equivalent_restored_exact",
        "b16_t768_equivalent_only_quantized",
        "b32_t1024_equivalent_restored_exact",
        "b32_t1024_equivalent_only_quantized",
        "c16_equal_byte_older_control_restored_exact",
        "c16_equal_byte_older_control_only_quantized",
        "c32_equal_byte_older_control_restored_exact",
        "c32_equal_byte_older_control_only_quantized",
    ]
    for component in ("k_only", "v_only", "both"):
        baseline = variants["all_old_quantized_baseline"][
            "projected_attention_output"
        ][component]
        assert baseline["error_norm_ratio_vs_all_old_component_baseline"] == 1.0
        assert baseline["error_norm_rescue_fraction"] == 0.0
        for name in (
            "b16_t768_equivalent_restored_exact",
            "b32_t1024_equivalent_restored_exact",
            "c16_equal_byte_older_control_restored_exact",
            "c32_equal_byte_older_control_restored_exact",
        ):
            metric = variants[name]["projected_attention_output"][component]
            assert "error_norm_rescue_fraction" in metric
            assert "error_norm_recreation_fraction" not in metric
        for name in (
            "b16_t768_equivalent_only_quantized",
            "b32_t1024_equivalent_only_quantized",
            "c16_equal_byte_older_control_only_quantized",
            "c32_equal_byte_older_control_only_quantized",
        ):
            metric = variants[name]["projected_attention_output"][component]
            assert "error_norm_recreation_fraction" in metric
            assert "error_norm_rescue_fraction" not in metric

    b16_k = variants["b16_t768_equivalent_restored_exact"][
        "projected_attention_output"
    ]["k_only"]
    assert b16_k["effective_key_tensor_policy"] == "restore_selected_exact"
    assert b16_k["effective_value_tensor_policy"] == "all_exact_reference"
    assert len(b16_k["effective_key_logical_pages"]) == 16
    b32_v = variants["b32_t1024_equivalent_only_quantized"][
        "projected_attention_output"
    ]["v_only"]
    assert b32_v["effective_key_tensor_policy"] == "all_exact_reference"
    assert b32_v["effective_value_tensor_policy"] == "only_selected_quantized"
    assert len(b32_v["effective_value_logical_pages"]) == 32
    storage = ablation["nominal_exact_fp16_storage_bytes"]
    assert storage["bytes_by_page_set_per_layer_request"] == {
        "B16": 2048,
        "B32": 4096,
        "C16": 2048,
        "C32": 4096,
    }


def test_pre_tail_band_aggregate_is_strict_one_by_32_ratio_of_sums() -> None:
    ablation, mapping = _small_pre_tail_band_ablation()
    target = {
        "step": 798,
        "request": 3,
        "page_mapping": mapping,
        "layers": [
            {
                "layer": layer,
                "kv_counterfactual": {"pre_tail_band_ablation": ablation},
            }
            for layer in range(32)
        ],
    }
    aggregate = MODULE.aggregate_pre_tail_band_ablation(
        [target], expected_layer_count=32
    )
    assert aggregate["target_count"] == 1
    assert aggregate["target_layer_rows"] == 32
    assert aggregate["frozen_s3_t512_target798_request3_gate_passed"] is True
    assert aggregate["ordered_complete_layer_ids_0_through_31"] is True
    assert aggregate["component_matched_denominators"] is True
    assert aggregate["ratio_of_sums_aggregate"] is True
    both = aggregate["variants"]["b16_t768_equivalent_restored_exact"][
        "components"
    ]["both"]
    assert "denominator_weighted_rescue_fraction" in both
    assert both["error_norm_rescue_fraction"]["count"] == 32
    recreation = aggregate["variants"][
        "b32_t1024_equivalent_only_quantized"
    ]["components"]["v_only"]
    assert "denominator_weighted_recreation_fraction" in recreation
    assert recreation["error_norm_recreation_fraction"]["count"] == 32
    assert set(aggregate["paired_contrasts"]) == {
        "B16_rescue_minus_equal_byte_C16",
        "B32_rescue_minus_equal_byte_C32",
        "B32_rescue_minus_nested_B16",
        "B16_recreation_minus_equal_byte_C16",
        "B32_recreation_minus_equal_byte_C32",
        "B32_recreation_minus_nested_B16",
    }
    assert aggregate["nominal_exact_fp16_storage_bytes"][
        "bytes_by_page_set_all_layers_one_request"
    ]["B32"] == 131_072
    assert "local layerwise representation counterfactual" in aggregate[
        "interpretation_guard"
    ]

    with pytest.raises(RuntimeError, match="exactly frozen TRAIN target 798:3"):
        MODULE.aggregate_pre_tail_band_ablation(
            [dict(target, request=2)], expected_layer_count=32
        )
    with pytest.raises(RuntimeError, match="ordered layers 0..31"):
        MODULE.aggregate_pre_tail_band_ablation(
            [dict(target, layers=target["layers"][:-1])],
            expected_layer_count=32,
        )
    with pytest.raises(ValueError, match="all 32 Mistral layers"):
        MODULE.aggregate_pre_tail_band_ablation(
            [target], expected_layer_count=31
        )


def test_page_ablation_uses_component_matched_baselines_and_all_variants() -> None:
    ablation = _small_page_ablation()
    assert ablation["page_sets_validated_inside_source_old_segment"] is True
    assert ablation["control_has_equal_fp16_page_bytes_to_s3_added_prefix"] is True
    assert "request-specific semantic anchor" in ablation["page2_interpretation"]
    assert "cross-request TRAIN" in ablation["s3_candidate_interpretation"]
    variants = ablation["variants"]
    assert list(variants) == [
        "all_old_quantized_baseline",
        "page2_k_restored",
        "page2_v_restored",
        "page2_k_and_v_restored",
        "contiguous_prefix_s3_candidate_pages1_2_restored",
        "equal_byte_control_pages3_4_restored",
        "page2_only_quantized_all_other_old_exact",
    ]
    for component in ("k_only", "v_only", "both"):
        baseline = variants["all_old_quantized_baseline"][
            "projected_attention_output"
        ][component]
        assert baseline["error_norm_ratio_vs_all_old_component_baseline"] == 1.0
        assert baseline["error_norm_rescue_fraction"] == 0.0
        recreation = variants["page2_only_quantized_all_other_old_exact"][
            "projected_attention_output"
        ][component]
        assert "error_norm_recreation_fraction" in recreation
        assert "error_norm_rescue_fraction" not in recreation

    unchanged_v = variants["page2_k_restored"]["projected_attention_output"][
        "v_only"
    ]
    unchanged_k = variants["page2_v_restored"]["projected_attention_output"][
        "k_only"
    ]
    assert unchanged_v["error_norm_ratio_vs_all_old_component_baseline"] == 1.0
    assert unchanged_k["error_norm_ratio_vs_all_old_component_baseline"] == 1.0
    assert unchanged_v["effective_key_tensor_policy"] == "all_exact_reference"
    assert unchanged_v["effective_value_tensor_policy"] == "all_old_quantized"
    page2_k_both = variants["page2_k_restored"]["projected_attention_output"][
        "both"
    ]
    assert page2_k_both["effective_key_tensor_policy"] == (
        "restore_selected_exact"
    )
    assert page2_k_both["effective_value_tensor_policy"] == "all_old_quantized"
    page2_v_only = variants["page2_v_restored"]["projected_attention_output"][
        "v_only"
    ]
    assert page2_v_only["effective_key_tensor_policy"] == "all_exact_reference"
    assert page2_v_only["effective_value_logical_pages"] == [2]
    storage = ablation["nominal_exact_fp16_storage_bytes"]
    assert storage["fp16_kv_bytes_per_page_per_layer_request"] == 512
    assert storage["s3_candidate_added_bytes_per_layer_request"] == 1024
    assert storage["control_added_bytes_per_layer_request"] == 1024


def test_nominal_storage_matches_mistral_geometry_and_equal_byte_control() -> None:
    storage = MODULE.nominal_exact_fp16_storage_bytes(
        kv_heads=8,
        head_dim=128,
        candidate_added_pages=2,
        control_added_pages=2,
    )
    assert storage["fp16_kv_bytes_per_page_per_layer_request"] == 65_536
    assert storage["s3_candidate_added_bytes_per_layer_request"] == 131_072
    assert storage["control_added_bytes_per_layer_request"] == 131_072
    assert 131_072 * 32 * 4 == 16_777_216
    with pytest.raises(ValueError, match="must match"):
        MODULE.nominal_exact_fp16_storage_bytes(
            kv_heads=8,
            head_dim=128,
            candidate_added_pages=2,
            control_added_pages=1,
        )


def test_false_prefill_population_gate_cannot_pass_diagnostic_protocol() -> None:
    passed = MODULE.prefill_population_protocol_gate(
        [
            {"sampled_population_gate_passed": True},
            {"sampled_population_gate_passed": False},
        ]
    )
    assert passed == {
        "record_count": 2,
        "all_sampled_population_gates_passed": False,
        "failed_record_indices": [1],
    }
    assert MODULE.diagnostic_protocol_passed(
        source_reproduction_passed=True,
        manual_reference_finite=True,
        sampled_prefill_population_gate_passed=passed[
            "all_sampled_population_gates_passed"
        ],
    ) is False
    with pytest.raises(RuntimeError, match="lacks its required gate"):
        MODULE.prefill_population_protocol_gate([{}])
    with pytest.raises(RuntimeError, match="no sampled records"):
        MODULE.prefill_population_protocol_gate([])


def test_page_ablation_aggregate_covers_complete_target_layer_grid() -> None:
    ablation = _small_page_ablation()
    targets = [
        {
            "step": step,
            "request": 0,
            "layers": [
                {"layer": 0, "kv_counterfactual": {"page_ablation": ablation}},
                {"layer": 1, "kv_counterfactual": {"page_ablation": ablation}},
            ]
        }
        for step in range(2)
    ]
    aggregate = MODULE.aggregate_page_ablation(
        targets,
        expected_layer_count=2,
        batch_size=2,
        require_frozen_source_grid=False,
    )
    assert aggregate["target_count"] == 2
    assert aggregate["layers_per_target"] == 2
    assert aggregate["target_layer_rows"] == 4
    assert aggregate["complete_selected_target_layer_cartesian_product"] is True
    assert aggregate["component_matched_denominators"] is True
    assert len(aggregate["per_layer_combined"]) == 2
    assert aggregate["frozen_source_four_by_32_gate"]["required"] is False
    assert aggregate["frozen_source_four_by_32_gate"]["passed"] is False
    assert aggregate["nominal_exact_fp16_storage_bytes"][
        "s1_to_s3_gross_added_bytes"
    ] == 4096
    baseline = aggregate["variants"]["all_old_quantized_baseline"]
    assert baseline["components"]["both"][
        "error_norm_ratio_vs_all_old_component_baseline"
    ]["mean"] == 1.0
    recreation = aggregate["variants"][
        "page2_only_quantized_all_other_old_exact"
    ]
    assert recreation["components"]["both"][
        "error_norm_recreation_fraction"
    ]["count"] == 4
    assert set(aggregate["paired_contrasts"]) == {
        "s3_rescue_minus_equal_byte_control",
        "page2_both_rescue_minus_s3_rescue",
        "page2_v_rescue_minus_page2_k_rescue",
    }
    both = aggregate["variants"]["page2_k_and_v_restored"]["components"][
        "both"
    ]
    assert both["denominator_weighted_error_norm_ratio_of_sums"] > 0.0
    assert "denominator_weighted_rescue_fraction" in both

    with pytest.raises(RuntimeError, match="frozen source target grid"):
        MODULE.aggregate_page_ablation(
            targets,
            expected_layer_count=2,
            batch_size=2,
            require_frozen_source_grid=True,
        )
    duplicate = [targets[0], targets[0]]
    with pytest.raises(RuntimeError, match="target keys are not unique"):
        MODULE.aggregate_page_ablation(
            duplicate,
            expected_layer_count=2,
            batch_size=2,
            require_frozen_source_grid=False,
        )

    malformed_layers = [dict(targets[0], layers=[*targets[0]["layers"]])]
    malformed_layers[0]["layers"][1] = dict(
        malformed_layers[0]["layers"][1], layer=0
    )
    with pytest.raises(RuntimeError, match="complete ordered range"):
        MODULE.aggregate_page_ablation(
            malformed_layers,
            expected_layer_count=2,
            batch_size=2,
            require_frozen_source_grid=False,
        )


def test_page_ablation_frozen_source_gate_requires_exact_four_by_32_grid() -> None:
    ablation = _small_page_ablation()
    targets = [
        {
            "step": step,
            "request": 3,
            "layers": [
                {
                    "layer": layer,
                    "kv_counterfactual": {"page_ablation": ablation},
                }
                for layer in range(32)
            ],
        }
        for step in (594, 798, 866, 900)
    ]
    aggregate = MODULE.aggregate_page_ablation(
        targets,
        expected_layer_count=32,
        batch_size=4,
        require_frozen_source_grid=True,
    )
    assert aggregate["target_layer_rows"] == 128
    assert aggregate["complete_selected_target_layer_cartesian_product"] is True
    assert aggregate["frozen_source_four_by_32_gate"]["passed"] is True
    assert aggregate["nominal_exact_fp16_storage_bytes"][
        "s1_to_s3_gross_added_bytes"
    ] == 131_072


def test_source_reproduction_requires_sink_identity_and_available_hashes() -> None:
    hf_logits = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.float16)
    candidate_logits = hf_logits.clone()
    teacher_inputs = torch.tensor([[11]], dtype=torch.long)
    tokens = torch.tensor([[7, 11]], dtype=torch.long)
    candidate_tokens = torch.tensor([[0]], dtype=torch.long)
    configuration = {
        "exact_tail_tokens": 512,
        "exact_sink_pages": 1,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
    }
    source = {
        "configuration": configuration,
        "pairing": {
            "configuration": {
                **configuration,
                "token_matrix_sha256": MODULE.SUSTAINED.sha256_tensors([tokens]),
                "teacher_inputs_sha256": MODULE.SUSTAINED.sha256_tensors(
                    [teacher_inputs]
                ),
            }
        },
        "attention_implementation": {
            "exact_tail_pages": 32,
            "exact_sink_pages": 1,
            "old_int8_value_scale_placement": "probability",
        },
        "correctness": {
            "backend_vs_hf_sdpa_fp16": {
                "logits": MODULE.SUSTAINED.compare_logits(
                    hf_logits, candidate_logits
                )
            },
            "hashes": {
                "hf_logits_sha256": MODULE.SUSTAINED.sha256_tensors([hf_logits]),
                "graph_generated_tokens_sha256": MODULE.SUSTAINED.sha256_tensors(
                    [candidate_tokens]
                ),
            },
            "same_backend_eager_vs_graph": {
                "passed": True,
                "eager_gate_passed": True,
                "logits": {"bitwise_identical": True},
                "generated_argmax_tokens": {"bitwise_identical": True},
            },
        },
    }
    reproduction = MODULE.source_reproduction(
        source,
        hf_logits,
        candidate_logits,
        teacher_inputs,
        tokens,
        candidate_tokens,
    )
    assert reproduction["passed"] is True
    assert reproduction["candidate_logits_source_hash_available"] is False
    assert reproduction["candidate_logits_hash_gate"] is None
    assert "no source candidate-logits hash" in reproduction[
        "candidate_logits_reproduction_claim"
    ]
    assert all(reproduction["hash_gates"].values())
    assert all(reproduction["source_pairing_configuration_gates"].values())
    assert all(reproduction["source_attention_identity_gates"].values())

    source["pairing"]["configuration"]["exact_sink_pages"] = 0
    mismatched = MODULE.source_reproduction(
        source,
        hf_logits,
        candidate_logits,
        teacher_inputs,
        tokens,
        candidate_tokens,
    )
    assert mismatched["passed"] is False
    assert mismatched["source_pairing_configuration_gates"][
        "exact_sink_pages"
    ] is False


def test_source_reproduction_requires_generic_s3_prefix_identity() -> None:
    hf_logits = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.float16)
    candidate_logits = hf_logits.clone()
    teacher_inputs = torch.tensor([[11]], dtype=torch.long)
    tokens = torch.tensor([[7, 11]], dtype=torch.long)
    candidate_tokens = torch.tensor([[0]], dtype=torch.long)
    configuration = {
        "exact_tail_tokens": 512,
        "exact_sink_pages": 3,
        "exact_prefix_pages": 3,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
    }
    source = {
        "configuration": configuration,
        "pairing": {
            "configuration": {
                **configuration,
                "token_matrix_sha256": MODULE.SUSTAINED.sha256_tensors([tokens]),
                "teacher_inputs_sha256": MODULE.SUSTAINED.sha256_tensors(
                    [teacher_inputs]
                ),
            }
        },
        "attention_implementation": {
            "exact_tail_pages": 32,
            "exact_sink_pages": 3,
            "exact_prefix_pages": 3,
            "exact_physical_layout": (
                "fixed slots 0..S-1 followed by modulo tail-ring slots S..S+T-1"
            ),
            "old_int8_value_scale_placement": "probability",
        },
        "correctness": {
            "backend_vs_hf_sdpa_fp16": {
                "logits": MODULE.SUSTAINED.compare_logits(
                    hf_logits, candidate_logits
                )
            },
            "hashes": {
                "hf_logits_sha256": MODULE.SUSTAINED.sha256_tensors([hf_logits]),
                "graph_generated_tokens_sha256": MODULE.SUSTAINED.sha256_tensors(
                    [candidate_tokens]
                ),
            },
            "same_backend_eager_vs_graph": {
                "passed": True,
                "eager_gate_passed": True,
                "logits": {"bitwise_identical": True},
                "generated_argmax_tokens": {"bitwise_identical": True},
            },
        },
    }
    reproduction = MODULE.source_reproduction(
        source,
        hf_logits,
        candidate_logits,
        teacher_inputs,
        tokens,
        candidate_tokens,
    )
    assert reproduction["passed"] is True
    assert reproduction["source_pairing_configuration_gates"][
        "exact_prefix_pages"
    ] is True
    assert reproduction["source_attention_identity_gates"][
        "exact_prefix_pages"
    ] is True
    assert reproduction["source_attention_identity_gates"][
        "fixed_prefix_physical_layout"
    ] is True

    source["attention_implementation"]["exact_prefix_pages"] = 2
    mismatched = MODULE.source_reproduction(
        source,
        hf_logits,
        candidate_logits,
        teacher_inputs,
        tokens,
        candidate_tokens,
    )
    assert mismatched["passed"] is False
    assert mismatched["source_attention_identity_gates"][
        "exact_prefix_pages"
    ] is False


def test_grouped_query_attention_matches_expanded_reference() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(4, 3, generator=generator)
    key = torch.randn(7, 2, 3, generator=generator)
    value = torch.randn(7, 2, 3, generator=generator)
    observed, probability = MODULE.grouped_query_attention(query, key, value)
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    score = torch.einsum("hd,thd->ht", query, expanded_key) / (3.0**0.5)
    expected_probability = torch.softmax(score, dim=-1)
    expected = torch.einsum("ht,thd->hd", expected_probability, expanded_value)
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(
        probability.reshape(4, 7),
        expected_probability,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_first_layer_divergence_is_diagnostic_not_gate_logic() -> None:
    good = {
        "cosine": 0.9999,
        "relative_l2": 0.001,
    }
    bad = {
        "cosine": 0.998,
        "relative_l2": 0.02,
    }
    rows = []
    for layer in range(3):
        rows.append(
            {
                "layer": layer,
                "input_hidden": good,
                "attention_delta": bad if layer == 1 else good,
                "post_attention_hidden": good,
                "output_hidden": good,
            }
        )
    first = MODULE.first_layer_divergence(
        rows, cosine_trigger=0.999, relative_l2_trigger=0.01
    )
    assert first["found"] is True
    assert first["layer"] == 1
    assert first["stage"] == "attention_delta"
    assert first["triggered_by"] == ["cosine", "relative_l2"]
