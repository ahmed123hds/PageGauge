#!/usr/bin/env python3
"""Attribute saved PageGauge logit outliers by layer and K/V component.

This is a diagnostic-only replay.  It imports the publication fixtures without
modifying them, selects exact request/step rows from a prior
``outlier_diagnostics`` payload, and replays the same frozen-HF trajectory.
Only selected rows retain layer checkpoints.  At each selected HF row, a
representation-level counterfactual independently quantizes old K pages, old V
pages, or both while retaining the identical FP16 exact-page segment.

The counterfactual does not replace the production kernel correctness result:
it evaluates the same PageGauge representation algebra with an explicit FP32
softmax so K and V can be varied independently.  No acceptance threshold,
quantizer, cache representation, serving path, or timed measurement is changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SUSTAINED_PATH = ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py"


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUSTAINED = load_local_module(
    "page_gauge_sustained_for_quality_layer_trace", SUSTAINED_PATH
)
PG = SUSTAINED.PG
BACKEND = SUSTAINED.BACKEND
PREFILL = SUSTAINED.PREFILL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="STEP:REQUEST",
        help=(
            "Trace an explicit row; repeatable. Without this option, every "
            "below-gate row retained in the source outlier report is traced."
        ),
    )
    parser.add_argument("--maximum-targets", type=int, default=8)
    parser.add_argument("--page-top-k", type=int, default=8)
    parser.add_argument(
        "--hidden-cosine-trigger",
        type=float,
        default=0.999,
        help="Diagnostic checkpoint trigger only; never an acceptance gate.",
    )
    parser.add_argument(
        "--hidden-relative-l2-trigger",
        type=float,
        default=0.01,
        help="Diagnostic checkpoint trigger only; never an acceptance gate.",
    )
    parser.add_argument(
        "--wikitext-zip",
        type=Path,
        default=ROOT / "data/wikitext-2-raw-v1.zip",
    )
    return parser.parse_args()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_target(value: str) -> tuple[int, int]:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError(f"target must be STEP:REQUEST, got {value!r}")
    try:
        step, request = (int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError(f"target must contain integers, got {value!r}") from error
    if step < 0 or request < 0:
        raise ValueError("target step and request must be non-negative")
    return step, request


def _quality_endpoint(result: dict[str, Any]) -> dict[str, Any]:
    try:
        endpoint = result["correctness"]["backend_vs_hf_sdpa_fp16"]
    except (KeyError, TypeError) as error:
        raise ValueError("source result lacks the HF correctness endpoint") from error
    if not isinstance(endpoint, dict):
        raise ValueError("source HF correctness endpoint is malformed")
    return endpoint


def select_trace_targets(
    result: dict[str, Any], explicit: list[str], maximum_targets: int
) -> list[dict[str, Any]]:
    """Select deterministic request/step rows without changing the source rank."""

    if maximum_targets <= 0:
        raise ValueError("maximum target count must be positive")
    configuration = result.get("configuration", {})
    decode_steps = int(configuration.get("decode_steps", 0))
    batch_size = int(configuration.get("batch_size", 0))
    if decode_steps <= 0 or batch_size <= 0:
        raise ValueError("source configuration has invalid decode/batch dimensions")

    source_rows: dict[tuple[int, int], dict[str, Any]] = {}
    diagnostic = _quality_endpoint(result).get("outlier_diagnostics")
    if isinstance(diagnostic, dict):
        for row in diagnostic.get("worst_rows", []):
            key = (int(row["step"]), int(row["request"]))
            source_rows[key] = dict(row)

    if explicit:
        keys = [parse_target(value) for value in explicit]
    else:
        if not source_rows:
            raise ValueError(
                "source result has no row localization; rerun the sustained "
                "worker with --quality-diagnostics-top-k before layer tracing"
            )
        keys = [
            key
            for key, row in sorted(
                source_rows.items(),
                key=lambda item: (
                    int(item[1].get("rank", 1 << 30)),
                    item[0][0],
                    item[0][1],
                ),
            )
            if bool(row.get("below_strict_gate"))
        ]
        if not keys:
            raise ValueError("source outlier report contains no below-gate rows")

    unique_keys: list[tuple[int, int]] = []
    for key in keys:
        if key not in unique_keys:
            unique_keys.append(key)
    if len(unique_keys) > maximum_targets:
        raise ValueError(
            f"selected {len(unique_keys)} rows, exceeding --maximum-targets "
            f"{maximum_targets}"
        )

    targets = []
    context = int(configuration.get("context", 0))
    for step, request in unique_keys:
        if not 0 <= step < decode_steps:
            raise ValueError(f"target step {step} lies outside D={decode_steps}")
        if not 0 <= request < batch_size:
            raise ValueError(f"target request {request} lies outside B={batch_size}")
        source = source_rows.get((step, request))
        if source is not None:
            source_position = int(source.get("absolute_position", context + step))
            if source_position != context + step:
                raise ValueError("source outlier absolute position is inconsistent")
        targets.append(
            {
                "step": step,
                "request": request,
                "absolute_position": context + step,
                "source_outlier_row": source,
            }
        )
    return targets


def source_exact_cache_policy(configuration: dict[str, Any]) -> dict[str, int]:
    """Read the frozen source cache policy without inventing a new setting."""

    exact_tail_tokens = int(configuration["exact_tail_tokens"])
    if exact_tail_tokens <= 0 or exact_tail_tokens % PG.PAGE:
        raise ValueError("source exact tail must contain complete pages")
    legacy_pages = int(configuration.get("exact_sink_pages", 0))
    exact_prefix_pages = PG.validate_exact_prefix_pages(
        int(configuration.get("exact_prefix_pages", legacy_pages))
    )
    if "exact_sink_pages" in configuration and legacy_pages != exact_prefix_pages:
        raise ValueError("source exact-prefix and legacy exact-sink fields disagree")
    if exact_prefix_pages and configuration.get("tail_attention") != "flashinfer_merge":
        raise ValueError("source exact prefix requires segmented flashinfer_merge")
    exact_tail_pages = exact_tail_tokens // PG.PAGE
    context = int(configuration.get("context", 0))
    if context and (
        context % PG.PAGE
        or context // PG.PAGE <= exact_tail_pages + exact_prefix_pages
    ):
        raise ValueError("source context must retain at least one quantized old page")
    return {
        "exact_tail_tokens": exact_tail_tokens,
        "exact_tail_pages": exact_tail_pages,
        "exact_prefix_pages": exact_prefix_pages,
        "exact_sink_pages": exact_prefix_pages,
        "exact_storage_pages": exact_tail_pages + exact_prefix_pages,
    }


def logical_page_record(
    logical_page: int,
    *,
    sequence_length: int,
    old_logical_begin: int,
    old_logical_end: int,
    exact_tail_pages: int,
    exact_sink_pages: int,
    context_pages: int,
) -> dict[str, Any]:
    total_pages = math.ceil(sequence_length / PG.PAGE)
    if not 0 <= logical_page < total_pages:
        raise ValueError("logical page lies outside the active sequence")
    page_begin = logical_page * PG.PAGE
    page_end = min(sequence_length, page_begin + PG.PAGE)
    newest_position = sequence_length - 1
    if logical_page < exact_sink_pages:
        segment = (
            "exact_fp16_sink"
            if exact_sink_pages == 1
            else "exact_fp16_prefix"
        )
    elif old_logical_begin <= logical_page < old_logical_end:
        segment = "old_int8"
    else:
        segment = "exact_fp16_tail"
    exact_storage_page = (
        PG.exact_physical_page_index(
            0,
            logical_page,
            exact_tail_pages,
            exact_sink_pages,
        )
        if segment != "old_int8"
        else None
    )
    return {
        "logical_page": logical_page,
        "segment": segment,
        "source": "initial_prefix" if logical_page < context_pages else "generated",
        "token_range_start_inclusive": page_begin,
        "token_range_end_exclusive": page_end,
        "oldest_token_age": newest_position - page_begin,
        "newest_token_age": newest_position - (page_end - 1),
        "physical_full_cache_page_within_request": logical_page,
        "physical_exact_storage_page_within_request": exact_storage_page,
        "physical_exact_prefix_page_within_request": (
            logical_page if logical_page < exact_sink_pages else None
        ),
        "physical_exact_ring_page_within_request": (
            logical_page % exact_tail_pages
            if segment == "exact_fp16_tail"
            else None
        ),
        "is_exact_attention_prefix": logical_page < exact_sink_pages,
        "is_exact_attention_sink": (
            exact_sink_pages > 0 and logical_page == 0
        ),
    }


def page_age_mapping(
    *,
    context: int,
    step: int,
    exact_tail_tokens: int,
    exact_sink_pages: int = 0,
) -> dict[str, Any]:
    """Return the planner-exact page split after appending one decode token."""

    if context <= 0 or context % PG.PAGE:
        raise ValueError("context must be positive and page aligned")
    if exact_tail_tokens <= 0 or exact_tail_tokens % PG.PAGE:
        raise ValueError("exact tail must contain complete pages")
    if step < 0:
        raise ValueError("decode step must be non-negative")
    position = context + step
    sequence_length = position + 1
    total_pages = math.ceil(sequence_length / PG.PAGE)
    exact_tail_pages = exact_tail_tokens // PG.PAGE
    exact_sink_pages = PG.validate_exact_prefix_pages(exact_sink_pages)
    partition = PG.page_gauge_logical_partition(
        total_pages,
        exact_tail_pages,
        exact_sink_pages,
        (sequence_length - 1) % PG.PAGE + 1,
    )
    old_logical_begin = exact_sink_pages
    old_logical_end = int(partition["tail_logical_begin"])
    old_pages = int(partition["old_page_count"])
    exact_pages = int(partition["exact_page_count"])
    context_pages = context // PG.PAGE
    generated_old_pages = max(0, old_logical_end - context_pages)
    initial_old_pages = max(
        0, min(old_logical_end, context_pages) - old_logical_begin
    )
    initial_exact_pages = exact_sink_pages + max(
        0, min(context_pages, total_pages) - max(old_logical_end, exact_sink_pages)
    )

    anchors = {
        0,
        old_logical_begin,
        old_logical_end - 1,
        old_logical_end,
        total_pages - 1,
        position // PG.PAGE,
    }
    anchors.update(
        range(max(0, old_logical_end - 2), min(total_pages, old_logical_end + 2))
    )
    anchor_rows = [
        logical_page_record(
            page,
            sequence_length=sequence_length,
            old_logical_begin=old_logical_begin,
            old_logical_end=old_logical_end,
            exact_tail_pages=exact_tail_pages,
            exact_sink_pages=exact_sink_pages,
            context_pages=context_pages,
        )
        for page in sorted(page for page in anchors if 0 <= page < total_pages)
    ]
    prefix_rows = [
        logical_page_record(
            page,
            sequence_length=sequence_length,
            old_logical_begin=old_logical_begin,
            old_logical_end=old_logical_end,
            exact_tail_pages=exact_tail_pages,
            exact_sink_pages=exact_sink_pages,
            context_pages=context_pages,
        )
        for page in range(exact_sink_pages)
    ]
    tail_rows = [
        logical_page_record(
            page,
            sequence_length=sequence_length,
            old_logical_begin=old_logical_begin,
            old_logical_end=old_logical_end,
            exact_tail_pages=exact_tail_pages,
            exact_sink_pages=exact_sink_pages,
            context_pages=context_pages,
        )
        for page in range(old_logical_end, total_pages)
    ]
    prefix_physical_pages = [
        int(row["physical_exact_storage_page_within_request"])
        for row in prefix_rows
    ]
    if prefix_physical_pages != list(range(exact_sink_pages)):
        raise RuntimeError("exact prefix does not occupy fixed slots 0..S-1")
    exact_physical_pages = prefix_physical_pages + [
        int(row["physical_exact_storage_page_within_request"])
        for row in tail_rows
    ]
    if len(set(exact_physical_pages)) != len(exact_physical_pages):
        raise RuntimeError("active exact prefix and tail physical pages alias")
    legacy_generated_count = max(
        0, (step + 1 - exact_tail_tokens) // PG.PAGE
    )
    return {
        "position": position,
        "sequence_length_after_append": sequence_length,
        "current_page_offset": position % PG.PAGE,
        "total_pages": total_pages,
        "context_pages": context_pages,
        "old_int8_pages": old_pages,
        "exact_fp16_pages": exact_pages,
        "exact_tail_pages": exact_tail_pages,
        "exact_prefix_pages": exact_sink_pages,
        "exact_sink_pages": exact_sink_pages,
        "old_logical_range_start_inclusive_end_exclusive": [
            old_logical_begin,
            old_logical_end,
        ],
        "exact_prefix_logical_pages": list(range(exact_sink_pages)),
        "exact_sink_logical_pages": list(range(exact_sink_pages)),
        "exact_tail_logical_range_start_inclusive_end_exclusive": [
            old_logical_end,
            total_pages,
        ],
        "old_token_range_start_inclusive_end_exclusive": [
            old_logical_begin * PG.PAGE,
            old_logical_end * PG.PAGE,
        ],
        "exact_prefix_token_range_start_inclusive_end_exclusive": [
            0,
            exact_sink_pages * PG.PAGE,
        ],
        "exact_sink_token_range_start_inclusive_end_exclusive": [
            0,
            exact_sink_pages * PG.PAGE,
        ],
        "exact_tail_token_range_start_inclusive_end_exclusive": [
            old_logical_end * PG.PAGE,
            sequence_length,
        ],
        "initial_prefix_pages_currently_old_int8": initial_old_pages,
        "initial_prefix_pages_currently_exact_fp16": initial_exact_pages,
        "generated_pages_currently_old_int8": generated_old_pages,
        "generated_pages_currently_exact_fp16": max(
            0, total_pages - max(old_logical_end, context_pages)
        ),
        "logical_page_sets_disjoint": bool(partition["coverage_disjoint"]),
        "logical_token_coverage_exactly_once": (
            partition["old_token_count"] + partition["exact_token_count"]
            == sequence_length
        ),
        "exact_prefix_fixed_slot_gate_passed": True,
        "exact_prefix_tail_physical_pages_unique": True,
        "legacy_floor_formula_generated_old_pages": legacy_generated_count,
        "legacy_floor_formula_matches_planner": (
            legacy_generated_count == generated_old_pages
        ),
        "anchor_pages": anchor_rows,
        "exact_prefix_mapping": prefix_rows,
        "exact_sink_mapping": prefix_rows,
        "exact_ring_mapping": tail_rows,
        "exact_page_mapping": prefix_rows + tail_rows,
    }


def vector_comparison(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("checkpoint tensors must have identical non-empty shapes")
    reference_f = reference.detach().float().reshape(-1)
    candidate_f = candidate.detach().float().reshape(-1)
    difference = candidate_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f)
    candidate_norm = torch.linalg.vector_norm(candidate_f)
    difference_norm = torch.linalg.vector_norm(difference)
    cosine = F.cosine_similarity(reference_f, candidate_f, dim=0)
    return {
        "cosine": float(cosine.item()),
        "relative_l2": float(
            (difference_norm / reference_norm.clamp_min(1.0e-30)).item()
        ),
        "maximum_absolute_error": float(difference.abs().max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "reference_l2": float(reference_norm.item()),
        "candidate_l2": float(candidate_norm.item()),
    }


def first_layer_divergence(
    layers: list[dict[str, Any]],
    *,
    cosine_trigger: float,
    relative_l2_trigger: float,
) -> dict[str, Any]:
    """Find first diagnostic trigger; this is explicitly not a quality gate."""

    stages = (
        "input_hidden",
        "attention_delta",
        "post_attention_hidden",
        "output_hidden",
    )
    for layer in layers:
        for stage in stages:
            metric = layer[stage]
            triggered_by = []
            if float(metric["cosine"]) < cosine_trigger:
                triggered_by.append("cosine")
            if float(metric["relative_l2"]) > relative_l2_trigger:
                triggered_by.append("relative_l2")
            if triggered_by:
                return {
                    "found": True,
                    "layer": int(layer["layer"]),
                    "stage": stage,
                    "triggered_by": triggered_by,
                    "metrics": metric,
                }
    return {"found": False, "layer": None, "stage": None, "triggered_by": []}


def _round_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.where(
        values >= 0,
        torch.floor(values + 0.5),
        -torch.floor(-values + 0.5),
    )


def reconstruct_old_pages(
    source: torch.Tensor,
    center: torch.Tensor,
    old_pages: int,
    *,
    old_logical_begin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize only the contiguous old logical range; preserve exact pages."""

    if source.ndim != 3 or source.shape[1:] != center.shape:
        raise ValueError("source/center must be [tokens,Hkv,D] and [Hkv,D]")
    old_token_begin = old_logical_begin * PG.PAGE
    old_token_end = old_token_begin + old_pages * PG.PAGE
    if old_pages <= 0 or not (
        0 <= old_token_begin < old_token_end <= int(source.shape[0])
    ):
        raise ValueError("old-page count lies outside the active sequence")
    centered = source.float() - center.float()[None]
    old = centered[old_token_begin:old_token_end].reshape(
        old_pages, PG.PAGE, int(center.shape[0]), int(center.shape[1])
    )
    page_head = old.permute(0, 2, 1, 3)
    scale = (page_head.abs().amax(dim=(2, 3)) / 127.0).clamp_min(2.0**-20)
    stored_scale = scale.half().float()
    quotient = page_head / stored_scale[:, :, None, None]
    codes = _round_away_from_zero(quotient).clamp(-127, 127).to(torch.int8)
    reconstructed_old = (
        codes.float() * stored_scale[:, :, None, None]
    ).permute(0, 2, 1, 3).reshape(old_pages * PG.PAGE, *center.shape)
    # Start from the exact centered FP16 representation for every logical page,
    # then replace only the old middle. This keeps every fixed prefix page and
    # the recent tail bitwise identical to the controlled reference.
    reconstructed = centered.half().float()
    reconstructed[old_token_begin:old_token_end] = reconstructed_old
    return reconstructed, codes, stored_scale


def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit FP32 GQA for one query and one request."""

    if query.ndim != 2 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("attention expects Q=[Hq,D], K/V=[T,Hkv,D]")
    hq, head_dim = map(int, query.shape)
    tokens, hkv, kv_dim = map(int, key.shape)
    if head_dim != kv_dim or hq % hkv:
        raise ValueError("incompatible GQA geometry")
    groups = hq // hkv
    query_grouped = query.float().reshape(hkv, groups, head_dim)
    score = torch.einsum("hgd,thd->hgt", query_grouped, key.float())
    probability = torch.softmax(score / math.sqrt(head_dim), dim=-1)
    output = torch.einsum("hgt,thd->hgd", probability, value.float())
    if tuple(output.shape) != (hkv, groups, head_dim) or tokens <= 0:
        raise RuntimeError("manual grouped attention produced an invalid shape")
    return output.reshape(hq, head_dim), probability


def page_ablation_page_sets(mapping: dict[str, Any]) -> dict[str, list[int]]:
    """Return predeclared page sets, failing if any page is not currently old."""

    old_begin, old_end = map(
        int, mapping["old_logical_range_start_inclusive_end_exclusive"]
    )
    sequence_length = int(mapping["sequence_length_after_append"])
    if int(mapping.get("exact_sink_pages", -1)) != 1:
        raise ValueError("page2 ablation requires source exact_sink_pages=1")
    if list(mapping.get("exact_sink_logical_pages", [])) != [0]:
        raise ValueError("page2 ablation requires exact logical sink page [0]")
    if old_begin != 1:
        raise ValueError("page2 ablation requires source old segment to begin at 1")
    page_sets = {
        "page2_semantic_anchor": [2],
        "contiguous_added_prefix_pages1_2": [1, 2],
        "equal_byte_fixed_control_pages3_4": [3, 4],
    }
    if len(page_sets["contiguous_added_prefix_pages1_2"]) != len(
        page_sets["equal_byte_fixed_control_pages3_4"]
    ):
        raise RuntimeError("fixed-prefix candidate/control byte counts differ")
    if not set(page_sets["page2_semantic_anchor"]).issubset(
        page_sets["contiguous_added_prefix_pages1_2"]
    ):
        raise RuntimeError("S3 candidate does not contain the page2 anchor")
    if set(page_sets["contiguous_added_prefix_pages1_2"]) & set(
        page_sets["equal_byte_fixed_control_pages3_4"]
    ):
        raise RuntimeError("fixed-prefix candidate/control pages overlap")
    for name, pages in page_sets.items():
        if not pages or len(set(pages)) != len(pages):
            raise ValueError(f"{name} must contain unique logical pages")
        for page in pages:
            if not old_begin <= page < old_end:
                raise ValueError(
                    f"{name} logical page {page} is not inside the old segment "
                    f"[{old_begin}, {old_end})"
                )
            if (page + 1) * PG.PAGE > sequence_length:
                raise ValueError(f"{name} contains an incomplete old page")
    return page_sets


def page_ablation_applicability(mapping: dict[str, Any]) -> dict[str, Any]:
    """Gate the frozen S1→S3 page2 ablation without blocking generic traces."""

    prefix_pages = int(
        mapping.get("exact_prefix_pages", mapping.get("exact_sink_pages", -1))
    )
    prefix_logical_pages = list(
        mapping.get(
            "exact_prefix_logical_pages",
            mapping.get("exact_sink_logical_pages", []),
        )
    )
    enabled = prefix_pages == 1 and prefix_logical_pages == [0]
    if enabled:
        reason = "legacy S1 source supports the frozen page2 necessity/sufficiency ablation"
    elif prefix_pages >= 3:
        reason = (
            "source exact prefix already contains logical page2; the legacy "
            "S1-to-S3 page2 rescue ablation is not applicable"
        )
    else:
        reason = (
            "page2 necessity/sufficiency was predeclared only for a legacy "
            "S1 source policy"
        )
    return {
        "enabled": enabled,
        "source_exact_prefix_pages": prefix_pages,
        "source_exact_prefix_logical_pages": prefix_logical_pages,
        "reason": reason,
    }


def pre_tail_band_ablation_source_gate(
    source: dict[str, Any],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Enable only the predeclared S3/T512 TRAIN-row causal diagnostic."""

    configuration = source["configuration"]
    exact_cache_policy = source_exact_cache_policy(configuration)
    exact_prefix_pages = PG.validate_exact_prefix_pages(
        int(exact_cache_policy["exact_prefix_pages"])
    )
    exact_tail_tokens = int(exact_cache_policy["exact_tail_tokens"])
    target_keys = [
        (int(target["step"]), int(target["request"])) for target in targets
    ]
    if exact_prefix_pages != 3:
        return {
            "enabled": False,
            "gate_passed": True,
            "source_exact_prefix_pages": exact_prefix_pages,
            "source_exact_tail_tokens": exact_tail_tokens,
            "observed_target_step_request": [list(key) for key in target_keys],
            "reason": (
                "nested pre-tail bands are predeclared only for the frozen "
                "S3/T512 TRAIN row; generic non-S3 tracing remains unchanged"
            ),
        }
    if exact_tail_tokens != 512 or int(exact_cache_policy["exact_tail_pages"]) != 32:
        raise ValueError("pre-tail band ablation requires the frozen S3/T512 source")
    frozen_configuration = {
        "context": 20480,
        "decode_steps": 1024,
        "batch_size": 4,
        "trajectory_mode": SUSTAINED.FROZEN_HF_TEACHER,
        "token_source": "wikitext2",
        "wikitext_member": "wikitext-2-raw/wiki.train.raw",
        "seed": 20260861,
        "token_offset": 100000,
        "token_stride": 21984,
        "min_logits_cosine": 0.995,
    }
    configuration_gates = {
        name: configuration.get(name) == expected
        for name, expected in frozen_configuration.items()
    }
    if not all(configuration_gates.values()):
        failed = [name for name, passed in configuration_gates.items() if not passed]
        raise ValueError(
            "pre-tail band ablation source is not the frozen TRAIN configuration; "
            f"failed fields: {failed}"
        )
    diagnostics = _quality_endpoint(source).get("outlier_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("pre-tail band source lacks outlier diagnostics")
    summary = diagnostics.get("summary")
    worst_rows = diagnostics.get("worst_rows")
    if not isinstance(summary, dict) or not isinstance(worst_rows, list):
        raise ValueError("pre-tail band source outlier diagnostics are malformed")
    expected_checked_rows = int(configuration["decode_steps"]) * int(
        configuration["batch_size"]
    )
    retained_below_gate = [
        row
        for row in worst_rows
        if isinstance(row, dict) and row.get("below_strict_gate") is True
    ]
    retained_below_keys = [
        (int(row["step"]), int(row["request"])) for row in retained_below_gate
    ]
    sole_below_gate_identity = bool(
        int(summary.get("checked_rows", -1)) == expected_checked_rows == 4096
        and int(summary.get("rows_below_gate", -1)) == 1
        and retained_below_keys == [(798, 3)]
        and float(diagnostics.get("threshold_unchanged", float("nan"))) == 0.995
    )
    if not sole_below_gate_identity:
        raise ValueError(
            "pre-tail band source must have exactly one below-gate row, 798:3, "
            "over the complete frozen 4096-row TRAIN matrix"
        )
    if target_keys != [(798, 3)]:
        raise ValueError(
            "pre-tail band ablation requires exactly frozen TRAIN target 798:3"
        )
    source_row = targets[0].get("source_outlier_row")
    if (
        not isinstance(source_row, dict)
        or source_row.get("below_strict_gate") is not True
        or source_row != retained_below_gate[0]
    ):
        raise ValueError("pre-tail band target 798:3 must be a retained below-gate row")
    return {
        "enabled": True,
        "gate_passed": True,
        "source_exact_prefix_pages": 3,
        "source_exact_tail_tokens": 512,
        "source_exact_tail_pages": 32,
        "required_target_step_request": [798, 3],
        "observed_target_step_request": [[798, 3]],
        "target_is_retained_below_gate_row": True,
        "frozen_train_configuration_gates": configuration_gates,
        "checked_source_rows": expected_checked_rows,
        "source_rows_below_gate": 1,
        "sole_source_below_gate_step_request": [798, 3],
        "scope": (
            "diagnostic-only local representation counterfactual on one frozen "
            "TRAIN row; it is not a serving-policy or cross-request claim"
        ),
    }


def pre_tail_band_definition(
    mapping: dict[str, Any],
    *,
    step: int,
    request: int,
) -> dict[str, Any]:
    """Return nested pre-tail bands and older equal-byte controls fail closed."""

    if (int(step), int(request)) != (798, 3):
        raise ValueError("pre-tail band ablation requires target step/request 798:3")
    context_pages = int(mapping["context_pages"])
    inferred_step = int(mapping["position"]) - context_pages * PG.PAGE
    if inferred_step != int(step):
        raise ValueError("pre-tail band mapping does not reproduce target step 798")
    exact_prefix_pages = PG.validate_exact_prefix_pages(
        int(mapping.get("exact_prefix_pages", mapping.get("exact_sink_pages", -1)))
    )
    if exact_prefix_pages != 3 or list(
        mapping.get("exact_prefix_logical_pages", [])
    ) != [0, 1, 2]:
        raise ValueError("pre-tail band ablation requires fixed exact prefix S3")
    exact_tail_pages = int(mapping["exact_tail_pages"])
    if exact_tail_pages != 32:
        raise ValueError("pre-tail band ablation requires source exact tail T512")
    old_begin, old_end = map(
        int, mapping["old_logical_range_start_inclusive_end_exclusive"]
    )
    tail_begin, tail_end = map(
        int,
        mapping[
            "exact_tail_logical_range_start_inclusive_end_exclusive"
        ],
    )
    total_pages = int(mapping["total_pages"])
    sequence_length = int(mapping["sequence_length_after_append"])
    if old_begin != 3 or old_end != tail_begin or tail_end != total_pages:
        raise ValueError("pre-tail band source logical partition is inconsistent")
    if tail_end - tail_begin != exact_tail_pages:
        raise ValueError("pre-tail band exact tail does not contain 32 active pages")
    if not (
        mapping.get("logical_page_sets_disjoint") is True
        and mapping.get("logical_token_coverage_exactly_once") is True
        and mapping.get("exact_prefix_fixed_slot_gate_passed") is True
        and mapping.get("exact_prefix_tail_physical_pages_unique") is True
    ):
        raise ValueError("pre-tail band source mapping lacks exact partition gates")

    tail_start = tail_begin
    page_sets = {
        "B16": list(range(tail_start - 16, tail_start)),
        "B32": list(range(tail_start - 32, tail_start)),
        "C16": list(range(tail_start - 64, tail_start - 48)),
        "C32": list(range(tail_start - 64, tail_start - 32)),
    }
    expected_ranges = {
        "B16": [tail_start - 16, tail_start],
        "B32": [tail_start - 32, tail_start],
        "C16": [tail_start - 64, tail_start - 48],
        "C32": [tail_start - 64, tail_start - 32],
    }
    expected_counts = {"B16": 16, "B32": 32, "C16": 16, "C32": 32}
    for name, pages in page_sets.items():
        if len(pages) != expected_counts[name] or len(set(pages)) != len(pages):
            raise RuntimeError(f"pre-tail band {name} page count/uniqueness failed")
        if not pages or [pages[0], pages[-1] + 1] != expected_ranges[name]:
            raise RuntimeError(f"pre-tail band {name} is not the predeclared range")
        for page in pages:
            if not old_begin <= page < old_end:
                raise ValueError(
                    f"pre-tail band {name} logical page {page} lies outside "
                    f"source old segment [{old_begin}, {old_end})"
                )
            if (page + 1) * PG.PAGE > sequence_length:
                raise ValueError(f"pre-tail band {name} contains an incomplete page")
    if not set(page_sets["B16"]).issubset(page_sets["B32"]):
        raise RuntimeError("B16 must be nested inside B32")
    if not set(page_sets["C16"]).issubset(page_sets["C32"]):
        raise RuntimeError("C16 must be nested inside C32")
    if set(page_sets["B32"]) & set(page_sets["C32"]):
        raise RuntimeError("pre-tail candidate and older control bands overlap")
    if len(page_sets["B16"]) != len(page_sets["C16"]) or len(
        page_sets["B32"]
    ) != len(page_sets["C32"]):
        raise RuntimeError("pre-tail candidate/control page counts are not equal-byte")

    return {
        "tail_start_logical_page_E": tail_start,
        "source_old_logical_range_start_inclusive_end_exclusive": [
            old_begin,
            old_end,
        ],
        "source_exact_prefix_pages": exact_prefix_pages,
        "source_exact_tail_pages": exact_tail_pages,
        "source_exact_tail_tokens": exact_tail_pages * PG.PAGE,
        "page_sets": page_sets,
        "logical_ranges_start_inclusive_end_exclusive": expected_ranges,
        "page_counts": expected_counts,
        "labels": {
            "B16": (
                "16-page band immediately before E; restoring it is the "
                "contiguous T768-equivalent local counterfactual"
            ),
            "B32": (
                "32-page band immediately before E; restoring it is the "
                "contiguous T1024-equivalent local counterfactual"
            ),
            "C16": "16-page equal-byte older control [E-64,E-48)",
            "C32": "32-page equal-byte older control [E-64,E-32)",
        },
        "tail_token_equivalents_if_restored": {"B16": 768, "B32": 1024},
        "nested_band_gates": {
            "B16_subset_B32": True,
            "C16_subset_C32": True,
            "B32_disjoint_C32": True,
            "B16_equal_page_bytes_C16": True,
            "B32_equal_page_bytes_C32": True,
            "all_sets_inside_old_segment": True,
        },
        "local_counterfactual_only": True,
    }


def pre_tail_band_storage_bytes(
    *,
    kv_heads: int,
    head_dim: int,
    definition: dict[str, Any],
) -> dict[str, Any]:
    """Attest nominal gross FP16 K/V bytes for both equal-page controls."""

    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("pre-tail storage geometry must be positive")
    page_counts = definition["page_counts"]
    bytes_per_page = 2 * PG.PAGE * kv_heads * head_dim * 2
    by_set = {
        name: int(page_counts[name]) * bytes_per_page
        for name in ("B16", "B32", "C16", "C32")
    }
    if by_set["B16"] != by_set["C16"] or by_set["B32"] != by_set["C32"]:
        raise RuntimeError("pre-tail candidate/control nominal bytes differ")
    return {
        "accounting": (
            "gross additive FP16 K+V bytes for a hypothetical exact band; "
            "the local counterfactual does not allocate a production cache"
        ),
        "page_tokens": PG.PAGE,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "fp16_bytes_per_element": 2,
        "fp16_kv_bytes_per_page_per_layer_request": bytes_per_page,
        "bytes_by_page_set_per_layer_request": by_set,
        "equal_byte_gates": {
            "B16_equals_C16": True,
            "B32_equals_C32": True,
        },
    }


def nominal_exact_fp16_storage_bytes(
    *,
    kv_heads: int,
    head_dim: int,
    candidate_added_pages: int,
    control_added_pages: int,
) -> dict[str, Any]:
    """Return gross FP16 K/V bytes for equal-page diagnostic policies."""

    if min(kv_heads, head_dim, candidate_added_pages, control_added_pages) <= 0:
        raise ValueError("nominal exact-storage geometry must be positive")
    fp16_bytes_per_element = 2
    kv_tensors = 2
    per_page = (
        kv_tensors
        * PG.PAGE
        * kv_heads
        * head_dim
        * fp16_bytes_per_element
    )
    candidate_bytes = candidate_added_pages * per_page
    control_bytes = control_added_pages * per_page
    if candidate_added_pages != control_added_pages or candidate_bytes != control_bytes:
        raise ValueError("candidate and control exact-storage bytes must match")
    return {
        "accounting": (
            "gross additive FP16 K+V storage; existing INT8 codes/scales "
            "remain allocated"
        ),
        "page_tokens": PG.PAGE,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "fp16_bytes_per_element": fp16_bytes_per_element,
        "fp16_kv_bytes_per_page_per_layer_request": per_page,
        "s3_candidate_added_pages": candidate_added_pages,
        "equal_byte_control_pages": control_added_pages,
        "s3_candidate_added_bytes_per_layer_request": candidate_bytes,
        "control_added_bytes_per_layer_request": control_bytes,
    }


def replace_logical_pages(
    base: torch.Tensor,
    donor: torch.Tensor,
    logical_pages: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Clone ``base`` and replace only complete selected pages from ``donor``."""

    if base.shape != donor.shape or base.ndim != 3:
        raise ValueError("page replacement requires matching [tokens,Hkv,D] tensors")
    if not logical_pages or len(set(logical_pages)) != len(logical_pages):
        raise ValueError("replacement pages must be non-empty and unique")
    output = base.clone()
    for logical_page in logical_pages:
        begin = int(logical_page) * PG.PAGE
        end = begin + PG.PAGE
        if logical_page < 0 or end > int(base.shape[0]):
            raise ValueError("replacement page lies outside the active sequence")
        output[begin:end].copy_(donor[begin:end])
    return output


def _project_attention_output(
    layer: Any,
    centered_output: torch.Tensor,
    value_center_gqa: torch.Tensor,
) -> torch.Tensor:
    return F.linear(
        (centered_output + value_center_gqa).reshape(1, -1).half(),
        layer.self_attn.o_proj.weight,
    )[0]


def _batched_projected_error_records(
    exact: torch.Tensor,
    projected_by_variant: dict[str, dict[str, torch.Tensor]],
    policy_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute every ablation metric with one device-to-host transfer."""

    components = ("k_only", "v_only", "both")
    ordered = [
        (variant_name, component)
        for variant_name in policy_specs
        for component in components
    ]
    candidates = torch.stack(
        [projected_by_variant[variant][component] for variant, component in ordered]
    ).detach().float()
    exact_f = exact.detach().float()[None]
    difference = candidates - exact_f
    error_l2 = torch.linalg.vector_norm(difference, dim=1)
    reference_l2 = torch.linalg.vector_norm(exact_f, dim=1).expand_as(error_l2)
    candidate_l2 = torch.linalg.vector_norm(candidates, dim=1)
    metric_matrix = torch.stack(
        (
            error_l2,
            error_l2 / reference_l2.clamp_min(1.0e-30),
            F.cosine_similarity(exact_f.expand_as(candidates), candidates, dim=1),
            difference.abs().amax(dim=1),
            difference.square().mean(dim=1).sqrt(),
            reference_l2,
            candidate_l2,
        ),
        dim=1,
    ).cpu()
    host_metrics = {
        key: [float(value) for value in metric_matrix[index].tolist()]
        for index, key in enumerate(ordered)
    }
    baseline_error_l2 = {
        component: host_metrics[("all_old_quantized_baseline", component)][0]
        for component in components
    }
    if not all(
        math.isfinite(value) and value > 0.0
        for value in baseline_error_l2.values()
    ):
        raise RuntimeError("page-ablation baseline errors must be finite and positive")

    records: dict[str, dict[str, dict[str, Any]]] = {}
    for variant_name, spec in policy_specs.items():
        relationship = str(spec["relationship"])
        records[variant_name] = {}
        for component in components:
            (
                component_error_l2,
                relative_l2,
                cosine,
                maximum_absolute_error,
                rmse,
                reference_norm,
                candidate_norm,
            ) = host_metrics[(variant_name, component)]
            denominator = baseline_error_l2[component]
            ratio = component_error_l2 / denominator
            effective_key_policy = (
                str(spec["key_policy"])
                if component in ("k_only", "both")
                else "all_exact_reference"
            )
            effective_value_policy = (
                str(spec["value_policy"])
                if component in ("v_only", "both")
                else "all_exact_reference"
            )
            record = {
                "effective_key_tensor_policy": effective_key_policy,
                "effective_value_tensor_policy": effective_value_policy,
                "effective_key_logical_pages": (
                    list(spec["key_pages"])
                    if component in ("k_only", "both")
                    else []
                ),
                "effective_value_logical_pages": (
                    list(spec["value_pages"])
                    if component in ("v_only", "both")
                    else []
                ),
                "comparison_vs_exact_reference": {
                    "cosine": cosine,
                    "relative_l2": relative_l2,
                    "maximum_absolute_error": maximum_absolute_error,
                    "rmse": rmse,
                    "reference_l2": reference_norm,
                    "candidate_l2": candidate_norm,
                },
                "error_l2": component_error_l2,
                "all_old_component_baseline_error_l2": denominator,
                "error_norm_ratio_vs_all_old_component_baseline": ratio,
            }
            if relationship == "baseline":
                record["error_norm_rescue_fraction"] = 0.0
            elif relationship == "rescue":
                # Deliberately not clamped: negative exposes a harmful rescue.
                record["error_norm_rescue_fraction"] = 1.0 - ratio
            elif relationship == "recreation":
                # 1.0 means page2 alone recreates the all-old error norm.
                record["error_norm_recreation_fraction"] = ratio
            else:
                raise ValueError(
                    f"unknown page-ablation relationship {relationship!r}"
                )
            records[variant_name][component] = record
    return records


def _materialize_page_policy(
    reference: torch.Tensor,
    all_old_quantized: torch.Tensor,
    *,
    policy: str,
    logical_pages: list[int],
) -> torch.Tensor:
    if policy == "all_old_quantized":
        return all_old_quantized
    if policy == "restore_selected_exact":
        return replace_logical_pages(
            all_old_quantized, reference, logical_pages
        )
    if policy == "only_selected_quantized":
        return replace_logical_pages(
            reference, all_old_quantized, logical_pages
        )
    raise ValueError(f"unknown page materialization policy {policy!r}")


def page_ablation_counterfactual(
    *,
    layer: Any,
    query: torch.Tensor,
    reference_key: torch.Tensor,
    reference_value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    reconstructed_value: torch.Tensor,
    reference_probability: torch.Tensor,
    quantized_key_probability: torch.Tensor,
    value_center_gqa: torch.Tensor,
    mapping: dict[str, Any],
    existing_projected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Necessity/sufficiency ablation for the frozen request3 page2 anchor."""

    page_sets = page_ablation_page_sets(mapping)
    exact_projected = existing_projected["reference_centered_form"]
    component_names = {
        "k_only": "old_k_quantized_only",
        "v_only": "old_v_quantized_only",
        "both": "old_k_and_v_quantized",
    }
    baseline_projected = {
        component: existing_projected[name]
        for component, name in component_names.items()
    }

    policy_specs = {
        "all_old_quantized_baseline": {
            "key_policy": "all_old_quantized",
            "value_policy": "all_old_quantized",
            "key_pages": [],
            "value_pages": [],
            "relationship": "baseline",
        },
        "page2_k_restored": {
            "key_policy": "restore_selected_exact",
            "value_policy": "all_old_quantized",
            "key_pages": page_sets["page2_semantic_anchor"],
            "value_pages": [],
            "relationship": "rescue",
        },
        "page2_v_restored": {
            "key_policy": "all_old_quantized",
            "value_policy": "restore_selected_exact",
            "key_pages": [],
            "value_pages": page_sets["page2_semantic_anchor"],
            "relationship": "rescue",
        },
        "page2_k_and_v_restored": {
            "key_policy": "restore_selected_exact",
            "value_policy": "restore_selected_exact",
            "key_pages": page_sets["page2_semantic_anchor"],
            "value_pages": page_sets["page2_semantic_anchor"],
            "relationship": "rescue",
        },
        "contiguous_prefix_s3_candidate_pages1_2_restored": {
            "key_policy": "restore_selected_exact",
            "value_policy": "restore_selected_exact",
            "key_pages": page_sets["contiguous_added_prefix_pages1_2"],
            "value_pages": page_sets["contiguous_added_prefix_pages1_2"],
            "relationship": "rescue",
        },
        "equal_byte_control_pages3_4_restored": {
            "key_policy": "restore_selected_exact",
            "value_policy": "restore_selected_exact",
            "key_pages": page_sets["equal_byte_fixed_control_pages3_4"],
            "value_pages": page_sets["equal_byte_fixed_control_pages3_4"],
            "relationship": "rescue",
        },
        "page2_only_quantized_all_other_old_exact": {
            "key_policy": "only_selected_quantized",
            "value_policy": "only_selected_quantized",
            "key_pages": page_sets["page2_semantic_anchor"],
            "value_pages": page_sets["page2_semantic_anchor"],
            "relationship": "recreation",
        },
    }
    projected_by_variant: dict[str, dict[str, torch.Tensor]] = {}
    key_policy_cache: dict[
        tuple[str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]
    ] = {
        ("all_old_quantized", ()): (
            quantized_key_probability,
            baseline_projected["k_only"],
        )
    }
    for name, spec in policy_specs.items():
        relationship = str(spec["relationship"])
        if relationship == "baseline":
            projected = baseline_projected
        else:
            key_signature = (
                str(spec["key_policy"]),
                tuple(int(page) for page in spec["key_pages"]),
            )
            if key_signature in key_policy_cache:
                key_probability, k_only_projected = key_policy_cache[
                    key_signature
                ]
            else:
                key_policy = _materialize_page_policy(
                    reference_key,
                    reconstructed_key,
                    policy=key_signature[0],
                    logical_pages=list(key_signature[1]),
                )
                k_only_output, key_probability = grouped_query_attention(
                    query, key_policy, reference_value
                )
                k_only_projected = _project_attention_output(
                    layer, k_only_output, value_center_gqa
                )
                key_policy_cache[key_signature] = (
                    key_probability,
                    k_only_projected,
                )
                del key_policy
            value_policy = _materialize_page_policy(
                reference_value,
                reconstructed_value,
                policy=str(spec["value_policy"]),
                logical_pages=list(spec["value_pages"]),
            )
            v_only_output = torch.einsum(
                "hgt,thd->hgd",
                reference_probability,
                value_policy.float(),
            ).reshape_as(query)
            both_output = torch.einsum(
                "hgt,thd->hgd",
                key_probability,
                value_policy.float(),
            ).reshape_as(query)
            projected = {
                "k_only": k_only_projected,
                "v_only": _project_attention_output(
                    layer, v_only_output, value_center_gqa
                ),
                "both": _project_attention_output(
                    layer, both_output, value_center_gqa
                ),
            }
            del value_policy
        projected_by_variant[name] = projected

    del key_policy_cache
    metric_records = _batched_projected_error_records(
        exact_projected,
        projected_by_variant,
        policy_specs,
    )
    variants = {
        name: {
            "key_tensor_policy": spec["key_policy"],
            "value_tensor_policy": spec["value_policy"],
            "key_logical_pages": list(spec["key_pages"]),
            "value_logical_pages": list(spec["value_pages"]),
            "relationship_to_all_old_baseline": spec["relationship"],
            "projected_attention_output": metric_records[name],
        }
        for name, spec in policy_specs.items()
    }

    return {
        "enabled": True,
        "source_exact_prefix_pages": int(mapping["exact_prefix_pages"]),
        "scope": (
            "diagnostic-only projected attention-output necessity/sufficiency; "
            "query, quantizer, exact source sink/tail, projection, and attention "
            "math unchanged"
        ),
        "page_sets_validated_inside_source_old_segment": True,
        "source_old_logical_range_start_inclusive_end_exclusive": mapping[
            "old_logical_range_start_inclusive_end_exclusive"
        ],
        "predeclared_page_sets": page_sets,
        "page2_interpretation": (
            "request-specific semantic anchor (tokens32-47), not a validated "
            "universal positional sink"
        ),
        "s3_candidate_interpretation": (
            "fixed contiguous prefix candidate requiring cross-request TRAIN "
            "confirmation before any production or TEST policy"
        ),
        "control_has_equal_fp16_page_bytes_to_s3_added_prefix": True,
        "nominal_exact_fp16_storage_bytes": nominal_exact_fp16_storage_bytes(
            kv_heads=int(reference_key.shape[1]),
            head_dim=int(reference_key.shape[2]),
            candidate_added_pages=len(
                page_sets["contiguous_added_prefix_pages1_2"]
            ),
            control_added_pages=len(
                page_sets["equal_byte_fixed_control_pages3_4"]
            ),
        ),
        "fraction_definitions": {
            "rescue": "1 - variant_error_l2/all_old_component_error_l2",
            "recreation": "page2_only_error_l2/all_old_component_error_l2",
            "fractions_are_not_clamped": True,
        },
        "variants": variants,
    }


def pre_tail_band_ablation_counterfactual(
    *,
    layer: Any,
    query: torch.Tensor,
    reference_key: torch.Tensor,
    reference_value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    reconstructed_value: torch.Tensor,
    reference_probability: torch.Tensor,
    value_center_gqa: torch.Tensor,
    mapping: dict[str, Any],
    existing_projected: dict[str, torch.Tensor],
    step: int,
    request: int,
) -> dict[str, Any]:
    """Causal rescue/recreation for nested bands immediately before the tail."""

    definition = pre_tail_band_definition(mapping, step=step, request=request)
    page_sets = definition["page_sets"]
    exact_projected = existing_projected["reference_centered_form"]
    baseline_projected = {
        "k_only": existing_projected["old_k_quantized_only"],
        "v_only": existing_projected["old_v_quantized_only"],
        "both": existing_projected["old_k_and_v_quantized"],
    }
    band_specs = {
        "b16_t768_equivalent": ("B16", "pre_tail_candidate"),
        "b32_t1024_equivalent": ("B32", "pre_tail_candidate"),
        "c16_equal_byte_older_control": ("C16", "older_control"),
        "c32_equal_byte_older_control": ("C32", "older_control"),
    }
    policy_specs: dict[str, dict[str, Any]] = {
        "all_old_quantized_baseline": {
            "key_policy": "all_old_quantized",
            "value_policy": "all_old_quantized",
            "key_pages": [],
            "value_pages": [],
            "relationship": "baseline",
            "band_symbol": None,
            "band_role": "component_matched_all_old_baseline",
        }
    }
    for label, (symbol, role) in band_specs.items():
        pages = list(page_sets[symbol])
        policy_specs[f"{label}_restored_exact"] = {
            "key_policy": "restore_selected_exact",
            "value_policy": "restore_selected_exact",
            "key_pages": pages,
            "value_pages": pages,
            "relationship": "rescue",
            "band_symbol": symbol,
            "band_role": role,
        }
        policy_specs[f"{label}_only_quantized"] = {
            "key_policy": "only_selected_quantized",
            "value_policy": "only_selected_quantized",
            "key_pages": pages,
            "value_pages": pages,
            "relationship": "recreation",
            "band_symbol": symbol,
            "band_role": role,
        }

    # Only projected hidden vectors survive each variant.  Full-length K/V
    # materializations are created and released sequentially, so the diagnostic
    # never retains a bank of eight full counterfactual cache tensors.
    projected_by_variant: dict[str, dict[str, torch.Tensor]] = {
        "all_old_quantized_baseline": baseline_projected
    }
    for name, spec in list(policy_specs.items())[1:]:
        key_policy = _materialize_page_policy(
            reference_key,
            reconstructed_key,
            policy=str(spec["key_policy"]),
            logical_pages=list(spec["key_pages"]),
        )
        k_only_output, key_probability = grouped_query_attention(
            query, key_policy, reference_value
        )
        k_only_projected = _project_attention_output(
            layer, k_only_output, value_center_gqa
        )
        del key_policy, k_only_output

        value_policy = _materialize_page_policy(
            reference_value,
            reconstructed_value,
            policy=str(spec["value_policy"]),
            logical_pages=list(spec["value_pages"]),
        )
        v_only_output = torch.einsum(
            "hgt,thd->hgd",
            reference_probability,
            value_policy.float(),
        ).reshape_as(query)
        both_output = torch.einsum(
            "hgt,thd->hgd",
            key_probability,
            value_policy.float(),
        ).reshape_as(query)
        projected_by_variant[name] = {
            "k_only": k_only_projected,
            "v_only": _project_attention_output(
                layer, v_only_output, value_center_gqa
            ),
            "both": _project_attention_output(
                layer, both_output, value_center_gqa
            ),
        }
        del value_policy, v_only_output, both_output, key_probability

    metric_records = _batched_projected_error_records(
        exact_projected,
        projected_by_variant,
        policy_specs,
    )
    variants = {
        name: {
            "band_symbol": spec["band_symbol"],
            "band_role": spec["band_role"],
            "key_tensor_policy": spec["key_policy"],
            "value_tensor_policy": spec["value_policy"],
            "key_logical_pages": list(spec["key_pages"]),
            "value_logical_pages": list(spec["value_pages"]),
            "relationship_to_all_old_baseline": spec["relationship"],
            "projected_attention_output": metric_records[name],
        }
        for name, spec in policy_specs.items()
    }
    return {
        "enabled": True,
        "source_gate": {
            "exact_prefix_pages_is_3": True,
            "exact_tail_tokens_is_512": True,
            "target_step_request_is_798_3": True,
        },
        "scope": (
            "diagnostic-only local representation counterfactual at one frozen "
            "TRAIN row and one layer; no production cache policy is evaluated"
        ),
        "attention_quantizer_and_threshold_math_unchanged": True,
        "materialization_schedule": (
            "one full-length K or V counterfactual tensor at a time; only projected "
            "hidden vectors retained across variants"
        ),
        "component_matched_denominators": True,
        "page_sets_validated_inside_source_old_segment": True,
        "predeclared_bands": definition,
        "nominal_exact_fp16_storage_bytes": pre_tail_band_storage_bytes(
            kv_heads=int(reference_key.shape[1]),
            head_dim=int(reference_key.shape[2]),
            definition=definition,
        ),
        "fraction_definitions": {
            "rescue": "1 - restored_band_error_l2/all_old_component_error_l2",
            "recreation": (
                "band_only_quantized_error_l2/all_old_component_error_l2"
            ),
            "primary_aggregate": (
                "ratio of summed variant error L2 to summed matching-baseline "
                "error L2 across all 32 layer rows"
            ),
            "fractions_are_not_clamped": True,
        },
        "variants": variants,
    }


def _page_error_rows(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    probability: torch.Tensor,
    *,
    mapping: dict[str, Any],
    page_top_k: int,
) -> dict[str, Any]:
    sequence_length = int(reference.shape[0])
    total_pages = int(mapping["total_pages"])
    padded_tokens = total_pages * PG.PAGE
    pad = padded_tokens - sequence_length
    difference = reconstructed.float() - reference.float()
    if pad:
        difference = F.pad(difference, (0, 0, 0, 0, 0, pad))
        reference_padded = F.pad(reference.float(), (0, 0, 0, 0, 0, pad))
        probability_padded = F.pad(probability, (0, pad))
    else:
        reference_padded = reference.float()
        probability_padded = probability
    difference_pages = difference.reshape(
        total_pages, PG.PAGE, int(reference.shape[1]), int(reference.shape[2])
    )
    reference_pages = reference_padded.reshape_as(difference_pages)
    mass = probability_padded.reshape(
        int(probability.shape[0]),
        int(probability.shape[1]),
        total_pages,
        PG.PAGE,
    ).sum(dim=-1).mean(dim=(0, 1))
    rmse = difference_pages.square().mean(dim=(1, 2, 3)).sqrt()
    relative = (
        torch.linalg.vector_norm(difference_pages.reshape(total_pages, -1), dim=1)
        / torch.linalg.vector_norm(
            reference_pages.reshape(total_pages, -1), dim=1
        ).clamp_min(1.0e-30)
    )
    maximum = difference_pages.abs().amax(dim=(1, 2, 3))
    retain = min(page_top_k, total_pages)
    selected = set(mass.topk(retain).indices.cpu().tolist())
    selected.update(rmse.topk(retain).indices.cpu().tolist())
    old_logical_begin, old_logical_end = map(
        int, mapping["old_logical_range_start_inclusive_end_exclusive"]
    )
    exact_tail_pages = int(mapping["exact_tail_pages"])
    exact_sink_pages = int(mapping["exact_sink_pages"])
    context_pages = int(mapping["context_pages"])
    records = []
    for page in sorted(selected):
        record = logical_page_record(
            page,
            sequence_length=sequence_length,
            old_logical_begin=old_logical_begin,
            old_logical_end=old_logical_end,
            exact_tail_pages=exact_tail_pages,
            exact_sink_pages=exact_sink_pages,
            context_pages=context_pages,
        )
        record.update(
            {
                "reference_attention_mass_mean_across_heads": float(
                    mass[page].item()
                ),
                "reconstruction_rmse": float(rmse[page].item()),
                "reconstruction_relative_l2": float(relative[page].item()),
                "reconstruction_maximum_absolute_error": float(
                    maximum[page].item()
                ),
            }
        )
        records.append(record)
    return {
        "retention": (
            "union of top pages by reference attention mass and reconstruction RMSE"
        ),
        "page_top_k_per_ranking": retain,
        "retained_pages": records,
    }


def kv_counterfactual(
    *,
    layer: Any,
    layer_input: torch.Tensor,
    hf_key: torch.Tensor,
    hf_value: torch.Tensor,
    key_center: torch.Tensor,
    value_center: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position: int,
    mapping: dict[str, Any],
    hf_attention_delta: torch.Tensor,
    page_top_k: int,
    pre_tail_band_target: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Vary old K and old V independently under identical explicit softmax."""

    normalized = layer.input_layernorm(layer_input.reshape(1, 1, -1))
    hq = int(layer.self_attn.config.num_attention_heads)
    query = layer.self_attn.q_proj(normalized).reshape(1, 1, hq, PG.DIM)[0, 0]
    query, _ = PG.BASE_E2E.apply_rope(
        query,
        query.new_zeros(query.shape),
        rope_cos[position],
        rope_sin[position],
    )
    old_logical_begin, old_logical_end = map(
        int, mapping["old_logical_range_start_inclusive_end_exclusive"]
    )
    old_pages = old_logical_end - old_logical_begin
    reference_key = (hf_key.float() - key_center.float()[None]).half().float()
    reference_value = (hf_value.float() - value_center.float()[None]).half().float()
    reconstructed_key, key_codes, key_scale = reconstruct_old_pages(
        hf_key,
        key_center,
        old_pages,
        old_logical_begin=old_logical_begin,
    )
    reconstructed_value, value_codes, value_scale = reconstruct_old_pages(
        hf_value,
        value_center,
        old_pages,
        old_logical_begin=old_logical_begin,
    )

    reference_output, reference_probability = grouped_query_attention(
        query, reference_key, reference_value
    )
    k_probability_output, quantized_k_probability = grouped_query_attention(
        query, reconstructed_key, reference_value
    )
    v_only_output = torch.einsum(
        "hgt,thd->hgd",
        reference_probability,
        reconstructed_value.float(),
    ).reshape_as(reference_output)
    both_output = torch.einsum(
        "hgt,thd->hgd",
        quantized_k_probability,
        reconstructed_value.float(),
    ).reshape_as(reference_output)
    value_center_gqa = value_center.repeat_interleave(
        hq // int(value_center.shape[0]), dim=0
    ).float()
    outputs = {
        "reference_centered_form": reference_output + value_center_gqa,
        "old_k_quantized_only": k_probability_output + value_center_gqa,
        "old_v_quantized_only": v_only_output + value_center_gqa,
        "old_k_and_v_quantized": both_output + value_center_gqa,
    }
    projected = {
        name: F.linear(
            output.reshape(1, -1).half(), layer.self_attn.o_proj.weight
        )[0]
        for name, output in outputs.items()
    }
    ablation_applicability = page_ablation_applicability(mapping)
    page_ablation = (
        page_ablation_counterfactual(
            layer=layer,
            query=query,
            reference_key=reference_key,
            reference_value=reference_value,
            reconstructed_key=reconstructed_key,
            reconstructed_value=reconstructed_value,
            reference_probability=reference_probability,
            quantized_key_probability=quantized_k_probability,
            value_center_gqa=value_center_gqa,
            mapping=mapping,
            existing_projected=projected,
        )
        if ablation_applicability["enabled"]
        else {
            **ablation_applicability,
            "scope": (
                "not executed; generic layer/K/V and page-error attribution "
                "remain active"
            ),
        }
    )
    pre_tail_band_ablation = (
        pre_tail_band_ablation_counterfactual(
            layer=layer,
            query=query,
            reference_key=reference_key,
            reference_value=reference_value,
            reconstructed_key=reconstructed_key,
            reconstructed_value=reconstructed_value,
            reference_probability=reference_probability,
            value_center_gqa=value_center_gqa,
            mapping=mapping,
            existing_projected=projected,
            step=int(pre_tail_band_target[0]),
            request=int(pre_tail_band_target[1]),
        )
        if pre_tail_band_target is not None
        else {
            "enabled": False,
            "scope": (
                "not executed; predeclared only for frozen S3/T512 TRAIN "
                "target 798:3"
            ),
        }
    )
    reference_name = "reference_centered_form"
    comparisons = {
        name: vector_comparison(outputs[reference_name], output)
        for name, output in outputs.items()
        if name != reference_name
    }
    projected_comparisons = {
        name: vector_comparison(projected[reference_name], output)
        for name, output in projected.items()
        if name != reference_name
    }
    k_delta = outputs["old_k_quantized_only"] - outputs[reference_name]
    v_delta = outputs["old_v_quantized_only"] - outputs[reference_name]
    both_delta = outputs["old_k_and_v_quantized"] - outputs[reference_name]
    interaction = both_delta - k_delta - v_delta
    denominator = torch.linalg.vector_norm(reference_output).clamp_min(1.0e-30)
    manual_projection = projected[reference_name]
    return {
        "scope": (
            "representation-level controlled counterfactual; explicit FP32 "
            "softmax; exact FP16 pages and query held fixed"
        ),
        "old_k_v_math_unchanged": True,
        "exact_segment_quantized": False,
        "quantized_old_logical_range_start_inclusive_end_exclusive": [
            old_logical_begin,
            old_logical_end,
        ],
        "preserved_exact_prefix_logical_pages": mapping[
            "exact_prefix_logical_pages"
        ],
        "preserved_exact_sink_logical_pages": mapping["exact_sink_logical_pages"],
        "preserved_exact_tail_logical_range_start_inclusive_end_exclusive": mapping[
            "exact_tail_logical_range_start_inclusive_end_exclusive"
        ],
        "page_ablation": page_ablation,
        "pre_tail_band_ablation": pre_tail_band_ablation,
        "output_comparisons_vs_unquantized_centered_reference": comparisons,
        "projected_comparisons_vs_unquantized_centered_reference": (
            projected_comparisons
        ),
        "component_delta_relative_l2": {
            "k_only": float((torch.linalg.vector_norm(k_delta) / denominator).item()),
            "v_only": float((torch.linalg.vector_norm(v_delta) / denominator).item()),
            "both": float((torch.linalg.vector_norm(both_delta) / denominator).item()),
            "nonadditive_interaction": float(
                (torch.linalg.vector_norm(interaction) / denominator).item()
            ),
        },
        "manual_reference_projection_vs_hf_sdpa_attention_delta": (
            vector_comparison(hf_attention_delta, manual_projection)
        ),
        "quantizer": {
            "rounding": "half-away-from-zero",
            "code_range": [-127, 127],
            "scale": "FP16(max_abs(centered_page_head)/127), floor 2^-20",
            "key_code_min_max": [int(key_codes.min()), int(key_codes.max())],
            "value_code_min_max": [
                int(value_codes.min()),
                int(value_codes.max()),
            ],
            "key_scale_min_max": [
                float(key_scale.min()),
                float(key_scale.max()),
            ],
            "value_scale_min_max": [
                float(value_scale.min()),
                float(value_scale.max()),
            ],
        },
        "key_page_localization": _page_error_rows(
            reference_key,
            reconstructed_key,
            reference_probability,
            mapping=mapping,
            page_top_k=page_top_k,
        ),
        "value_page_localization": _page_error_rows(
            reference_value,
            reconstructed_value,
            reference_probability,
            mapping=mapping,
            page_top_k=page_top_k,
        ),
    }


class HFTargetInterceptor:
    """Observe selected B1 HF decode calls inside the coherent fixture."""

    def __init__(
        self,
        *,
        model: Any,
        cache: Any,
        targets: list[dict[str, Any]],
        context: int,
        exact_tail_tokens: int,
        exact_sink_pages: int,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        page_top_k: int,
        pre_tail_band_ablation_enabled: bool,
    ) -> None:
        self.model = model
        self.cache = cache
        self.context = context
        self.exact_tail_tokens = exact_tail_tokens
        self.exact_sink_pages = exact_sink_pages
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.page_top_k = page_top_k
        self.pre_tail_band_ablation_enabled = bool(
            pre_tail_band_ablation_enabled
        )
        self.targets = {(row["step"], row["request"]) for row in targets}
        self.target_steps_by_request: dict[int, set[int]] = {}
        for step, request in self.targets:
            self.target_steps_by_request.setdefault(request, set()).add(step)
        self.traces: dict[tuple[int, int], dict[str, Any]] = {}
        self._original_forward = None
        self._handles: list[Any] = []
        self._active: tuple[int, int] | None = None
        self._layer_inputs: dict[int, torch.Tensor] = {}
        self._attention_delta: dict[int, torch.Tensor] = {}
        self._layer_outputs: dict[int, torch.Tensor] = {}
        self._current_request = -1
        self._last_step = -1

    @staticmethod
    def _first_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(
            output[0], torch.Tensor
        ):
            return output[0]
        raise RuntimeError("HF hook output does not begin with a tensor")

    def _layer_pre_hook(self, layer_index: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if self._active is not None:
                self._layer_inputs[layer_index] = inputs[0].detach()

        return hook

    def _attention_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._active is not None:
                self._attention_delta[layer_index] = self._first_tensor(output).detach()

        return hook

    def _layer_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._active is not None:
                self._layer_outputs[layer_index] = self._first_tensor(output).detach()

        return hook

    def __enter__(self) -> "HFTargetInterceptor":
        for layer_index, layer in enumerate(self.model.model.layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._layer_pre_hook(layer_index))
            )
            self._handles.append(
                layer.self_attn.register_forward_hook(
                    self._attention_hook(layer_index)
                )
            )
            self._handles.append(
                layer.register_forward_hook(self._layer_hook(layer_index))
            )
        self._original_forward = self.model.model.forward

        def traced_forward(*args: Any, **kwargs: Any):
            dynamic_cache = kwargs.get("past_key_values")
            input_ids = kwargs.get("input_ids")
            sequence_before = (
                int(dynamic_cache.get_seq_length())
                if dynamic_cache is not None
                else -1
            )
            is_decode = (
                sequence_before >= self.context
                and isinstance(input_ids, torch.Tensor)
                and int(input_ids.numel()) == 1
            )
            step = sequence_before - self.context if is_decode else -1
            if is_decode and step == 0:
                self._current_request += 1
                self._last_step = -1
            if is_decode:
                if step != self._last_step + 1:
                    raise RuntimeError("HF decode calls are not sequential")
                self._last_step = step
            key = (step, self._current_request)
            self._active = key if is_decode and key in self.targets else None
            if self._active is not None:
                self._layer_inputs.clear()
                self._attention_delta.clear()
                self._layer_outputs.clear()
            assert self._original_forward is not None
            output = self._original_forward(*args, **kwargs)
            if self._active is not None:
                self._finalize(output.past_key_values)
            self._active = None
            return output

        self.model.model.forward = traced_forward
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_forward is not None:
            self.model.model.forward = self._original_forward
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _finalize(self, dynamic_cache: Any) -> None:
        assert self._active is not None
        step, request = self._active
        layers = len(self.model.model.layers)
        if not (
            len(self._layer_inputs)
            == len(self._attention_delta)
            == len(self._layer_outputs)
            == layers
        ):
            raise RuntimeError("HF target hooks did not cover every decoder layer")
        mapping = page_age_mapping(
            context=self.context,
            step=step,
            exact_tail_tokens=self.exact_tail_tokens,
            exact_sink_pages=self.exact_sink_pages,
        )
        layer_records = []
        for layer_index, layer in enumerate(self.model.model.layers):
            layer_input = self._layer_inputs[layer_index][0, -1]
            attention_delta = self._attention_delta[layer_index][0, -1]
            output_hidden = self._layer_outputs[layer_index][0, -1]
            hf_key, hf_value = PREFILL.cache_layer_tensors(
                dynamic_cache, layer_index
            )
            source_key = hf_key[0].transpose(0, 1).contiguous()
            source_value = hf_value[0].transpose(0, 1).contiguous()
            expected_tokens = int(mapping["sequence_length_after_append"])
            if int(source_key.shape[0]) != expected_tokens:
                raise RuntimeError("HF target cache length is inconsistent")
            counterfactual = kv_counterfactual(
                layer=layer,
                layer_input=layer_input,
                hf_key=source_key,
                hf_value=source_value,
                key_center=self.cache.key_center[layer_index, request],
                value_center=self.cache.value_center[layer_index, request],
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                position=self.context + step,
                mapping=mapping,
                hf_attention_delta=attention_delta,
                page_top_k=self.page_top_k,
                pre_tail_band_target=(step, request)
                if self.pre_tail_band_ablation_enabled
                else None,
            )
            layer_records.append(
                {
                    "layer": layer_index,
                    "reference_vectors": {
                        "input_hidden": layer_input.detach().cpu(),
                        "attention_delta": attention_delta.detach().cpu(),
                        "post_attention_hidden": (
                            layer_input + attention_delta
                        ).detach().cpu(),
                        "output_hidden": output_hidden.detach().cpu(),
                    },
                    "kv_counterfactual": counterfactual,
                }
            )
            del source_key, source_value
        self.traces[(step, request)] = {
            "step": step,
            "request": request,
            "absolute_position": self.context + step,
            "page_mapping": mapping,
            "layers": layer_records,
        }


def _candidate_layer(
    decoder: Any,
    layer_index: int,
    hidden: torch.Tensor,
    position: int,
) -> dict[str, torch.Tensor]:
    layer = decoder.model.model.layers[layer_index]
    residual = hidden
    normalized = layer.input_layernorm(hidden)
    attention = layer.self_attn
    qkv = F.linear(
        normalized,
        attention.qkv_weight,
        getattr(attention, "qkv_bias", None),
    )
    q_flat, k_flat, v_flat = torch.split(
        qkv,
        (
            attention._pkv_q_width,
            attention._pkv_kv_width,
            attention._pkv_kv_width,
        ),
        dim=-1,
    )
    query = q_flat.reshape(decoder.batch_size, decoder.hq, PG.DIM)
    key = k_flat.reshape(decoder.batch_size, decoder.hkv, PG.DIM)
    value = v_flat.reshape(decoder.batch_size, decoder.hkv, PG.DIM)
    decoder.append(layer_index, query, key, value, position)
    attended = decoder.attention(layer_index)
    attention_delta = F.linear(
        attended.reshape(decoder.batch_size, -1),
        attention.o_proj.weight,
        decoder.output_projection_biases[layer_index],
    )
    post_attention_hidden = residual + attention_delta
    mlp_input = layer.post_attention_layernorm(post_attention_hidden)
    mlp = layer.mlp
    gate_up = F.linear(
        mlp_input,
        mlp.gate_up_weight,
        getattr(mlp, "gate_up_bias", None),
    )
    gate, up = gate_up.split(mlp._pkv_intermediate, dim=-1)
    activated = F.silu(gate) * up
    output_hidden = post_attention_hidden + mlp.down_proj(activated)
    return {
        "input_hidden": hidden,
        "attention_output": attended,
        "attention_delta": attention_delta,
        "post_attention_hidden": post_attention_hidden,
        "output_hidden": output_hidden,
    }


@torch.inference_mode()
def trace_candidate_teacher(
    decoder: Any,
    *,
    teacher_inputs: torch.Tensor,
    context: int,
    targets: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], dict[str, Any]]]:
    target_requests_by_step: dict[int, list[int]] = {}
    for target in targets:
        target_requests_by_step.setdefault(target["step"], []).append(
            target["request"]
        )
    traces: dict[tuple[int, int], dict[str, Any]] = {}
    logits_blocks: list[torch.Tensor] = []
    token_blocks: list[torch.Tensor] = []
    pending_logits: list[torch.Tensor] = []
    pending_tokens: list[torch.Tensor] = []
    for step in range(int(teacher_inputs.shape[0])):
        position = context + step
        requests = sorted(set(target_requests_by_step.get(step, [])))
        if not requests:
            logits = decoder.step(teacher_inputs[step], position)[0]
        else:
            decoder.plan(position + 1)
            hidden = decoder.model.model.embed_tokens(
                teacher_inputs[step].reshape(decoder.batch_size, 1)
            )[:, 0]
            for layer_index in range(decoder.layers):
                stages = _candidate_layer(decoder, layer_index, hidden, position)
                for request in requests:
                    target = traces.setdefault(
                        (step, request),
                        {
                            "step": step,
                            "request": request,
                            "absolute_position": position,
                            "layers": [],
                        },
                    )
                    target["layers"].append(
                        {
                            "layer": layer_index,
                            "candidate_vectors": {
                                name: stages[name][request].detach().cpu()
                                for name in (
                                    "input_hidden",
                                    "attention_delta",
                                    "post_attention_hidden",
                                    "output_hidden",
                                )
                            },
                        }
                    )
                hidden = stages["output_hidden"]
            logits = decoder.model.lm_head(
                decoder.model.model.norm(hidden)
            ).float()
        generated = logits.argmax(dim=-1)
        pending_logits.append(logits.detach().clone())
        pending_tokens.append(generated.detach().clone())
        if (step + 1) % PG.PAGE == 0:
            logits_blocks.append(torch.stack(pending_logits).cpu())
            token_blocks.append(torch.stack(pending_tokens).cpu())
            pending_logits.clear()
            pending_tokens.clear()
    if pending_logits or pending_tokens:
        raise RuntimeError("candidate trace requires page-aligned decode steps")
    return torch.cat(logits_blocks), torch.cat(token_blocks), traces


def _top1_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    top_vocab: int = 12,
) -> dict[str, Any]:
    width = min(top_vocab, int(reference.numel()))
    reference_values, reference_ids = reference.float().topk(width)
    candidate_values, candidate_ids = candidate.float().topk(width)
    reference_margin = float((reference_values[0] - reference_values[1]).item())
    candidate_margin = float((candidate_values[0] - candidate_values[1]).item())
    return {
        "reference_top1_id": int(reference_ids[0]),
        "candidate_top1_id": int(candidate_ids[0]),
        "top1_match": int(reference_ids[0]) == int(candidate_ids[0]),
        "reference_top1_margin": reference_margin,
        "candidate_top1_margin": candidate_margin,
        "reference_top_ids": reference_ids.tolist(),
        "reference_top_values": reference_values.tolist(),
        "candidate_top_ids": candidate_ids.tolist(),
        "candidate_top_values": candidate_values.tolist(),
    }


def merge_target_traces(
    *,
    targets: list[dict[str, Any]],
    hf_traces: dict[tuple[int, int], dict[str, Any]],
    candidate_traces: dict[tuple[int, int], dict[str, Any]],
    hf_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    hidden_cosine_trigger: float,
    hidden_relative_l2_trigger: float,
) -> list[dict[str, Any]]:
    merged = []
    for target in targets:
        key = (target["step"], target["request"])
        if key not in hf_traces or key not in candidate_traces:
            raise RuntimeError(f"target {key} lacks a complete HF/candidate trace")
        reference = hf_traces[key]
        candidate = candidate_traces[key]
        if len(reference["layers"]) != len(candidate["layers"]):
            raise RuntimeError("HF/candidate layer trace lengths differ")
        layers = []
        previous_output_relative = 0.0
        for reference_layer, candidate_layer in zip(
            reference["layers"], candidate["layers"]
        ):
            if reference_layer["layer"] != candidate_layer["layer"]:
                raise RuntimeError("HF/candidate layer indices differ")
            comparisons = {
                stage: vector_comparison(
                    reference_layer["reference_vectors"][stage],
                    candidate_layer["candidate_vectors"][stage],
                )
                for stage in (
                    "input_hidden",
                    "attention_delta",
                    "post_attention_hidden",
                    "output_hidden",
                )
            }
            output_relative = comparisons["output_hidden"]["relative_l2"]
            layers.append(
                {
                    "layer": reference_layer["layer"],
                    **comparisons,
                    "output_relative_l2_increment_from_prior_layer": (
                        output_relative - previous_output_relative
                    ),
                    "kv_counterfactual": reference_layer["kv_counterfactual"],
                }
            )
            previous_output_relative = output_relative
        step, request = key
        logit_reference = hf_logits[step, request]
        logit_candidate = candidate_logits[step, request]
        merged.append(
            {
                **target,
                "page_mapping": reference["page_mapping"],
                "logits": vector_comparison(logit_reference, logit_candidate),
                "top1": _top1_summary(logit_reference, logit_candidate),
                "first_diagnostic_divergence": first_layer_divergence(
                    layers,
                    cosine_trigger=hidden_cosine_trigger,
                    relative_l2_trigger=hidden_relative_l2_trigger,
                ),
                "layers": layers,
            }
        )
    return merged


def _finite_scalar_summary(values: list[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("page-ablation aggregate requires finite values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "count": len(values),
        "minimum": ordered[0],
        "mean": sum(values) / len(values),
        "median": median,
        "maximum": ordered[-1],
    }


def prefill_population_protocol_gate(
    prefill_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate every sampled prefill-population record fail closed."""

    if not prefill_records:
        raise RuntimeError("prefill population protocol has no sampled records")
    failed_record_indices = []
    for index, record in enumerate(prefill_records):
        if not isinstance(record, dict) or "sampled_population_gate_passed" not in record:
            raise RuntimeError(
                f"prefill population record {index} lacks its required gate"
            )
        if record["sampled_population_gate_passed"] is not True:
            failed_record_indices.append(index)
    return {
        "record_count": len(prefill_records),
        "all_sampled_population_gates_passed": not failed_record_indices,
        "failed_record_indices": failed_record_indices,
    }


def diagnostic_protocol_passed(
    *,
    source_reproduction_passed: bool,
    manual_reference_finite: bool,
    sampled_prefill_population_gate_passed: bool,
) -> bool:
    """Compose the diagnostic protocol gate without omitting cache population."""

    return bool(
        source_reproduction_passed
        and manual_reference_finite
        and sampled_prefill_population_gate_passed
    )


def aggregate_page_ablation(
    targets: list[dict[str, Any]],
    *,
    expected_layer_count: int,
    batch_size: int,
    require_frozen_source_grid: bool,
) -> dict[str, Any]:
    """Aggregate paired page ablations over the full target-by-layer grid."""

    if not targets or expected_layer_count <= 0 or batch_size <= 0:
        raise ValueError("page-ablation aggregate dimensions must be positive")
    layer_count = expected_layer_count
    target_keys = [(int(target["step"]), int(target["request"])) for target in targets]
    if len(set(target_keys)) != len(target_keys):
        raise RuntimeError("page-ablation target keys are not unique")
    frozen_target_keys = {(594, 3), (798, 3), (866, 3), (900, 3)}
    frozen_grid_passed = (
        set(target_keys) == frozen_target_keys
        and len(targets) == 4
        and expected_layer_count == 32
    )
    if require_frozen_source_grid and not frozen_grid_passed:
        raise RuntimeError("page-ablation frozen source target grid is incomplete")
    if any(len(target["layers"]) != layer_count for target in targets):
        raise RuntimeError("page-ablation targets do not share one layer count")
    for target in targets:
        layer_ids = [int(layer["layer"]) for layer in target["layers"]]
        if layer_ids != list(range(layer_count)):
            raise RuntimeError(
                "page-ablation layer ids are not the complete ordered range"
            )
    target_layer_keys = {
        (int(target["step"]), int(target["request"]), int(layer["layer"]))
        for target in targets
        for layer in target["layers"]
    }
    if len(target_layer_keys) != len(targets) * layer_count:
        raise RuntimeError("page-ablation target-layer observations are not unique")
    rows = [
        layer["kv_counterfactual"]["page_ablation"]
        for target in targets
        for layer in target["layers"]
    ]
    if len(rows) != len(targets) * layer_count:
        raise RuntimeError("page-ablation target-layer grid is incomplete")
    if not all(
        row["page_sets_validated_inside_source_old_segment"] for row in rows
    ):
        raise RuntimeError("page-ablation includes a page outside the old segment")
    variant_names = tuple(rows[0]["variants"])
    component_names = ("k_only", "v_only", "both")
    for row in rows:
        if tuple(row["variants"]) != variant_names:
            raise RuntimeError("page-ablation variant identities differ across rows")

    aggregate_variants: dict[str, Any] = {}
    for variant_name in variant_names:
        variant_rows = [row["variants"][variant_name] for row in rows]
        relationships = {
            str(row["relationship_to_all_old_baseline"])
            for row in variant_rows
        }
        if len(relationships) != 1:
            raise RuntimeError("page-ablation relationship differs across rows")
        relationship = relationships.pop()
        component_aggregate = {}
        for component in component_names:
            metrics = [
                row["projected_attention_output"][component]
                for row in variant_rows
            ]
            baseline_sum = sum(
                float(metric["all_old_component_baseline_error_l2"])
                for metric in metrics
            )
            error_sum = sum(float(metric["error_l2"]) for metric in metrics)
            if not math.isfinite(baseline_sum) or baseline_sum <= 0.0:
                raise RuntimeError("page-ablation aggregate baseline sum is invalid")
            weighted_ratio = error_sum / baseline_sum
            component_result = {
                "error_l2": _finite_scalar_summary(
                    [float(metric["error_l2"]) for metric in metrics]
                ),
                "error_norm_ratio_vs_all_old_component_baseline": (
                    _finite_scalar_summary(
                        [
                            float(
                                metric[
                                    "error_norm_ratio_vs_all_old_component_baseline"
                                ]
                            )
                            for metric in metrics
                        ]
                    )
                ),
                "all_old_component_baseline_error_l2_sum": baseline_sum,
                "variant_error_l2_sum": error_sum,
                "denominator_weighted_error_norm_ratio_of_sums": weighted_ratio,
            }
            if relationship in ("baseline", "rescue"):
                rescue = [
                    float(metric["error_norm_rescue_fraction"])
                    for metric in metrics
                ]
                component_result["error_norm_rescue_fraction"] = (
                    _finite_scalar_summary(rescue)
                )
                component_result["rows_with_positive_rescue"] = sum(
                    value > 0.0 for value in rescue
                )
                component_result["denominator_weighted_rescue_fraction"] = (
                    1.0 - weighted_ratio
                )
            elif relationship == "recreation":
                recreation = [
                    float(metric["error_norm_recreation_fraction"])
                    for metric in metrics
                ]
                component_result["error_norm_recreation_fraction"] = (
                    _finite_scalar_summary(recreation)
                )
                component_result["rows_recreating_at_least_half_baseline"] = sum(
                    value >= 0.5 for value in recreation
                )
                component_result["rows_recreating_at_least_full_baseline"] = sum(
                    value >= 1.0 for value in recreation
                )
                component_result[
                    "denominator_weighted_recreation_fraction"
                ] = weighted_ratio
            component_aggregate[component] = component_result
        aggregate_variants[variant_name] = {
            "relationship_to_all_old_baseline": relationship,
            "key_tensor_policy": variant_rows[0]["key_tensor_policy"],
            "value_tensor_policy": variant_rows[0]["value_tensor_policy"],
            "key_logical_pages": variant_rows[0]["key_logical_pages"],
            "value_logical_pages": variant_rows[0]["value_logical_pages"],
            "components": component_aggregate,
        }

    paired_contrast_specs = {
        "s3_rescue_minus_equal_byte_control": (
            "contiguous_prefix_s3_candidate_pages1_2_restored",
            "equal_byte_control_pages3_4_restored",
        ),
        "page2_both_rescue_minus_s3_rescue": (
            "page2_k_and_v_restored",
            "contiguous_prefix_s3_candidate_pages1_2_restored",
        ),
        "page2_v_rescue_minus_page2_k_rescue": (
            "page2_v_restored",
            "page2_k_restored",
        ),
    }
    paired_contrasts = {}
    for contrast_name, (left_name, right_name) in paired_contrast_specs.items():
        paired_contrasts[contrast_name] = {
            component: _finite_scalar_summary(
                [
                    float(
                        row["variants"][left_name]["projected_attention_output"][
                            component
                        ]["error_norm_rescue_fraction"]
                    )
                    - float(
                        row["variants"][right_name]["projected_attention_output"][
                            component
                        ]["error_norm_rescue_fraction"]
                    )
                    for row in rows
                ]
            )
            for component in component_names
        }

    per_layer_combined = []
    for layer_index in range(layer_count):
        layer_rows = [
            target["layers"][layer_index]["kv_counterfactual"][
                "page_ablation"
            ]
            for target in targets
        ]
        layer_variants = {}
        for variant_name in variant_names:
            metrics = [
                row["variants"][variant_name]["projected_attention_output"][
                    "both"
                ]
                for row in layer_rows
            ]
            relationship = rows[0]["variants"][variant_name][
                "relationship_to_all_old_baseline"
            ]
            result = {
                "error_norm_ratio_vs_all_old_component_baseline": (
                    _finite_scalar_summary(
                        [
                            float(
                                metric[
                                    "error_norm_ratio_vs_all_old_component_baseline"
                                ]
                            )
                            for metric in metrics
                        ]
                    )
                )
            }
            fraction_name = (
                "error_norm_recreation_fraction"
                if relationship == "recreation"
                else "error_norm_rescue_fraction"
            )
            result[fraction_name] = _finite_scalar_summary(
                [float(metric[fraction_name]) for metric in metrics]
            )
            layer_variants[variant_name] = result
        per_layer_combined.append(
            {
                "layer": layer_index,
                "target_rows": len(layer_rows),
                "variants": layer_variants,
            }
        )

    storage = rows[0]["nominal_exact_fp16_storage_bytes"]
    if any(row["nominal_exact_fp16_storage_bytes"] != storage for row in rows):
        raise RuntimeError("page-ablation storage geometry differs across rows")
    candidate_bytes = int(storage["s3_candidate_added_bytes_per_layer_request"])
    control_bytes = int(storage["control_added_bytes_per_layer_request"])
    if candidate_bytes != control_bytes:
        raise RuntimeError("page-ablation S3/control nominal bytes differ")

    return {
        "target_count": len(targets),
        "layers_per_target": layer_count,
        "target_layer_rows": len(rows),
        "complete_selected_target_layer_cartesian_product": True,
        "unique_target_keys": True,
        "unique_target_layer_observations": True,
        "ordered_complete_layer_ids": True,
        "frozen_source_four_by_32_gate": {
            "required": require_frozen_source_grid,
            "expected_target_step_request": [
                [step, request] for step, request in sorted(frozen_target_keys)
            ],
            "observed_target_step_request": [
                [step, request] for step, request in target_keys
            ],
            "expected_layers": 32,
            "passed": frozen_grid_passed,
        },
        "all_page_sets_validated_inside_source_old_segment": True,
        "component_matched_denominators": True,
        "aggregate_weighting": {
            "per_row_fraction_summaries": "equal target-layer weight",
            "primary_ratio_of_sums": (
                "sum variant error L2 / sum matching all-old baseline error L2"
            ),
        },
        "nominal_exact_fp16_storage_bytes": {
            **storage,
            "batch_size": batch_size,
            "layers": layer_count,
            "s1_to_s3_gross_added_bytes": (
                candidate_bytes * layer_count * batch_size
            ),
            "equal_byte_control_gross_bytes": (
                control_bytes * layer_count * batch_size
            ),
        },
        "variants": aggregate_variants,
        "paired_contrasts": paired_contrasts,
        "per_layer_combined": per_layer_combined,
        "interpretation_guard": (
            "page2 is a request-specific semantic anchor; S3 remains a fixed-prefix "
            "candidate requiring cross-request TRAIN confirmation"
        ),
    }


def aggregate_pre_tail_band_ablation(
    targets: list[dict[str, Any]],
    *,
    expected_layer_count: int,
) -> dict[str, Any]:
    """Aggregate the strict one-row nested-band diagnostic over all layers."""

    if expected_layer_count != 32:
        raise ValueError("pre-tail band aggregate requires all 32 Mistral layers")
    target_keys = [
        (int(target["step"]), int(target["request"])) for target in targets
    ]
    if target_keys != [(798, 3)]:
        raise RuntimeError(
            "pre-tail band aggregate requires exactly frozen TRAIN target 798:3"
        )
    target = targets[0]
    layer_ids = [int(layer["layer"]) for layer in target["layers"]]
    if layer_ids != list(range(expected_layer_count)):
        raise RuntimeError("pre-tail band aggregate lacks ordered layers 0..31")
    expected_definition = pre_tail_band_definition(
        target["page_mapping"], step=798, request=3
    )
    rows = [
        layer["kv_counterfactual"]["pre_tail_band_ablation"]
        for layer in target["layers"]
    ]
    if len(rows) != expected_layer_count or any(row.get("enabled") is not True for row in rows):
        raise RuntimeError("pre-tail band aggregate layer grid is incomplete")
    if any(
        row.get("page_sets_validated_inside_source_old_segment") is not True
        or row.get("component_matched_denominators") is not True
        or row.get("predeclared_bands") != expected_definition
        for row in rows
    ):
        raise RuntimeError("pre-tail band row identity or page-set gate failed")

    expected_variant_names = (
        "all_old_quantized_baseline",
        "b16_t768_equivalent_restored_exact",
        "b16_t768_equivalent_only_quantized",
        "b32_t1024_equivalent_restored_exact",
        "b32_t1024_equivalent_only_quantized",
        "c16_equal_byte_older_control_restored_exact",
        "c16_equal_byte_older_control_only_quantized",
        "c32_equal_byte_older_control_restored_exact",
        "c32_equal_byte_older_control_only_quantized",
    )
    for row in rows:
        if tuple(row["variants"]) != expected_variant_names:
            raise RuntimeError("pre-tail band variant identity/order differs")
    component_names = ("k_only", "v_only", "both")
    aggregate_variants: dict[str, Any] = {}
    for variant_name in expected_variant_names:
        variant_rows = [row["variants"][variant_name] for row in rows]
        relationships = {
            str(variant["relationship_to_all_old_baseline"])
            for variant in variant_rows
        }
        if len(relationships) != 1:
            raise RuntimeError("pre-tail band relationship differs across layers")
        relationship = relationships.pop()
        components: dict[str, Any] = {}
        for component in component_names:
            metrics = [
                variant["projected_attention_output"][component]
                for variant in variant_rows
            ]
            baseline_metrics = [
                row["variants"]["all_old_quantized_baseline"][
                    "projected_attention_output"
                ][component]
                for row in rows
            ]
            if any(
                float(metric["all_old_component_baseline_error_l2"])
                != float(baseline["error_l2"])
                for metric, baseline in zip(metrics, baseline_metrics)
            ):
                raise RuntimeError("pre-tail component baseline pairing failed")
            baseline_sum = sum(float(metric["error_l2"]) for metric in baseline_metrics)
            variant_sum = sum(float(metric["error_l2"]) for metric in metrics)
            if not math.isfinite(baseline_sum) or baseline_sum <= 0.0:
                raise RuntimeError("pre-tail aggregate baseline sum is invalid")
            ratio_of_sums = variant_sum / baseline_sum
            component_result = {
                "error_l2": _finite_scalar_summary(
                    [float(metric["error_l2"]) for metric in metrics]
                ),
                "error_norm_ratio_vs_all_old_component_baseline": (
                    _finite_scalar_summary(
                        [
                            float(
                                metric[
                                    "error_norm_ratio_vs_all_old_component_baseline"
                                ]
                            )
                            for metric in metrics
                        ]
                    )
                ),
                "all_old_component_baseline_error_l2_sum": baseline_sum,
                "variant_error_l2_sum": variant_sum,
                "denominator_weighted_error_norm_ratio_of_sums": ratio_of_sums,
            }
            if relationship in ("baseline", "rescue"):
                fractions = [
                    float(metric["error_norm_rescue_fraction"])
                    for metric in metrics
                ]
                component_result["error_norm_rescue_fraction"] = (
                    _finite_scalar_summary(fractions)
                )
                component_result["rows_with_positive_rescue"] = sum(
                    fraction > 0.0 for fraction in fractions
                )
                component_result["denominator_weighted_rescue_fraction"] = (
                    1.0 - ratio_of_sums
                )
            elif relationship == "recreation":
                fractions = [
                    float(metric["error_norm_recreation_fraction"])
                    for metric in metrics
                ]
                component_result["error_norm_recreation_fraction"] = (
                    _finite_scalar_summary(fractions)
                )
                component_result["rows_recreating_at_least_half_baseline"] = sum(
                    fraction >= 0.5 for fraction in fractions
                )
                component_result["rows_recreating_at_least_full_baseline"] = sum(
                    fraction >= 1.0 for fraction in fractions
                )
                component_result["denominator_weighted_recreation_fraction"] = (
                    ratio_of_sums
                )
            else:
                raise RuntimeError("pre-tail band relationship is invalid")
            components[component] = component_result
        first = variant_rows[0]
        aggregate_variants[variant_name] = {
            "band_symbol": first["band_symbol"],
            "band_role": first["band_role"],
            "relationship_to_all_old_baseline": relationship,
            "key_tensor_policy": first["key_tensor_policy"],
            "value_tensor_policy": first["value_tensor_policy"],
            "key_logical_pages": first["key_logical_pages"],
            "value_logical_pages": first["value_logical_pages"],
            "components": components,
        }

    contrast_specs = {
        "B16_rescue_minus_equal_byte_C16": (
            "b16_t768_equivalent_restored_exact",
            "c16_equal_byte_older_control_restored_exact",
            "denominator_weighted_rescue_fraction",
            "error_norm_rescue_fraction",
        ),
        "B32_rescue_minus_equal_byte_C32": (
            "b32_t1024_equivalent_restored_exact",
            "c32_equal_byte_older_control_restored_exact",
            "denominator_weighted_rescue_fraction",
            "error_norm_rescue_fraction",
        ),
        "B32_rescue_minus_nested_B16": (
            "b32_t1024_equivalent_restored_exact",
            "b16_t768_equivalent_restored_exact",
            "denominator_weighted_rescue_fraction",
            "error_norm_rescue_fraction",
        ),
        "B16_recreation_minus_equal_byte_C16": (
            "b16_t768_equivalent_only_quantized",
            "c16_equal_byte_older_control_only_quantized",
            "denominator_weighted_recreation_fraction",
            "error_norm_recreation_fraction",
        ),
        "B32_recreation_minus_equal_byte_C32": (
            "b32_t1024_equivalent_only_quantized",
            "c32_equal_byte_older_control_only_quantized",
            "denominator_weighted_recreation_fraction",
            "error_norm_recreation_fraction",
        ),
        "B32_recreation_minus_nested_B16": (
            "b32_t1024_equivalent_only_quantized",
            "b16_t768_equivalent_only_quantized",
            "denominator_weighted_recreation_fraction",
            "error_norm_recreation_fraction",
        ),
    }
    paired_contrasts: dict[str, Any] = {}
    for contrast_name, (
        left_name,
        right_name,
        weighted_field,
        row_field,
    ) in contrast_specs.items():
        paired_contrasts[contrast_name] = {}
        for component in component_names:
            row_differences = [
                float(
                    row["variants"][left_name]["projected_attention_output"][
                        component
                    ][row_field]
                )
                - float(
                    row["variants"][right_name]["projected_attention_output"][
                        component
                    ][row_field]
                )
                for row in rows
            ]
            paired_contrasts[contrast_name][component] = {
                "per_layer_fraction_difference": _finite_scalar_summary(
                    row_differences
                ),
                "denominator_weighted_fraction_difference": (
                    float(
                        aggregate_variants[left_name]["components"][component][
                            weighted_field
                        ]
                    )
                    - float(
                        aggregate_variants[right_name]["components"][component][
                            weighted_field
                        ]
                    )
                ),
            }

    storage = rows[0]["nominal_exact_fp16_storage_bytes"]
    if any(row["nominal_exact_fp16_storage_bytes"] != storage for row in rows):
        raise RuntimeError("pre-tail band storage identity differs across layers")
    bytes_by_set = storage["bytes_by_page_set_per_layer_request"]
    if bytes_by_set["B16"] != bytes_by_set["C16"] or bytes_by_set[
        "B32"
    ] != bytes_by_set["C32"]:
        raise RuntimeError("pre-tail equal-byte control gate failed")
    return {
        "enabled": True,
        "target_count": 1,
        "layers_per_target": expected_layer_count,
        "target_layer_rows": len(rows),
        "frozen_s3_t512_target798_request3_gate_passed": True,
        "ordered_complete_layer_ids_0_through_31": True,
        "all_page_sets_validated_inside_source_old_segment": True,
        "component_matched_denominators": True,
        "ratio_of_sums_aggregate": True,
        "aggregate_weighting": {
            "per_layer_fraction_summaries": "equal layer weight",
            "primary_ratio_of_sums": (
                "sum variant error L2 / sum matching component baseline error L2"
            ),
        },
        "predeclared_bands": expected_definition,
        "nominal_exact_fp16_storage_bytes": {
            **storage,
            "layers": expected_layer_count,
            "bytes_by_page_set_all_layers_one_request": {
                name: int(value) * expected_layer_count
                for name, value in bytes_by_set.items()
            },
        },
        "variants": aggregate_variants,
        "paired_contrasts": paired_contrasts,
        "interpretation_guard": (
            "local layerwise representation counterfactual on frozen TRAIN row "
            "798:3 only; T768/T1024 labels denote contiguous exact-tail-equivalent "
            "page coverage, not measured full-decoder policies"
        ),
    }


def source_reproduction(
    source: dict[str, Any],
    hf_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    teacher_inputs: torch.Tensor,
    tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> dict[str, Any]:
    observed = SUSTAINED.compare_logits(hf_logits, candidate_logits)
    source_endpoint = _quality_endpoint(source)
    expected = source_endpoint["logits"]
    metric_deltas = {
        name: abs(float(observed[name]) - float(expected[name]))
        for name in (
            "maximum_absolute_error",
            "minimum_cosine",
            "mean_cosine",
            "top1_agreement_fraction",
        )
    }
    pairing = source.get("pairing", {}).get("configuration", {})
    source_hashes = source.get("correctness", {}).get("hashes", {})
    configuration = source["configuration"]
    hashes = {
        "token_matrix_sha256": SUSTAINED.sha256_tensors([tokens]),
        "teacher_inputs_sha256": SUSTAINED.sha256_tensors([teacher_inputs]),
        "hf_logits_sha256": SUSTAINED.sha256_tensors([hf_logits]),
        "candidate_generated_tokens_sha256": SUSTAINED.sha256_tensors(
            [candidate_tokens]
        ),
    }
    expected_hashes = {
        "token_matrix_sha256": pairing.get("token_matrix_sha256"),
        "teacher_inputs_sha256": pairing.get("teacher_inputs_sha256"),
        "hf_logits_sha256": source_hashes.get("hf_logits_sha256"),
        "candidate_generated_tokens_sha256": source_hashes.get(
            "graph_generated_tokens_sha256"
        ),
    }
    hash_gates = {
        name: expected_hashes[name] is not None
        and hashes[name] == expected_hashes[name]
        for name in hashes
    }
    cache_identity_fields = [
        "exact_tail_tokens",
        "exact_sink_pages",
        "tail_attention",
        "old_value_scale_placement",
    ]
    if "exact_prefix_pages" in configuration or "exact_prefix_pages" in pairing:
        cache_identity_fields.append("exact_prefix_pages")
    source_pairing_configuration_gates = {
        name: name in configuration
        and name in pairing
        and configuration[name] == pairing[name]
        for name in cache_identity_fields
    }
    cache_policy = source_exact_cache_policy(configuration)
    attention = source.get("attention_implementation", {})
    source_attention_identity_gates = {
        "exact_tail_pages": attention.get("exact_tail_pages")
        == cache_policy["exact_tail_pages"],
        "exact_sink_pages": attention.get("exact_sink_pages")
        == cache_policy["exact_sink_pages"],
        "old_value_scale_placement": attention.get(
            "old_int8_value_scale_placement"
        )
        == configuration.get("old_value_scale_placement"),
    }
    if (
        "exact_prefix_pages" in configuration
        or "exact_prefix_pages" in attention
    ):
        source_attention_identity_gates["exact_prefix_pages"] = (
            attention.get("exact_prefix_pages")
            == cache_policy["exact_prefix_pages"]
        )
        source_attention_identity_gates["fixed_prefix_physical_layout"] = bool(
            attention.get("exact_physical_layout")
            == (
                "fixed slots 0..S-1 followed by modulo tail-ring slots "
                "S..S+T-1"
            )
        )
    same_backend = source.get("correctness", {}).get(
        "same_backend_eager_vs_graph", {}
    )
    same_backend_gate = bool(
        same_backend.get("passed")
        and same_backend.get("eager_gate_passed")
        and same_backend.get("logits", {}).get("bitwise_identical")
        and same_backend.get("generated_argmax_tokens", {}).get(
            "bitwise_identical"
        )
    )
    metric_gate = all(delta <= 1.0e-7 for delta in metric_deltas.values())
    candidate_logits_source_hash = source_hashes.get("candidate_logits_sha256")
    candidate_logits_hash_gate = None
    observed_candidate_logits_sha256 = None
    if candidate_logits_source_hash is not None:
        observed_candidate_logits_sha256 = SUSTAINED.sha256_tensors(
            [candidate_logits]
        )
        candidate_logits_hash_gate = (
            observed_candidate_logits_sha256 == candidate_logits_source_hash
        )
    return {
        "passed": (
            metric_gate
            and all(hash_gates.values())
            and all(source_pairing_configuration_gates.values())
            and all(source_attention_identity_gates.values())
            and same_backend_gate
            and candidate_logits_hash_gate is not False
        ),
        "metric_tolerance": 1.0e-7,
        "metric_gate_passed": metric_gate,
        "observed_logits": observed,
        "source_logits": expected,
        "metric_absolute_deltas": metric_deltas,
        "observed_hashes": hashes,
        "source_hashes": expected_hashes,
        "hash_gates": hash_gates,
        "all_available_required_hashes_gate_passed": all(hash_gates.values()),
        "source_pairing_configuration_gates": (
            source_pairing_configuration_gates
        ),
        "source_attention_identity_gates": source_attention_identity_gates,
        "same_backend_eager_graph_equivalence_relied_upon": same_backend_gate,
        "candidate_logits_source_hash_available": (
            candidate_logits_source_hash is not None
        ),
        "candidate_logits_source_sha256": candidate_logits_source_hash,
        "observed_candidate_logits_sha256": observed_candidate_logits_sha256,
        "candidate_logits_hash_gate": candidate_logits_hash_gate,
        "candidate_logits_reproduction_claim": (
            "bitwise candidate-logits hash reproduced"
            if candidate_logits_source_hash is not None
            else "no source candidate-logits hash; reproduction is limited to "
            "the 1e-7 scalar-metric tolerance, all available required tensor "
            "hashes, exact cache identity, and the source's attested bitwise "
            "same-backend eager/graph equivalence"
        ),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.maximum_targets <= 0 or args.page_top_k <= 0:
        raise SystemExit("target and page top-k limits must be positive")
    if not -1.0 <= args.hidden_cosine_trigger <= 1.0:
        raise SystemExit("hidden cosine trigger lies outside [-1,1]")
    if args.hidden_relative_l2_trigger < 0:
        raise SystemExit("hidden relative-L2 trigger must be non-negative")
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    if source.get("backend") != "page_gauge":
        raise SystemExit("source result must be a PageGauge backend run")
    configuration = source["configuration"]
    if configuration.get("trajectory_mode") != SUSTAINED.FROZEN_HF_TEACHER:
        raise SystemExit("layer attribution requires a frozen-HF teacher trajectory")
    targets = select_trace_targets(source, args.target, args.maximum_targets)
    exact_cache_policy = source_exact_cache_policy(configuration)
    pre_tail_band_source_gate = pre_tail_band_ablation_source_gate(
        source, targets
    )

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    seed = int(configuration["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from transformers import AutoModelForCausalLM

    model_name = str(configuration["model"])
    print("Loading the exact unpacked HF SDPA model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    layers, hq, hkv, hidden = PG.BASE_E2E.check_model(model)
    if (layers, hq, hkv, PG.DIM) != (32, 32, 8, 128):
        raise RuntimeError("layer trace requires Mistral-7B 32/32/8/128 geometry")
    token_args = SimpleNamespace(
        model=model_name,
        context=int(configuration["context"]),
        decode_steps=int(configuration["decode_steps"]),
        batch_size=int(configuration["batch_size"]),
        token_source=str(configuration["token_source"]),
        seed=seed,
        wikitext_zip=args.wikitext_zip,
        wikitext_member=str(configuration["wikitext_member"]),
        token_offset=int(configuration["token_offset"]),
        token_stride=int(configuration["token_stride"]),
    )
    tokens, token_provenance = PREFILL.build_token_matrix(
        token_args, int(model.config.vocab_size)
    )
    model = model.cuda()
    context = int(configuration["context"])
    decode_steps = int(configuration["decode_steps"])
    exact_tail = exact_cache_policy["exact_tail_tokens"]
    exact_sink_pages = exact_cache_policy["exact_sink_pages"]
    max_context = context + decode_steps
    served_pages = math.ceil(max_context / PG.PAGE)
    pages = served_pages + 1
    initial_pages = context // PG.PAGE
    exact_pages = exact_tail // PG.PAGE
    cache = BACKEND.allocate_backend_cache(
        "page_gauge",
        layers,
        pages,
        initial_pages,
        exact_pages,
        int(configuration["batch_size"]),
        hkv,
        exact_sink_pages=exact_sink_pages,
    )
    positions = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
    rope_probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
    rope_cos, rope_sin = model.model.rotary_emb(rope_probe, positions)
    if rope_cos.dim() == 3:
        rope_cos, rope_sin = rope_cos[0], rope_sin[0]
    rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
    rope_sin = rope_sin.to(dtype=torch.float16).contiguous()
    del positions, rope_probe

    print(
        f"Replaying the coherent HF fixture and tracing {len(targets)} rows...",
        flush=True,
    )
    with HFTargetInterceptor(
        model=model,
        cache=cache,
        targets=targets,
        context=context,
        exact_tail_tokens=exact_tail,
        exact_sink_pages=exact_sink_pages,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        page_top_k=args.page_top_k,
        pre_tail_band_ablation_enabled=bool(
            pre_tail_band_source_gate["enabled"]
        ),
    ) as hf_interceptor:
        hf_logits_list, hf_generated, prefill_records, kv_sample_sha256 = (
            BACKEND.build_coherent_fixture(
                model=model,
                tokens=tokens,
                cache=cache,
                backend="page_gauge",
                layers=layers,
                hkv=hkv,
                pages=pages,
                initial_pages=initial_pages,
                exact_pages=exact_pages,
                context=context,
                decode_steps=decode_steps,
                chunk_tokens=int(configuration["prefill_chunk_tokens"]),
                exact_sink_pages=exact_sink_pages,
            )
        )
    prefill_population_gate = prefill_population_protocol_gate(prefill_records)
    if not prefill_population_gate["all_sampled_population_gates_passed"]:
        raise RuntimeError(
            "sampled prefill cache population gate failed for record indices "
            f"{prefill_population_gate['failed_record_indices']}"
        )
    if set(hf_interceptor.traces) != {
        (target["step"], target["request"]) for target in targets
    }:
        raise RuntimeError("not every requested HF row was traced")
    hf_logits = torch.stack(hf_logits_list)
    initial_token = tokens[:, context].to(device="cuda", dtype=torch.long)
    teacher_inputs = SUSTAINED.build_teacher_inputs(initial_token, hf_generated)

    print("Packing the shared model and replaying PageGauge eagerly...", flush=True)
    PG.BASE_E2E.pack_model_projections(model)
    gc.collect()
    torch.cuda.empty_cache()
    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    decoder = PG.TransformerDecoder(
        model,
        flashinfer,
        append_extension,
        "page_gauge",
        cache,
        max_context + PG.PAGE,
        exact_tail,
        int(configuration["baseline_split_pages"]),
        int(configuration["candidate_split_pages"]),
        rope_cos,
        rope_sin,
        "attention_add",
        str(configuration["tail_attention"]),
        int(configuration["batch_size"]),
        device_dynamic_decoder_layer_graphs=False,
        old_value_scale_placement=str(
            configuration.get("old_value_scale_placement", "probability")
        ),
        exact_sink_pages=exact_sink_pages,
    )
    decoder.plan(context)
    candidate_logits, candidate_tokens, candidate_traces = trace_candidate_teacher(
        decoder,
        teacher_inputs=teacher_inputs,
        context=context,
        targets=targets,
    )
    torch.cuda.synchronize()
    merged = merge_target_traces(
        targets=targets,
        hf_traces=hf_interceptor.traces,
        candidate_traces=candidate_traces,
        hf_logits=hf_logits,
        candidate_logits=candidate_logits,
        hidden_cosine_trigger=args.hidden_cosine_trigger,
        hidden_relative_l2_trigger=args.hidden_relative_l2_trigger,
    )
    ablation_applicability = page_ablation_applicability(
        merged[0]["page_mapping"]
    )
    if any(
        page_ablation_applicability(target["page_mapping"])
        != ablation_applicability
        for target in merged
    ):
        raise RuntimeError("page-ablation applicability differs across targets")
    page_ablation_aggregate = (
        aggregate_page_ablation(
            merged,
            expected_layer_count=layers,
            batch_size=int(configuration["batch_size"]),
            require_frozen_source_grid=not bool(args.target),
        )
        if ablation_applicability["enabled"]
        else {
            **ablation_applicability,
            "target_count": len(merged),
            "layers_per_target": layers,
            "generic_layer_kv_and_page_error_attribution_executed": True,
        }
    )
    pre_tail_band_ablation_aggregate = (
        aggregate_pre_tail_band_ablation(
            merged,
            expected_layer_count=layers,
        )
        if pre_tail_band_source_gate["enabled"]
        else {
            **pre_tail_band_source_gate,
            "target_count": len(merged),
            "layers_per_target": layers,
        }
    )
    reproduction = source_reproduction(
        source,
        hf_logits,
        candidate_logits,
        teacher_inputs,
        tokens,
        candidate_tokens,
    )
    manual_reference_finite = all(
        math.isfinite(
            float(
                layer["kv_counterfactual"][
                    "manual_reference_projection_vs_hf_sdpa_attention_delta"
                ]["cosine"]
            )
        )
        for target in merged
        for layer in target["layers"]
    )
    result = {
        "schema_version": 4,
        "experiment": "page_gauge_strict_quality_layer_kv_attribution",
        "protocol_passed": diagnostic_protocol_passed(
            source_reproduction_passed=bool(reproduction["passed"]),
            manual_reference_finite=manual_reference_finite,
            sampled_prefill_population_gate_passed=bool(
                prefill_population_gate[
                    "all_sampled_population_gates_passed"
                ]
            ),
        ),
        "strict_quality_gate_passed": bool(_quality_endpoint(source)["passed"]),
        "claim_scope": (
            "diagnostic-only exact-row localization; no timing or revised quality claim"
        ),
        "configuration": {
            "source_result": str(args.source_result.resolve()),
            "source_result_sha256": source_sha256(args.source_result),
            "output": str(args.output),
            "maximum_targets": args.maximum_targets,
            "page_top_k": args.page_top_k,
            "hidden_cosine_trigger": args.hidden_cosine_trigger,
            "hidden_relative_l2_trigger": args.hidden_relative_l2_trigger,
            "source_configuration": configuration,
            "reproduced_exact_cache_policy": exact_cache_policy,
            "pre_tail_band_ablation_source_gate": pre_tail_band_source_gate,
        },
        "protocol": {
            "production_math_modified": False,
            "production_files_modified": False,
            "strict_threshold_unchanged": _quality_endpoint(source)[
                "minimum_logits_cosine"
            ],
            "target_selection": (
                "all retained source rows below the unchanged strict gate"
                if not args.target
                else "explicit STEP:REQUEST rows"
            ),
            "hf_execution": (
                "existing request-serialized coherent DynamicCache fixture with "
                "hooks enabled only at selected decode rows"
            ),
            "candidate_execution": (
                "existing PageGauge cache/decoder, frozen HF tokens, eager layers; "
                "same-backend eager/graph bitwise equivalence is re-attested from source"
            ),
            "kv_attribution": (
                "same centered PageGauge old-page scalar INT8 quantizer; old K and V "
                "varied independently; exact FP16 prefix+tail logical pages and query "
                "held fixed"
            ),
            "page2_necessity_sufficiency": (
                "diagnostic-only component-matched projected-output ablation; "
                "page2 semantic anchor, contiguous pages1-2 S3 candidate, and "
                "equal-byte pages3-4 control all validated within the source old "
                "segment; S3 requires cross-request TRAIN confirmation"
                if ablation_applicability["enabled"]
                else ablation_applicability["reason"]
            ),
            "nested_pre_tail_band_necessity_sufficiency": (
                "diagnostic-only component-matched K-only/V-only/both rescue and "
                "recreation for B16=[E-16,E), B32=[E-32,E), and equal-byte older "
                "controls C16=[E-64,E-48), C32=[E-64,E-32); T768/T1024 are "
                "local contiguous-tail-equivalent labels, not measured policies"
                if pre_tail_band_source_gate["enabled"]
                else pre_tail_band_source_gate["reason"]
            ),
            "timing_valid": False,
            "diagnostic_triggers_are_acceptance_gates": False,
        },
        "protocol_gates": {
            "source_reproduction_passed": bool(reproduction["passed"]),
            "manual_reference_finite": manual_reference_finite,
            "all_sampled_prefill_population_gates_passed": bool(
                prefill_population_gate[
                    "all_sampled_population_gates_passed"
                ]
            ),
            "pre_tail_band_ablation_source_gate_passed": bool(
                pre_tail_band_source_gate["gate_passed"]
            ),
        },
        "source_reproduction": reproduction,
        "page_ablation_aggregate": page_ablation_aggregate,
        "pre_tail_band_ablation_aggregate": (
            pre_tail_band_ablation_aggregate
        ),
        "targets": merged,
        "prefill_population": {
            **prefill_population_gate,
            "kv_sample_sha256": kv_sample_sha256,
            "records": prefill_records,
        },
        "token_source": token_provenance,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "flashinfer": str(flashinfer.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): source_sha256(path)
            for path in (
                Path(__file__),
                SUSTAINED_PATH,
                SUSTAINED.BACKEND_WORKER_PATH,
                ROOT / "scripts/benchmark_page_gauge_transformer.py",
                ROOT / "scripts/benchmark_page_gauge_overheads.py",
                ROOT / "tests/page_gauge_append_extension.cu",
            )
        },
    }
    return result


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for target in result["targets"]:
            first = target["first_diagnostic_divergence"]
            print(
                f"step={target['step']} request={target['request']} "
                f"logit_cos={target['logits']['cosine']:.9f} "
                f"first={first['layer']}:{first['stage']}",
                flush=True,
            )
        print(f"wrote {args.output}", flush=True)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 4,
            "experiment": "page_gauge_strict_quality_layer_kv_attribution",
            "protocol_passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
