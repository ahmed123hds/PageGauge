#!/usr/bin/env python3
"""Backend-exclusive sustained decode with reusable device-dynamic layer graphs.

The publication boundary is a continuous, fixed-batch 512-token decode. Model
loading, coherent HF prefill/oracle generation, exhaustive planner preflight,
graph-bank capture, cache restore, cache scrub, and hot preconditioning are
outside timing. Every timed token includes the supported FlashInfer planner
metadata update, device-position fill, embedding, 32 complete-layer graph
replays, final norm, LM head, and GPU argmax. No eager fallback is permitted.
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
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BACKEND_WORKER_PATH = ROOT / "diagnostics/benchmark_backend_exclusive.py"
GREEDY_FEEDBACK = "greedy_feedback"
FROZEN_HF_TEACHER = "frozen_hf_teacher_forced"
DYNAMIC_GRAPH_SCOPE = "decoder_layer_device_dynamic"


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BACKEND = load_local_module(
    "page_gauge_backend_exclusive_for_sustained_graphs", BACKEND_WORKER_PATH
)
PG = BACKEND.PG
PREFILL = BACKEND.PREFILL
TOKEN = BACKEND.TOKEN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("flashinfer_fp16", "page_gauge"), required=True
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=20480)
    parser.add_argument("--decode-steps", type=int, default=512)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument(
        "--exact-sink-pages",
        type=int,
        default=0,
        help=(
            "Keep a contiguous exact FP16 prefix of this many complete pages "
            "in fixed slots inside the existing exact wrapper. Nonnegative; "
            "zero keeps the original tail-only layout."
        ),
    )
    parser.add_argument(
        "--exact-static-suffix-pages",
        type=int,
        default=0,
        help="Fixed exact FP16 suffix pages from the original prefill cache.",
    )
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=256)
    parser.add_argument(
        "--tail-attention",
        choices=("flashinfer_merge", "fused_kernel", "heterogeneous_fa2"),
        default="flashinfer_merge",
    )
    parser.add_argument(
        "--old-value-scale-placement",
        choices=("probability", "value_fragment"),
        default="probability",
        help=(
            "Algebraically equivalent placement of the old-page V scale. "
            "value_fragment applies the scale after INT8-to-FP16 V conversion "
            "instead of multiplying diffuse FP16 softmax probabilities."
        ),
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=(GREEDY_FEEDBACK, FROZEN_HF_TEACHER),
        default=GREEDY_FEEDBACK,
    )
    parser.add_argument("--seed", type=int, default=20260861)
    parser.add_argument(
        "--token-source", choices=("wikitext2", "random"), default="wikitext2"
    )
    parser.add_argument(
        "--wikitext-zip",
        type=Path,
        default=ROOT / "data/wikitext-2-raw-v1.zip",
    )
    parser.add_argument(
        "--wikitext-member", default="wikitext-2-raw/wiki.train.raw"
    )
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--token-stride", type=int, default=0)
    parser.add_argument("--capture-warmups", type=int, default=1)
    parser.add_argument("--maximum-graph-banks", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument(
        "--quality-diagnostics-top-k",
        type=int,
        default=0,
        help=(
            "Record a posthoc, non-publication outlier report for the K lowest "
            "HF-vs-backend logit-cosine rows. Zero disables the report. This "
            "does not alter the strict quality thresholds or timed kernels."
        ),
    )
    parser.add_argument(
        "--quality-diagnostics-top-vocab",
        type=int,
        default=8,
        help="Vocabulary entries retained for each diagnosed outlier row.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.context <= args.exact_tail or args.context % PG.PAGE:
        raise SystemExit("context must be page aligned and exceed the exact tail")
    if args.exact_tail <= 0 or args.exact_tail % PG.PAGE:
        raise SystemExit("exact tail must contain whole pages")
    try:
        exact_prefix_pages = PG.validate_exact_prefix_attention_path(
            args.exact_sink_pages, args.tail_attention
        )
        exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
            args.exact_static_suffix_pages,
            exact_prefix_pages=exact_prefix_pages,
            initial_context_pages=args.context // PG.PAGE,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.backend != "page_gauge" and (
        exact_prefix_pages or exact_static_suffix_pages
    ):
        raise SystemExit("fixed exact pages are supported only by PageGauge")
    if exact_static_suffix_pages and args.tail_attention != "flashinfer_merge":
        raise SystemExit("an exact static suffix requires flashinfer_merge")
    if (
        args.context // PG.PAGE
        <= args.exact_tail // PG.PAGE
        + exact_prefix_pages
        + exact_static_suffix_pages
    ):
        raise SystemExit("context must contain at least one quantized old page")
    if args.decode_steps < 512 or args.decode_steps % PG.PAGE:
        raise SystemExit("sustained protocol requires >=512 page-aligned decode steps")
    if args.prefill_chunk_tokens <= 0:
        raise SystemExit("prefill chunk size must be positive")
    if args.capture_warmups <= 0 or args.maximum_graph_banks <= 0:
        raise SystemExit("capture warmups and graph-bank limit must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("warmups must be non-negative and repeats must be positive")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    if args.baseline_split_pages < 0 or args.candidate_split_pages < 0:
        raise SystemExit("fixed split page counts cannot be negative")
    if not -1.0 <= args.min_logits_cosine <= 1.0:
        raise SystemExit("logits cosine threshold lies outside [-1,1]")
    if not 0.0 <= args.min_top1_agreement <= 1.0:
        raise SystemExit("top-1 threshold lies outside [0,1]")
    if args.quality_diagnostics_top_k < 0:
        raise SystemExit("quality diagnostic top-k cannot be negative")
    if args.quality_diagnostics_top_vocab <= 0:
        raise SystemExit("quality diagnostic vocabulary width must be positive")


def sha256_tensors(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def cache_boundary_digest(decoder: Any, logical_page: int) -> dict[str, Any]:
    full_pages = torch.tensor(
        [
            request * decoder.max_pages + logical_page
            for request in range(decoder.batch_size)
        ],
        device="cuda",
        dtype=torch.long,
    )
    if decoder.backend == "flashinfer_fp16":
        tensors = [
            decoder.cache.key.index_select(1, full_pages),
            decoder.cache.value.index_select(1, full_pages),
        ]
        names = ["key", "value"]
        ring_page = None
    else:
        ring_pages = torch.tensor(
            [
                decoder.exact_physical_page(request, logical_page)
                for request in range(decoder.batch_size)
            ],
            device="cuda",
            dtype=torch.long,
        )
        tensors = [
            decoder.cache.exact_key.index_select(1, ring_pages),
            decoder.cache.exact_value.index_select(1, ring_pages),
            decoder.cache.key_codes.index_select(1, full_pages),
            decoder.cache.value_codes.index_select(1, full_pages),
            decoder.cache.key_scales.index_select(1, full_pages),
            decoder.cache.value_scales.index_select(1, full_pages),
        ]
        names = [
            "exact_key",
            "exact_value",
            "key_codes",
            "value_codes",
            "key_scales",
            "value_scales",
        ]
        ring_page = decoder.exact_physical_page(0, logical_page)
    torch.cuda.synchronize()
    records = {name: sha256_tensors([tensor]) for name, tensor in zip(names, tensors)}
    return {
        "logical_page": logical_page,
        "exact_ring_page": ring_page,
        "tensor_sha256": records,
        "combined_sha256": sha256_tensors(tensors),
    }


def build_teacher_inputs(
    initial_token: torch.Tensor, hf_generated: list[torch.Tensor]
) -> torch.Tensor:
    if not hf_generated:
        raise ValueError("HF generated-token oracle must be non-empty")
    rows = [initial_token.detach().cpu()]
    rows.extend(token.detach().cpu() for token in hf_generated[:-1])
    return torch.stack(rows, dim=0).to(device="cuda", dtype=torch.long)


@torch.inference_mode()
def execute_sustained(
    decoder: Any,
    *,
    initial_token: torch.Tensor,
    teacher_inputs: torch.Tensor,
    trajectory_mode: str,
    start_position: int,
    decode_steps: int,
    dynamic_token: torch.Tensor,
    argmax_token: torch.Tensor,
    collect: bool,
) -> dict[str, Any]:
    if tuple(initial_token.shape) != (decoder.batch_size,):
        raise ValueError("initial token must have shape [B]")
    if tuple(teacher_inputs.shape) != (decode_steps, decoder.batch_size):
        raise ValueError("teacher inputs must have shape [decode_steps,B]")
    if trajectory_mode not in (GREEDY_FEEDBACK, FROZEN_HF_TEACHER):
        raise ValueError(f"unknown trajectory mode {trajectory_mode}")

    if trajectory_mode == GREEDY_FEEDBACK:
        dynamic_token.copy_(initial_token)
    output_blocks: list[torch.Tensor] = []
    token_blocks: list[torch.Tensor] = []
    pending_logits: list[torch.Tensor] = []
    pending_tokens: list[torch.Tensor] = []
    boundaries: list[dict[str, Any]] = []
    for offset in range(decode_steps):
        token = (
            dynamic_token
            if trajectory_mode == GREEDY_FEEDBACK
            else teacher_inputs[offset]
        )
        logits = decoder.step(token, start_position + offset)[0]
        destination = dynamic_token if trajectory_mode == GREEDY_FEEDBACK else argmax_token
        torch.argmax(logits, dim=-1, out=destination)
        if collect:
            pending_logits.append(logits.detach().clone())
            pending_tokens.append(destination.detach().clone())
        if (offset + 1) % PG.PAGE == 0:
            if collect:
                output_blocks.append(torch.stack(pending_logits).cpu())
                token_blocks.append(torch.stack(pending_tokens).cpu())
                pending_logits.clear()
                pending_tokens.clear()
                boundaries.append(
                    cache_boundary_digest(
                        decoder, (start_position + offset) // PG.PAGE
                    )
                )
    if pending_logits or pending_tokens:
        raise RuntimeError("sustained collection ended with a partial page")
    return {
        "logits": torch.cat(output_blocks, dim=0) if collect else None,
        "generated_tokens": torch.cat(token_blocks, dim=0) if collect else None,
        "page_boundaries": boundaries,
    }


def compare_logits(expected: torch.Tensor, observed: torch.Tensor) -> dict[str, Any]:
    if expected.shape != observed.shape or expected.ndim != 3:
        raise ValueError("logit tensors must be matching [steps,B,V]")
    difference = (expected.float() - observed.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        expected.float(), observed.float(), dim=-1
    )
    top1 = expected.argmax(dim=-1).eq(observed.argmax(dim=-1))
    boundary_offsets = list(range(PG.PAGE - 1, int(expected.shape[0]), PG.PAGE))
    return {
        "checked_steps": int(expected.shape[0]),
        "checked_request_steps": int(top1.numel()),
        "bitwise_identical": bool(torch.equal(expected, observed)),
        "maximum_absolute_error": float(difference.max().item()),
        "minimum_cosine": float(cosine.min().item()),
        "mean_cosine": float(cosine.mean().item()),
        "top1_agreement_fraction": float(top1.float().mean().item()),
        "page_boundary_maximum_absolute_error": {
            str(offset): float(difference[offset].max().item())
            for offset in boundary_offsets
        },
    }


def _quality_quantiles(values: torch.Tensor) -> dict[str, float]:
    """Return fixed quantiles for a finite one-dimensional diagnostic tensor."""

    flat = values.detach().float().reshape(-1).cpu()
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("quality diagnostics require finite non-empty values")
    probabilities = (
        ("minimum", 0.0),
        ("p0_1", 0.001),
        ("p0_5", 0.005),
        ("p1", 0.01),
        ("p5", 0.05),
        ("median", 0.50),
        ("p95", 0.95),
        ("p99", 0.99),
        ("maximum", 1.0),
    )
    return {
        name: float(torch.quantile(flat, probability).item())
        for name, probability in probabilities
    }


def _consecutive_failure_clusters(
    below_gate: torch.Tensor,
) -> list[dict[str, int]]:
    """Describe consecutive below-threshold step runs independently per request."""

    if below_gate.ndim != 2 or below_gate.dtype != torch.bool:
        raise ValueError("failure mask must be boolean [steps,batch]")
    steps, batch_size = below_gate.shape
    clusters: list[dict[str, int]] = []
    mask = below_gate.cpu()
    for request in range(batch_size):
        start: int | None = None
        for step in range(steps + 1):
            active = step < steps and bool(mask[step, request])
            if active and start is None:
                start = step
            elif not active and start is not None:
                clusters.append(
                    {
                        "request": request,
                        "start_step": start,
                        "end_step_inclusive": step - 1,
                        "length": step - start,
                    }
                )
                start = None
    return clusters


def generated_int8_pages_visible_count(
    generated_tokens_including_current: int, exact_tail_pages: int
) -> int:
    """Count generated pages that have aged beyond a page-granular exact tail."""

    if generated_tokens_including_current <= 0 or exact_tail_pages <= 0:
        raise ValueError("generated-token and exact-tail page counts must be positive")
    generated_pages = (
        generated_tokens_including_current + PG.PAGE - 1
    ) // PG.PAGE
    return max(0, generated_pages - exact_tail_pages)


def outlier_page_visibility_counts(
    *,
    start_position: int,
    generated_tokens_including_current: int,
    exact_tail_pages: int,
    exact_prefix_pages: int,
) -> dict[str, int]:
    """Return the planner-exact old/exact page decomposition for one row.

    As decoding advances, pages that belonged to the initial exact tail move
    into the old INT8 segment before generated pages themselves age out of the
    tail.  Counting only the initial old pages plus generated-old pages omits
    those displaced initial-tail pages, so derive every count from the active
    page-table boundary instead.
    """

    if start_position <= 0 or start_position % PG.PAGE:
        raise ValueError("outlier page visibility requires a page-aligned prefix")
    if generated_tokens_including_current <= 0:
        raise ValueError("outlier page visibility requires a generated token")
    if exact_tail_pages <= 0:
        raise ValueError("outlier page visibility requires an exact tail")
    exact_prefix_pages = PG.validate_exact_prefix_pages(exact_prefix_pages)

    initial_pages = start_position // PG.PAGE
    generated_pages = (
        generated_tokens_including_current + PG.PAGE - 1
    ) // PG.PAGE
    total_pages = initial_pages + generated_pages
    tail_logical_begin = total_pages - exact_tail_pages
    if tail_logical_begin <= exact_prefix_pages:
        raise ValueError("outlier row has no quantized old segment")

    initial_old_at_prefix = initial_pages - exact_tail_pages - exact_prefix_pages
    if initial_old_at_prefix <= 0:
        raise ValueError("initial prefix has no quantized old segment")
    initial_pages_currently_old = (
        min(initial_pages, tail_logical_begin) - exact_prefix_pages
    )
    generated_pages_currently_old = max(0, tail_logical_begin - initial_pages)
    total_pages_currently_old = tail_logical_begin - exact_prefix_pages
    displaced_initial_tail_pages = (
        initial_pages_currently_old - initial_old_at_prefix
    )
    if (
        initial_pages_currently_old < initial_old_at_prefix
        or displaced_initial_tail_pages < 0
        or displaced_initial_tail_pages > exact_tail_pages
        or initial_pages_currently_old + generated_pages_currently_old
        != total_pages_currently_old
    ):
        raise RuntimeError("planner-exact outlier page counts are inconsistent")

    return {
        "initial_quantized_old_pages_at_prefix": initial_old_at_prefix,
        "initial_prefix_pages_currently_old_int8": initial_pages_currently_old,
        "displaced_initial_exact_tail_pages_now_old_int8": (
            displaced_initial_tail_pages
        ),
        "generated_int8_pages_visible": generated_pages_currently_old,
        "total_quantized_pages_visible": total_pages_currently_old,
        "planner_total_pages_visible": total_pages,
        "planner_tail_logical_begin": tail_logical_begin,
    }


def diagnose_logit_outliers(
    expected: torch.Tensor,
    observed: torch.Tensor,
    *,
    input_tokens: torch.Tensor,
    start_position: int,
    exact_tail_tokens: int,
    exact_prefix_pages: int,
    minimum_cosine: float,
    top_rows: int,
    top_vocab: int,
) -> dict[str, Any]:
    """Localize strict HF-vs-backend logit-cosine failures without reranking.

    The function is intentionally posthoc.  It consumes the already-collected
    CPU logits and teacher inputs, performs no model or cache mutation, changes
    no acceptance threshold, and is excluded from publication timing claims.
    """

    if expected.shape != observed.shape or expected.ndim != 3:
        raise ValueError("diagnostic logits must be matching [steps,batch,vocab]")
    steps, batch_size, vocab_size = expected.shape
    if tuple(input_tokens.shape) != (steps, batch_size):
        raise ValueError("diagnostic input tokens must have shape [steps,batch]")
    if top_rows <= 0:
        raise ValueError("diagnostic top-row count must be positive")
    if top_vocab <= 0:
        raise ValueError("diagnostic top-vocabulary width must be positive")
    if exact_tail_tokens <= 0 or exact_tail_tokens % PG.PAGE:
        raise ValueError("diagnostic exact tail must contain complete pages")
    exact_prefix_pages = PG.validate_exact_prefix_pages(exact_prefix_pages)
    if start_position <= 0 or start_position % PG.PAGE:
        raise ValueError("diagnostic start position must be positive and page aligned")
    initial_pages = int(start_position) // PG.PAGE
    exact_tail_pages = int(exact_tail_tokens) // PG.PAGE
    prefix_quantized_pages = initial_pages - exact_tail_pages - exact_prefix_pages
    if prefix_quantized_pages <= 0:
        raise ValueError("diagnostic context must contain a quantized old segment")
    quantized_old_begin = exact_prefix_pages
    quantized_old_end = initial_pages - exact_tail_pages

    reference = expected.detach().float().cpu()
    candidate = observed.detach().float().cpu()
    inputs = input_tokens.detach().long().cpu()
    difference = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
    difference_norm = torch.linalg.vector_norm(difference, dim=-1)
    relative_l2 = difference_norm / reference_norm.clamp_min(1.0e-30)
    maximum_absolute_error = difference.abs().amax(dim=-1)
    rmse = difference.square().mean(dim=-1).sqrt()
    top1_match = reference.argmax(dim=-1).eq(candidate.argmax(dim=-1))
    below_gate = cosine < float(minimum_cosine)

    retained_vocab = min(int(top_vocab), int(vocab_size))
    reference_top_values, reference_top_ids = reference.topk(
        retained_vocab, dim=-1
    )
    candidate_top_values, candidate_top_ids = candidate.topk(
        retained_vocab, dim=-1
    )
    pair_width = min(2, int(vocab_size))
    reference_pair = reference.topk(pair_width, dim=-1)
    candidate_pair = candidate.topk(pair_width, dim=-1)
    reference_margin = (
        reference_pair.values[..., 0] - reference_pair.values[..., 1]
        if pair_width > 1
        else torch.full_like(reference_pair.values[..., 0], float("inf"))
    )
    candidate_margin = (
        candidate_pair.values[..., 0] - candidate_pair.values[..., 1]
        if pair_width > 1
        else torch.full_like(candidate_pair.values[..., 0], float("inf"))
    )

    flattened_indices = list(range(int(steps * batch_size)))
    flattened_cosine = cosine.reshape(-1)
    flattened_indices.sort(
        key=lambda index: (
            float(flattened_cosine[index].item()),
            index // batch_size,
            index % batch_size,
        )
    )
    selected_indices = flattened_indices[: min(top_rows, len(flattened_indices))]

    worst_rows: list[dict[str, Any]] = []
    for rank, flattened_index in enumerate(selected_indices, start=1):
        step = flattened_index // batch_size
        request = flattened_index % batch_size
        position = int(start_position) + step
        generated_tokens_including_current = step + 1
        page_visibility = outlier_page_visibility_counts(
            start_position=start_position,
            generated_tokens_including_current=generated_tokens_including_current,
            exact_tail_pages=exact_tail_pages,
            exact_prefix_pages=exact_prefix_pages,
        )
        reference_top1 = int(reference_pair.indices[step, request, 0].item())
        candidate_top1 = int(candidate_pair.indices[step, request, 0].item())
        candidate_at_reference_top1 = float(
            candidate[step, request, reference_top1].item()
        )
        reference_at_candidate_top1 = float(
            reference[step, request, candidate_top1].item()
        )
        reference_ids = reference_top_ids[step, request].tolist()
        candidate_ids = candidate_top_ids[step, request].tolist()
        overlap = len(set(reference_ids) & set(candidate_ids))
        worst_rows.append(
            {
                "rank": rank,
                "step": step,
                "request": request,
                "absolute_position": position,
                "input_token_id": int(inputs[step, request].item()),
                "page_offset": position % PG.PAGE,
                "is_page_close": position % PG.PAGE == PG.PAGE - 1,
                "generated_tokens_including_current": (
                    generated_tokens_including_current
                ),
                "generated_int8_pages_visible": page_visibility[
                    "generated_int8_pages_visible"
                ],
                "exact_prefix_pages": exact_prefix_pages,
                "prefix_quantized_pages": prefix_quantized_pages,
                "initial_quantized_old_pages_at_prefix": page_visibility[
                    "initial_quantized_old_pages_at_prefix"
                ],
                "initial_prefix_pages_currently_old_int8": page_visibility[
                    "initial_prefix_pages_currently_old_int8"
                ],
                "displaced_initial_exact_tail_pages_now_old_int8": (
                    page_visibility[
                        "displaced_initial_exact_tail_pages_now_old_int8"
                    ]
                ),
                "total_quantized_pages_visible": page_visibility[
                    "total_quantized_pages_visible"
                ],
                "planner_total_pages_visible": page_visibility[
                    "planner_total_pages_visible"
                ],
                "planner_tail_logical_begin": page_visibility[
                    "planner_tail_logical_begin"
                ],
                "cosine": float(cosine[step, request].item()),
                "below_strict_gate": bool(below_gate[step, request].item()),
                "relative_l2": float(relative_l2[step, request].item()),
                "rmse": float(rmse[step, request].item()),
                "maximum_absolute_error": float(
                    maximum_absolute_error[step, request].item()
                ),
                "reference_l2": float(reference_norm[step, request].item()),
                "candidate_l2": float(candidate_norm[step, request].item()),
                "reference_mean": float(reference[step, request].mean().item()),
                "candidate_mean": float(candidate[step, request].mean().item()),
                "reference_std": float(
                    reference[step, request].std(unbiased=False).item()
                ),
                "candidate_std": float(
                    candidate[step, request].std(unbiased=False).item()
                ),
                "reference_top1_id": reference_top1,
                "candidate_top1_id": candidate_top1,
                "top1_match": reference_top1 == candidate_top1,
                "reference_top1_margin": float(
                    reference_margin[step, request].item()
                ),
                "candidate_top1_margin": float(
                    candidate_margin[step, request].item()
                ),
                "candidate_logit_at_reference_top1": (
                    candidate_at_reference_top1
                ),
                "reference_logit_at_candidate_top1": (
                    reference_at_candidate_top1
                ),
                "candidate_reference_top1_advantage": float(
                    (
                        candidate[step, request, reference_top1]
                        - candidate[step, request, candidate_top1]
                    ).item()
                ),
                "reference_candidate_top1_advantage": float(
                    (
                        reference[step, request, candidate_top1]
                        - reference[step, request, reference_top1]
                    ).item()
                ),
                "top_vocab_overlap_count": overlap,
                "top_vocab_overlap_fraction": overlap / retained_vocab,
                "reference_top_ids": reference_ids,
                "reference_top_values": reference_top_values[
                    step, request
                ].tolist(),
                "candidate_top_ids": candidate_ids,
                "candidate_top_values": candidate_top_values[
                    step, request
                ].tolist(),
            }
        )

    per_request = []
    for request in range(batch_size):
        request_cosine = cosine[:, request]
        minimum_step = int(request_cosine.argmin().item())
        request_below = below_gate[:, request]
        request_top1 = top1_match[:, request]
        per_request.append(
            {
                "request": request,
                "minimum_cosine": float(request_cosine[minimum_step].item()),
                "minimum_step": minimum_step,
                "minimum_absolute_position": int(start_position) + minimum_step,
                "mean_cosine": float(request_cosine.mean().item()),
                "rows_below_gate": int(request_below.sum().item()),
                "top1_mismatches": int((~request_top1).sum().item()),
            }
        )

    per_page_offset = []
    positions = torch.arange(steps, dtype=torch.long) + int(start_position)
    for offset in range(PG.PAGE):
        selected_steps = positions.remainder(PG.PAGE).eq(offset)
        selected_cosine = cosine[selected_steps]
        selected_below = below_gate[selected_steps]
        selected_top1 = top1_match[selected_steps]
        per_page_offset.append(
            {
                "page_offset": offset,
                "rows": int(selected_cosine.numel()),
                "minimum_cosine": float(selected_cosine.min().item()),
                "mean_cosine": float(selected_cosine.mean().item()),
                "rows_below_gate": int(selected_below.sum().item()),
                "top1_mismatches": int((~selected_top1).sum().item()),
            }
        )

    return {
        "schema_version": 3,
        "purpose": (
            "posthoc localization of strict HF-vs-backend logit outliers; "
            "not a replacement quality endpoint and not publication timing"
        ),
        "threshold_unchanged": float(minimum_cosine),
        "configuration": {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "vocab_size": int(vocab_size),
            "start_position": int(start_position),
            "exact_tail_tokens": int(exact_tail_tokens),
            "exact_tail_pages": exact_tail_pages,
            "exact_prefix_pages": exact_prefix_pages,
            "exact_prefix_logical_pages": list(range(exact_prefix_pages)),
            "initial_quantized_old_pages": prefix_quantized_pages,
            "initial_quantized_old_logical_range_start_inclusive_end_exclusive": [
                quantized_old_begin,
                quantized_old_end,
            ],
            "initial_exact_tail_logical_range_start_inclusive_end_exclusive": [
                quantized_old_end,
                initial_pages,
            ],
            "initial_page_partition_disjoint_and_complete": bool(
                exact_prefix_pages
                + prefix_quantized_pages
                + exact_tail_pages
                == initial_pages
            ),
            "page_size": int(PG.PAGE),
            "top_rows_retained": len(worst_rows),
            "top_vocab_retained": retained_vocab,
        },
        "summary": {
            "checked_rows": int(steps * batch_size),
            "rows_below_gate": int(below_gate.sum().item()),
            "fraction_below_gate": float(below_gate.float().mean().item()),
            "top1_mismatches": int((~top1_match).sum().item()),
            "top1_agreement_fraction": float(top1_match.float().mean().item()),
            "cosine_quantiles": _quality_quantiles(cosine),
            "relative_l2_quantiles": _quality_quantiles(relative_l2),
            "maximum_absolute_error_quantiles": _quality_quantiles(
                maximum_absolute_error
            ),
            "failure_clusters": _consecutive_failure_clusters(below_gate),
        },
        "per_request": per_request,
        "per_page_offset": per_page_offset,
        "worst_rows": worst_rows,
        "row_matrices": {
            "cosine_by_step_request": cosine.tolist(),
            "relative_l2_by_step_request": relative_l2.tolist(),
            "maximum_absolute_error_by_step_request": (
                maximum_absolute_error.tolist()
            ),
            "top1_match_by_step_request": top1_match.tolist(),
            "reference_l2_by_step_request": reference_norm.tolist(),
            "candidate_l2_by_step_request": candidate_norm.tolist(),
        },
    }


def compare_tokens(expected: torch.Tensor, observed: torch.Tensor) -> dict[str, Any]:
    if expected.shape != observed.shape or expected.ndim != 2:
        raise ValueError("token tensors must be matching [steps,B]")
    agreement = expected.eq(observed)
    return {
        "checked_steps": int(expected.shape[0]),
        "checked_request_steps": int(agreement.numel()),
        "bitwise_identical": bool(agreement.all().item()),
        "agreement_fraction": float(agreement.float().mean().item()),
        "expected_sha256": sha256_tensors([expected]),
        "observed_sha256": sha256_tensors([observed]),
    }


def compare_boundary_digests(
    expected: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("boundary digest counts differ")
    records = []
    for reference, result in zip(expected, observed):
        same_page = (
            reference["logical_page"] == result["logical_page"]
            and reference["exact_ring_page"] == result["exact_ring_page"]
        )
        identical = same_page and reference["tensor_sha256"] == result["tensor_sha256"]
        records.append(
            {
                "logical_page": reference["logical_page"],
                "exact_ring_page": reference["exact_ring_page"],
                "bitwise_identical": identical,
                "expected_combined_sha256": reference["combined_sha256"],
                "observed_combined_sha256": result["combined_sha256"],
            }
        )
    return {
        "passed": all(record["bitwise_identical"] for record in records),
        "checked_page_closes": len(records),
        "records": records,
    }


def compare_serving_metadata(
    expected: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    """Compare active page tables and planner materialization fail closed."""

    expected_payload = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    observed_payload = json.dumps(observed, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    observed_hash = hashlib.sha256(observed_payload.encode("utf-8")).hexdigest()
    wrapper_names = sorted(set(expected.get("wrappers", {})) | set(observed.get("wrappers", {})))
    wrapper_records = {
        name: {
            "bitwise_identical": expected.get("wrappers", {}).get(name)
            == observed.get("wrappers", {}).get(name),
            "expected_active_indices_sha256": expected.get("wrappers", {})
            .get(name, {})
            .get("active_indices_sha256"),
            "observed_active_indices_sha256": observed.get("wrappers", {})
            .get(name, {})
            .get("active_indices_sha256"),
            "expected_semantic_int_workspace_sha256": expected.get("wrappers", {})
            .get(name, {})
            .get("semantic_int_workspace_sha256"),
            "observed_semantic_int_workspace_sha256": observed.get("wrappers", {})
            .get(name, {})
            .get("semantic_int_workspace_sha256"),
        }
        for name in wrapper_names
    }
    return {
        "passed": expected == observed,
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "wrappers": wrapper_records,
        "current_pages_identical": expected.get("current_pages")
        == observed.get("current_pages"),
        "old_pages_identical": expected.get("old_pages")
        == observed.get("old_pages"),
        "exact_pages_identical": expected.get("exact_pages")
        == observed.get("exact_pages"),
        "exact_sink_pages_identical": expected.get("exact_sink_pages")
        == observed.get("exact_sink_pages"),
        "exact_prefix_pages_identical": expected.get("exact_prefix_pages")
        == observed.get("exact_prefix_pages"),
        "logical_partition_identical": expected.get(
            "old_logical_range_start_inclusive_end_exclusive"
        )
        == observed.get("old_logical_range_start_inclusive_end_exclusive")
        and expected.get(
            "exact_tail_logical_range_start_inclusive_end_exclusive"
        )
        == observed.get(
            "exact_tail_logical_range_start_inclusive_end_exclusive"
        ),
        "heterogeneous_old_kv_len_identical": expected.get(
            "heterogeneous_old_kv_len"
        )
        == observed.get("heterogeneous_old_kv_len"),
    }


def initialize_following_page_canary(
    decoder: Any, start_position: int, decode_steps: int
) -> int:
    """Initialize one allocated-but-unserved page with deterministic sentinels."""

    logical_page = (start_position + decode_steps) // PG.PAGE
    if logical_page >= decoder.max_pages:
        raise ValueError("decoder has no capacity for a following-page canary")
    physical_pages = [
        request * decoder.max_pages + logical_page
        for request in range(decoder.batch_size)
    ]
    if decoder.backend == "flashinfer_fp16":
        for physical_page in physical_pages:
            decoder.cache.key[:, physical_page].fill_(0.125)
            decoder.cache.value[:, physical_page].fill_(-0.25)
    else:
        for physical_page in physical_pages:
            decoder.cache.key_codes[:, physical_page].fill_(-113)
            decoder.cache.value_codes[:, physical_page].fill_(107)
            decoder.cache.key_scales[:, physical_page].fill_(0.75)
            decoder.cache.value_scales[:, physical_page].fill_(1.25)
    representative = (
        decoder.cache.key
        if decoder.backend == "flashinfer_fp16"
        else decoder.cache.key_codes
    )
    if representative.device.type == "cuda":
        torch.cuda.synchronize(representative.device)
    return logical_page


def immutable_page_digest(decoder: Any, logical_page: int) -> dict[str, Any]:
    """Hash one full-cache page without touching the mutable exact ring."""

    if not 0 <= logical_page < decoder.max_pages:
        raise ValueError("canary logical page lies outside cache capacity")
    physical_pages = torch.tensor(
        [
            request * decoder.max_pages + logical_page
            for request in range(decoder.batch_size)
        ],
        device="cuda",
        dtype=torch.long,
    )
    if decoder.backend == "flashinfer_fp16":
        tensors = (
            decoder.cache.key.index_select(1, physical_pages),
            decoder.cache.value.index_select(1, physical_pages),
        )
        names = ("key", "value")
    else:
        tensors = (
            decoder.cache.key_codes.index_select(1, physical_pages),
            decoder.cache.value_codes.index_select(1, physical_pages),
            decoder.cache.key_scales.index_select(1, physical_pages),
            decoder.cache.value_scales.index_select(1, physical_pages),
        )
        names = ("key_codes", "value_codes", "key_scales", "value_scales")
    records = {
        name: sha256_tensors([tensor]) for name, tensor in zip(names, tensors)
    }
    return {
        "logical_page": logical_page,
        "tensor_sha256": records,
        "combined_sha256": sha256_tensors(tensors),
    }


def immutable_exact_prefix_digest(decoder: Any) -> dict[str, Any]:
    """Hash every layer/request fixed prefix/suffix slot outside restores."""

    prefix_pages = int(
        getattr(
            decoder,
            "exact_prefix_pages",
            getattr(decoder, "exact_sink_pages", 0),
        )
    )
    static_suffix_pages = int(
        getattr(decoder, "exact_static_suffix_pages", 0)
    )
    fixed_pages = prefix_pages + static_suffix_pages
    enabled = bool(
        decoder.backend == "page_gauge"
        and fixed_pages > 0
    )
    if not enabled:
        return {
            "enabled": False,
            "exact_prefix_pages": 0,
            "exact_static_suffix_pages": 0,
            "fixed_exact_pages": 0,
            "logical_pages": [],
            "physical_pages": [],
            "tensor_sha256": {},
        }
    if decoder.cache.exact_key.shape != decoder.cache.exact_value.shape:
        raise RuntimeError("exact-prefix K/V storage geometry differs")
    total_storage_pages = int(decoder.cache.exact_key.shape[1])
    if total_storage_pages % decoder.batch_size:
        raise RuntimeError("exact-prefix storage is not request-major")
    storage_pages = total_storage_pages // decoder.batch_size
    if storage_pages != fixed_pages + int(decoder.exact_tail_pages):
        raise RuntimeError("fixed-exact storage stride disagrees with S+A+T")
    initial_pages = (
        int(decoder.initial_context_pages) if static_suffix_pages else 0
    )
    logical_pages = list(range(prefix_pages)) + list(
        range(initial_pages - static_suffix_pages, initial_pages)
    )
    physical_page_values = [
        decoder.exact_physical_page(request, logical_page)
        for request in range(decoder.batch_size)
        for logical_page in logical_pages
    ]
    expected_physical_page_values = [
        request * storage_pages + local_page
        for request in range(decoder.batch_size)
        for local_page in range(fixed_pages)
    ]
    if physical_page_values != expected_physical_page_values:
        raise RuntimeError("decoder exact-prefix mapping disagrees with fixed slots")
    if len(set(physical_page_values)) != decoder.batch_size * fixed_pages:
        raise RuntimeError("fixed-exact physical slots are not request-major unique")
    physical_pages = torch.tensor(
        physical_page_values,
        device=decoder.cache.exact_key.device,
        dtype=torch.long,
    )
    staged = torch.stack(
        (
            decoder.cache.exact_key.index_select(1, physical_pages),
            decoder.cache.exact_value.index_select(1, physical_pages),
        )
    ).detach().contiguous().cpu()
    tensors = (staged[0], staged[1])
    names = ("exact_key", "exact_value")
    records = {
        name: sha256_tensors([tensor]) for name, tensor in zip(names, tensors)
    }
    return {
        "enabled": True,
        "exact_prefix_pages": prefix_pages,
        "exact_static_suffix_pages": static_suffix_pages,
        "fixed_exact_pages": fixed_pages,
        "logical_pages": logical_pages,
        "logical_to_physical_request_major": [
            {
                "request": request,
                "logical_pages": logical_pages,
                "physical_pages": [
                    decoder.exact_physical_page(request, logical_page)
                    for logical_page in logical_pages
                ],
            }
            for request in range(decoder.batch_size)
        ],
        "physical_pages": physical_page_values,
        "physical_page_count": decoder.batch_size * fixed_pages,
        "tensor_sha256": records,
        "combined_sha256": sha256_tensors(tensors),
    }


def immutable_exact_sink_digest(decoder: Any) -> dict[str, Any]:
    """Backward-compatible name for the generic exact-prefix digest."""

    return immutable_exact_prefix_digest(decoder)


def immutable_adjacent_page_digests(
    decoder: Any, start_position: int, decode_steps: int
) -> dict[str, Any]:
    preceding = start_position // PG.PAGE - 1
    following = (start_position + decode_steps) // PG.PAGE
    if preceding < 0 or following >= decoder.max_pages:
        raise ValueError("sustained range requires allocated adjacent canary pages")
    records = {
        "preceding": immutable_page_digest(decoder, preceding),
        "following": immutable_page_digest(decoder, following),
    }
    if decoder.backend == "page_gauge" and (
        getattr(decoder, "exact_sink_pages", 0)
        or getattr(decoder, "exact_static_suffix_pages", 0)
    ):
        prefix_digest = immutable_exact_prefix_digest(decoder)
        records["exact_prefix"] = prefix_digest
        if int(getattr(decoder, "exact_sink_pages", 0)) == 1:
            records["exact_sink"] = prefix_digest
    return records


def compare_exact_sink_canary(
    expected: dict[str, Any] | None,
    decoder: Any,
    *,
    phase: str,
) -> dict[str, Any]:
    enabled = bool(expected and expected.get("enabled"))
    observed = immutable_exact_prefix_digest(decoder)
    passed = expected == observed if enabled else not observed.get("enabled")
    return {
        "phase": phase,
        "enabled": enabled,
        "passed": passed,
        "exact_prefix_pages": int(observed.get("exact_prefix_pages", 0)),
        "exact_static_suffix_pages": int(
            observed.get("exact_static_suffix_pages", 0)
        ),
        "fixed_exact_pages": int(observed.get("fixed_exact_pages", 0)),
        "logical_pages": observed.get("logical_pages", []),
        "physical_pages": observed.get("physical_pages", []),
        "physical_page_count": int(observed.get("physical_page_count", 0)),
        "expected_combined_sha256": (
            expected.get("combined_sha256") if expected else None
        ),
        "observed_combined_sha256": observed.get("combined_sha256"),
        "expected_tensor_sha256": expected.get("tensor_sha256", {})
        if expected
        else {},
        "observed_tensor_sha256": observed.get("tensor_sha256", {}),
    }


def compare_canary(
    expected: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "passed": expected == observed,
        "logical_pages": {
            name: record.get("logical_page")
            for name, record in expected.items()
        },
        "expected_combined_sha256": {
            name: record.get("combined_sha256")
            for name, record in expected.items()
        },
        "observed_combined_sha256": {
            name: record.get("combined_sha256")
            for name, record in observed.items()
        },
    }


def expected_runtime_counts(decoder: Any, decode_steps: int) -> dict[str, Any]:
    wrappers = 1 if decoder.backend == "flashinfer_fp16" else (
        1 if decoder.tail_attention == "heterogeneous_fa2" else 2
    )
    # prepare_run() restores the prefix plan.  For a page-aligned prefix the
    # first generated token opens a new page, so D512 contains 32 real planner
    # topology/page-table transitions, including offset zero.
    page_transitions = decode_steps // PG.PAGE
    if decoder.backend == "flashinfer_fp16":
        plan_rebuilds_by_wrapper = {"baseline": page_transitions}
    elif decoder.tail_attention == "heterogeneous_fa2":
        plan_rebuilds_by_wrapper = {"heterogeneous": page_transitions}
    else:
        suffix_pages = int(getattr(decoder, "exact_static_suffix_pages", 0))
        tail_pages = int(decoder.exact_tail_pages)
        # While the dynamic tail displaces original pages already retained in
        # the fixed suffix, the old logical set is unchanged.  Thereafter it
        # grows at every page boundary.  The exact table changes throughout.
        old_rebuilds = max(
            0, page_transitions - min(suffix_pages, tail_pages)
        )
        plan_rebuilds_by_wrapper = {
            "old_int8": old_rebuilds,
            "exact_fp16": page_transitions,
        }
    return {
        "decoder_plan_calls": decode_steps,
        "device_position_fills": decode_steps,
        "wrapper_count": wrappers,
        "plan_invocations_per_wrapper": decode_steps,
        "plan_rebuilds_by_wrapper": plan_rebuilds_by_wrapper,
        "last_page_len_device_fills_per_wrapper": decode_steps,
        "decoder_layer_graph_replays": decoder.layers * decode_steps,
        "decoder_layer_eager_calls": 0,
        "graph_bank_misses": 0,
        "nested_attention_dispatch_calls": 0,
        "heterogeneous_page_table_updates": (
            page_transitions
            if decoder.backend == "page_gauge"
            and decoder.tail_attention == "heterogeneous_fa2"
            else 0
        ),
        "exact_page_table_updates": (
            page_transitions
            if decoder.backend == "page_gauge"
            and (
                getattr(decoder, "exact_sink_pages", 0)
                or getattr(decoder, "exact_static_suffix_pages", 0)
            )
            else 0
        ),
    }


def runtime_count_gate(
    decoder: Any,
    dispatch: dict[str, Any],
    attention_dispatch: dict[str, Any],
    operations: dict[str, Any],
    decode_steps: int,
) -> dict[str, Any]:
    expected = expected_runtime_counts(decoder, decode_steps)
    wrapper_records = operations["wrappers"]
    wrapper_passed = bool(
        len(wrapper_records) == expected["wrapper_count"]
        and set(wrapper_records) == set(expected["plan_rebuilds_by_wrapper"])
        and all(
            record["plan_invocations"] == expected["plan_invocations_per_wrapper"]
            and record["plan_rebuilds"]
            == expected["plan_rebuilds_by_wrapper"][name]
            and record["last_page_len_device_fills"]
            == expected["last_page_len_device_fills_per_wrapper"]
            for name, record in wrapper_records.items()
        )
    )
    passed = bool(
        dispatch["graph_replays"] == expected["decoder_layer_graph_replays"]
        and dispatch["eager_calls"] == 0
        and dispatch["graph_bank_misses"] == 0
        and operations["decoder_plan_calls"] == decode_steps
        and operations["device_position_fills"] == decode_steps
        and operations["heterogeneous_page_table_updates"]
        == expected["heterogeneous_page_table_updates"]
        and operations.get("exact_page_table_updates", 0)
        == expected["exact_page_table_updates"]
        and attention_dispatch["total_calls"]
        == expected["nested_attention_dispatch_calls"]
        and wrapper_passed
    )
    return {
        "passed": passed,
        "expected": expected,
        "observed_dispatch": dispatch,
        "observed_nested_attention_dispatch": attention_dispatch,
        "observed_operations": operations,
        "wrapper_counts_passed": wrapper_passed,
    }


def prepare_run(
    decoder: Any, initial_cache: dict[str, Any], start_position: int
) -> None:
    PG.restore_mutated_cache_range(decoder, initial_cache)
    # Restore the supported wrapper to the prefix state.  The first timed
    # decoder.step(start_position) must perform the real transition to
    # context=start_position+1; preplanning that state would hide one of the
    # 32 D512 page-boundary scheduling costs.
    decoder.plan(start_position)
    torch.cuda.synchronize()


def initialize_nonsemantic_capture_pages(
    decoder: Any, start_position: int, decode_steps: int
) -> None:
    """Make future-page capture warmups finite without changing validation state."""

    pages = PG.mutated_cache_page_indices(decoder, start_position, decode_steps)[
        "full"
    ]
    if decoder.backend == "flashinfer_fp16":
        tensors = (decoder.cache.key, decoder.cache.value)
    else:
        # Preserve the populated exact ring. Sequential validation overwrites
        # these future code/scale pages before they enter the old INT8 segment.
        tensors = (
            decoder.cache.key_codes,
            decoder.cache.value_codes,
            decoder.cache.key_scales,
            decoder.cache.value_scales,
        )
    for physical_page in pages:
        for tensor in tensors:
            tensor[:, physical_page].zero_()
    torch.cuda.synchronize()


@torch.inference_mode()
def timed_block(
    decoder: Any,
    *,
    initial_token: torch.Tensor,
    teacher_inputs: torch.Tensor,
    trajectory_mode: str,
    start_position: int,
    decode_steps: int,
    dynamic_token: torch.Tensor,
    argmax_token: torch.Tensor,
) -> dict[str, Any]:
    decoder.reset_attention_dispatch_counts()
    decoder.reset_runtime_operation_counts()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    begin.record()
    execute_sustained(
        decoder,
        initial_token=initial_token,
        teacher_inputs=teacher_inputs,
        trajectory_mode=trajectory_mode,
        start_position=start_position,
        decode_steps=decode_steps,
        dynamic_token=dynamic_token,
        argmax_token=argmax_token,
        collect=False,
    )
    end.record()
    torch.cuda.synchronize()
    cuda_ms = float(begin.elapsed_time(end))
    wall_ms = (time.perf_counter() - wall_started) * 1e3
    gate = runtime_count_gate(
        decoder,
        decoder.decoder_layer_dispatch_counts(),
        decoder.attention_dispatch_counts(),
        decoder.runtime_operation_counts(),
        decode_steps,
    )
    if not gate["passed"]:
        raise RuntimeError(f"timed graph dispatch/operation gate failed: {gate}")
    return {"cuda_ms": cuda_ms, "wall_ms": wall_ms, "runtime_gate": gate}


def summarize_samples(values: list[float], steps: int, batch_size: int) -> dict[str, Any]:
    ordered = sorted(values)
    output_tokens = steps * batch_size
    return {
        "raw_ms": values,
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "mean_ms_per_decode_step": statistics.mean(values) / steps,
        "mean_ms_per_output_token": statistics.mean(values) / output_tokens,
        "output_tokens_per_second": output_tokens / (statistics.mean(values) / 1e3),
    }


@torch.inference_mode()
def measure_modes(
    decoder: Any,
    *,
    initial_cache: dict[str, Any],
    initial_token: torch.Tensor,
    teacher_inputs: torch.Tensor,
    trajectory_mode: str,
    start_position: int,
    decode_steps: int,
    dynamic_token: torch.Tensor,
    argmax_token: torch.Tensor,
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    expected_sink_canary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {}
    for mode in ("cache_neutral", "cache_hot"):
        sink_canary_records = []
        warmup_records = []
        for warmup in range(warmups):
            prepare_run(decoder, initial_cache, start_position)
            precondition = None
            if mode == "cache_neutral":
                cache_scrub.add_(1)
                torch.cuda.synchronize()
            else:
                precondition = timed_block(
                    decoder,
                    initial_token=initial_token,
                    teacher_inputs=teacher_inputs[: PG.PAGE],
                    trajectory_mode=trajectory_mode,
                    start_position=start_position,
                    decode_steps=PG.PAGE,
                    dynamic_token=dynamic_token,
                    argmax_token=argmax_token,
                )
                decoder.plan(start_position)
                torch.cuda.synchronize()
                precondition["precondition_steps"] = PG.PAGE
                precondition["planner_reset_without_cache_restore"] = True
                precondition["exact_prefix_attestation"] = (
                    "covered jointly by the post-sample canary so no "
                    "candidate-only digest perturbs the hot precondition"
                )
                precondition["no_restore_before_timed_sample"] = True
            sample = timed_block(
                decoder,
                initial_token=initial_token,
                teacher_inputs=teacher_inputs,
                trajectory_mode=trajectory_mode,
                start_position=start_position,
                decode_steps=decode_steps,
                dynamic_token=dynamic_token,
                argmax_token=argmax_token,
            )
            sample_canary = compare_exact_sink_canary(
                expected_sink_canary,
                decoder,
                phase=f"{mode}.warmup.{warmup}.sample",
            )
            sink_canary_records.append(sample_canary)
            warmup_records.append(
                {
                    "warmup": warmup,
                    "precondition": precondition,
                    "exact_sink_canary": sample_canary,
                    "exact_prefix_canary": sample_canary,
                    **sample,
                }
            )

        samples = []
        for sample_index in range(repeats):
            prepare_run(decoder, initial_cache, start_position)
            precondition = None
            if mode == "cache_neutral":
                cache_scrub.add_(1)
                torch.cuda.synchronize()
            else:
                precondition = timed_block(
                    decoder,
                    initial_token=initial_token,
                    teacher_inputs=teacher_inputs[: PG.PAGE],
                    trajectory_mode=trajectory_mode,
                    start_position=start_position,
                    decode_steps=PG.PAGE,
                    dynamic_token=dynamic_token,
                    argmax_token=argmax_token,
                )
                decoder.plan(start_position)
                torch.cuda.synchronize()
                precondition["precondition_steps"] = PG.PAGE
                precondition["planner_reset_without_cache_restore"] = True
                precondition["exact_prefix_attestation"] = (
                    "covered jointly by the post-sample canary so no "
                    "candidate-only digest perturbs the hot precondition"
                )
                precondition["no_restore_before_timed_sample"] = True
            sample = timed_block(
                decoder,
                initial_token=initial_token,
                teacher_inputs=teacher_inputs,
                trajectory_mode=trajectory_mode,
                start_position=start_position,
                decode_steps=decode_steps,
                dynamic_token=dynamic_token,
                argmax_token=argmax_token,
            )
            sample_canary = compare_exact_sink_canary(
                expected_sink_canary,
                decoder,
                phase=f"{mode}.sample.{sample_index}.sample",
            )
            sink_canary_records.append(sample_canary)
            samples.append(
                {
                    "sample_index": sample_index,
                    "precondition": precondition,
                    "exact_sink_canary": sample_canary,
                    "exact_prefix_canary": sample_canary,
                    **sample,
                }
            )
        cuda_values = [record["cuda_ms"] for record in samples]
        wall_values = [record["wall_ms"] for record in samples]
        result[mode] = {
            "raw_samples": samples,
            "warmup_samples": warmup_records,
            "cuda": summarize_samples(cuda_values, decode_steps, decoder.batch_size),
            "wall": summarize_samples(wall_values, decode_steps, decoder.batch_size),
            "restore_inside_timed_boundary": False,
            "precondition_inside_timed_boundary": False,
            "no_restore_between_hot_precondition_and_sample": True,
            "hot_precondition_steps": PG.PAGE,
            "planner_reset_without_cache_restore": True,
            "exact_sink_canary": {
                "enabled": bool(expected_sink_canary and expected_sink_canary.get("enabled")),
                "passed": all(record["passed"] for record in sink_canary_records),
                "check_count": len(sink_canary_records),
                "checks": sink_canary_records,
                "checked_after_terminal_synchronize_before_restore": True,
                "no_prefix_digest_between_hot_precondition_and_sample": True,
                "inside_timed_boundary": False,
            },
            "exact_prefix_canary": {
                "enabled": bool(
                    expected_sink_canary and expected_sink_canary.get("enabled")
                ),
                "passed": all(record["passed"] for record in sink_canary_records),
                "check_count": len(sink_canary_records),
                "checks": sink_canary_records,
                "checked_after_terminal_synchronize_before_restore": True,
                "no_prefix_digest_between_hot_precondition_and_sample": True,
                "inside_timed_boundary": False,
            },
        }
    prepare_run(decoder, initial_cache, start_position)
    return result


def scheduler_capacity(
    *,
    num_sms: int,
    batch_size: int,
    hkv: int,
    maximum_active_pages: int,
    fixed_split_pages: int,
) -> dict[str, Any]:
    max_batch_size_if_split = (2 * num_sms) // hkv
    max_chunks_per_request = max_batch_size_if_split // batch_size
    chunks = (
        math.ceil(maximum_active_pages / fixed_split_pages)
        if fixed_split_pages > 0
        else None
    )
    minimum_safe_split_pages = (
        math.ceil(maximum_active_pages / max_chunks_per_request)
        if max_chunks_per_request > 0
        else None
    )
    return {
        "formula": "max_batch_size_if_split=(2*num_sms)//Hkv",
        "num_sms": num_sms,
        "hkv": hkv,
        "batch_size": batch_size,
        "maximum_active_pages_per_request": maximum_active_pages,
        "fixed_split_pages": fixed_split_pages,
        "max_batch_size_if_split": max_batch_size_if_split,
        "max_chunks_per_request": max_chunks_per_request,
        "required_chunks_per_request": chunks,
        "required_split_tiles": chunks * batch_size if chunks is not None else None,
        "minimum_safe_fixed_split_pages": minimum_safe_split_pages,
        "analytically_within_capacity": (
            chunks is None or chunks * batch_size <= max_batch_size_if_split
        ),
        "authoritative_gate": "exhaustive supported-wrapper plan preflight",
        "split_was_automatically_clamped": False,
    }


def scheduler_capacity_report(
    *,
    backend: str,
    tail_attention: str,
    num_sms: int,
    batch_size: int,
    hkv: int,
    maximum_pages: int,
    exact_pages: int,
    baseline_split_pages: int,
    candidate_split_pages: int,
    exact_sink_pages: int = 0,
    exact_static_suffix_pages: int = 0,
) -> dict[str, Any]:
    """Report the capacity calculation for every active FlashInfer wrapper."""

    exact_sink_pages = PG.validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=maximum_pages,
    )
    if backend == "page_gauge":
        PG.validate_exact_prefix_attention_path(exact_sink_pages, tail_attention)
    elif exact_sink_pages or exact_static_suffix_pages:
        raise ValueError("fixed exact pages are supported only by PageGauge")
    if backend == "flashinfer_fp16":
        specifications = {
            "baseline": (maximum_pages, baseline_split_pages),
        }
    elif tail_attention == "heterogeneous_fa2":
        # The unified heterogeneous wrapper attends the full page table.  Its
        # active-page count is not reduced by the exact FP16 tail.
        specifications = {
            "heterogeneous": (maximum_pages, candidate_split_pages),
        }
    else:
        specifications = {
            "old_int8": (
                maximum_pages
                - exact_pages
                - exact_sink_pages
                - exact_static_suffix_pages,
                candidate_split_pages,
            ),
            "exact_fp16": (
                exact_pages + exact_sink_pages + exact_static_suffix_pages,
                candidate_split_pages,
            ),
        }
    if any(active_pages <= 0 for active_pages, _ in specifications.values()):
        raise ValueError("scheduler segment must contain at least one active page")
    wrappers = {
        name: scheduler_capacity(
            num_sms=num_sms,
            batch_size=batch_size,
            hkv=hkv,
            maximum_active_pages=active_pages,
            fixed_split_pages=split_pages,
        )
        for name, (active_pages, split_pages) in specifications.items()
    }
    return {
        "wrappers": wrappers,
        "all_wrappers_analytically_within_capacity": all(
            record["analytically_within_capacity"]
            for record in wrappers.values()
        ),
        "authoritative_gate": "exhaustive supported-wrapper plan preflight",
        "split_was_automatically_clamped": False,
    }


def planner_boundary_positions(
    start_position: int, decode_steps: int
) -> list[int]:
    """Positions whose step opens a page beyond the restored prefix plan."""

    return [
        position
        for position in range(start_position, start_position + decode_steps)
        if position == start_position or position % PG.PAGE == 0
    ]


def runtime_finalization_consumption(
    decoder: Any, context: int, decode_steps: int
) -> dict[str, Any]:
    first_generated_page = context // PG.PAGE
    last_generated_page = (context + decode_steps - 1) // PG.PAGE
    final_pages = math.ceil((context + decode_steps) / PG.PAGE)
    if decoder.backend != "page_gauge":
        return {
            "first_generated_logical_page": first_generated_page,
            "last_generated_logical_page": last_generated_page,
            "generated_pages": last_generated_page - first_generated_page + 1,
            "final_old_pages": None,
            "runtime_finalized_pages_consumed_as_int8": [],
            "runtime_finalized_int8_pages_consumed_count": 0,
            "final_attention_page_table_gate_passed": True,
            "passed": True,
        }

    last_page_len = (context + decode_steps - 1) % PG.PAGE + 1
    partition = PG.page_gauge_logical_partition(
        final_pages,
        decoder.exact_tail_pages,
        decoder.exact_sink_pages,
        last_page_len,
        getattr(decoder, "exact_static_suffix_pages", 0),
        getattr(decoder, "initial_context_pages", None),
    )
    prefix_pages = list(partition["prefix_logical_pages"])
    static_suffix_pages = list(partition["static_suffix_logical_pages"])
    old_pages = list(partition["old_logical_pages"])
    tail_pages = list(partition["tail_logical_pages"])
    consumed = [
        page
        for page in old_pages
        if first_generated_page <= page <= last_generated_page
    ]
    old_table_gate = True
    exact_table_gate = True
    if decoder.tail_attention == "heterogeneous_fa2":
        table = decoder.heterogeneous_wrapper.indices[
            : decoder.batch_size * final_pages
        ].view(decoder.batch_size, final_pages)
        for request in range(decoder.batch_size):
            expected = torch.tensor(
                [request * decoder.max_pages + page for page in consumed],
                device=table.device,
                dtype=table.dtype,
            )
            old_table_gate = old_table_gate and bool(
                torch.equal(table[request, consumed[0] : consumed[-1] + 1], expected)
            )
    else:
        old_table = decoder.old_wrapper.indices[
            : decoder.batch_size * len(old_pages)
        ].view(decoder.batch_size, len(old_pages))
        exact_logical_pages = list(partition["exact_logical_pages"])
        exact_table = decoder.exact_wrapper.indices[
            : decoder.batch_size * len(exact_logical_pages)
        ].view(decoder.batch_size, len(exact_logical_pages))
        for request in range(decoder.batch_size):
            expected_old = torch.tensor(
                [request * decoder.max_pages + page for page in old_pages],
                device=old_table.device,
                dtype=old_table.dtype,
            )
            expected_exact = torch.tensor(
                [
                    decoder.exact_physical_page(request, page)
                    for page in exact_logical_pages
                ],
                device=exact_table.device,
                dtype=exact_table.dtype,
            )
            old_table_gate = old_table_gate and bool(
                torch.equal(old_table[request], expected_old)
            )
            exact_table_gate = exact_table_gate and bool(
                torch.equal(exact_table[request], expected_exact)
                and len(set(expected_exact.detach().cpu().tolist()))
                == len(exact_logical_pages)
            )
    logical_sets_disjoint = (
        not (set(prefix_pages) & set(old_pages))
        and not (set(static_suffix_pages) & set(old_pages))
        and not (set(static_suffix_pages) & set(tail_pages))
        and not (set(prefix_pages) & set(static_suffix_pages))
        and not (set(prefix_pages) & set(tail_pages))
        and not (set(old_pages) & set(tail_pages))
    )
    logical_coverage = (
        logical_sets_disjoint
        and set(prefix_pages + static_suffix_pages + old_pages + tail_pages)
        == set(range(final_pages))
        and partition["old_token_count"] + partition["exact_token_count"]
        == partition["context_token_count"]
    )
    prefix_exclusion_gate = not (set(prefix_pages) & set(old_pages))
    static_suffix_exclusion_gate = not (
        set(static_suffix_pages) & set(old_pages)
    )
    table_gate = old_table_gate and exact_table_gate
    return {
        "first_generated_logical_page": first_generated_page,
        "last_generated_logical_page": last_generated_page,
        "generated_pages": last_generated_page - first_generated_page + 1,
        "final_old_pages": len(old_pages),
        "final_old_logical_page_end_exclusive": partition["tail_logical_begin"],
        "final_old_logical_pages": old_pages,
        "final_exact_prefix_logical_pages": prefix_pages,
        "final_exact_sink_logical_pages": prefix_pages,
        "final_exact_static_suffix_logical_pages": static_suffix_pages,
        "final_exact_tail_logical_pages": tail_pages,
        "runtime_finalized_pages_consumed_as_int8": consumed,
        "runtime_finalized_int8_pages_consumed_count": len(consumed),
        "final_attention_page_table_gate_passed": table_gate,
        "old_attention_page_table_gate_passed": old_table_gate,
        "exact_attention_page_table_gate_passed": exact_table_gate,
        "logical_page_sets_disjoint": logical_sets_disjoint,
        "logical_token_coverage_exactly_once": logical_coverage,
        "exact_prefix_excluded_from_old_segment": prefix_exclusion_gate,
        "page_zero_excluded_from_old_segment": 0 not in old_pages,
        "prefix_exclusion_gate_passed": prefix_exclusion_gate,
        "static_suffix_exclusion_gate_passed": static_suffix_exclusion_gate,
        "sink_exclusion_gate_passed": prefix_exclusion_gate,
        "passed": (
            bool(consumed)
            and table_gate
            and logical_coverage
            and prefix_exclusion_gate
            and static_suffix_exclusion_gate
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    memory = {"process_start": TOKEN.gpu_memory_state()}

    from transformers import AutoModelForCausalLM

    print(f"Loading exclusive {args.backend} sustained fixture...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    layers, hq, hkv, hidden = PG.BASE_E2E.check_model(model)
    if (layers, hq, hkv, PG.DIM) != (32, 32, 8, 128):
        raise RuntimeError("sustained worker requires Mistral-7B 32/32/8/128 geometry")
    config_payload = model.config.to_dict()
    parameter_sample = BACKEND.sampled_model_parameter_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokens, token_provenance = PREFILL.build_token_matrix(
        args, int(model.config.vocab_size)
    )
    quality_windows = PREFILL.request_window_metadata(
        token_provenance, args.context, args.decode_steps, args.batch_size
    )
    model = model.cuda()
    torch.cuda.synchronize()
    memory["model_loaded"] = TOKEN.gpu_memory_state()

    max_context = args.context + args.decode_steps
    served_pages = math.ceil(max_context / PG.PAGE)
    # One extra physical page is never planned or served.  It is initialized
    # as a following-page canary so an append destination overrun fails the
    # correctness protocol instead of landing outside allocated storage.
    pages = served_pages + 1
    initial_pages = args.context // PG.PAGE
    exact_pages = args.exact_tail // PG.PAGE
    cache = BACKEND.allocate_backend_cache(
        args.backend,
        layers,
        pages,
        initial_pages,
        exact_pages,
        args.batch_size,
        hkv,
        exact_sink_pages=args.exact_sink_pages,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
    )
    torch.cuda.synchronize()
    memory["selected_backend_cache_allocated"] = TOKEN.gpu_memory_state()
    cache_manifest = BACKEND.cache_tensor_manifest(cache, args.backend)
    if args.backend == "page_gauge":
        prefix_gross_bytes = (
            layers
            * args.batch_size
            * args.exact_sink_pages
            * PG.PAGE
            * hkv
            * PG.DIM
            * 2
            * torch.empty((), dtype=torch.float16).element_size()
        )
        static_suffix_gross_bytes = (
            layers
            * args.batch_size
            * args.exact_static_suffix_pages
            * PG.PAGE
            * hkv
            * PG.DIM
            * 2
            * torch.empty((), dtype=torch.float16).element_size()
        )
        cache_manifest["exact_layout"] = {
            "tail_pages_per_request": exact_pages,
            "prefix_pages_per_request": args.exact_sink_pages,
            "sink_pages_per_request": args.exact_sink_pages,
            "static_suffix_pages_per_request": args.exact_static_suffix_pages,
            "fixed_exact_pages_per_request": (
                args.exact_sink_pages + args.exact_static_suffix_pages
            ),
            "storage_pages_per_request": (
                exact_pages
                + args.exact_sink_pages
                + args.exact_static_suffix_pages
            ),
            "physical_order": (
                "[fixed contiguous prefix slots 0..S-1, modulo tail-ring "
                "slots S..S+T-1]"
            ),
            "prefix_logical_pages": list(range(args.exact_sink_pages)),
            "sink_logical_pages": list(range(args.exact_sink_pages)),
            "static_suffix_logical_pages": list(
                range(
                    initial_pages - args.exact_static_suffix_pages,
                    initial_pages,
                )
            ),
            "prefix_int8_code_and_scale_storage_retained": bool(
                args.exact_sink_pages
            ),
            "all_exact_prefix_int8_storage_retained": bool(
                args.exact_sink_pages
            ),
            "page_zero_int8_storage_retained": bool(args.exact_sink_pages),
            "prefix_gross_allocated_bytes": prefix_gross_bytes,
            "sink_gross_allocated_bytes": prefix_gross_bytes,
            "static_suffix_gross_allocated_bytes": static_suffix_gross_bytes,
            "compact_replacement_not_claimed": True,
        }
    canary_full_cache_tensor_names = (
        ("key", "value")
        if args.backend == "flashinfer_fp16"
        else ("key_codes", "value_codes", "key_scales", "value_scales")
    )
    following_canary_storage_bytes = sum(
        int(getattr(cache, name)[:, 0].numel() * getattr(cache, name).element_size())
        * args.batch_size
        for name in canary_full_cache_tensor_names
    )

    hf_logits, hf_generated, prefill_records, kv_sample_sha256 = (
        BACKEND.build_coherent_fixture(
            model=model,
            tokens=tokens,
            cache=cache,
            backend=args.backend,
            layers=layers,
            hkv=hkv,
            pages=pages,
            initial_pages=initial_pages,
            exact_pages=exact_pages,
            context=args.context,
            decode_steps=args.decode_steps,
            chunk_tokens=args.prefill_chunk_tokens,
            exact_sink_pages=args.exact_sink_pages,
            exact_static_suffix_pages=args.exact_static_suffix_pages,
        )
    )
    if not all(record["sampled_population_gate_passed"] for record in prefill_records):
        raise RuntimeError("direct backend-cache population gate failed")
    memory["after_hf_dynamic_caches_released"] = TOKEN.gpu_memory_state()

    PG.BASE_E2E.pack_model_projections(model)
    gc.collect()
    torch.cuda.empty_cache()
    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    with torch.inference_mode():
        positions = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        rope_probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(rope_probe, positions)
        if rope_cos.dim() == 3:
            rope_cos, rope_sin = rope_cos[0], rope_sin[0]
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()
    decoder = PG.TransformerDecoder(
        model,
        flashinfer,
        append_extension,
        args.backend,
        cache,
        max_context + PG.PAGE,
        args.exact_tail,
        args.baseline_split_pages,
        args.candidate_split_pages,
        rope_cos,
        rope_sin,
        "attention_add",
        args.tail_attention,
        args.batch_size,
        device_dynamic_decoder_layer_graphs=True,
        old_value_scale_placement=args.old_value_scale_placement,
        exact_sink_pages=(
            args.exact_sink_pages if args.backend == "page_gauge" else 0
        ),
        exact_static_suffix_pages=(
            args.exact_static_suffix_pages
            if args.backend == "page_gauge"
            else 0
        ),
        initial_context_pages=initial_pages,
    )
    del positions, rope_probe
    torch.cuda.synchronize()
    memory["decoder_constructed"] = TOKEN.gpu_memory_state()

    device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    capacity = scheduler_capacity_report(
        backend=args.backend,
        tail_attention=args.tail_attention,
        num_sms=int(device_properties.multi_processor_count),
        batch_size=args.batch_size,
        hkv=hkv,
        maximum_pages=served_pages,
        exact_pages=exact_pages,
        baseline_split_pages=args.baseline_split_pages,
        candidate_split_pages=args.candidate_split_pages,
        exact_sink_pages=args.exact_sink_pages,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
    )
    if not capacity["all_wrappers_analytically_within_capacity"]:
        raise RuntimeError(
            "selected fixed split violates derived scheduler capacity; "
            "the configured split is never silently clamped"
        )

    initial_token = tokens[:, args.context].to(device="cuda", dtype=torch.long)
    teacher_inputs = build_teacher_inputs(initial_token, hf_generated)
    dynamic_token = torch.empty_like(initial_token)
    argmax_token = torch.empty_like(initial_token)
    following_canary_logical_page = initialize_following_page_canary(
        decoder, args.context, args.decode_steps
    )
    initial_gpu_snapshot = PG.snapshot_mutated_cache_range(
        decoder, args.context, args.decode_steps
    )
    torch.cuda.synchronize()
    initial_cache = PG.offload_cache_range_snapshot(
        initial_gpu_snapshot, pin_memory=True
    )
    initial_adjacent_canaries = immutable_adjacent_page_digests(
        decoder, args.context, args.decode_steps
    )
    initial_prefix_canary = initial_adjacent_canaries.get(
        "exact_prefix", initial_adjacent_canaries.get("exact_sink")
    )
    del initial_gpu_snapshot
    gc.collect()
    torch.cuda.empty_cache()

    print("Validating sustained eager reference...", flush=True)
    prepare_run(decoder, initial_cache, args.context)
    decoder.decoder_layer_graph_bank_replay_enabled = False
    decoder.decoder_layer_graph_bank_strict = False
    decoder.reset_attention_dispatch_counts()
    decoder.reset_runtime_operation_counts()
    eager = execute_sustained(
        decoder,
        initial_token=initial_token,
        teacher_inputs=teacher_inputs,
        trajectory_mode=args.trajectory_mode,
        start_position=args.context,
        decode_steps=args.decode_steps,
        dynamic_token=dynamic_token,
        argmax_token=argmax_token,
        collect=True,
    )
    torch.cuda.synchronize()
    eager_dispatch = decoder.decoder_layer_dispatch_counts()
    eager_attention_dispatch = decoder.attention_dispatch_counts()
    eager_operations = decoder.runtime_operation_counts()
    eager_serving_metadata = decoder.serving_metadata_snapshot()
    eager_adjacent_canaries = immutable_adjacent_page_digests(
        decoder, args.context, args.decode_steps
    )
    eager_gpu_snapshot = PG.snapshot_mutated_cache_range(
        decoder, args.context, args.decode_steps
    )
    eager_cache = PG.offload_cache_range_snapshot(eager_gpu_snapshot)
    del eager_gpu_snapshot

    prepare_run(decoder, initial_cache, args.context)
    initialize_nonsemantic_capture_pages(decoder, args.context, args.decode_steps)
    memory["before_graph_bank_preflight_and_capture"] = TOKEN.gpu_memory_state()
    print("Exhaustively preflighting and capturing structural graph banks...", flush=True)
    graph_bank_build_started = time.perf_counter()
    decoder.capture_dynamic_decoder_layer_graph_banks(
        args.context,
        args.decode_steps,
        capture_warmups=args.capture_warmups,
        maximum_graph_banks=args.maximum_graph_banks,
        strict=True,
    )
    torch.cuda.synchronize()
    graph_bank_build_wall_seconds = time.perf_counter() - graph_bank_build_started
    memory["after_graph_bank_preflight_and_capture"] = TOKEN.gpu_memory_state()
    prepare_run(decoder, initial_cache, args.context)

    print("Validating sustained strict graph replay...", flush=True)
    decoder.reset_attention_dispatch_counts()
    decoder.reset_runtime_operation_counts()
    graph = execute_sustained(
        decoder,
        initial_token=initial_token,
        teacher_inputs=teacher_inputs,
        trajectory_mode=args.trajectory_mode,
        start_position=args.context,
        decode_steps=args.decode_steps,
        dynamic_token=dynamic_token,
        argmax_token=argmax_token,
        collect=True,
    )
    torch.cuda.synchronize()
    graph_dispatch = decoder.decoder_layer_dispatch_counts()
    graph_attention_dispatch = decoder.attention_dispatch_counts()
    graph_operations = decoder.runtime_operation_counts()
    graph_runtime_gate = runtime_count_gate(
        decoder,
        graph_dispatch,
        graph_attention_dispatch,
        graph_operations,
        args.decode_steps,
    )
    graph_serving_metadata = decoder.serving_metadata_snapshot()
    graph_adjacent_canaries = immutable_adjacent_page_digests(
        decoder, args.context, args.decode_steps
    )
    graph_gpu_snapshot = PG.snapshot_mutated_cache_range(
        decoder, args.context, args.decode_steps
    )
    graph_cache = PG.offload_cache_range_snapshot(graph_gpu_snapshot)
    del graph_gpu_snapshot

    # A complete D512 PageGauge run advances through the entire exact ring.
    # Repeating without a restore would start from a semantically different
    # tail, so determinism is tested only after restoring the complete initial
    # cache range and the prefix planner state.
    prepare_run(decoder, initial_cache, args.context)
    print("Repeating strict graph replay from restored initial state...", flush=True)
    decoder.reset_attention_dispatch_counts()
    decoder.reset_runtime_operation_counts()
    graph_repeat = execute_sustained(
        decoder,
        initial_token=initial_token,
        teacher_inputs=teacher_inputs,
        trajectory_mode=args.trajectory_mode,
        start_position=args.context,
        decode_steps=args.decode_steps,
        dynamic_token=dynamic_token,
        argmax_token=argmax_token,
        collect=True,
    )
    torch.cuda.synchronize()
    graph_repeat_dispatch = decoder.decoder_layer_dispatch_counts()
    graph_repeat_attention_dispatch = decoder.attention_dispatch_counts()
    graph_repeat_operations = decoder.runtime_operation_counts()
    graph_repeat_runtime_gate = runtime_count_gate(
        decoder,
        graph_repeat_dispatch,
        graph_repeat_attention_dispatch,
        graph_repeat_operations,
        args.decode_steps,
    )
    graph_repeat_serving_metadata = decoder.serving_metadata_snapshot()
    graph_repeat_adjacent_canaries = immutable_adjacent_page_digests(
        decoder, args.context, args.decode_steps
    )
    graph_repeat_gpu_snapshot = PG.snapshot_mutated_cache_range(
        decoder, args.context, args.decode_steps
    )
    graph_repeat_cache = PG.offload_cache_range_snapshot(
        graph_repeat_gpu_snapshot
    )
    del graph_repeat_gpu_snapshot

    eager_vs_graph_logits = compare_logits(eager["logits"], graph["logits"])
    eager_vs_graph_tokens = compare_tokens(
        eager["generated_tokens"], graph["generated_tokens"]
    )
    eager_vs_graph_cache = PG.compare_mutated_cache_ranges(eager_cache, graph_cache)
    eager_vs_graph_boundaries = compare_boundary_digests(
        eager["page_boundaries"], graph["page_boundaries"]
    )
    eager_vs_graph_serving_metadata = compare_serving_metadata(
        eager_serving_metadata, graph_serving_metadata
    )
    graph_repeat_logits = compare_logits(graph["logits"], graph_repeat["logits"])
    graph_repeat_tokens = compare_tokens(
        graph["generated_tokens"], graph_repeat["generated_tokens"]
    )
    graph_repeat_cache_comparison = PG.compare_mutated_cache_ranges(
        graph_cache, graph_repeat_cache
    )
    graph_repeat_boundaries = compare_boundary_digests(
        graph["page_boundaries"], graph_repeat["page_boundaries"]
    )
    graph_repeat_metadata = compare_serving_metadata(
        graph_serving_metadata, graph_repeat_serving_metadata
    )
    canary_gates = {
        "eager": compare_canary(initial_adjacent_canaries, eager_adjacent_canaries),
        "graph_run_1": compare_canary(
            initial_adjacent_canaries, graph_adjacent_canaries
        ),
        "graph_run_2_restored": compare_canary(
            initial_adjacent_canaries, graph_repeat_adjacent_canaries
        ),
    }
    graph_repeat_passed = bool(
        graph_repeat_logits["bitwise_identical"]
        and graph_repeat_tokens["bitwise_identical"]
        and graph_repeat_cache_comparison["passed"]
        and graph_repeat_boundaries["passed"]
        and graph_repeat_metadata["passed"]
        and graph_repeat_runtime_gate["passed"]
    )
    hf_logits_tensor = torch.stack(hf_logits)
    hf_tokens_tensor = torch.stack(hf_generated)
    hf_vs_graph_logits = compare_logits(hf_logits_tensor, graph["logits"])
    hf_vs_graph_tokens = compare_tokens(hf_tokens_tensor, graph["generated_tokens"])
    quality_outlier_diagnostics = None
    if args.quality_diagnostics_top_k:
        print(
            "Building posthoc strict-quality outlier localization report...",
            flush=True,
        )
        quality_outlier_diagnostics = diagnose_logit_outliers(
            hf_logits_tensor,
            graph["logits"],
            input_tokens=teacher_inputs,
            start_position=args.context,
            exact_tail_tokens=args.exact_tail,
            exact_prefix_pages=args.exact_sink_pages,
            minimum_cosine=args.min_logits_cosine,
            top_rows=args.quality_diagnostics_top_k,
            top_vocab=args.quality_diagnostics_top_vocab,
        )
    finalization = runtime_finalization_consumption(
        decoder, args.context, args.decode_steps
    )
    graph_provenance = decoder.decoder_layer_graph_provenance()
    graph_structure_passed = bool(
        graph_provenance["enabled"]
        and graph_provenance["scope"]
        == "complete_decoder_layer_device_dynamic_position"
        and graph_provenance["graph_bank_count"] > 0
        and graph_provenance["graphs_per_bank"] == layers
        and graph_provenance["total_graphs"]
        == graph_provenance["graph_bank_count"] * layers
        and graph_provenance["graph_pools"]
        == graph_provenance["graph_bank_count"]
        and graph_provenance["replays_per_decode_step"] == layers
        and graph_provenance["nested_attention_graphs"] is False
        and graph_provenance["preflight_position_count"] == args.decode_steps
        and graph_provenance[
            "preflight_range_start_inclusive_end_exclusive"
        ]
        == [args.context, args.context + args.decode_steps]
        and graph_provenance["strict_missing_bucket_failure"]
        and graph_provenance["graph_bank_misses"] == 0
        and graph_runtime_gate["passed"]
        and graph_repeat_runtime_gate["passed"]
    )
    expected_counts = expected_runtime_counts(decoder, args.decode_steps)
    eager_wrapper_records = eager_operations["wrappers"]
    eager_wrapper_gate = bool(
        len(eager_wrapper_records) == expected_counts["wrapper_count"]
        and set(eager_wrapper_records)
        == set(expected_counts["plan_rebuilds_by_wrapper"])
        and all(
            record["plan_invocations"]
            == expected_counts["plan_invocations_per_wrapper"]
            and record["plan_rebuilds"]
            == expected_counts["plan_rebuilds_by_wrapper"][name]
            and record["last_page_len_device_fills"]
            == expected_counts["last_page_len_device_fills_per_wrapper"]
            for name, record in eager_wrapper_records.items()
        )
    )
    eager_gate = bool(
        eager_dispatch["eager_calls"] == layers * args.decode_steps
        and eager_dispatch["graph_replays"] == 0
        and eager_attention_dispatch["eager_calls"] == layers * args.decode_steps
        and eager_attention_dispatch["graph_replays"] == 0
        and eager_operations["decoder_plan_calls"] == args.decode_steps
        and eager_operations["device_position_fills"] == args.decode_steps
        and eager_operations["heterogeneous_page_table_updates"]
        == expected_counts["heterogeneous_page_table_updates"]
        and eager_operations.get("exact_page_table_updates", 0)
        == expected_counts["exact_page_table_updates"]
        and eager_wrapper_gate
    )
    same_backend_passed = bool(
        eager_gate
        and eager_vs_graph_logits["bitwise_identical"]
        and eager_vs_graph_tokens["bitwise_identical"]
        and eager_vs_graph_cache["passed"]
        and eager_vs_graph_boundaries["passed"]
        and eager_vs_graph_serving_metadata["passed"]
        and all(record["passed"] for record in canary_gates.values())
        and graph_repeat_passed
        and graph_structure_passed
        and finalization["passed"]
    )
    cross_reference_passed = bool(
        hf_vs_graph_logits["minimum_cosine"] >= args.min_logits_cosine
        and hf_vs_graph_logits["top1_agreement_fraction"]
        >= args.min_top1_agreement
        and (
            hf_vs_graph_tokens["bitwise_identical"]
            if args.trajectory_mode == GREEDY_FEEDBACK
            else True
        )
    )

    del (
        eager_cache,
        graph_cache,
        graph_repeat_cache,
        eager["logits"],
        graph["logits"],
        graph_repeat["logits"],
    )
    gc.collect()
    prepare_run(decoder, initial_cache, args.context)
    cache_scrub = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024,
        device="cuda",
        dtype=torch.uint8,
    )
    memory["before_timing"] = TOKEN.gpu_memory_state()
    print(
        f"Timing {args.decode_steps} continuous steps: "
        f"warmups={args.warmups}, repeats={args.repeats}...",
        flush=True,
    )
    timing_modes = measure_modes(
        decoder,
        initial_cache=initial_cache,
        initial_token=initial_token,
        teacher_inputs=teacher_inputs,
        trajectory_mode=args.trajectory_mode,
        start_position=args.context,
        decode_steps=args.decode_steps,
        dynamic_token=dynamic_token,
        argmax_token=argmax_token,
        cache_scrub=cache_scrub,
        warmups=args.warmups,
        repeats=args.repeats,
        expected_sink_canary=initial_prefix_canary,
    )
    timed_sink_canary_gate = {
        "passed": all(
            mode["exact_sink_canary"]["passed"]
            for mode in timing_modes.values()
        ),
        "check_count": sum(
            mode["exact_sink_canary"]["check_count"]
            for mode in timing_modes.values()
        ),
        "per_mode": {
            name: mode["exact_sink_canary"]
            for name, mode in timing_modes.items()
        },
    }
    timed_prefix_canary_gate = {
        **timed_sink_canary_gate,
        "semantic_name": "immutable_exact_prefix",
        "legacy_exact_sink_alias": True,
    }
    same_backend_passed = bool(
        same_backend_passed and timed_prefix_canary_gate["passed"]
    )
    memory["after_timing"] = TOKEN.gpu_memory_state()

    model_revision = getattr(model.config, "_commit_hash", None)
    if model_revision is None:
        model_revision = token_provenance.get("tokenizer", {}).get(
            "resolved_snapshot_revision"
        )
    if model_revision is None:
        model_revision = f"sampled-parameters:{parameter_sample['sha256']}"
    flashinfer_package_root = Path(flashinfer.__file__).resolve().parent
    flashinfer_abi_paths = {
        "decode.py": flashinfer_package_root / "decode.py",
        "scheduler.cuh": flashinfer_package_root
        / "data/include/flashinfer/attention/scheduler.cuh",
    }
    missing_flashinfer_abi_sources = [
        str(path) for path in flashinfer_abi_paths.values() if not path.is_file()
    ]
    if missing_flashinfer_abi_sources:
        raise RuntimeError(
            "cannot attest installed FlashInfer graph/planner ABI sources: "
            + ", ".join(missing_flashinfer_abi_sources)
        )
    flashinfer_abi = {
        "package_root": str(flashinfer_package_root),
        "version": getattr(flashinfer, "__version__", "unknown"),
        "source_sha256": {
            name: BACKEND.sha256_file(path)
            for name, path in flashinfer_abi_paths.items()
        },
    }
    config = {
        "backend": args.backend,
        "model": args.model,
        "model_revision": model_revision,
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail_tokens": args.exact_tail,
        "exact_sink_pages": args.exact_sink_pages,
        "exact_prefix_pages": args.exact_sink_pages,
        "exact_static_suffix_pages": args.exact_static_suffix_pages,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "tail_attention": args.tail_attention,
        "old_value_scale_placement": args.old_value_scale_placement,
        "trajectory_mode": args.trajectory_mode,
        "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
        "token_source": args.token_source,
        "wikitext_member": args.wikitext_member,
        "wikitext_archive_sha256": token_provenance.get("archive_sha256"),
        "min_logits_cosine": args.min_logits_cosine,
        "min_top1_agreement": args.min_top1_agreement,
        "seed": args.seed,
        "token_offset": args.token_offset,
        "token_stride": args.token_stride,
        "capture_warmups": args.capture_warmups,
        "maximum_graph_banks": args.maximum_graph_banks,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cache_scrub_mib": args.cache_scrub_mib,
        "quality_diagnostics_top_k": args.quality_diagnostics_top_k,
        "quality_diagnostics_top_vocab": args.quality_diagnostics_top_vocab,
    }
    pairing_configuration = {
        key: config[key]
        for key in (
            "model",
            "model_revision",
            "batch_size",
            "context",
            "decode_steps",
            "exact_tail_tokens",
            "exact_sink_pages",
            "exact_prefix_pages",
            "exact_static_suffix_pages",
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
        )
    }
    pairing_configuration["teacher_inputs_sha256"] = sha256_tensors(
        [teacher_inputs]
    )
    pairing_configuration["token_matrix_sha256"] = sha256_tensors([tokens])
    pairing_configuration["token_source_provenance_sha256"] = (
        PREFILL.canonical_json_sha256(token_provenance)
    )
    pairing_configuration["model_config_sha256"] = (
        PREFILL.canonical_json_sha256(config_payload)
    )
    pairing_configuration["sampled_model_parameters_sha256"] = parameter_sample[
        "sha256"
    ]
    pairing_configuration["flashinfer_abi_source_sha256"] = flashinfer_abi[
        "source_sha256"
    ]
    pairing_key_sha256 = hashlib.sha256(
        json.dumps(
            pairing_configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    passed = bool(same_backend_passed and cross_reference_passed)
    source_paths = (
        Path(__file__),
        BACKEND_WORKER_PATH,
        ROOT / "diagnostics/benchmark_full_sequence_graph.py",
        ROOT / "scripts/benchmark_page_gauge_transformer.py",
        ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
        ROOT / "scripts/page_gauge_heterogeneous_fa2.py",
        ROOT / "scripts/prepare_flashinfer_page_gauge.py",
        ROOT / "scripts/prepare_flashinfer_page_gauge_heterogeneous.py",
        ROOT / "scripts/benchmark_page_gauge_overheads.py",
        ROOT / "scripts/benchmark_e2e_transformer.py",
        ROOT / "scripts/page_gauge_runtime.py",
        ROOT / "tests/page_gauge_append_extension.cu",
        ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch",
        ROOT / "patches/flashinfer-0.6.17-page-gauge-heterogeneous.patch",
        ROOT / "docs/decoder_layer_cuda_graphs.md",
        BACKEND.PREFILL_PATH,
        BACKEND.TOKEN_PATH,
    )
    return {
        "schema_version": 3,
        "experiment": "page_gauge_backend_exclusive_sustained_dynamic_graphs",
        "passed": passed,
        "backend": args.backend,
        "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
        "configuration": config,
        "pairing": {
            "configuration": pairing_configuration,
            "pairing_key_sha256": pairing_key_sha256,
            "backend_excluded_from_key": True,
            "cuda_graph_scope_must_match_exactly": True,
            "fixed_window_decoder_layer_scope_is_incompatible": True,
        },
        "claim_scope": (
            f"backend-exclusive fixed-B{args.batch_size} continuous D{args.decode_steps} "
            "GPU-resident decode with structural graph-bank pre-capture; excludes "
            "model load, prefill, capture, cache restore/scrub, and preconditioning"
        ),
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
            "mode": args.trajectory_mode,
            "generated_feedback": args.trajectory_mode == GREEDY_FEEDBACK,
            "frozen_identical_hf_input_chain": args.trajectory_mode
            == FROZEN_HF_TEACHER,
            "gpu_argmax_timed_every_step": True,
            "argmax_fed_to_next_step": args.trajectory_mode == GREEDY_FEEDBACK,
            "cross_backend_pairing_requires_identical_generated_trajectory_hash": (
                args.trajectory_mode == GREEDY_FEEDBACK
            ),
            "cross_backend_pairing_requires_identical_frozen_input_chain_hash": (
                args.trajectory_mode == FROZEN_HF_TEACHER
            ),
            "teacher_inputs_sha256": sha256_tensors([teacher_inputs]),
        },
        "timed_work": {
            "continuous_decoder_steps": args.decode_steps,
            "output_tokens": args.decode_steps * args.batch_size,
            "decoder_plan_calls": args.decode_steps,
            "wrapper_plan_invocations": args.decode_steps
            * expected_counts["wrapper_count"],
            "wrapper_last_page_len_device_fills": args.decode_steps
            * expected_counts["wrapper_count"],
            "device_position_fills": args.decode_steps,
            "greedy_seed_token_device_copy": (
                1 if args.trajectory_mode == GREEDY_FEEDBACK else 0
            ),
            "token_embedding_calls": args.decode_steps,
            "embedding_to_graph_bank_input_device_copies": args.decode_steps,
            "complete_layer_graph_replays": layers * args.decode_steps,
            "captured_persistent_layer_output_writes": layers
            * args.decode_steps,
            "final_model_norm_calls": args.decode_steps,
            "lm_head_calls": args.decode_steps,
            "gpu_argmax_operations": args.decode_steps,
            "runtime_page_table_and_plan_updates_included": True,
            "prefix_plan_restored_before_timing": True,
            "first_generated_page_transition_included": True,
            "one_time_graph_bank_build_included": False,
            "planner_boundary_positions": planner_boundary_positions(
                args.context, args.decode_steps
            ),
            "host_planner_boundary_points": len(
                planner_boundary_positions(args.context, args.decode_steps)
            ),
            "flashinfer_full_plan_host_barriers_per_wrapper": len(
                planner_boundary_positions(args.context, args.decode_steps)
            ),
            "flashinfer_full_plan_calls_all_wrappers": len(
                planner_boundary_positions(args.context, args.decode_steps)
            )
            * expected_counts["wrapper_count"],
            "blocking_d2h_metadata_copies": 2
            * len(
                planner_boundary_positions(args.context, args.decode_steps)
            )
            * expected_counts["wrapper_count"],
            "blocking_d2h_metadata_copies_per_full_plan": 2,
            "host_synchronization_free_between_tokens": False,
            "host_planner_behavior": (
                "supported FlashInfer plan performs D2H metadata reads and host "
                "scheduling at each page-boundary rebuild; these stream-idle gaps "
                "are inside both CUDA-event and wall-clock intervals"
            ),
            "no_host_token_id_readback": True,
        },
        "correctness": {
            "passed": same_backend_passed and cross_reference_passed,
            "same_backend_eager_vs_graph": {
                "passed": same_backend_passed,
                "logits": eager_vs_graph_logits,
                "generated_argmax_tokens": eager_vs_graph_tokens,
                "full_mutated_cache_range": eager_vs_graph_cache,
                "every_page_close_cache_digest": eager_vs_graph_boundaries,
                "final_serving_metadata": eager_vs_graph_serving_metadata,
                "immutable_adjacent_page_canaries": canary_gates,
                "immutable_exact_prefix_timed_sample_canary": (
                    timed_prefix_canary_gate
                ),
                "immutable_exact_sink_timed_sample_canary": timed_sink_canary_gate,
                "eager_dispatch": eager_dispatch,
                "eager_attention_dispatch": eager_attention_dispatch,
                "eager_operations": eager_operations,
                "eager_gate_passed": eager_gate,
                "graph_runtime_gate": graph_runtime_gate,
                "restored_graph_repeat": {
                    "passed": graph_repeat_passed,
                    "logits": graph_repeat_logits,
                    "generated_argmax_tokens": graph_repeat_tokens,
                    "full_mutated_cache_range": graph_repeat_cache_comparison,
                    "every_page_close_cache_digest": graph_repeat_boundaries,
                    "final_serving_metadata": graph_repeat_metadata,
                    "runtime_gate": graph_repeat_runtime_gate,
                    "restore_before_repeat": True,
                    "no_restore_repeat_performed": False,
                },
            },
            "backend_vs_hf_sdpa_fp16": {
                "passed": cross_reference_passed,
                "logits": hf_vs_graph_logits,
                "generated_argmax_tokens": hf_vs_graph_tokens,
                "exact_tokens_required": args.trajectory_mode == GREEDY_FEEDBACK,
                "minimum_logits_cosine": args.min_logits_cosine,
                "minimum_top1_agreement": args.min_top1_agreement,
                "outlier_diagnostics": quality_outlier_diagnostics,
            },
            "runtime_page_finalization_and_consumption": finalization,
            "hashes": {
                "hf_logits_sha256": sha256_tensors([hf_logits_tensor]),
                "hf_generated_tokens_sha256": sha256_tensors([hf_tokens_tensor]),
                "eager_generated_tokens_sha256": sha256_tensors(
                    [eager["generated_tokens"]]
                ),
                "graph_generated_tokens_sha256": sha256_tensors(
                    [graph["generated_tokens"]]
                ),
                "restored_graph_repeat_generated_tokens_sha256": sha256_tensors(
                    [graph_repeat["generated_tokens"]]
                ),
            },
        },
        "cuda_graph_provenance": {
            **graph_provenance,
            "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
            "structure_gate_passed": graph_structure_passed,
            "page_table_allocator_invariant": (
                "monotone request-major identity allocation; every table-content "
                "change coincides with active-page-count or exact-slice identity"
            ),
            "general_allocator_requirement": (
                "eviction, compaction, or same-shape page remapping requires an "
                "explicit host page-table epoch in the plan-rebuild key"
            ),
            "timed_page_table_content_hash": False,
            "captured_model_lifecycle_invariant": (
                "model packing and CUDA placement complete before capture; model, "
                "decoder, parameter, bias, and module-buffer objects remain alive "
                "and immutable with no to(), repack, parameter replacement, or "
                "torch.cuda.empty_cache() after capture"
            ),
            "outer_worker_capture_memory_delta": {
                field: memory["after_graph_bank_preflight_and_capture"][field]
                - memory["before_graph_bank_preflight_and_capture"][field]
                for field in (
                    "torch_allocated_bytes",
                    "torch_reserved_bytes",
                    "cuda_mem_get_info_free_bytes",
                )
            },
            "one_time_exhaustive_preflight_and_graph_bank_build_wall_seconds": float(
                graph_bank_build_wall_seconds
            ),
        },
        "scheduler_capacity": capacity,
        "timing_modes": timing_modes,
        "exclusivity": {
            "fresh_process_required": True,
            "selected_persistent_backend": args.backend,
            "opposite_backend_full_gpu_cache_allocated": False,
            "hf_dynamic_cache_scope": (
                "one B1 request during coherent fixture construction only"
            ),
            "hf_dynamic_caches_released_before_decoder_construction": True,
            "cache_serialization_or_reload": False,
            "direct_destination_construction": True,
        },
        "cache_build": {
            "served_pages_per_request": served_pages,
            "allocated_pages_per_request": pages,
            "initial_pages_per_request": initial_pages,
            "mutated_generated_pages_per_request": args.decode_steps // PG.PAGE,
            "exact_ring_pages_per_request": exact_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_tail_pages_per_request": exact_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_sink_pages_per_request": args.exact_sink_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_prefix_pages_per_request": args.exact_sink_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_static_suffix_pages_per_request": (
                args.exact_static_suffix_pages
                if args.backend == "page_gauge"
                else 0
            ),
            "fixed_exact_pages_per_request": (
                args.exact_sink_pages + args.exact_static_suffix_pages
                if args.backend == "page_gauge"
                else 0
            ),
            "exact_storage_pages_per_request": (
                exact_pages
                + args.exact_sink_pages
                + args.exact_static_suffix_pages
            )
            if args.backend == "page_gauge"
            else 0,
            "exact_physical_layout": (
                "slots0..S-1=fixed prefix; slotsS..S+A-1=fixed original-"
                "prefill suffix; slotsS+A..S+A+T-1=tail ring"
                if args.backend == "page_gauge"
                and (args.exact_sink_pages or args.exact_static_suffix_pages)
                else "slots0..T-1=tail ring"
            ),
            "prefix_storage_accounting": (
                "gross FP16 allocation; prefix INT8 codes/scales remain allocated"
                if args.backend == "page_gauge" and args.exact_sink_pages
                else "no exact prefix allocation"
            ),
            "sink_storage_accounting": (
                "legacy alias: gross FP16 prefix allocation; prefix INT8 "
                "codes/scales remain allocated"
                if args.backend == "page_gauge" and args.exact_sink_pages
                else "legacy alias: no exact prefix allocation"
            ),
            "snapshot_offloaded_to_cpu_before_capture_and_timing": True,
            "full_served_mutation_range_restored_before_every_validation_and_sample": True,
            "preceding_and_following_page_canaries_checked": True,
            "following_canary_logical_page": following_canary_logical_page,
            "selected_backend_cache": cache_manifest,
            "selected_backend_cache_served_bytes_excluding_following_canary": (
                cache_manifest["total_bytes"] - following_canary_storage_bytes
            ),
            "following_canary_storage_bytes": following_canary_storage_bytes,
            "opposite_backend_full_gpu_cache_allocated": False,
            "sampled_boundary_kv_sha256": kv_sample_sha256,
            "request_records": prefill_records,
        },
        "token_source": token_provenance,
        "quality_windows": quality_windows,
        "attention_implementation": decoder.attention_implementation_provenance(),
        "model_provenance": {
            "requested_name_or_path": args.model,
            "resolved_revision": model_revision,
            "config_sha256": PREFILL.canonical_json_sha256(config_payload),
            "sampled_parameters": parameter_sample,
            "parameter_count": parameter_count,
        },
        "memory": memory,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "multiprocessor_count": int(device_properties.multi_processor_count),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "flashinfer_abi": flashinfer_abi,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "local_files_only": True,
            "orchestration_environment": BACKEND.orchestration_environment(),
        },
        "invocation": {
            "argv": sys.argv,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "cwd": str(Path.cwd()),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): BACKEND.sha256_file(path)
            for path in source_paths
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except BaseException as error:
        failure = {
            "schema_version": 2,
            "experiment": "page_gauge_backend_exclusive_sustained_dynamic_graphs",
            "passed": False,
            "backend": getattr(args, "backend", None),
            "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
            "configuration": vars(args),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "invocation": {
                "argv": sys.argv,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "cwd": str(Path.cwd()),
            },
        }
        args.output.write_text(json.dumps(failure, indent=2, default=str) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {args.output}", flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
