#!/usr/bin/env python3
"""Trace where batched PageGauge and FP16 decoder states first diverge.

This diagnostic deliberately does not modify the publication runner.  It uses
that runner's model packing, cache construction, request-major cache views,
append extension, FlashInfer planners, and eager/graph attention dispatch.  A
small copy of the layer arithmetic from ``TransformerDecoder.step`` exposes
per-[step, layer, request] checkpoints that the production API does not return.

The isolation control is designed for reproducibility: ``--batch-size 4
--isolate-request R`` first constructs the exact canonical B4 random cache and
token matrix, then runs B1 through request R's storage views and token column.
Thus a B4 row and its isolated run start from identical model weights, random
cache values, centers, quantized pages, and tokens without duplicating the B4
cache allocation.
"""

from __future__ import annotations

import argparse
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
RUNNER_PATH = ROOT / "scripts/benchmark_page_gauge_transformer.py"
DEPENDENCY_PATHS = (
    RUNNER_PATH,
    ROOT / "tests/page_gauge_append_extension.cu",
    ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
    ROOT / "scripts/benchmark_page_gauge_overheads.py",
    ROOT / "scripts/benchmark_e2e_transformer.py",
    ROOT / "scripts/page_gauge_runtime.py",
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "page_gauge_batched_correctness_trace_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PG = load_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--context", type=int, default=20480)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 4),
        default=4,
        help="Canonical source batch. This is also the active batch unless isolated.",
    )
    parser.add_argument(
        "--isolate-request",
        type=int,
        help=(
            "Run only this request from the canonical source batch. For example, "
            "--batch-size 4 --isolate-request 0 is a B1 isolation of B4 row 0."
        ),
    )
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=128)
    parser.add_argument(
        "--center-restore",
        choices=("attention_add", "projection_bias"),
        default="attention_add",
    )
    parser.add_argument(
        "--tail-attention",
        choices=("flashinfer_merge", "fused_kernel"),
        default="flashinfer_merge",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("graph", "eager"),
        default="graph",
        help="Use the same per-layer attention graphs as the production run, or eager.",
    )
    parser.add_argument("--seed", type=int, default=20260861)
    parser.add_argument("--trigger-cosine", type=float, default=0.995)
    parser.add_argument("--trigger-relative-l2", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_request_pages(
    tensor: torch.Tensor,
    source_batch_size: int,
    pages_per_request: int,
    request: int,
) -> torch.Tensor:
    """Return [L,P,...] storage for one request without a multi-GB copy."""
    expected_pages = source_batch_size * pages_per_request
    if tensor.dim() < 2 or int(tensor.shape[1]) != expected_pages:
        raise ValueError(
            f"expected page dimension {expected_pages}, got {tuple(tensor.shape)}"
        )
    shape = (
        int(tensor.shape[0]),
        source_batch_size,
        pages_per_request,
        *tensor.shape[2:],
    )
    selected = tensor.view(shape)[:, request]
    # The full [L,P,...] view is not globally contiguous because its layer
    # stride retains B4 storage.  Production passes one layer at a time; those
    # layer slices must be contiguous for the CUDA extension and FlashInfer.
    if not selected[0].is_contiguous():
        raise RuntimeError("isolated per-layer request storage is not contiguous")
    return selected


def isolate_caches(
    baseline_cache,
    gauge_cache,
    *,
    source_batch_size: int,
    pages: int,
    exact_pages: int,
    request: int,
):
    """Select an exact canonical B4 request as an active B1 cache."""
    baseline = PG.BaselineCache(
        select_request_pages(
            baseline_cache.key, source_batch_size, pages, request
        ),
        select_request_pages(
            baseline_cache.value, source_batch_size, pages, request
        ),
    )
    gauge = PG.GaugeCache(
        select_request_pages(
            gauge_cache.exact_key,
            source_batch_size,
            exact_pages,
            request,
        ),
        select_request_pages(
            gauge_cache.exact_value,
            source_batch_size,
            exact_pages,
            request,
        ),
        select_request_pages(
            gauge_cache.key_codes, source_batch_size, pages, request
        ),
        select_request_pages(
            gauge_cache.value_codes, source_batch_size, pages, request
        ),
        select_request_pages(
            gauge_cache.key_scales, source_batch_size, pages, request
        ),
        select_request_pages(
            gauge_cache.value_scales, source_batch_size, pages, request
        ),
        gauge_cache.key_center[:, request : request + 1],
        gauge_cache.value_center[:, request : request + 1],
    )
    return baseline, gauge


def row_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Compare B corresponding rows and search for accidental row permutation."""
    if reference.shape != candidate.shape or reference.dim() < 1:
        raise ValueError(
            f"checkpoint shapes differ: {reference.shape} versus {candidate.shape}"
        )
    batch_size = int(reference.shape[0])
    reference_rows = reference.detach().float().reshape(batch_size, -1)
    candidate_rows = candidate.detach().float().reshape(batch_size, -1)
    difference = candidate_rows - reference_rows
    reference_norm = torch.linalg.vector_norm(reference_rows, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate_rows, dim=1)
    difference_norm = torch.linalg.vector_norm(difference, dim=1)
    denominator = (reference_norm * candidate_norm).clamp_min(1.0e-30)
    cosine = (reference_rows * candidate_rows).sum(dim=1) / denominator
    relative_l2 = difference_norm / reference_norm.clamp_min(1.0e-30)
    max_abs = difference.abs().amax(dim=1)

    normalized_reference = F.normalize(reference_rows, p=2, dim=1, eps=1.0e-30)
    normalized_candidate = F.normalize(candidate_rows, p=2, dim=1, eps=1.0e-30)
    cross_cosine = normalized_candidate @ normalized_reference.transpose(0, 1)
    nearest = cross_cosine.argmax(dim=1)
    diagonal = cross_cosine.diagonal()
    same_request_rank = (cross_cosine > diagonal[:, None]).sum(dim=1) + 1
    return {
        "cosine": cosine.cpu().tolist(),
        "relative_l2": relative_l2.cpu().tolist(),
        "max_abs": max_abs.cpu().tolist(),
        "reference_l2": reference_norm.cpu().tolist(),
        "candidate_l2": candidate_norm.cpu().tolist(),
        "nearest_reference_request": nearest.cpu().tolist(),
        "nearest_reference_cosine": cross_cosine.amax(dim=1).cpu().tolist(),
        "same_reference_cosine": diagonal.cpu().tolist(),
        "nearest_reference_margin": (
            cross_cosine.amax(dim=1) - diagonal
        ).cpu().tolist(),
        "same_request_rank": same_request_rank.cpu().tolist(),
        "cross_request_cosine": cross_cosine.cpu().tolist(),
    }


def record_checkpoint(
    records: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
    *,
    stage: str,
    step: int,
    position: int,
    layer: int,
    source_requests: list[int],
    reference: torch.Tensor,
    candidate: torch.Tensor,
    trigger_cosine: float,
    trigger_relative_l2: float,
) -> None:
    comparison = row_comparison(reference, candidate)
    cross_request_cosine = comparison.pop("cross_request_cosine")
    for active_request, source_request in enumerate(source_requests):
        record = {
            "step": step,
            "position": position,
            "layer": layer,
            "request": active_request,
            "source_request": source_request,
            "stage": stage,
            **{
                name: values[active_request]
                for name, values in comparison.items()
            },
        }
        records.append(record)
        triggered_by = []
        if record["cosine"] < trigger_cosine:
            triggered_by.append("cosine")
        if record["relative_l2"] > trigger_relative_l2:
            triggered_by.append("relative_l2")
        # Equal hidden rows (for example, duplicate input token IDs at layer
        # zero) can make argmax choose the first request.  Only treat another
        # row as evidence when it is strictly better beyond roundoff.
        if (
            record["nearest_reference_request"] != active_request
            and record["nearest_reference_margin"] > 1.0e-6
        ):
            triggered_by.append("nearest_reference_request")
        if triggered_by:
            triggers.append(
                {
                    **record,
                    "triggered_by": triggered_by,
                    "active_request_order": source_requests,
                    "cross_request_cosine": cross_request_cosine,
                }
            )


def advance_layer(decoder, layer_index: int, hidden: torch.Tensor, position: int):
    """One production decoder layer, returning non-mutating checkpoints."""
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
    projected = F.linear(
        attended.reshape(decoder.batch_size, -1),
        attention.o_proj.weight,
        decoder.output_projection_biases[layer_index],
    )
    post_attention_hidden = residual + projected
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
        "post_attention_hidden": post_attention_hidden,
        "output_hidden": output_hidden,
    }


def topk_summary(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    source_requests: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    top_k = min(top_k, int(reference_logits.shape[-1]))
    reference_values, reference_ids = reference_logits.topk(top_k, dim=-1)
    candidate_values, candidate_ids = candidate_logits.topk(top_k, dim=-1)
    reference_pair = reference_logits.topk(min(2, reference_logits.shape[-1]), dim=-1)
    candidate_pair = candidate_logits.topk(min(2, candidate_logits.shape[-1]), dim=-1)
    summaries = []
    for request, source_request in enumerate(source_requests):
        reference_top1 = int(reference_pair.indices[request, 0].item())
        candidate_top1 = int(candidate_pair.indices[request, 0].item())
        reference_margin = (
            float(
                (
                    reference_pair.values[request, 0]
                    - reference_pair.values[request, 1]
                ).item()
            )
            if reference_pair.values.shape[1] > 1
            else float("inf")
        )
        candidate_margin = (
            float(
                (
                    candidate_pair.values[request, 0]
                    - candidate_pair.values[request, 1]
                ).item()
            )
            if candidate_pair.values.shape[1] > 1
            else float("inf")
        )
        candidate_at_reference = candidate_logits[request, reference_top1]
        if candidate_top1 == reference_top1 and candidate_pair.values.shape[1] > 1:
            candidate_competitor = candidate_pair.values[request, 1]
        else:
            candidate_competitor = candidate_pair.values[request, 0]
        summaries.append(
            {
                "request": request,
                "source_request": source_request,
                "reference_top1_id": reference_top1,
                "candidate_top1_id": candidate_top1,
                "top1_match": reference_top1 == candidate_top1,
                "reference_top1_margin": reference_margin,
                "candidate_top1_margin": candidate_margin,
                "reference_top1_advantage_in_candidate": float(
                    (candidate_at_reference - candidate_competitor).item()
                ),
                "reference_topk_ids": reference_ids[request].cpu().tolist(),
                "reference_topk_values": reference_values[request].cpu().tolist(),
                "candidate_topk_ids": candidate_ids[request].cpu().tolist(),
                "candidate_topk_values": candidate_values[request].cpu().tolist(),
            }
        )
    return summaries


@torch.inference_mode()
def run_trace(
    model,
    baseline,
    candidate,
    tokens: torch.Tensor,
    context: int,
    source_requests: list[int],
    trigger_cosine: float,
    trigger_relative_l2: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoints: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    batch_size = len(source_requests)
    for step_index in range(int(tokens.shape[0])):
        position = context + step_index
        token = tokens[step_index]
        baseline.plan(position + 1)
        candidate.plan(position + 1)
        # Two calls match the production A/B executions. Embedding lookup is
        # deterministic, and the layer-0 checkpoint verifies that directly.
        reference_hidden = model.model.embed_tokens(
            token.reshape(batch_size, 1)
        )[:, 0]
        candidate_hidden = model.model.embed_tokens(
            token.reshape(batch_size, 1)
        )[:, 0]
        for layer_index in range(baseline.layers):
            reference_stages = advance_layer(
                baseline, layer_index, reference_hidden, position
            )
            candidate_stages = advance_layer(
                candidate, layer_index, candidate_hidden, position
            )
            for stage in (
                "input_hidden",
                "attention_output",
                "post_attention_hidden",
                "output_hidden",
            ):
                record_checkpoint(
                    checkpoints,
                    triggers,
                    stage=stage,
                    step=step_index,
                    position=position,
                    layer=layer_index,
                    source_requests=source_requests,
                    reference=reference_stages[stage],
                    candidate=candidate_stages[stage],
                    trigger_cosine=trigger_cosine,
                    trigger_relative_l2=trigger_relative_l2,
                )
            reference_hidden = reference_stages["output_hidden"]
            candidate_hidden = candidate_stages["output_hidden"]
        reference_logits = model.lm_head(model.model.norm(reference_hidden)).float()
        candidate_logits = model.lm_head(model.model.norm(candidate_hidden)).float()
        logits_comparison = row_comparison(reference_logits, candidate_logits)
        logits_comparison.pop("cross_request_cosine")
        steps.append(
            {
                "step": step_index,
                "position": position,
                "token_ids": token.cpu().tolist(),
                "logits": logits_comparison,
                "topk": topk_summary(
                    reference_logits, candidate_logits, source_requests, top_k
                ),
            }
        )
        print(
            f"step={step_index:02d} position={position} "
            f"minimum_logits_cosine={min(logits_comparison['cosine']):.6f}",
            flush=True,
        )
    return checkpoints, triggers, steps


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.context <= args.exact_tail:
        raise SystemExit("context must exceed the exact tail")
    if args.context % PG.PAGE or args.exact_tail % PG.PAGE:
        raise SystemExit("context and exact tail must be page aligned")
    if args.decode_steps <= 0:
        raise SystemExit("decode steps must be positive")
    if args.attention_mode == "graph" and args.decode_steps > PG.PAGE:
        raise SystemExit(
            "one captured plan bucket covers at most the next 16 aligned tokens"
        )
    if not 0 < args.trigger_cosine <= 1:
        raise SystemExit("trigger cosine must lie in (0,1]")
    if args.trigger_relative_l2 < 0:
        raise SystemExit("trigger relative L2 must be non-negative")
    if args.top_k <= 0:
        raise SystemExit("top-k must be positive")
    if args.isolate_request is not None and not (
        0 <= args.isolate_request < args.batch_size
    ):
        raise SystemExit("isolated request must lie within the canonical batch")
    active_batch_size = 1 if args.isolate_request is not None else args.batch_size
    if active_batch_size > 1 and args.center_restore == "projection_bias":
        raise SystemExit(
            "batched request-specific centers require --center-restore attention_add"
        )
    if (
        args.tail_attention == "fused_kernel"
        and args.center_restore != "attention_add"
    ):
        raise SystemExit("fused tail requires --center-restore attention_add")


def main() -> None:
    args = parse_args()
    validate_args(args)
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    model_args = SimpleNamespace(
        model=args.model,
        context=args.context,
        decode_steps=args.decode_steps,
        local_files_only=args.local_files_only,
    )
    print("Loading packed model and current production kernels...", flush=True)
    model = PG.load_model(model_args)
    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    layers, _, hkv, hidden = PG.BASE_E2E.check_model(model)
    max_context = args.context + args.decode_steps + 1
    pages = math.ceil(max_context / PG.PAGE)
    initial_pages = args.context // PG.PAGE
    exact_pages = args.exact_tail // PG.PAGE
    with torch.inference_mode():
        positions = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(probe, positions)
        if rope_cos.dim() == 3:
            rope_cos, rope_sin = rope_cos[0], rope_sin[0]
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()

    print(
        f"Constructing canonical B{args.batch_size} caches for seed={args.seed}...",
        flush=True,
    )
    source_baseline_cache, source_gauge_cache = PG.build_caches(
        layers,
        pages,
        initial_pages,
        exact_pages,
        hkv,
        args.seed + 1000,
        args.batch_size,
    )
    source_cache_storage = PG.cache_storage(
        source_baseline_cache, source_gauge_cache
    )
    if args.isolate_request is None:
        baseline_cache, gauge_cache = source_baseline_cache, source_gauge_cache
        active_batch_size = args.batch_size
        source_requests = list(range(args.batch_size))
    else:
        baseline_cache, gauge_cache = isolate_caches(
            source_baseline_cache,
            source_gauge_cache,
            source_batch_size=args.batch_size,
            pages=pages,
            exact_pages=exact_pages,
            request=args.isolate_request,
        )
        active_batch_size = 1
        source_requests = [args.isolate_request]
    active_cache_storage = PG.cache_storage(baseline_cache, gauge_cache)

    common = (model, flashinfer, append_extension)
    baseline = PG.TransformerDecoder(
        *common,
        "flashinfer_fp16",
        baseline_cache,
        max_context,
        args.exact_tail,
        args.baseline_split_pages,
        args.candidate_split_pages,
        rope_cos,
        rope_sin,
        args.center_restore,
        args.tail_attention,
        active_batch_size,
    )
    candidate = PG.TransformerDecoder(
        *common,
        "page_gauge",
        gauge_cache,
        max_context,
        args.exact_tail,
        args.baseline_split_pages,
        args.candidate_split_pages,
        rope_cos,
        rope_sin,
        args.center_restore,
        args.tail_attention,
        active_batch_size,
    )

    generator = torch.Generator().manual_seed(args.seed)
    source_tokens = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.decode_steps + 1, args.batch_size),
        generator=generator,
        dtype=torch.long,
    )
    if args.isolate_request is None:
        active_tokens = source_tokens[: args.decode_steps]
    else:
        active_tokens = source_tokens[
            : args.decode_steps, args.isolate_request : args.isolate_request + 1
        ]
    active_tokens = active_tokens.to(device="cuda")

    if args.attention_mode == "graph":
        print("Capturing matched production per-layer attention graphs...", flush=True)
        baseline.capture_attention_graphs(args.context + 1)
        candidate.capture_attention_graphs(args.context + 1)
    baseline.reset_attention_dispatch_counts()
    candidate.reset_attention_dispatch_counts()
    checkpoints, triggers, steps = run_trace(
        model,
        baseline,
        candidate,
        active_tokens,
        args.context,
        source_requests,
        args.trigger_cosine,
        args.trigger_relative_l2,
        args.top_k,
    )
    torch.cuda.synchronize()

    source_paths = (*DEPENDENCY_PATHS, Path(__file__))
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_layer_interleaved_batched_correctness_trace",
        "claim_scope": "diagnostic correctness trace; no timing claim",
        "configuration": {
            **{key: value for key, value in vars(args).items() if key != "output"},
            "output": str(args.output),
            "source_batch_size": args.batch_size,
            "active_batch_size": active_batch_size,
            "active_source_requests": source_requests,
            "source_token_matrix": source_tokens.cpu().tolist(),
        },
        "protocol": {
            "checkpoint_index": "flat records keyed by [step, layer, request, stage]",
            "checkpoint_stages": [
                "input_hidden",
                "attention_output",
                "post_attention_hidden",
                "output_hidden",
            ],
            "layer_execution_order": (
                "FP16 layer N then PageGauge layer N, followed by checkpoint "
                "materialization; caches and decoder scratch are backend-local"
            ),
            "production_components_reused": [
                "load_model and packed projections",
                "build_caches",
                "TransformerDecoder.plan",
                "TransformerDecoder.append",
                "TransformerDecoder.attention",
                "attention CUDA-graph capture and signature guard",
            ],
            "duplicated_for_observability": (
                "the arithmetic in TransformerDecoder.step around append/attention"
            ),
            "cache_initialization": (
                "publication runner build_caches: independent Gaussian K/V scaled "
                "by 0.35, seed+1000"
            ),
            "timing_valid": False,
        },
        "controls": {
            "isolation": {
                "enabled": args.isolate_request is not None,
                "source_request": args.isolate_request,
                "method": (
                    "zero-copy request-major views of a canonical source-batch "
                    "cache plus the corresponding canonical token column"
                ),
                "retains_full_source_cache_allocation": (
                    args.isolate_request is not None
                ),
            },
            "permutation": {
                "implemented": False,
                "reason": (
                    "the production append ABI derives request-major physical "
                    "pages from the active request index; a faithful permutation "
                    "would require another full cache copy or a production ABI "
                    "change, neither is appropriate for this standalone diagnostic"
                ),
            },
            "cross_request_nearest_neighbor": (
                "every checkpoint compares each candidate row with every FP16 row; "
                "nearest row, rank of the same row, and trigger matrices are retained"
            ),
        },
        "thresholds": {
            "cosine": args.trigger_cosine,
            "relative_l2": args.trigger_relative_l2,
            "cross_request_trigger_margin": 1.0e-6,
        },
        "checkpoints": checkpoints,
        "triggered_checkpoints": triggers,
        "step_summaries": steps,
        "attention_dispatch": {
            "flashinfer_fp16": baseline.attention_dispatch_counts(),
            "page_gauge": candidate.attention_dispatch_counts(),
        },
        "cache_storage": {
            "canonical_source": source_cache_storage,
            "active_view_accounting": active_cache_storage,
            "note": (
                "active view accounting excludes inactive elements by numel, but an "
                "isolation run retains the full canonical source allocation"
            ),
        },
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
            for path in source_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"wrote {args.output} with {len(checkpoints)} checkpoints and "
        f"{len(triggers)} triggers",
        flush=True,
    )


if __name__ == "__main__":
    main()
