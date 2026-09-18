#!/usr/bin/env python3
"""Falsification probe for certified hierarchical KV aggregation.

This script deliberately does not benchmark a CUDA kernel.  It extracts real
post-RoPE queries and cached keys/values from a pretrained GQA model, replaces
old-token clusters by aggregate nodes, and asks two prior questions:

1. Can interval arithmetic certify the aggregate attention output?
2. After charging summary metadata and GQA-head union, are at least 80% of the
   dense K/V bytes avoided often enough to justify implementing a kernel?

The certificate uses only data that an aggregate node can store: a key
centroid, coordinate-wise key bounds, a token count, and positive/negative
value sums.  Refinement converts an uncertain aggregate node back into exact
token attention.  The recent window is always exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = Path(
    "/home/filliones/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B/snapshots"
)


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--corpus", choices=("paper", "code", "mixed", "random"), default="paper"
    )
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--layers", type=str, default="0,9,18,27")
    parser.add_argument("--cluster-counts", type=str, default="64,128")
    parser.add_argument("--bound-modes", type=str, default="box,ball,oracle")
    parser.add_argument("--relative-tolerances", type=str, default="0.01,0.005,0.001")
    parser.add_argument("--kmeans-iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--required-byte-reduction", type=float, default=0.80)
    parser.add_argument("--required-fast-path-rate", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    snapshots = sorted(path for path in DEFAULT_MODEL_ROOT.glob("*") if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No local model snapshot under {DEFAULT_MODEL_ROOT}")
    return snapshots[-1]


def candidate_text_files(corpus: str) -> list[Path]:
    paper_files = [
        REPO_ROOT / "paper/persistentkv_qifk_blind.tex",
        REPO_ROOT / "paper/persistentkv_mlsys_workshop.tex",
        REPO_ROOT / "paper/persistent_kv_attention_proposal.tex",
    ]
    code_files = [
        REPO_ROOT / "kernels/persistentkv_attention.cuh",
        REPO_ROOT / "scripts/run_coda_experiments.py",
        REPO_ROOT / "scripts/benchmark_e2e_transformer.py",
        REPO_ROOT / "tests/benchmark_serving_trace.py",
        REPO_ROOT / "persistentkv/router.py",
    ]
    if corpus == "paper":
        return paper_files
    if corpus == "code":
        return code_files
    if corpus == "mixed":
        return paper_files + code_files
    return []


def load_corpus_text(corpus: str) -> tuple[str, list[str]]:
    if corpus == "random":
        return "", []
    files = candidate_text_files(corpus)
    pieces: list[str] = []
    used: list[str] = []
    for path in files:
        if not path.exists():
            continue
        pieces.append(f"\n\nFILE {path.name}\n\n{path.read_text(errors='replace')}")
        used.append(str(path))
    if not pieces:
        raise FileNotFoundError(f"No source text found for corpus={corpus}")
    return "".join(pieces), used


def make_input_ids(
    tokenizer: Any,
    corpus: str,
    context: int,
    offset: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    text, files = load_corpus_text(corpus)
    if corpus == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vocab_size = int(tokenizer.vocab_size)
        ids = torch.randint(0, vocab_size, (context,), generator=generator)
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        return ids.unsqueeze(0), {
            "corpus": corpus,
            "files": files,
            "available_tokens": context,
            "repeated": False,
            "sha256": digest,
        }

    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    available = int(encoded.numel())
    required = offset + context
    repeated = False
    if available < required:
        repeated = True
        repeats = math.ceil(required / max(available, 1))
        encoded = encoded.repeat(repeats)
    ids = encoded[offset : offset + context].contiguous()
    digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    return ids.unsqueeze(0), {
        "corpus": corpus,
        "files": files,
        "available_tokens": available,
        "repeated": repeated,
        "sha256": digest,
    }


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class QueryCapture:
    def __init__(self, model: Any, layers: Iterable[int]) -> None:
        self.latest: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_idx in layers:
            module = model.model.layers[layer_idx].self_attn
            self.handles.append(
                module.register_forward_pre_hook(
                    self._make_hook(layer_idx), with_kwargs=True
                )
            )

    def _make_hook(self, layer_idx: int):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden is None or position_embeddings is None:
                raise RuntimeError("Qwen attention hook did not receive expected inputs")
            cos, sin = position_embeddings
            input_shape = hidden.shape[:-1]
            q = module.q_proj(hidden).view(
                *input_shape, module.config.num_attention_heads, module.head_dim
            )
            q = q.transpose(1, 2)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q = (q * cos) + (rotate_half(q) * sin)
            self.latest[layer_idx] = q[:, :, -1, :].detach().float().cpu()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def extract_real_qkv(
    model_path: Path,
    input_ids: torch.Tensor,
    layers: list[int],
    decode_steps: int,
    prefill_chunk: int,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list[dict[int, torch.Tensor]], list[int], dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval().cuda()
    config = model.config
    if max(layers) >= config.num_hidden_layers:
        raise ValueError(
            f"Layer {max(layers)} is invalid for {config.num_hidden_layers} layers"
        )

    device_ids = input_ids.cuda(non_blocking=False)
    past = None
    output = None
    with torch.inference_mode():
        for begin in range(0, device_ids.shape[1], prefill_chunk):
            chunk = device_ids[:, begin : begin + prefill_chunk]
            output = model(
                input_ids=chunk,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            print(
                f"prefill {min(begin + prefill_chunk, device_ids.shape[1])}/"
                f"{device_ids.shape[1]}",
                flush=True,
            )

        assert output is not None and past is not None
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture = QueryCapture(model, layers)
        queries: list[dict[int, torch.Tensor]] = []
        context_lengths: list[int] = []
        try:
            for step in range(decode_steps):
                capture.latest.clear()
                output = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
                past = output.past_key_values
                missing = set(layers) - set(capture.latest)
                if missing:
                    raise RuntimeError(f"Missing captured queries for layers {sorted(missing)}")
                queries.append({idx: capture.latest[idx].clone() for idx in layers})
                context_lengths.append(int(past.get_seq_length()))
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                print(f"decode capture {step + 1}/{decode_steps}", flush=True)
        finally:
            capture.close()

    kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in layers:
        cache_layer = past.layers[layer_idx]
        kv[layer_idx] = (
            cache_layer.keys[0].detach().half().cpu().contiguous(),
            cache_layer.values[0].detach().half().cpu().contiguous(),
        )

    metadata = {
        "model_type": config.model_type,
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "gqa_group_size": int(config.num_attention_heads // config.num_key_value_heads),
        "model_load_and_capture_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }

    del model, past, output, device_ids
    torch.cuda.empty_cache()
    return kv, queries, context_lengths, metadata


def kmeans(
    x: torch.Tensor,
    clusters: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if x.ndim != 2 or clusters < 1 or clusters > x.shape[0]:
        raise ValueError(f"Invalid k-means shape={tuple(x.shape)}, clusters={clusters}")
    x = x.float()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    initial = torch.randperm(x.shape[0], generator=generator, device=x.device)[:clusters]
    centers = x[initial].clone()
    min_distance = None
    labels = None
    for _ in range(iterations):
        x_norm = (x * x).sum(dim=1, keepdim=True)
        c_norm = (centers * centers).sum(dim=1).unsqueeze(0)
        distance = (x_norm + c_norm - 2.0 * (x @ centers.T)).clamp_min_(0.0)
        min_distance, labels = distance.min(dim=1)
        counts = torch.bincount(labels, minlength=clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = (~nonempty).nonzero(as_tuple=False).flatten()
        if empty.numel():
            farthest = torch.argsort(min_distance, descending=True)
            for empty_idx, point_idx in zip(empty.tolist(), farthest.tolist()):
                centers[empty_idx] = x[point_idx]
    x_norm = (x * x).sum(dim=1, keepdim=True)
    c_norm = (centers * centers).sum(dim=1).unsqueeze(0)
    distance = (x_norm + c_norm - 2.0 * (x @ centers.T)).clamp_min_(0.0)
    min_distance, labels = distance.min(dim=1)
    counts = torch.bincount(labels, minlength=clusters)
    if bool((counts == 0).any()):
        raise RuntimeError("k-means produced empty clusters after repair")
    distortion = float(min_distance.mean().sqrt().item())
    return labels, centers, distortion


@dataclass
class ClusterSummary:
    labels: torch.Tensor
    centers: torch.Tensor
    counts: torch.Tensor
    key_min: torch.Tensor
    key_max: torch.Tensor
    key_radius: torch.Tensor
    value_positive: torch.Tensor
    value_negative: torch.Tensor
    distortion: float


def build_cluster_summary(
    keys: torch.Tensor,
    values: torch.Tensor,
    clusters: int,
    iterations: int,
    seed: int,
) -> ClusterSummary:
    labels, centers, distortion = kmeans(keys, clusters, iterations, seed)
    dim = keys.shape[1]
    expanded = labels[:, None].expand(-1, dim)
    key_min = torch.full(
        (clusters, dim), float("inf"), dtype=torch.float32, device=keys.device
    )
    key_max = torch.full(
        (clusters, dim), -float("inf"), dtype=torch.float32, device=keys.device
    )
    key_min.scatter_reduce_(0, expanded, keys.float(), reduce="amin", include_self=True)
    key_max.scatter_reduce_(0, expanded, keys.float(), reduce="amax", include_self=True)
    residual_norm = (keys.float() - centers[labels]).norm(dim=1)
    key_radius = torch.zeros(clusters, dtype=torch.float32, device=keys.device)
    key_radius.scatter_reduce_(
        0, labels, residual_norm, reduce="amax", include_self=True
    )
    value_positive = torch.zeros(
        (clusters, dim), dtype=torch.float64, device=keys.device
    )
    value_negative = torch.zeros_like(value_positive)
    value64 = values.double()
    value_positive.index_add_(0, labels, value64.clamp_min(0.0))
    value_negative.index_add_(0, labels, value64.clamp_max(0.0))
    counts = torch.bincount(labels, minlength=clusters).double()
    return ClusterSummary(
        labels=labels,
        centers=centers,
        counts=counts,
        key_min=key_min,
        key_max=key_max,
        key_radius=key_radius,
        value_positive=value_positive,
        value_negative=value_negative,
        distortion=distortion,
    )


@dataclass
class HeadCertificateState:
    aggregate_z: torch.Tensor
    aggregate_a: torch.Tensor
    exact_z: torch.Tensor
    exact_a: torch.Tensor
    lower_z: torch.Tensor
    upper_z: torch.Tensor
    lower_a: torch.Tensor
    upper_a: torch.Tensor
    local_z: torch.Tensor
    local_a: torch.Tensor
    dense_output: torch.Tensor
    risk_per_token: torch.Tensor


def build_head_state(
    query: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    summary: ClusterSummary,
    bound_mode: str = "box",
) -> HeadCertificateState:
    dim = old_keys.shape[1]
    scaled_query = query.float() / math.sqrt(dim)
    old_scores = old_keys.float() @ scaled_query
    local_scores = local_keys.float() @ scaled_query
    center_scores = summary.centers.float() @ scaled_query
    if bound_mode == "box":
        product_min = summary.key_min * scaled_query
        product_max = summary.key_max * scaled_query
        score_lower = torch.minimum(product_min, product_max).sum(dim=1)
        score_upper = torch.maximum(product_min, product_max).sum(dim=1)
    elif bound_mode == "ball":
        delta = scaled_query.norm() * summary.key_radius
        score_lower = center_scores - delta
        score_upper = center_scores + delta
    elif bound_mode == "oracle":
        # Diagnostic lower bound on certificate difficulty.  These ranges are
        # exact for this query, so obtaining them online requires reading every
        # old key and cannot itself produce a large-margin implementation.
        clusters = summary.centers.shape[0]
        score_lower = torch.full(
            (clusters,), float("inf"), dtype=torch.float32, device=old_keys.device
        )
        score_upper = torch.full(
            (clusters,), -float("inf"), dtype=torch.float32, device=old_keys.device
        )
        score_lower.scatter_reduce_(
            0, summary.labels, old_scores, reduce="amin", include_self=True
        )
        score_upper.scatter_reduce_(
            0, summary.labels, old_scores, reduce="amax", include_self=True
        )
    else:
        raise ValueError(f"Unknown bound mode: {bound_mode}")
    shift_candidates = [float(score_upper.max().item()), float(old_scores.max().item())]
    if local_scores.numel():
        shift_candidates.append(float(local_scores.max().item()))
    shift = max(shift_candidates)

    old_weight = torch.exp(old_scores.double() - shift)
    local_weight = torch.exp(local_scores.double() - shift)
    center_weight = torch.exp(center_scores.double() - shift)
    lower_weight = torch.exp(score_lower.double() - shift)
    upper_weight = torch.exp(score_upper.double() - shift)

    clusters = summary.centers.shape[0]
    exact_z = torch.zeros(clusters, dtype=torch.float64, device=old_keys.device)
    exact_z.index_add_(0, summary.labels, old_weight)
    exact_a = torch.zeros(
        (clusters, dim), dtype=torch.float64, device=old_keys.device
    )
    exact_a.index_add_(0, summary.labels, old_weight[:, None] * old_values.double())

    value_sum = summary.value_positive + summary.value_negative
    aggregate_z = center_weight * summary.counts
    aggregate_a = center_weight[:, None] * value_sum
    lower_z = lower_weight * summary.counts
    upper_z = upper_weight * summary.counts
    lower_a = (
        lower_weight[:, None] * summary.value_positive
        + upper_weight[:, None] * summary.value_negative
    )
    upper_a = (
        upper_weight[:, None] * summary.value_positive
        + lower_weight[:, None] * summary.value_negative
    )

    local_z = local_weight.sum()
    local_a = (local_weight[:, None] * local_values.double()).sum(dim=0)
    dense_z = exact_z.sum() + local_z
    dense_a = exact_a.sum(dim=0) + local_a
    dense_output = dense_a / dense_z

    all_aggregate_output = (aggregate_a.sum(dim=0) + local_a) / (
        aggregate_z.sum() + local_z
    )
    numerator_width = (upper_a - lower_a).norm(dim=1)
    denominator_width = upper_z - lower_z
    risk = numerator_width + all_aggregate_output.norm() * denominator_width
    risk_per_token = risk / summary.counts.clamp_min(1.0)
    return HeadCertificateState(
        aggregate_z=aggregate_z,
        aggregate_a=aggregate_a,
        exact_z=exact_z,
        exact_a=exact_a,
        lower_z=lower_z,
        upper_z=upper_z,
        lower_a=lower_a,
        upper_a=upper_a,
        local_z=local_z,
        local_a=local_a,
        dense_output=dense_output,
        risk_per_token=risk_per_token,
    )


def divide_interval(
    numerator_lower: torch.Tensor,
    numerator_upper: torch.Tensor,
    denominator_lower: torch.Tensor,
    denominator_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(denominator_lower.item()) <= 0.0:
        inf = torch.full_like(numerator_lower, float("inf"))
        return -inf, inf
    candidates = torch.stack(
        (
            numerator_lower / denominator_lower,
            numerator_lower / denominator_upper,
            numerator_upper / denominator_lower,
            numerator_upper / denominator_upper,
        ),
        dim=0,
    )
    return candidates.min(dim=0).values, candidates.max(dim=0).values


def evaluate_mask(
    state: HeadCertificateState, exact_mask: torch.Tensor
) -> dict[str, Any]:
    z = torch.where(exact_mask, state.exact_z, state.aggregate_z).sum() + state.local_z
    a = torch.where(
        exact_mask[:, None], state.exact_a, state.aggregate_a
    ).sum(dim=0) + state.local_a
    output = a / z

    lower_z = torch.where(exact_mask, state.exact_z, state.lower_z).sum() + state.local_z
    upper_z = torch.where(exact_mask, state.exact_z, state.upper_z).sum() + state.local_z
    lower_a = torch.where(
        exact_mask[:, None], state.exact_a, state.lower_a
    ).sum(dim=0) + state.local_a
    upper_a = torch.where(
        exact_mask[:, None], state.exact_a, state.upper_a
    ).sum(dim=0) + state.local_a
    output_lower, output_upper = divide_interval(lower_a, upper_a, lower_z, upper_z)
    component_bound = torch.maximum(
        (output - output_lower).abs(), (output_upper - output).abs()
    )
    # Score/bound construction starts from FP32 tensors, and CUDA reductions do
    # not use directed rounding.  Inflate the real-arithmetic interval by a
    # small, explicit FP32 allowance so a numerically collapsed interval cannot
    # be mistaken for a zero-error certificate.  This is negligible relative
    # to the screening tolerances; a production kernel would need an operation-
    # level rounding proof instead of this conservative probe allowance.
    fp32_eps = torch.finfo(torch.float32).eps
    roundoff_component = (
        128.0
        * fp32_eps
        * torch.maximum(
            torch.maximum(output.abs(), state.dense_output.abs()),
            torch.maximum(output_lower.abs(), output_upper.abs()),
        ).clamp_min(1e-12)
        + 1e-8
    )
    interval_l2 = component_bound.norm()
    roundoff_l2 = roundoff_component.norm()
    certificate_l2 = interval_l2 + roundoff_l2
    observed_l2 = (output - state.dense_output).norm()
    output_norm = output.norm().clamp_min(1e-12)
    dense_norm = state.dense_output.norm().clamp_min(1e-12)
    containment_slack = torch.maximum(
        output_lower - state.dense_output - roundoff_component,
        state.dense_output - output_upper - roundoff_component,
    ).max()
    if float(containment_slack.item()) > 1e-12:
        raise AssertionError(
            f"Dense output escaped interval by {float(containment_slack.item()):.3e}"
        )
    if float(observed_l2.item()) > float(certificate_l2.item()) + 1e-12:
        raise AssertionError(
            "Observed error exceeded certificate: "
            f"{float(observed_l2.item()):.3e} > {float(certificate_l2.item()):.3e}"
        )
    return {
        "output": output,
        "interval_l2": float(interval_l2.item()),
        "roundoff_l2": float(roundoff_l2.item()),
        "certificate_l2": float(certificate_l2.item()),
        "certificate_relative_to_approx": float((certificate_l2 / output_norm).item()),
        "observed_l2": float(observed_l2.item()),
        "observed_relative_to_dense": float((observed_l2 / dense_norm).item()),
        "dense_norm": float(dense_norm.item()),
    }


def summary_bytes(clusters: int, dim: int, bound_mode: str = "box") -> int:
    # All modes store FP16 centroids, FP32 signed V sums, and int32 counts.
    base = (dim * 2) + (2 * dim * 4) + 4
    if bound_mode == "box":
        per_cluster = base + (2 * dim * 2)  # FP16 coordinate min/max.
    elif bound_mode == "ball":
        per_cluster = base + 4  # FP32 radius.
    elif bound_mode == "oracle":
        per_cluster = base + 8  # Query-specific FP32 score min/max.
    else:
        raise ValueError(f"Unknown bound mode: {bound_mode}")
    return clusters * per_cluster


def byte_accounting(
    exact_mask: torch.Tensor,
    counts: torch.Tensor,
    old_tokens: int,
    local_tokens: int,
    clusters: int,
    dim: int,
    element_bytes: int = 2,
    bound_mode: str = "box",
) -> dict[str, Any]:
    exact_old_tokens = int(counts[exact_mask].sum().item())
    total_tokens = old_tokens + local_tokens
    kv_per_token = 2 * dim * element_bytes
    dense_bytes = total_tokens * kv_per_token
    metadata_bytes = summary_bytes(clusters, dim, bound_mode)
    bound_acquisition_bytes = old_tokens * dim * element_bytes if bound_mode == "oracle" else 0
    gather_index_bytes = exact_old_tokens * 4
    candidate_bytes = (
        (exact_old_tokens + local_tokens) * kv_per_token
        + metadata_bytes
        + gather_index_bytes
        + bound_acquisition_bytes
    )
    raw_fraction = candidate_bytes / dense_bytes
    return {
        "exact_old_tokens": exact_old_tokens,
        "exact_old_token_fraction": exact_old_tokens / max(old_tokens, 1),
        "local_tokens": local_tokens,
        "dense_bytes": dense_bytes,
        "summary_bytes": metadata_bytes,
        "gather_index_bytes": gather_index_bytes,
        "bound_acquisition_bytes": bound_acquisition_bytes,
        "candidate_bytes": candidate_bytes,
        "raw_candidate_byte_fraction": raw_fraction,
        "effective_byte_fraction_with_dense_fallback": min(raw_fraction, 1.0),
        "optimistic_byte_speedup": 1.0 / min(raw_fraction, 1.0),
    }


def refine_group(
    states: list[HeadCertificateState],
    counts: torch.Tensor,
    tolerance: float,
    old_tokens: int,
    local_tokens: int,
    dim: int,
    enforce_dense_fallback: bool,
    bound_mode: str = "box",
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    clusters = int(counts.numel())
    mask = torch.zeros(clusters, dtype=torch.bool, device=counts.device)
    initial = [evaluate_mask(state, mask) for state in states]
    metrics = initial
    fallback_dense = False
    while True:
        failing = [
            idx
            for idx, metric in enumerate(metrics)
            if metric["certificate_relative_to_approx"] > tolerance
        ]
        accounting = byte_accounting(
            mask, counts, old_tokens, local_tokens, clusters, dim, bound_mode=bound_mode
        )
        if not failing:
            break
        if enforce_dense_fallback and accounting["raw_candidate_byte_fraction"] >= 1.0:
            fallback_dense = True
            break
        available = ~mask
        if not bool(available.any()):
            fallback_dense = True
            break
        risk = torch.stack([states[idx].risk_per_token for idx in failing]).amax(dim=0)
        risk = risk.masked_fill(~available, -float("inf"))
        chosen = int(risk.argmax().item())
        mask[chosen] = True
        metrics = [evaluate_mask(state, mask) for state in states]

    accounting = byte_accounting(
        mask, counts, old_tokens, local_tokens, clusters, dim, bound_mode=bound_mode
    )
    if fallback_dense:
        returned = [
            {
                **metric,
                "certificate_l2": 0.0,
                "certificate_relative_to_approx": 0.0,
                "observed_l2": 0.0,
                "observed_relative_to_dense": 0.0,
            }
            for metric in metrics
        ]
        accounting["effective_byte_fraction_with_dense_fallback"] = 1.0
        accounting["optimistic_byte_speedup"] = 1.0
    else:
        returned = metrics
    accounting.update(
        {
            "fallback_dense": fallback_dense,
            "fast_path": not fallback_dense,
            "exact_clusters": int(mask.sum().item()),
            "aggregate_clusters": int((~mask).sum().item()),
            "initial_max_certificate_relative": max(
                item["certificate_relative_to_approx"] for item in initial
            ),
            "initial_max_observed_relative": max(
                item["observed_relative_to_dense"] for item in initial
            ),
            "last_candidate_max_certificate_relative": max(
                item["certificate_relative_to_approx"] for item in metrics
            ),
            "last_candidate_max_observed_relative": max(
                item["observed_relative_to_dense"] for item in metrics
            ),
        }
    )
    return mask, returned, accounting


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def aggregate_records(
    records: list[dict[str, Any]],
    cluster_counts: list[int],
    tolerances: list[float],
    bound_modes: list[str],
    required_reduction: float,
    required_fast_rate: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bound_mode in bound_modes:
      for clusters in cluster_counts:
        for tolerance in tolerances:
            selected = [
                item
                for item in records
                if item["clusters"] == clusters
                and item["bound_mode"] == bound_mode
                and math.isclose(item["relative_tolerance"], tolerance)
            ]
            fractions = [
                item["effective_byte_fraction_with_dense_fallback"]
                for item in selected
            ]
            raw_fractions = [
                item["raw_candidate_byte_fraction"] for item in selected
            ]
            exact_fractions = [
                item["exact_old_token_fraction"] for item in selected
            ]
            speedups = [item["optimistic_byte_speedup"] for item in selected]
            observed = [item["max_observed_relative"] for item in selected]
            certificates = [item["max_certificate_relative"] for item in selected]
            initial_observed = [
                item["initial_max_observed_relative"] for item in selected
            ]
            initial_certificates = [
                item["initial_max_certificate_relative"] for item in selected
            ]
            last_certificates = [
                item["last_candidate_max_certificate_relative"] for item in selected
            ]
            union_inflation = [
                item["gqa_union_inflation_vs_mean_individual"] for item in selected
            ]
            fast_rate = statistics.mean(float(item["fast_path"]) for item in selected)
            p95_fraction = quantile(fractions, 0.95)
            gate = (
                bound_mode != "oracle"
                and
                fast_rate >= required_fast_rate
                and p95_fraction <= (1.0 - required_reduction)
            )
            output[f"{bound_mode}_clusters_{clusters}_tol_{tolerance:g}"] = {
                "records": len(selected),
                "fast_path_rate": fast_rate,
                "byte_fraction_p50": quantile(fractions, 0.50),
                "byte_fraction_p95": p95_fraction,
                "raw_candidate_byte_fraction_at_stop_p50": quantile(
                    raw_fractions, 0.50
                ),
                "exact_old_token_fraction_at_stop_p50": quantile(
                    exact_fractions, 0.50
                ),
                "exact_old_token_fraction_at_stop_p95": quantile(
                    exact_fractions, 0.95
                ),
                "optimistic_byte_speedup_p50": quantile(speedups, 0.50),
                "optimistic_byte_speedup_p05": quantile(speedups, 0.05),
                "initial_observed_relative_error_p95": quantile(
                    initial_observed, 0.95
                ),
                "initial_certificate_relative_p95": quantile(
                    initial_certificates, 0.95
                ),
                "last_candidate_certificate_relative_p95": quantile(
                    last_certificates, 0.95
                ),
                "gqa_union_inflation_p95": quantile(union_inflation, 0.95),
                "observed_relative_error_p95": quantile(observed, 0.95),
                "certificate_relative_p95": quantile(certificates, 0.95),
                "passes_traffic_gate": gate,
                "diagnostic_only": bound_mode == "oracle",
            }
    return output


def analyze(
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    queries: list[dict[int, torch.Tensor]],
    context_lengths: list[int],
    context: int,
    local_window: int,
    layers: list[int],
    cluster_counts: list[int],
    bound_modes: list[str],
    tolerances: list[float],
    kmeans_iters: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_tokens = context - local_window
    if old_tokens < max(cluster_counts):
        raise ValueError("Cluster count exceeds the old-token region")
    records: list[dict[str, Any]] = []
    clustering: dict[str, Any] = {}
    for layer_idx in layers:
        layer_keys = kv[layer_idx][0].cuda().float()
        layer_values = kv[layer_idx][1].cuda().float()
        hkv, _, dim = layer_keys.shape
        hq = queries[0][layer_idx].shape[1]
        group_size = hq // hkv
        for kv_head in range(hkv):
            old_keys = layer_keys[kv_head, :old_tokens]
            old_values = layer_values[kv_head, :old_tokens]
            for clusters in cluster_counts:
                print(
                    f"cluster layer={layer_idx} kv_head={kv_head} C={clusters}",
                    flush=True,
                )
                summary = build_cluster_summary(
                    old_keys,
                    old_values,
                    clusters,
                    kmeans_iters,
                    seed + layer_idx * 1009 + kv_head * 101 + clusters,
                )
                clustering[f"layer_{layer_idx}_kv_{kv_head}_c_{clusters}"] = {
                    "rms_key_distortion": summary.distortion,
                    "min_cluster_tokens": int(summary.counts.min().item()),
                    "max_cluster_tokens": int(summary.counts.max().item()),
                }
                for step_idx, length in enumerate(context_lengths):
                    local_keys = layer_keys[kv_head, old_tokens:length]
                    local_values = layer_values[kv_head, old_tokens:length]
                    query_group = queries[step_idx][layer_idx][0, kv_head * group_size : (kv_head + 1) * group_size].cuda()
                    for bound_mode in bound_modes:
                      states = [
                          build_head_state(
                              query_group[head],
                              old_keys,
                              old_values,
                              local_keys,
                              local_values,
                              summary,
                              bound_mode=bound_mode,
                          )
                          for head in range(group_size)
                      ]
                      for tolerance in tolerances:
                        individual_masks: list[torch.Tensor] = []
                        individual_exact_tokens: list[int] = []
                        for state in states:
                            individual_mask, _, individual_accounting = refine_group(
                                [state],
                                summary.counts,
                                tolerance,
                                old_tokens,
                                length - old_tokens,
                                dim,
                                enforce_dense_fallback=False,
                                bound_mode=bound_mode,
                            )
                            individual_masks.append(individual_mask)
                            individual_exact_tokens.append(
                                individual_accounting["exact_old_tokens"]
                            )
                        independent_union = torch.stack(individual_masks).any(dim=0)
                        independent_union_tokens = int(
                            summary.counts[independent_union].sum().item()
                        )

                        mask, metrics, accounting = refine_group(
                            states,
                            summary.counts,
                            tolerance,
                            old_tokens,
                            length - old_tokens,
                            dim,
                            enforce_dense_fallback=True,
                            bound_mode=bound_mode,
                        )
                        mean_individual = statistics.mean(individual_exact_tokens)
                        record = {
                            "layer": layer_idx,
                            "kv_head": kv_head,
                            "decode_step": step_idx,
                            "context_length": length,
                            "gqa_group_size": group_size,
                            "clusters": clusters,
                            "bound_mode": bound_mode,
                            "relative_tolerance": tolerance,
                            "max_certificate_relative": max(
                                item["certificate_relative_to_approx"] for item in metrics
                            ),
                            "max_observed_relative": max(
                                item["observed_relative_to_dense"] for item in metrics
                            ),
                            "mean_individual_exact_old_tokens": mean_individual,
                            "independent_union_exact_old_tokens": independent_union_tokens,
                            "gqa_union_inflation_vs_mean_individual": (
                                independent_union_tokens / max(mean_individual, 1.0)
                            ),
                            **accounting,
                        }
                        records.append(record)
                del summary
        del layer_keys, layer_values
        torch.cuda.empty_cache()
    return records, clustering


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.context <= args.local_window:
        raise SystemExit("context must be greater than local-window")
    layers = parse_int_csv(args.layers)
    cluster_counts = parse_int_csv(args.cluster_counts)
    bound_modes = parse_str_csv(args.bound_modes)
    invalid_modes = set(bound_modes) - {"box", "ball", "oracle"}
    if invalid_modes:
        raise SystemExit(f"Unknown bound modes: {sorted(invalid_modes)}")
    tolerances = parse_float_csv(args.relative_tolerances)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path = resolve_model_path(args.model)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    input_ids, corpus_metadata = make_input_ids(
        tokenizer, args.corpus, args.context, args.token_offset, args.seed
    )
    print(
        f"model={model_path.name} corpus={args.corpus} context={args.context} "
        f"layers={layers}",
        flush=True,
    )
    kv, queries, context_lengths, model_metadata = extract_real_qkv(
        model_path,
        input_ids,
        layers,
        args.decode_steps,
        args.prefill_chunk,
    )
    records, clustering = analyze(
        kv,
        queries,
        context_lengths,
        args.context,
        args.local_window,
        layers,
        cluster_counts,
        bound_modes,
        tolerances,
        args.kmeans_iters,
        args.seed,
    )
    aggregate = aggregate_records(
        records,
        cluster_counts,
        tolerances,
        bound_modes,
        args.required_byte_reduction,
        args.required_fast_path_rate,
    )
    result = {
        "schema_version": 1,
        "experiment": "certified_hierarchical_attention_falsification_probe",
        "claim_scope": (
            "Offline mathematical and useful-byte feasibility over real post-RoPE "
            "Q/K/V; not a kernel timing or end-to-end quality result."
        ),
        "environment": {
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "model_path": str(model_path),
        "model": model_metadata,
        "corpus": corpus_metadata,
        "configuration": {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "context_lengths": context_lengths,
            "prefill_chunk": args.prefill_chunk,
            "local_window": args.local_window,
            "clustered_old_tokens": args.context - args.local_window,
            "layers": layers,
            "cluster_counts": cluster_counts,
            "bound_modes": bound_modes,
            "relative_tolerances": tolerances,
            "kmeans_iters": args.kmeans_iters,
            "seed": args.seed,
            "required_byte_reduction": args.required_byte_reduction,
            "required_fast_path_rate": args.required_fast_path_rate,
        },
        "clustering": clustering,
        "aggregate": aggregate,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
