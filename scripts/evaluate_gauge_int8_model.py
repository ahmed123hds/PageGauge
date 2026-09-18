#!/usr/bin/env python3
"""Evaluate calibration-frozen GaugeINT8 inside the actual Qwen attention stack."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_certified_hierarchical_attention import (  # noqa: E402
    make_input_ids,
    resolve_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--calibration-captures", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--corpus", choices=("paper", "code", "mixed", "random"), default="paper"
    )
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--token-offset", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--range-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=("global_gauge", "page_affine", "page_symmetric", "page_gauge"),
        default="global_gauge",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--page-scale-ratio",
        type=float,
        default=1.0,
        help="Fraction of page absmax represented before INT8 clipping.",
    )
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--value-page-group-size",
        type=int,
        default=128,
        help="Channel group size for page-gauge V (K always uses one scale per head).",
    )
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--exact-sink", type=int, default=0)
    parser.add_argument(
        "--orthogonal-gauge",
        choices=("none", "hadamard"),
        default="none",
        help="Token-invariant orthogonal channel gauge applied before page quantization.",
    )
    parser.add_argument("--orthogonal-seed", type=int, default=20260815)
    parser.add_argument(
        "--channel-normalization-power",
        type=float,
        default=0.0,
        help="Global diagonal gauge strength: 0 disables, 1 fully equalizes calibrated ranges.",
    )
    parser.add_argument("--sample-logit-rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_capture(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_affine_parameters(
    captures: list[dict[str, Any]], range_multiplier: float, device: torch.device
) -> dict[int, dict[str, torch.Tensor]]:
    if range_multiplier <= 0:
        raise ValueError("range multiplier must be positive")
    layers = sorted(int(layer) for layer in captures[0]["kv"])
    parameters: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        layer_params: dict[str, torch.Tensor] = {}
        for name, index in (("k", 0), ("v", 1)):
            samples = torch.cat(
                [capture["kv"][layer][index].float() for capture in captures], dim=1
            )
            minimum = samples.amin(dim=1)
            maximum = samples.amax(dim=1)
            center = ((minimum + maximum) * 0.5).half()
            scale = (
                ((maximum - minimum) * (range_multiplier / 254.0))
                .clamp_min(2.0**-20)
                .half()
            )
            layer_params[f"{name}_center"] = center.to(device)
            layer_params[f"{name}_scale"] = scale.to(device)
        parameters[layer] = layer_params
    return parameters


def quantize_signed_affine(
    values: torch.Tensor, center: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = (values.float() - center.float()[None, :, None, :]) / scale.float()[
        None, :, None, :
    ]
    clipped = normalized.abs() > 127
    codes = normalized.round().clamp(-127, 127).to(torch.int8)
    return codes, clipped


def make_orthogonal_gauge(
    num_heads: int,
    head_dim: int,
    layer: int,
    seed: int,
    stream: int,
    device: torch.device,
) -> torch.Tensor:
    """Return per-head randomized normalized Hadamard matrices [H, D, D]."""
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("Hadamard gauge requires a power-of-two head dimension")
    hadamard = torch.ones((1, 1), dtype=torch.float32)
    while hadamard.shape[0] < head_dim:
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            ),
            dim=0,
        )
    hadamard /= math.sqrt(head_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729 * layer + 1000003 * stream)
    signs = torch.randint(
        0, 2, (num_heads, head_dim), generator=generator, dtype=torch.int8
    ).float()
    signs = signs.mul_(2.0).sub_(1.0)
    # R = diag(sign) H. Applying the same R to Q and K preserves QK^T.
    return (signs[:, :, None] * hadamard[None]).to(device)


class GaugeAttention:
    def __init__(
        self, parameters: dict[int, dict[str, torch.Tensor]], local_window: int
    ) -> None:
        self.parameters = parameters
        self.local_window = local_window
        self.key_clip_counts: dict[int, list[float]] = {}
        self.value_clip_counts: dict[int, list[float]] = {}

    def __call__(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        params = self.parameters[int(module.layer_idx)]
        key_codes, key_clipped = quantize_signed_affine(
            key, params["k_center"], params["k_scale"]
        )
        value_codes, value_clipped = quantize_signed_affine(
            value, params["v_center"], params["v_scale"]
        )
        self.key_clip_counts.setdefault(int(module.layer_idx), []).append(
            float(key_clipped.float().mean())
        )
        self.value_clip_counts.setdefault(int(module.layer_idx), []).append(
            float(value_clipped.float().mean())
        )

        groups = int(module.num_key_value_groups)
        key_scale = repeat_kv(params["k_scale"][None, :, None, :], groups)
        transformed_query = query * key_scale.to(query.dtype)
        repeated_key_codes = repeat_kv(key_codes, groups).to(query.dtype)
        quantized_scores = (
            torch.matmul(
                transformed_query.float(),
                repeated_key_codes.float().transpose(2, 3),
            )
            * scaling
        )
        # Use the same key gauge for the exact tail.  Omitting this term would
        # shift old and recent logits relative to each other even though the
        # center cancels when applied to every token.
        centered_exact_key = key.float() - params["k_center"].float()[
            None, :, None, :
        ]
        repeated_exact_key = repeat_kv(centered_exact_key, groups)
        exact_scores = torch.matmul(
            query.float(), repeated_exact_key.float().transpose(2, 3)
        ) * scaling
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        query_positions = (
            torch.arange(query_length, device=query.device) + key_length - query_length
        )
        key_positions = torch.arange(key_length, device=query.device)
        old_mask = key_positions[None, :] < (
            query_positions[:, None] - self.local_window
        )
        old_mask = old_mask[None, None]
        attention_weights = torch.where(old_mask, quantized_scores, exact_scores)
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32)
        if dropout:
            attention_weights = F.dropout(
                attention_weights, p=dropout, training=module.training
            )

        old_weights = attention_weights * old_mask
        recent_weights = attention_weights - old_weights
        repeated_value_codes = repeat_kv(value_codes, groups)
        output = torch.matmul(old_weights, repeated_value_codes.float())
        value_scale = repeat_kv(params["v_scale"][None, :, None, :], groups)
        value_center = repeat_kv(params["v_center"][None, :, None, :], groups)
        old_mass = old_weights.sum(dim=-1, keepdim=True)
        output = output * value_scale.float() + old_mass * value_center.float()
        repeated_exact_value = repeat_kv(value, groups)
        output = output + torch.matmul(recent_weights, repeated_exact_value.float())
        return output.to(query.dtype).transpose(1, 2).contiguous(), None


class ExactAttention:
    def __call__(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        repeated_key = repeat_kv(key, int(module.num_key_value_groups))
        repeated_value = repeat_kv(value, int(module.num_key_value_groups))
        weights = torch.matmul(
            query.float(), repeated_key.float().transpose(2, 3)
        ) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask.float()
        weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        if dropout:
            weights = F.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, repeated_value.float()).to(query.dtype)
        return output.transpose(1, 2).contiguous(), None


def page_affine_reconstruct(
    values: torch.Tensor, page_size: int, group_size: int
) -> torch.Tensor:
    batch, heads, tokens, dim = values.shape
    if tokens % page_size or dim % group_size:
        raise ValueError("token and head dimensions must be page/group aligned")
    tiles = values.float().reshape(
        batch, heads, tokens // page_size, page_size, dim // group_size, group_size
    )
    minimum = tiles.amin(dim=(3, 5), keepdim=True)
    maximum = tiles.amax(dim=(3, 5), keepdim=True)
    stored_minimum = minimum.half().float()
    stored_scale = ((maximum - minimum) / 255.0).clamp_min(2.0**-20).half().float()
    codes = ((tiles - stored_minimum) / stored_scale).round().clamp(0, 255)
    return (codes * stored_scale + stored_minimum).reshape_as(values)


def page_symmetric_reconstruct(
    values: torch.Tensor,
    page_size: int,
    group_size: int,
    scale_ratio: float = 1.0,
) -> torch.Tensor:
    """Signed INT8 with zero center, allowing scale movement across the MMA."""
    batch, heads, tokens, dim = values.shape
    if tokens % page_size or dim % group_size:
        raise ValueError("token and head dimensions must be page/group aligned")
    tiles = values.float().reshape(
        batch, heads, tokens // page_size, page_size, dim // group_size, group_size
    )
    stored_scale = (
        tiles.abs().amax(dim=(3, 5), keepdim=True) * scale_ratio / 127.0
    ).clamp_min(2.0**-20).half().float()
    codes = (tiles / stored_scale).round().clamp(-127, 127)
    return (codes * stored_scale).reshape_as(values)


class PageAffineAttention:
    def __init__(
        self,
        page_size: int,
        group_size: int,
        local_window: int,
        symmetric: bool = False,
    ) -> None:
        self.page_size = page_size
        self.group_size = group_size
        self.local_window = local_window
        self.symmetric = symmetric

    def __call__(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        exact_key = key
        exact_value = value
        reconstruct = (
            page_symmetric_reconstruct if self.symmetric else page_affine_reconstruct
        )
        quantized_key = reconstruct(exact_key, self.page_size, self.group_size)
        quantized_value = reconstruct(exact_value, self.page_size, self.group_size)
        groups = int(module.num_key_value_groups)
        repeated_quantized_key = repeat_kv(quantized_key, groups)
        repeated_quantized_value = repeat_kv(quantized_value, groups)
        repeated_exact_key = repeat_kv(exact_key, groups)
        repeated_exact_value = repeat_kv(exact_value, groups)
        # The current/recent window remains FP16.  This is expressed as a
        # query-dependent mask so prefill faithfully simulates decode semantics.
        quantized_scores = torch.matmul(
            query.float(), repeated_quantized_key.float().transpose(2, 3)
        ) * scaling
        exact_scores = torch.matmul(
            query.float(), repeated_exact_key.float().transpose(2, 3)
        ) * scaling
        query_length = query.shape[-2]
        key_length = exact_key.shape[-2]
        query_positions = (
            torch.arange(query_length, device=query.device) + key_length - query_length
        )
        key_positions = torch.arange(key_length, device=query.device)
        old_mask = key_positions[None, :] < (
            query_positions[:, None] - self.local_window
        )
        old_mask = old_mask[None, None]
        weights = torch.where(old_mask, quantized_scores, exact_scores)
        if attention_mask is not None:
            weights = weights + attention_mask.float()
        weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        if dropout:
            weights = F.dropout(weights, p=dropout, training=module.training)
        old_weights = weights * old_mask
        recent_weights = weights - old_weights
        output = torch.matmul(
            old_weights, repeated_quantized_value.float()
        ) + torch.matmul(recent_weights, repeated_exact_value.float())
        return output.to(query.dtype).transpose(1, 2).contiguous(), None


class PageGaugeAttention:
    """Global channel gauge plus one symmetric INT8 scale per page and KV head."""

    def __init__(
        self,
        parameters: dict[int, dict[str, torch.Tensor]],
        page_size: int,
        local_window: int,
        channel_normalization_power: float = 0.0,
        value_page_group_size: int = 128,
        exact_sink: int = 0,
        orthogonal_gauge: str = "none",
        orthogonal_seed: int = 20260815,
        page_scale_ratio: float = 1.0,
    ) -> None:
        self.parameters = parameters
        self.page_size = page_size
        self.local_window = local_window
        self.channel_normalization_power = channel_normalization_power
        self.value_page_group_size = value_page_group_size
        self.exact_sink = exact_sink
        self.orthogonal_gauge = orthogonal_gauge
        self.orthogonal_seed = orthogonal_seed
        self.page_scale_ratio = page_scale_ratio

    def channel_normalizer(
        self, params: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        raw = params[f"{prefix}_scale"].float()
        median = raw.median(dim=-1, keepdim=True).values.clamp_min(2.0**-20)
        ratio = (raw / median).clamp(1.0 / 16.0, 16.0)
        return ratio.pow(self.channel_normalization_power)

    def __call__(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        params = self.parameters[int(module.layer_idx)]
        key_center = params["k_center"].float()[None, :, None, :]
        value_center = params["v_center"].float()[None, :, None, :]
        key_normalizer = self.channel_normalizer(params, "k")[None, :, None, :]
        value_normalizer = self.channel_normalizer(params, "v")[None, :, None, :]
        centered_key = (key.float() - key_center) / key_normalizer
        centered_value = (value.float() - value_center) / value_normalizer
        key_rotation = None
        value_rotation = None
        if self.orthogonal_gauge == "hadamard":
            layer = int(module.layer_idx)
            key_rotation = make_orthogonal_gauge(
                centered_key.shape[1],
                centered_key.shape[-1],
                layer,
                self.orthogonal_seed,
                0,
                centered_key.device,
            )
            value_rotation = make_orthogonal_gauge(
                centered_value.shape[1],
                centered_value.shape[-1],
                layer,
                self.orthogonal_seed,
                1,
                centered_value.device,
            )
            centered_key = torch.einsum(
                "bhtd,hde->bhte", centered_key, key_rotation
            )
            centered_value = torch.einsum(
                "bhtd,hde->bhte", centered_value, value_rotation
            )
        quantized_key = page_symmetric_reconstruct(
            centered_key,
            self.page_size,
            centered_key.shape[-1],
            self.page_scale_ratio,
        )
        quantized_value = page_symmetric_reconstruct(
            centered_value,
            self.page_size,
            self.value_page_group_size,
            self.page_scale_ratio,
        )
        groups = int(module.num_key_value_groups)
        repeated_quantized_key = repeat_kv(quantized_key, groups)
        repeated_quantized_value = repeat_kv(quantized_value, groups)
        repeated_centered_key = repeat_kv(centered_key, groups)
        repeated_centered_value = repeat_kv(centered_value, groups)

        repeated_key_normalizer = repeat_kv(key_normalizer, groups)
        transformed_query = query.float() * repeated_key_normalizer
        if key_rotation is not None:
            repeated_key_rotation = key_rotation.repeat_interleave(groups, dim=0)
            transformed_query = torch.einsum(
                "bhtd,hde->bhte", transformed_query, repeated_key_rotation
            )
        quantized_scores = torch.matmul(
            transformed_query, repeated_quantized_key.transpose(2, 3)
        ) * scaling
        exact_scores = torch.matmul(
            transformed_query, repeated_centered_key.transpose(2, 3)
        ) * scaling
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        query_positions = (
            torch.arange(query_length, device=query.device) + key_length - query_length
        )
        key_positions = torch.arange(key_length, device=query.device)
        old_mask = (key_positions[None, :] >= self.exact_sink) & (
            key_positions[None, :] < query_positions[:, None] - self.local_window
        )
        old_mask = old_mask[None, None]
        weights = torch.where(old_mask, quantized_scores, exact_scores)
        if attention_mask is not None:
            weights = weights + attention_mask.float()
        weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        if dropout:
            weights = F.dropout(weights, p=dropout, training=module.training)
        old_weights = weights * old_mask
        recent_weights = weights - old_weights
        output = torch.matmul(
            old_weights, repeated_quantized_value
        ) + torch.matmul(recent_weights, repeated_centered_value)
        # Restore the diagonal V gauge, then its additive gauge.  Both can be
        # folded into the output projection in an integrated model.
        if value_rotation is not None:
            repeated_value_rotation = value_rotation.repeat_interleave(groups, dim=0)
            output = torch.einsum(
                "bhtd,hde->bhte",
                output,
                repeated_value_rotation.transpose(-1, -2),
            )
        repeated_value_normalizer = repeat_kv(value_normalizer, groups)
        output = output * repeated_value_normalizer
        repeated_value_center = repeat_kv(value_center, groups)
        output = output + repeated_value_center
        return output.to(query.dtype).transpose(1, 2).contiguous(), None


def token_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    )


def choose_rows(context: int, requested: int) -> torch.Tensor:
    count = min(context - 1, requested)
    if count <= 0:
        raise ValueError("context must contain at least two tokens")
    return torch.linspace(0, context - 2, count).round().long().unique()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not 0.0 <= args.channel_normalization_power <= 1.0:
        raise SystemExit("--channel-normalization-power must be in [0, 1]")
    if args.value_page_group_size <= 0 or 128 % args.value_page_group_size:
        raise SystemExit("--value-page-group-size must be a positive divisor of 128")
    if args.exact_sink < 0:
        raise SystemExit("--exact-sink must be non-negative")
    if not 0.0 < args.page_scale_ratio <= 1.0:
        raise SystemExit("--page-scale-ratio must be in (0, 1]")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2

    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    input_ids, corpus_metadata = make_input_ids(
        tokenizer, args.corpus, args.context, args.token_offset, args.seed
    )
    input_ids = input_ids.cuda()
    captures = [load_capture(path) for path in args.calibration_captures]
    parameters = fit_affine_parameters(
        captures, args.range_multiplier, torch.device("cuda")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    ).cuda().eval()
    rows = choose_rows(args.context, args.sample_logit_rows).cuda()

    exact_implementation_name = "gauge_exact_fp32_eval"
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register(
        exact_implementation_name, ExactAttention()
    )
    model.config._attn_implementation = exact_implementation_name
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = exact_implementation_name

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    with torch.inference_mode():
        baseline_logits = model(input_ids, use_cache=False).logits
        baseline_nll = token_nll(baseline_logits, input_ids)
        baseline_top1 = baseline_logits[:, :-1].argmax(dim=-1)
        baseline_sample = baseline_logits[0, rows].float().cpu()
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started
    baseline_peak = torch.cuda.max_memory_allocated()
    del baseline_logits
    torch.cuda.empty_cache()

    if args.mode == "global_gauge":
        candidate_attention: Any = GaugeAttention(parameters, args.local_window)
        implementation_name = "gauge_int8_eval"
    elif args.mode == "page_gauge":
        candidate_attention = PageGaugeAttention(
            parameters,
            args.page_size,
            args.local_window,
            args.channel_normalization_power,
            args.value_page_group_size,
            args.exact_sink,
            args.orthogonal_gauge,
            args.orthogonal_seed,
            args.page_scale_ratio,
        )
        implementation_name = "page_gauge_int8_eval"
    else:
        candidate_attention = PageAffineAttention(
            args.page_size,
            args.group_size,
            args.local_window,
            symmetric=args.mode == "page_symmetric",
        )
        implementation_name = f"{args.mode}_int8_eval"
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register(
        implementation_name, candidate_attention
    )
    model.config._attn_implementation = implementation_name
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = implementation_name

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    candidate_started = time.perf_counter()
    with torch.inference_mode():
        candidate_logits = model(input_ids, use_cache=False).logits
        candidate_nll = token_nll(candidate_logits, input_ids)
        candidate_top1 = candidate_logits[:, :-1].argmax(dim=-1)
        candidate_sample = candidate_logits[0, rows].float().cpu()
    torch.cuda.synchronize()
    candidate_seconds = time.perf_counter() - candidate_started
    candidate_peak = torch.cuda.max_memory_allocated()

    reference_log_probability = F.log_softmax(baseline_sample, dim=-1)
    candidate_log_probability = F.log_softmax(candidate_sample, dim=-1)
    reference_probability = reference_log_probability.exp()
    sampled_kl = (
        reference_probability * (reference_log_probability - candidate_log_probability)
    ).sum(dim=-1)
    sampled_relative = (candidate_sample - baseline_sample).norm(dim=-1) / baseline_sample.norm(
        dim=-1
    ).clamp_min(1e-12)
    nll_delta = candidate_nll - baseline_nll
    nll_delta_values = nll_delta.float().cpu().tolist()
    relative_values = sampled_relative.tolist()
    kl_values = sampled_kl.tolist()

    payload = {
        "schema_version": 1,
        "experiment": "int8_kv_end_to_end_model_quality",
        "model": str(model_path),
        "calibration_captures": [str(path.resolve()) for path in args.calibration_captures],
        "evaluation": {
            "corpus": args.corpus,
            "context": args.context,
            "token_offset": args.token_offset,
            "seed": args.seed,
            "range_multiplier": args.range_multiplier,
            "mode": args.mode,
            "page_size": args.page_size,
            "page_scale_ratio": args.page_scale_ratio,
            "group_size": args.group_size,
            "local_window": args.local_window,
            "channel_normalization_power": args.channel_normalization_power,
            "value_page_group_size": args.value_page_group_size,
            "exact_sink": args.exact_sink,
            "orthogonal_gauge": args.orthogonal_gauge,
            "orthogonal_seed": args.orthogonal_seed,
            "corpus_metadata": corpus_metadata,
        },
        "quality": {
            "baseline_mean_nll": float(baseline_nll.mean()),
            "candidate_mean_nll": float(candidate_nll.mean()),
            "mean_nll_delta": float(nll_delta.mean()),
            "p95_absolute_nll_delta": quantile(
                [abs(value) for value in nll_delta_values], 0.95
            ),
            "baseline_perplexity": math.exp(float(baseline_nll.mean())),
            "candidate_perplexity": math.exp(float(candidate_nll.mean())),
            "next_token_top1_agreement": float(
                (baseline_top1 == candidate_top1).float().mean()
            ),
            "sampled_logit_relative_error_p50": quantile(relative_values, 0.50),
            "sampled_logit_relative_error_p95": quantile(relative_values, 0.95),
            "sampled_kl_mean": statistics.mean(kl_values),
            "sampled_kl_p95": quantile(kl_values, 0.95),
        },
        "clipping": (
            {
                "key_mean_by_layer": {
                    str(layer): statistics.mean(values)
                    for layer, values in candidate_attention.key_clip_counts.items()
                },
                "value_mean_by_layer": {
                    str(layer): statistics.mean(values)
                    for layer, values in candidate_attention.value_clip_counts.items()
                },
                "key_max_layer_mean": max(
                    statistics.mean(values)
                    for values in candidate_attention.key_clip_counts.values()
                ),
                "value_max_layer_mean": max(
                    statistics.mean(values)
                    for values in candidate_attention.value_clip_counts.values()
                ),
            }
            if args.mode == "global_gauge"
            else {"not_applicable": "page-local ranges do not clip"}
        ),
        "diagnostics": {
            "baseline_seconds": baseline_seconds,
            "candidate_seconds_quality_simulation_only": candidate_seconds,
            "baseline_peak_allocated_bytes": baseline_peak,
            "candidate_peak_allocated_bytes": candidate_peak,
            "sampled_logit_rows": rows.cpu().tolist(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
