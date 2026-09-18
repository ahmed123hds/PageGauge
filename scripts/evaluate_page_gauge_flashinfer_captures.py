#!/usr/bin/env python3
"""Evaluate the fused PageGauge FlashInfer kernel on captured real Q/K/V states."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_flashinfer_page_affine_int8 as kernel_bench  # noqa: E402
from evaluate_gauge_int8_model import (  # noqa: E402
    fit_affine_parameters,
    load_capture,
    make_orthogonal_gauge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-captures", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluation-capture", type=Path, required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--exact-sink", type=int, default=0)
    parser.add_argument("--fixed-split-pages", type=int, default=256)
    parser.add_argument("--channel-normalization-power", type=float, default=0.0)
    parser.add_argument("--value-page-group-size", type=int, default=128)
    parser.add_argument("--page-scale-ratio", type=float, default=1.0)
    parser.add_argument(
        "--orthogonal-gauge", choices=("none", "hadamard"), default="none"
    )
    parser.add_argument("--orthogonal-seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tokens_to_pages(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Convert [HKV, tokens, dim] to zero-padded [pages, 16, HKV, dim]."""
    if values.ndim != 3:
        raise ValueError(f"expected [HKV, tokens, dim], received {tuple(values.shape)}")
    tokens = values.shape[1]
    pages = math.ceil(tokens / kernel_bench.PAGE)
    padded_tokens = pages * kernel_bench.PAGE
    token_major = values.permute(1, 0, 2).contiguous()
    if padded_tokens != tokens:
        token_major = torch.nn.functional.pad(
            token_major, (0, 0, 0, 0, 0, padded_tokens - tokens)
        )
    return token_major.reshape(
        pages, kernel_bench.PAGE, values.shape[0], values.shape[2]
    ).contiguous(), tokens % kernel_bench.PAGE or kernel_bench.PAGE


def quantize_centered_pages(
    values: torch.Tensor,
    center: torch.Tensor,
    normalizer: torch.Tensor,
    group_size: int = 128,
    rotation: torch.Tensor | None = None,
    affine_offset: bool = False,
    scale_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [HKV, old_tokens, dim] with a fixed channel gauge."""
    centered = (values.float() - center.float()[:, None, :]) / normalizer.float()[
        :, None, :
    ]
    if rotation is not None:
        centered = torch.einsum("htd,hde->hte", centered, rotation)
    pages, _ = tokens_to_pages(centered.half())
    groups = pages.shape[-1] // group_size
    page_head = pages.permute(0, 2, 1, 3).float().reshape(
        pages.shape[0], pages.shape[2], pages.shape[1], groups, group_size
    )
    if affine_offset:
        minimum = page_head.amin(dim=(2, 4))
        maximum = page_head.amax(dim=(2, 4))
        offset = ((minimum + maximum) * 0.5).half()
        scale = ((maximum - minimum) / 254.0).clamp_min(2.0**-20).half()
    else:
        offset = torch.zeros_like(page_head[:, :, 0, :, 0], dtype=torch.float16)
        scale = (
            page_head.abs().amax(dim=(2, 4)) * scale_ratio / 127.0
        ).clamp_min(2.0**-20).half()
    codes = (
        (
            (page_head - offset.float()[:, :, None, :, None])
            / scale.float()[:, :, None, :, None]
        )
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .reshape(pages.shape[0], pages.shape[2], pages.shape[1], pages.shape[3])
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    reconstructed = (
        codes.permute(0, 2, 1, 3)
        .float()
        .reshape(pages.shape[0], pages.shape[2], pages.shape[1], groups, group_size)
        * scale.float()[:, :, None, :, None]
        + offset.float()[:, :, None, :, None]
    ).reshape(pages.shape[0], pages.shape[2], pages.shape[1], pages.shape[3]).permute(
        0, 2, 1, 3
    ).half().contiguous()
    squeezed_scale = scale.squeeze(-1) if groups == 1 else scale
    return codes, squeezed_scale.contiguous(), reconstructed


def page_table(pages: int, last_page_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0, pages], device="cuda", dtype=torch.int32),
        torch.arange(pages, device="cuda", dtype=torch.int32),
        torch.tensor([last_page_len], device="cuda", dtype=torch.int32),
    )


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def channel_normalizer(
    params: dict[str, torch.Tensor], prefix: str, power: float
) -> torch.Tensor:
    raw = params[f"{prefix}_scale"].float()
    median = raw.median(dim=-1, keepdim=True).values.clamp_min(2.0**-20)
    return (raw / median).clamp(1.0 / 16.0, 16.0).pow(power)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.exact_tail <= 0:
        raise SystemExit("--exact-tail must be positive")
    if args.exact_sink < 0 or args.exact_sink % kernel_bench.PAGE:
        raise SystemExit("--exact-sink must be a non-negative multiple of page size")
    if not 0.0 <= args.channel_normalization_power <= 1.0:
        raise SystemExit("--channel-normalization-power must be in [0, 1]")
    if args.value_page_group_size <= 0 or kernel_bench.DIM % args.value_page_group_size:
        raise SystemExit("--value-page-group-size must divide head dim")
    if not 0.0 < args.page_scale_ratio <= 1.0:
        raise SystemExit("--page-scale-ratio must be in (0, 1]")
    import flashinfer

    calibrations = [load_capture(path) for path in args.calibration_captures]
    evaluation = load_capture(args.evaluation_capture)
    parameters = fit_affine_parameters(
        calibrations, range_multiplier=1.0, device=torch.device("cuda")
    )
    step = args.step if args.step >= 0 else len(evaluation["queries"]) + args.step
    if step < 0 or step >= len(evaluation["queries"]):
        raise SystemExit(f"invalid step {args.step}")
    context_length = int(evaluation["context_lengths"][step])
    old_start = args.exact_sink
    old_end = (
        (context_length - args.exact_tail) // kernel_bench.PAGE
    ) * kernel_bench.PAGE
    old_tokens = old_end - old_start
    if old_tokens <= 0:
        raise SystemExit("capture is too short for the requested exact sink and tail")
    tail_tokens = context_length - old_end
    exact_tokens = args.exact_sink + tail_tokens
    layers = sorted(int(layer) for layer in evaluation["kv"] if int(layer) in parameters)
    if not layers:
        raise SystemExit("no common layers between calibration and evaluation captures")

    sample_key = evaluation["kv"][layers[0]][0]
    hkv, _, dim = sample_key.shape
    sample_query = evaluation["queries"][step][layers[0]]
    hq = sample_query.shape[1]
    if dim != kernel_bench.DIM:
        raise SystemExit(
            f"kernel specialization requires head dim {kernel_bench.DIM}; capture has {dim}"
        )

    full_pages = math.ceil(context_length / kernel_bench.PAGE)
    old_pages = old_tokens // kernel_bench.PAGE
    exact_pages = math.ceil(exact_tokens / kernel_bench.PAGE)
    full_table = page_table(
        full_pages, context_length % kernel_bench.PAGE or kernel_bench.PAGE
    )
    old_table = page_table(old_pages, kernel_bench.PAGE)
    exact_table = page_table(
        exact_pages, exact_tokens % kernel_bench.PAGE or kernel_bench.PAGE
    )

    baseline = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
    )
    compressed = kernel_bench.make_page_gauge_wrapper(
        flashinfer,
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    exact_segment = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
    )
    exact_old = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
    )
    for wrapper, table, dtype in (
        (baseline, full_table, torch.float16),
        (compressed, old_table, torch.int8),
        (exact_old, old_table, torch.float16),
        (exact_segment, exact_table, torch.float16),
    ):
        kernel_bench.plan(
            wrapper,
            *table,
            dtype,
            args.fixed_split_pages,
            num_qo_heads=hq,
            num_kv_heads=hkv,
            head_dim=dim,
        )

    layer_results: dict[str, dict[str, dict[str, float]]] = {}
    all_head_relative: dict[str, list[float]] = {
        "gauge_segment_merge": [],
        "quantized_reconstruction": [],
        "key_quantized_only": [],
        "value_quantized_only": [],
        "grouped_value_reconstruction": [],
        "value_page_offset_reconstruction": [],
        "key_page_offset_reconstruction": [],
        "both_page_offset_reconstruction": [],
        "factorized_kernel": [],
        "factorization_only": [],
    }
    for layer in layers:
        key_cpu, value_cpu = evaluation["kv"][layer]
        key = key_cpu[:, :context_length].cuda(non_blocking=False)
        value = value_cpu[:, :context_length].cuda(non_blocking=False)
        query = evaluation["queries"][step][layer].half().cuda(non_blocking=False)
        key_center = parameters[layer]["k_center"]
        value_center = parameters[layer]["v_center"]
        key_normalizer = channel_normalizer(
            parameters[layer], "k", args.channel_normalization_power
        )
        value_normalizer = channel_normalizer(
            parameters[layer], "v", args.channel_normalization_power
        )
        repeated_key_normalizer = key_normalizer.repeat_interleave(hq // hkv, dim=0)
        transformed_query = (query.float() * repeated_key_normalizer[None]).half()
        key_rotation = None
        value_rotation = None
        if args.orthogonal_gauge == "hadamard":
            key_rotation = make_orthogonal_gauge(
                hkv, dim, layer, args.orthogonal_seed, 0, key.device
            )
            value_rotation = make_orthogonal_gauge(
                hkv, dim, layer, args.orthogonal_seed, 1, value.device
            )
            repeated_key_rotation = key_rotation.repeat_interleave(hq // hkv, dim=0)
            transformed_query = torch.einsum(
                "bhd,hde->bhe", transformed_query.float(), repeated_key_rotation
            ).half()

        full_key, _ = tokens_to_pages(key)
        full_value, _ = tokens_to_pages(value)
        old_key = key[:, old_start:old_end]
        old_value = value[:, old_start:old_end]
        exact_key_source = torch.cat((key[:, :old_start], key[:, old_end:]), dim=1)
        exact_value_source = torch.cat(
            (value[:, :old_start], value[:, old_end:]), dim=1
        )
        old_key_gauge = (
            (old_key.float() - key_center[:, None, :].float())
            / key_normalizer[:, None, :]
        )
        old_value_gauge = (
            (old_value.float() - value_center[:, None, :].float())
            / value_normalizer[:, None, :]
        )
        exact_key_gauge = (
            (exact_key_source.float() - key_center[:, None, :].float())
            / key_normalizer[:, None, :]
        )
        exact_value_gauge = (
            (exact_value_source.float() - value_center[:, None, :].float())
            / value_normalizer[:, None, :]
        )
        if key_rotation is not None:
            old_key_gauge = torch.einsum(
                "htd,hde->hte", old_key_gauge, key_rotation
            )
            exact_key_gauge = torch.einsum(
                "htd,hde->hte", exact_key_gauge, key_rotation
            )
            old_value_gauge = torch.einsum(
                "htd,hde->hte", old_value_gauge, value_rotation
            )
            exact_value_gauge = torch.einsum(
                "htd,hde->hte", exact_value_gauge, value_rotation
            )
        old_key_centered, _ = tokens_to_pages(
            old_key_gauge.half()
        )
        old_value_centered, _ = tokens_to_pages(
            old_value_gauge.half()
        )
        k_codes, k_scale, reconstructed_k = quantize_centered_pages(
            old_key,
            key_center,
            key_normalizer,
            rotation=key_rotation,
            scale_ratio=args.page_scale_ratio,
        )
        v_codes, v_scale, reconstructed_v = quantize_centered_pages(
            old_value,
            value_center,
            value_normalizer,
            rotation=value_rotation,
            scale_ratio=args.page_scale_ratio,
        )
        _, _, reconstructed_v_grouped = quantize_centered_pages(
            old_value,
            value_center,
            value_normalizer,
            args.value_page_group_size,
            value_rotation,
            scale_ratio=args.page_scale_ratio,
        )
        _, _, reconstructed_k_offset = quantize_centered_pages(
            old_key,
            key_center,
            key_normalizer,
            rotation=key_rotation,
            affine_offset=True,
        )
        _, _, reconstructed_v_offset = quantize_centered_pages(
            old_value,
            value_center,
            value_normalizer,
            rotation=value_rotation,
            affine_offset=True,
        )
        exact_key, _ = tokens_to_pages(exact_key_gauge.half())
        exact_value, _ = tokens_to_pages(exact_value_gauge.half())

        reference = baseline.run(query, (full_key, full_value))
        old_output, old_lse = compressed.run(
            transformed_query,
            (k_codes, v_codes),
            k_scale,
            v_scale,
            1.0 / math.sqrt(dim),
            return_lse=True,
        )
        exact_output, exact_lse = exact_segment.run(
            transformed_query, (exact_key, exact_value), return_lse=True
        )
        gauge_output, gauge_lse = exact_old.run(
            transformed_query,
            (old_key_centered, old_value_centered),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            gauge_output, gauge_lse, exact_output, exact_lse
        )
        reconstructed_output, reconstructed_lse = exact_old.run(
            transformed_query, (reconstructed_k, reconstructed_v), return_lse=True
        )
        flashinfer.merge_state_in_place(
            reconstructed_output, reconstructed_lse, exact_output, exact_lse
        )
        key_only_output, key_only_lse = exact_old.run(
            transformed_query,
            (reconstructed_k, old_value_centered),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            key_only_output, key_only_lse, exact_output, exact_lse
        )
        value_only_output, value_only_lse = exact_old.run(
            transformed_query,
            (old_key_centered, reconstructed_v),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            value_only_output, value_only_lse, exact_output, exact_lse
        )
        grouped_value_output, grouped_value_lse = exact_old.run(
            transformed_query,
            (reconstructed_k, reconstructed_v_grouped),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            grouped_value_output,
            grouped_value_lse,
            exact_output,
            exact_lse,
        )
        value_offset_output, value_offset_lse = exact_old.run(
            transformed_query,
            (reconstructed_k, reconstructed_v_offset),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            value_offset_output, value_offset_lse, exact_output, exact_lse
        )
        key_offset_output, key_offset_lse = exact_old.run(
            transformed_query,
            (reconstructed_k_offset, reconstructed_v),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            key_offset_output, key_offset_lse, exact_output, exact_lse
        )
        both_offset_output, both_offset_lse = exact_old.run(
            transformed_query,
            (reconstructed_k_offset, reconstructed_v_offset),
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            both_offset_output, both_offset_lse, exact_output, exact_lse
        )
        flashinfer.merge_state_in_place(
            old_output, old_lse, exact_output, exact_lse
        )
        if value_rotation is not None:
            repeated_value_rotation = value_rotation.repeat_interleave(
                hq // hkv, dim=0
            )
            inverse_value_rotation = repeated_value_rotation.transpose(-1, -2)

            def restore_rotation(candidate: torch.Tensor) -> torch.Tensor:
                return torch.einsum(
                    "bhd,hde->bhe", candidate.float(), inverse_value_rotation
                ).half()

            gauge_output = restore_rotation(gauge_output)
            reconstructed_output = restore_rotation(reconstructed_output)
            key_only_output = restore_rotation(key_only_output)
            value_only_output = restore_rotation(value_only_output)
            grouped_value_output = restore_rotation(grouped_value_output)
            value_offset_output = restore_rotation(value_offset_output)
            key_offset_output = restore_rotation(key_offset_output)
            both_offset_output = restore_rotation(both_offset_output)
            old_output = restore_rotation(old_output)
        output_center = value_center.repeat_interleave(hq // hkv, dim=0).unsqueeze(0)
        output_normalizer = value_normalizer.repeat_interleave(
            hq // hkv, dim=0
        ).unsqueeze(0)
        gauge_output = gauge_output * output_normalizer + output_center
        reconstructed_output = reconstructed_output * output_normalizer + output_center
        key_only_output = key_only_output * output_normalizer + output_center
        value_only_output = value_only_output * output_normalizer + output_center
        grouped_value_output = grouped_value_output * output_normalizer + output_center
        value_offset_output = value_offset_output * output_normalizer + output_center
        key_offset_output = key_offset_output * output_normalizer + output_center
        both_offset_output = both_offset_output * output_normalizer + output_center
        actual = old_output * output_normalizer + output_center
        torch.cuda.synchronize()

        variants = {
            "gauge_segment_merge": gauge_output,
            "quantized_reconstruction": reconstructed_output,
            "key_quantized_only": key_only_output,
            "value_quantized_only": value_only_output,
            "grouped_value_reconstruction": grouped_value_output,
            "value_page_offset_reconstruction": value_offset_output,
            "key_page_offset_reconstruction": key_offset_output,
            "both_page_offset_reconstruction": both_offset_output,
            "factorized_kernel": actual,
        }
        per_layer: dict[str, dict[str, float]] = {}
        reference_norm = reference.float().norm(dim=-1).clamp_min(1e-8)
        for name, candidate in variants.items():
            difference = candidate.float() - reference.float()
            relative = difference.norm(dim=-1) / reference_norm
            cosine = torch.nn.functional.cosine_similarity(
                candidate.float(), reference.float(), dim=-1
            )
            head_values = relative.flatten().cpu().tolist()
            all_head_relative[name].extend(head_values)
            per_layer[name] = {
                "relative_l2_mean": statistics.mean(head_values),
                "relative_l2_max": max(head_values),
                "cosine_mean": float(cosine.mean()),
                "absolute_max": float(difference.abs().max()),
            }
        factorization_difference = actual.float() - reconstructed_output.float()
        factorization_relative = factorization_difference.norm(dim=-1) / reconstructed_output.float().norm(
            dim=-1
        ).clamp_min(1e-8)
        factorization_values = factorization_relative.flatten().cpu().tolist()
        all_head_relative["factorization_only"].extend(factorization_values)
        per_layer["factorization_only"] = {
            "relative_l2_mean": statistics.mean(factorization_values),
            "relative_l2_max": max(factorization_values),
            "cosine_mean": float(
                torch.nn.functional.cosine_similarity(
                    actual.float(), reconstructed_output.float(), dim=-1
                ).mean()
            ),
            "absolute_max": float(factorization_difference.abs().max()),
        }
        layer_results[str(layer)] = per_layer

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "page_gauge_fused_kernel_real_qkv_quality",
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
        },
        "inputs": {
            "calibration_captures": [str(path.resolve()) for path in args.calibration_captures],
            "evaluation_capture": str(args.evaluation_capture.resolve()),
            "step": step,
            "context_length": context_length,
            "old_int8_tokens": old_tokens,
            "old_int8_start": old_start,
            "old_int8_end": old_end,
            "exact_sink_tokens": args.exact_sink,
            "exact_tail_tokens": tail_tokens,
            "exact_segment_tokens": exact_tokens,
            "fixed_split_pages": args.fixed_split_pages,
            "channel_normalization_power": args.channel_normalization_power,
            "value_page_group_size": args.value_page_group_size,
            "page_scale_ratio": args.page_scale_ratio,
            "orthogonal_gauge": args.orthogonal_gauge,
            "orthogonal_seed": args.orthogonal_seed,
            "layers": layers,
        },
        "aggregate": {
            name: {
                "head_relative_l2_mean": statistics.mean(values),
                "head_relative_l2_p50": quantile(values, 0.50),
                "head_relative_l2_p95": quantile(values, 0.95),
                "head_relative_l2_max": max(values),
            }
            for name, values in all_head_relative.items()
        },
        "per_layer": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
