#!/usr/bin/env python3
"""Fresh-process, single-backend PageGauge publication diagnostic.

Each invocation constructs exactly one persistent GPU KV-cache backend.  A
request-local Hugging Face DynamicCache is used only while producing the
coherent WikiText prefix and the greedy HF oracle, then destroyed before the
custom decoder is captured or timed.  In particular, the PageGauge worker
never allocates a full FP16 paged cache and the FlashInfer worker never
allocates PageGauge codes, scales, centers, or an exact ring.

The timed boundary is one fixed-batch, 16-step greedy decode block.  It
contains one device-to-device seed copy, 16 decoder steps, the selected
matched CUDA-graph boundary, and 16 GPU argmax operations. Cache restoration,
cache scrubbing, hot preconditioning, model loading, prefill, and graph capture
are deliberately outside the boundary.
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
PREFILL_PATH = ROOT / "diagnostics/benchmark_model_prefill_correctness.py"
TOKEN_PATH = ROOT / "diagnostics/benchmark_token_step_graphs.py"


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREFILL = load_local_module(
    "page_gauge_model_prefill_for_backend_exclusive", PREFILL_PATH
)
TOKEN = load_local_module(
    "page_gauge_token_step_for_backend_exclusive", TOKEN_PATH
)
PG = PREFILL.PG
OUTER = TOKEN.OUTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("flashinfer_fp16", "page_gauge"),
        required=True,
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=20480)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument(
        "--exact-sink-pages",
        type=int,
        default=0,
        help=(
            "Length S of the fixed contiguous exact FP16 prefix; legacy option "
            "name retained for pairing compatibility."
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
    parser.add_argument("--candidate-split-pages", type=int, default=128)
    parser.add_argument(
        "--tail-attention",
        choices=("flashinfer_merge", "fused_kernel", "heterogeneous_fa2"),
        default="flashinfer_merge",
    )
    parser.add_argument(
        "--cuda-graph-scope",
        choices=("attention", "decoder_layer"),
        default="attention",
        help=(
            "Matched graph boundary for both backends. decoder_layer is the "
            "opt-in 16-offset x 32-layer dynamic-token diagnostic."
        ),
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
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument(
        "--residency-ratio-limit",
        type=float,
        default=4.0,
        help="maximum selected-layer attention/whole-model median ratio",
    )
    parser.add_argument(
        "--residency-absolute-limit-ms",
        type=float,
        default=1.0,
        help="minimum absolute ceiling used by the residency gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.decode_steps != PG.PAGE:
        raise SystemExit("backend-exclusive timing requires exactly 16 decode steps")
    if args.context <= args.exact_tail:
        raise SystemExit("context must exceed the exact-tail window")
    if args.context % PG.PAGE or args.exact_tail % PG.PAGE:
        raise SystemExit("context and exact tail must be page aligned")
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
    if args.context // PG.PAGE <= (
        args.exact_tail // PG.PAGE
        + exact_prefix_pages
        + exact_static_suffix_pages
    ):
        raise SystemExit("context must contain at least one quantized old page")
    if args.prefill_chunk_tokens <= 0:
        raise SystemExit("prefill chunk size must be positive")
    if args.token_offset < 0 or args.token_stride < 0:
        raise SystemExit("token offset and stride cannot be negative")
    if args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("warmups must be non-negative and repeats must be positive")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    if not 0.0 <= args.min_top1_agreement <= 1.0:
        raise SystemExit("top-1 threshold must lie in [0,1]")
    if not -1.0 <= args.min_logits_cosine <= 1.0:
        raise SystemExit("cosine threshold must lie in [-1,1]")
    if args.residency_ratio_limit <= 1.0:
        raise SystemExit("residency ratio limit must exceed one")
    if args.residency_absolute_limit_ms <= 0.0:
        raise SystemExit("residency absolute limit must be positive")
    if args.backend == "page_gauge" and args.exact_tail % PG.PAGE:
        raise SystemExit("PageGauge exact tail must contain whole pages")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sampled_model_parameter_sha256(model: Any) -> dict[str, Any]:
    """Hash a deterministic systematic sample from every model parameter."""
    digest = hashlib.sha256()
    parameter_count = 0
    sampled_values = 0
    parameter_tensors = 0
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        parameter_count += int(flat.numel())
        parameter_tensors += 1
        stride = max(1, math.ceil(int(flat.numel()) / 64))
        sample = flat[::stride][:64].contiguous().cpu()
        sampled_values += int(sample.numel())
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(json.dumps(list(parameter.shape)).encode("ascii"))
        digest.update(sample.numpy().tobytes())
    return {
        "method": "up to 64 systematic values from every named parameter",
        "sha256": digest.hexdigest(),
        "parameter_tensors": parameter_tensors,
        "parameter_count": parameter_count,
        "sampled_values": sampled_values,
    }


def cache_tensor_manifest(cache: Any, backend: str) -> dict[str, Any]:
    attributes = (
        ("key", "value")
        if backend == "flashinfer_fp16"
        else (
            "exact_key",
            "exact_value",
            "key_codes",
            "value_codes",
            "key_scales",
            "value_scales",
            "key_center",
            "value_center",
            "output_center",
        )
    )
    records: dict[str, Any] = {}
    total = 0
    pointers: set[int] = set()
    for attribute in attributes:
        tensor = getattr(cache, attribute)
        byte_count = int(tensor.numel() * tensor.element_size())
        pointer = int(tensor.data_ptr())
        if pointer in pointers:
            raise RuntimeError(f"cache tensor storage alias detected at {attribute}")
        pointers.add(pointer)
        total += byte_count
        records[attribute] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": byte_count,
            "data_ptr": pointer,
        }
    return {
        "backend": backend,
        "tensor_attributes": list(attributes),
        "tensors": records,
        "total_bytes": total,
        "all_storage_pointers_distinct": len(pointers) == len(attributes),
    }


def allocate_backend_cache(
    backend: str,
    layers: int,
    pages: int,
    initial_pages: int,
    exact_pages: int,
    batch_size: int,
    hkv: int,
    exact_sink_pages: int = 0,
    exact_static_suffix_pages: int = 0,
) -> Any:
    exact_sink_pages = PG.validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=initial_pages,
    )
    if backend != "page_gauge" and (exact_sink_pages or exact_static_suffix_pages):
        raise ValueError("fixed exact pages are supported only by PageGauge")
    shape = (layers, batch_size * pages, PG.PAGE, hkv, PG.DIM)
    if backend == "flashinfer_fp16":
        cache = PG.BaselineCache(
            torch.empty(shape, device="cuda", dtype=torch.float16),
            torch.empty(shape, device="cuda", dtype=torch.float16),
        )
        for request in range(batch_size):
            output_page = request * pages + initial_pages
            cache.key[:, output_page].zero_()
            cache.value[:, output_page].zero_()
        return cache

    code_key = torch.empty(shape, device="cuda", dtype=torch.int8)
    code_value = torch.empty_like(code_key)
    scale_shape = (layers, batch_size * pages, hkv)
    key_scales = torch.empty(scale_shape, device="cuda", dtype=torch.float16)
    value_scales = torch.empty_like(key_scales)
    center_shape = (layers, batch_size, hkv, PG.DIM)
    key_center = torch.empty(center_shape, device="cuda", dtype=torch.float16)
    value_center = torch.empty_like(key_center)
    exact_shape = (
        layers,
        batch_size
        * (exact_pages + exact_sink_pages + exact_static_suffix_pages),
        PG.PAGE,
        hkv,
        PG.DIM,
    )
    exact_key = torch.empty(exact_shape, device="cuda", dtype=torch.float16)
    exact_value = torch.empty_like(exact_key)
    cache = PG.GaugeCache(
        exact_key,
        exact_value,
        code_key,
        code_value,
        key_scales,
        value_scales,
        key_center,
        value_center,
        exact_tail_pages=exact_pages,
        exact_sink_pages=exact_sink_pages,
        exact_static_suffix_pages=exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )
    for request in range(batch_size):
        output_page = request * pages + initial_pages
        cache.key_codes[:, output_page].zero_()
        cache.value_codes[:, output_page].zero_()
        cache.key_scales[:, output_page].zero_()
        cache.value_scales[:, output_page].zero_()
    return cache


def _source_boundary_sample(
    source_key: torch.Tensor, source_value: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        (
            source_key[0].reshape(-1)[:64],
            source_key[-1].reshape(-1)[:64],
            source_value[0].reshape(-1)[:64],
            source_value[-1].reshape(-1)[:64],
        )
    ).contiguous()


def token_major_centers(
    source_key: torch.Tensor, source_value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce only T in token-major [T,Hkv,D] K/V tensors.

    This deliberately lives outside the CUDA-only fixture so its shape and
    per-head semantics have a cheap CPU regression.  Collapsing Hkv here is a
    numerically plausible broadcasting bug, so fail before cache population
    rather than relying only on the downstream quantizer's shape check.
    """
    if source_key.shape != source_value.shape or source_key.ndim != 3:
        raise ValueError("token-major K/V must have identical [T,Hkv,D] shapes")
    if source_key.shape[0] <= 0 or source_key.shape[1] <= 0:
        raise ValueError("token-major K/V must contain tokens and heads")
    if int(source_key.shape[2]) != PG.DIM:
        raise ValueError(f"token-major K/V head dimension must be {PG.DIM}")
    key_center = source_key.float().mean(dim=0).half()
    value_center = source_value.float().mean(dim=0).half()
    expected = (int(source_key.shape[1]), PG.DIM)
    if tuple(key_center.shape) != expected or tuple(value_center.shape) != expected:
        raise RuntimeError(
            f"token-axis center reduction produced K={tuple(key_center.shape)}, "
            f"V={tuple(value_center.shape)}; expected {expected}"
        )
    return key_center, value_center


@torch.inference_mode()
def populate_backend_cache_layer(
    *,
    backend: str,
    cache: Any,
    layer: int,
    request: int,
    pages: int,
    initial_pages: int,
    exact_pages: int,
    source_key: torch.Tensor,
    source_value: torch.Tensor,
    exact_sink_pages: int = 0,
    exact_static_suffix_pages: int = 0,
) -> bool:
    """Populate one request/layer without materializing the other backend."""
    exact_sink_pages = PG.validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=initial_pages,
    )
    if backend != "page_gauge" and (exact_sink_pages or exact_static_suffix_pages):
        raise ValueError("fixed exact pages are supported only by PageGauge")
    if tuple(source_key.shape) != tuple(source_value.shape):
        raise RuntimeError("HF K/V shapes differ")
    physical_begin = request * pages
    physical_end = physical_begin + initial_pages
    if backend == "flashinfer_fp16":
        destination_key = cache.key[layer, physical_begin:physical_end]
        destination_value = cache.value[layer, physical_begin:physical_end]
        destination_key.copy_(source_key.view_as(destination_key))
        destination_value.copy_(source_value.view_as(destination_value))
        return bool(
            torch.equal(destination_key[0], source_key[: PG.PAGE])
            and torch.equal(destination_key[-1], source_key[-PG.PAGE :])
            and torch.equal(destination_value[0], source_value[: PG.PAGE])
            and torch.equal(destination_value[-1], source_value[-PG.PAGE :])
        )

    # The equivalent paged representation reduces axes P and 16.  Preserve
    # Hkv and D exactly; token_major_centers has a CPU regression for this.
    key_center, value_center = token_major_centers(source_key, source_value)
    cache.key_center[layer, request].copy_(key_center)
    cache.value_center[layer, request].copy_(value_center)
    # GaugeCache normally derives output_center in its constructor.  This
    # direct-build path intentionally allocates the cache before the HF prefix
    # exists, so refresh the derived GQA-expanded value center here.
    cache.output_center[layer, request].copy_(
        value_center.repeat_interleave(4, dim=0)
    )
    paged_key = source_key.view(initial_pages, PG.PAGE, *source_key.shape[1:])
    paged_value = source_value.view(initial_pages, PG.PAGE, *source_value.shape[1:])
    PG.OVERHEAD.quantize_completed_kv_pages(
        paged_key,
        paged_value,
        cache.key_center[layer, request],
        cache.value_center[layer, request],
        cache.key_codes[layer, physical_begin:physical_end],
        cache.value_codes[layer, physical_begin:physical_end],
        cache.key_scales[layer, physical_begin:physical_end],
        cache.value_scales[layer, physical_begin:physical_end],
    )
    if (
        cache.exact_tail_pages != exact_pages
        or cache.exact_sink_pages != exact_sink_pages
        or cache.exact_static_suffix_pages != exact_static_suffix_pages
        or cache.initial_context_pages != initial_pages
    ):
        raise ValueError(
            "fixture fixed-exact/tail layout does not match population policy"
        )
    if initial_pages <= (
        exact_pages + exact_sink_pages + exact_static_suffix_pages
    ):
        raise ValueError("fixture must contain at least one quantized old page")
    tail_logical_begin = initial_pages - exact_pages
    partition = PG.page_gauge_logical_partition(
        initial_pages,
        exact_pages,
        exact_sink_pages,
        PG.PAGE,
        exact_static_suffix_pages,
        initial_pages,
    )
    exact_logical_pages = tuple(partition["exact_logical_pages"])
    for logical_page in exact_logical_pages:
        source_begin = logical_page * PG.PAGE
        source_end = source_begin + PG.PAGE
        ring_page = PG.exact_physical_page_index(
            request,
            logical_page,
            exact_pages,
            exact_sink_pages,
            exact_static_suffix_pages,
            initial_pages,
        )
        centered_key = (
            source_key[source_begin:source_end].float()
            - cache.key_center[layer, request][None].float()
        ).half()
        centered_value = (
            source_value[source_begin:source_end].float()
            - cache.value_center[layer, request][None].float()
        ).half()
        expected_exact_shape = (PG.PAGE, source_key.shape[1], source_key.shape[2])
        if (
            tuple(centered_key.shape) != expected_exact_shape
            or tuple(centered_value.shape) != expected_exact_shape
        ):
            raise RuntimeError(
                "centered exact-ring page has an unexpected shape: "
                f"K={tuple(centered_key.shape)}, V={tuple(centered_value.shape)}, "
                f"expected={expected_exact_shape}"
            )
        cache.exact_key[layer, ring_page].copy_(centered_key)
        cache.exact_value[layer, ring_page].copy_(centered_value)
    sampled_logical_pages = list(range(exact_sink_pages))
    sampled_logical_pages.extend(partition["static_suffix_logical_pages"])
    sampled_logical_pages.extend((tail_logical_begin, initial_pages - 1))
    sampled_logical_pages = list(dict.fromkeys(sampled_logical_pages))
    sampled_physical_pages = torch.tensor(
        [
            PG.exact_physical_page_index(
                request,
                logical_page,
                exact_pages,
                exact_sink_pages,
                exact_static_suffix_pages,
                initial_pages,
            )
            for logical_page in sampled_logical_pages
        ],
        device=source_key.device,
        dtype=torch.long,
    )
    expected_key = torch.stack(
        [
            (
                source_key[
                    logical_page * PG.PAGE : (logical_page + 1) * PG.PAGE
                ].float()
                - cache.key_center[layer, request][None].float()
            ).half()
            for logical_page in sampled_logical_pages
        ]
    )
    expected_value = torch.stack(
        [
            (
                source_value[
                    logical_page * PG.PAGE : (logical_page + 1) * PG.PAGE
                ].float()
                - cache.value_center[layer, request][None].float()
            ).half()
            for logical_page in sampled_logical_pages
        ]
    )
    observed_key = cache.exact_key[layer].index_select(0, sampled_physical_pages)
    observed_value = cache.exact_value[layer].index_select(
        0, sampled_physical_pages
    )
    return bool(
        torch.equal(
            torch.stack((observed_key, observed_value)),
            torch.stack((expected_key, expected_value)),
        )
    )


@torch.inference_mode()
def build_coherent_fixture(
    *,
    model: Any,
    tokens: torch.Tensor,
    cache: Any,
    backend: str,
    layers: int,
    hkv: int,
    pages: int,
    initial_pages: int,
    exact_pages: int,
    context: int,
    decode_steps: int,
    chunk_tokens: int,
    exact_sink_pages: int = 0,
    exact_static_suffix_pages: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]], str]:
    """Stream B1 HF prefixes directly into the selected persistent cache."""
    from transformers.cache_utils import DynamicCache

    batch_size = int(tokens.shape[0])
    logits_by_request: list[list[torch.Tensor]] = []
    generated_by_request: list[list[torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    sampled_fingerprint = hashlib.sha256()
    for request in range(batch_size):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        request_started = time.perf_counter()
        dynamic_cache = DynamicCache()
        exact_prefix_destination_fingerprint = hashlib.sha256()
        sampled_prefix_layers = sorted({0, layers - 1})
        for begin in range(0, context, chunk_tokens):
            end = min(context, begin + chunk_tokens)
            input_ids = tokens[request, begin:end].to("cuda")[None]
            positions = torch.arange(begin, end, device="cuda", dtype=torch.long)
            outputs = model.model(
                input_ids=input_ids,
                position_ids=positions[None],
                cache_position=positions,
                past_key_values=dynamic_cache,
                use_cache=True,
                return_dict=True,
            )
            dynamic_cache = outputs.past_key_values
            del input_ids, positions, outputs
        if int(dynamic_cache.get_seq_length()) != context:
            raise RuntimeError(
                f"request {request} HF cache has length "
                f"{dynamic_cache.get_seq_length()}, expected {context}"
            )

        sampled_copy_gate = True
        for layer in range(layers):
            hf_key, hf_value = PREFILL.cache_layer_tensors(dynamic_cache, layer)
            expected_shape = (1, hkv, context, PG.DIM)
            if tuple(hf_key.shape) != expected_shape or tuple(hf_value.shape) != expected_shape:
                raise RuntimeError(
                    f"HF layer {layer} cache K={tuple(hf_key.shape)}, "
                    f"V={tuple(hf_value.shape)}; expected {expected_shape}"
                )
            source_key = hf_key[0].transpose(0, 1).contiguous()
            source_value = hf_value[0].transpose(0, 1).contiguous()
            layer_population_gate = populate_backend_cache_layer(
                backend=backend,
                cache=cache,
                layer=layer,
                request=request,
                pages=pages,
                initial_pages=initial_pages,
                exact_pages=exact_pages,
                source_key=source_key,
                source_value=source_value,
                exact_sink_pages=exact_sink_pages,
                exact_static_suffix_pages=exact_static_suffix_pages,
            )
            sampled_copy_gate = sampled_copy_gate and layer_population_gate
            if layer in (0, layers - 1):
                sample = _source_boundary_sample(source_key, source_value).cpu()
                sampled_fingerprint.update(sample.numpy().tobytes())
                if backend == "page_gauge" and (
                    exact_sink_pages or exact_static_suffix_pages
                ):
                    fixed_logical_pages = list(range(exact_sink_pages))
                    fixed_logical_pages.extend(
                        range(
                            initial_pages - exact_static_suffix_pages,
                            initial_pages,
                        )
                    )
                    for logical_page in fixed_logical_pages:
                        physical_page = PG.exact_physical_page_index(
                            request,
                            logical_page,
                            exact_pages,
                            exact_sink_pages,
                            exact_static_suffix_pages,
                            initial_pages,
                        )
                        exact_prefix_destination_fingerprint.update(
                            f"layer={layer};request={request};logical={logical_page}".encode(
                                "utf-8"
                            )
                        )
                        for tensor in (
                            cache.exact_key[layer, physical_page],
                            cache.exact_value[layer, physical_page],
                        ):
                            exact_prefix_destination_fingerprint.update(
                                tensor.detach().contiguous().cpu().numpy().tobytes()
                            )
            del source_key, source_value

        request_logits: list[torch.Tensor] = []
        request_generated: list[torch.Tensor] = []
        current_token = tokens[request, context].reshape(1).to("cuda")
        for step in range(decode_steps):
            position = context + step
            position_tensor = torch.tensor([position], device="cuda", dtype=torch.long)
            outputs = model.model(
                input_ids=current_token.reshape(1, 1),
                position_ids=position_tensor[None],
                cache_position=position_tensor,
                past_key_values=dynamic_cache,
                use_cache=True,
                return_dict=True,
            )
            dynamic_cache = outputs.past_key_values
            logits = model.lm_head(outputs.last_hidden_state[:, -1]).float()
            next_token = logits.argmax(dim=-1)
            request_logits.append(logits.detach().cpu())
            request_generated.append(next_token.detach().cpu())
            current_token = next_token
            del position_tensor, outputs, logits
        torch.cuda.synchronize()
        records.append(
            {
                "request": request,
                "prefix_tokens": context,
                "reference_greedy_steps": decode_steps,
                "prefill_chunks": math.ceil(context / chunk_tokens),
                "chunk_tokens": chunk_tokens,
                "direct_selected_backend_population": True,
                "opposite_backend_full_cache_allocated": False,
                "sampled_population_gate_passed": sampled_copy_gate,
                "sampled_exact_prefix_population": {
                    "enabled": backend == "page_gauge"
                    and (exact_sink_pages + exact_static_suffix_pages) > 0,
                    "logical_pages": (
                        list(range(exact_sink_pages))
                        + list(
                            range(
                                initial_pages - exact_static_suffix_pages,
                                initial_pages,
                            )
                        )
                        if backend == "page_gauge"
                        else []
                    ),
                    "sampled_layers": (
                        sampled_prefix_layers
                        if backend == "page_gauge"
                        and (exact_sink_pages + exact_static_suffix_pages)
                        else []
                    ),
                    "all_prefix_pages_checked_in_every_layer": (
                        backend == "page_gauge"
                        and (exact_sink_pages + exact_static_suffix_pages) > 0
                    ),
                    "sampled_destination_sha256": (
                        exact_prefix_destination_fingerprint.hexdigest()
                        if backend == "page_gauge"
                        and (exact_sink_pages + exact_static_suffix_pages)
                        else None
                    ),
                },
                "allocated_before_bytes": allocated_before,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "incremental_peak_allocated_bytes": max(
                    0, int(torch.cuda.max_memory_allocated()) - allocated_before
                ),
                "wall_ms": (time.perf_counter() - request_started) * 1e3,
            }
        )
        logits_by_request.append(request_logits)
        generated_by_request.append(request_generated)
        del dynamic_cache, current_token, request_logits, request_generated
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logits = [
        torch.cat(
            [logits_by_request[request][step] for request in range(batch_size)],
            dim=0,
        )
        for step in range(decode_steps)
    ]
    generated = [
        torch.cat(
            [generated_by_request[request][step] for request in range(batch_size)],
            dim=0,
        )
        for step in range(decode_steps)
    ]
    return logits, generated, records, sampled_fingerprint.hexdigest()


def summarize_latency(values: list[float], steps: int, batch_size: int) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        coordinate = probability * (len(ordered) - 1)
        lower = math.floor(coordinate)
        upper = math.ceil(coordinate)
        if lower == upper:
            return float(ordered[lower])
        weight = coordinate - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    output_tokens = steps * batch_size
    mean_ms = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms_per_sequence": mean_ms,
        "median_ms_per_sequence": float(statistics.median(values)),
        "minimum_ms_per_sequence": float(min(values)),
        "maximum_ms_per_sequence": float(max(values)),
        "p95_ms_per_sequence": percentile(0.95),
        "mean_ms_per_decode_step": mean_ms / steps,
        "mean_ms_per_output_token": mean_ms / output_tokens,
        "output_tokens_per_second": output_tokens / (mean_ms / 1e3),
    }


@torch.inference_mode()
def timed_generated_block(
    decoder: Any,
    initial_token: torch.Tensor,
    dynamic_token: torch.Tensor,
    start_position: int,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    begin.record()
    TOKEN.execute_direct_feedback(
        decoder,
        initial_token,
        start_position,
        dynamic_token,
        collect=False,
        synchronize_each_token=False,
    )
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end)), (time.perf_counter() - wall_started) * 1e3


def snapshot_mutated_page(decoder: Any, start_position: int) -> dict[str, Any]:
    """Snapshot the generated page without aliasing an immutable exact prefix.

    The legacy full-sequence helper assumes that the exact cache is only a
    T-page ring.  A PageGauge cache with a fixed S-page prefix instead has a
    request stride of S+T and maps generated pages to S+(logical mod T).
    Keep that policy local to this worker so the publication-independent
    legacy diagnostic retains its original ABI.
    """

    if decoder.backend != "page_gauge":
        return OUTER.snapshot_mutated_page(decoder, start_position)
    logical_page, offset = divmod(start_position, PG.PAGE)
    if offset:
        raise ValueError("the backend-exclusive graph must begin on a page boundary")
    prefix_pages = int(
        getattr(
            decoder,
            "exact_prefix_pages",
            getattr(decoder, "exact_sink_pages", 0),
        )
    )
    if logical_page < prefix_pages:
        raise ValueError("a generated page cannot overlap the immutable exact prefix")
    exact_physical_pages = tuple(
        decoder.exact_physical_page(request, logical_page)
        for request in range(decoder.batch_size)
    )
    exact_storage_pages = prefix_pages + int(decoder.exact_tail_pages)
    expected_exact_pages = tuple(
        request * exact_storage_pages
        + prefix_pages
        + logical_page % int(decoder.exact_tail_pages)
        for request in range(decoder.batch_size)
    )
    if exact_physical_pages != expected_exact_pages:
        raise RuntimeError("decoder exact-page mapping disagrees with prefix+tail ABI")
    if len(set(exact_physical_pages)) != decoder.batch_size:
        raise RuntimeError("request-major exact generated pages must be unique")
    for request, physical_page in enumerate(exact_physical_pages):
        request_base = request * exact_storage_pages
        if not request_base + prefix_pages <= physical_page < request_base + exact_storage_pages:
            raise RuntimeError("generated exact page aliases a prefix or another request")
    code_physical_pages = tuple(
        request * decoder.max_pages + logical_page
        for request in range(decoder.batch_size)
    )
    return {
        "backend": decoder.backend,
        "logical_page": logical_page,
        "ring_page": logical_page % int(decoder.exact_tail_pages),
        "exact_prefix_pages": prefix_pages,
        "exact_storage_pages_per_request": exact_storage_pages,
        "physical_pages": {
            "exact": exact_physical_pages,
            "codes": code_physical_pages,
        },
        "tensors": {
            "exact_key": torch.stack(
                [decoder.cache.exact_key[:, page] for page in exact_physical_pages],
                dim=1,
            ).clone(),
            "exact_value": torch.stack(
                [decoder.cache.exact_value[:, page] for page in exact_physical_pages],
                dim=1,
            ).clone(),
            "key_codes": torch.stack(
                [decoder.cache.key_codes[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
            "value_codes": torch.stack(
                [decoder.cache.value_codes[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
            "key_scales": torch.stack(
                [decoder.cache.key_scales[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
            "value_scales": torch.stack(
                [decoder.cache.value_scales[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
        },
    }


def restore_mutated_page(decoder: Any, snapshot: dict[str, Any]) -> None:
    """Validate the frozen mapping before delegating tensor restoration."""

    if decoder.backend == "page_gauge":
        prefix_pages = int(
            getattr(
                decoder,
                "exact_prefix_pages",
                getattr(decoder, "exact_sink_pages", 0),
            )
        )
        if snapshot.get("exact_prefix_pages") != prefix_pages:
            raise ValueError("cache snapshot belongs to a different exact-prefix policy")
        logical_page = int(snapshot["logical_page"])
        if logical_page < prefix_pages:
            raise ValueError("cache snapshot logical page overlaps the exact prefix")
        expected_storage_pages = prefix_pages + int(decoder.exact_tail_pages)
        if snapshot.get("exact_storage_pages_per_request") != expected_storage_pages:
            raise ValueError("cache snapshot exact-storage request stride is stale")
        expected = tuple(
            decoder.exact_physical_page(request, logical_page)
            for request in range(decoder.batch_size)
        )
        if tuple(snapshot["physical_pages"]["exact"]) != expected:
            raise ValueError("cache snapshot exact-page mapping is stale")
    OUTER.restore_mutated_page(decoder, snapshot)


def immutable_exact_prefix_digest(decoder: Any) -> dict[str, Any]:
    """Hash every request/layer prefix slot with one staged device transfer."""

    prefix_pages = int(
        getattr(
            decoder,
            "exact_prefix_pages",
            getattr(decoder, "exact_sink_pages", 0),
        )
    )
    if decoder.backend != "page_gauge" or prefix_pages == 0:
        return {
            "enabled": False,
            "exact_prefix_pages": 0,
            "logical_pages": [],
            "physical_pages": [],
        }
    if decoder.cache.exact_key.shape != decoder.cache.exact_value.shape:
        raise RuntimeError("exact-prefix K/V storage geometry differs")
    total_storage_pages = int(decoder.cache.exact_key.shape[1])
    if total_storage_pages % decoder.batch_size:
        raise RuntimeError("exact-prefix storage is not request-major")
    storage_pages = total_storage_pages // decoder.batch_size
    if storage_pages != prefix_pages + int(decoder.exact_tail_pages):
        raise RuntimeError("exact-prefix storage stride disagrees with S+T")
    logical_pages = list(range(prefix_pages))
    physical_pages = [
        decoder.exact_physical_page(request, logical_page)
        for request in range(decoder.batch_size)
        for logical_page in logical_pages
    ]
    expected_physical_pages = [
        request * storage_pages + logical_page
        for request in range(decoder.batch_size)
        for logical_page in logical_pages
    ]
    if physical_pages != expected_physical_pages:
        raise RuntimeError("decoder exact-prefix mapping disagrees with fixed slots")
    if len(set(physical_pages)) != decoder.batch_size * prefix_pages:
        raise RuntimeError("exact-prefix physical slots are not request-major unique")
    index = torch.tensor(
        physical_pages,
        device=decoder.cache.exact_key.device,
        dtype=torch.long,
    )
    staged = torch.stack(
        (
            decoder.cache.exact_key.index_select(1, index),
            decoder.cache.exact_value.index_select(1, index),
        )
    ).detach().contiguous().cpu()
    key_payload = staged[0].numpy().tobytes()
    value_payload = staged[1].numpy().tobytes()
    combined = hashlib.sha256()
    combined.update(key_payload)
    combined.update(value_payload)
    return {
        "enabled": True,
        "exact_prefix_pages": prefix_pages,
        "logical_pages": logical_pages,
        "physical_pages": physical_pages,
        "physical_page_count": len(physical_pages),
        "tensor_sha256": {
            "exact_key": hashlib.sha256(key_payload).hexdigest(),
            "exact_value": hashlib.sha256(value_payload).hexdigest(),
        },
        "combined_sha256": combined.hexdigest(),
    }


def compare_exact_prefix_digest(
    expected: dict[str, Any], decoder: Any, phase: str
) -> dict[str, Any]:
    observed = immutable_exact_prefix_digest(decoder)
    return {
        "phase": phase,
        "passed": expected == observed,
        "enabled": bool(expected.get("enabled")),
        "exact_prefix_pages": int(observed.get("exact_prefix_pages", 0)),
        "physical_page_count": int(observed.get("physical_page_count", 0)),
        "expected_combined_sha256": expected.get("combined_sha256"),
        "observed_combined_sha256": observed.get("combined_sha256"),
    }


@torch.inference_mode()
def measure_timing_modes(
    *,
    decoder: Any,
    initial_token: torch.Tensor,
    dynamic_token: torch.Tensor,
    initial_cache: dict[str, Any],
    start_position: int,
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    expected_prefix_digest: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    experiment_started = time.perf_counter()
    for mode in ("cache_neutral", "cache_hot"):
        prefix_canary_records: list[dict[str, Any]] = []
        warmup_samples: list[dict[str, Any]] = []
        for warmup in range(warmups):
            restore_mutated_page(decoder, initial_cache)
            torch.cuda.synchronize()
            precondition = None
            if mode == "cache_neutral":
                cache_scrub.add_(1)
                torch.cuda.synchronize()
            else:
                hot_cuda, hot_wall = timed_generated_block(
                    decoder, initial_token, dynamic_token, start_position
                )
                decoder.plan(start_position)
                torch.cuda.synchronize()
                precondition = {
                    "cuda_ms": hot_cuda,
                    "wall_ms": hot_wall,
                    "precondition_steps": PG.PAGE,
                    "planner_reset_without_cache_restore": True,
                    "exact_prefix_attestation": (
                        "covered jointly by the post-sample canary so no "
                        "candidate-only digest perturbs the hot precondition"
                    ),
                    "no_restore_before_timed_sample": True,
                }
            cuda_ms, wall_ms = timed_generated_block(
                decoder, initial_token, dynamic_token, start_position
            )
            sample_canary = compare_exact_prefix_digest(
                expected_prefix_digest,
                decoder,
                f"{mode}.warmup.{warmup}.sample",
            )
            prefix_canary_records.append(sample_canary)
            warmup_samples.append(
                {
                    "warmup_index": warmup,
                    "elapsed_seconds": time.perf_counter() - experiment_started,
                    "precondition": precondition,
                    "cuda_ms": cuda_ms,
                    "wall_ms": wall_ms,
                    "exact_prefix_canary": sample_canary,
                }
            )

        raw_samples: list[dict[str, Any]] = []
        for sample_index in range(repeats):
            sample_started = time.perf_counter()
            restore_mutated_page(decoder, initial_cache)
            torch.cuda.synchronize()
            precondition = None
            if mode == "cache_neutral":
                cache_scrub.add_(1)
                torch.cuda.synchronize()
            else:
                hot_cuda, hot_wall = timed_generated_block(
                    decoder, initial_token, dynamic_token, start_position
                )
                decoder.plan(start_position)
                torch.cuda.synchronize()
                precondition = {
                    "cuda_ms": hot_cuda,
                    "wall_ms": hot_wall,
                    "precondition_steps": PG.PAGE,
                    "planner_reset_without_cache_restore": True,
                    "exact_prefix_attestation": (
                        "covered jointly by the post-sample canary so no "
                        "candidate-only digest perturbs the hot precondition"
                    ),
                    "no_restore_before_timed_sample": True,
                }
            cuda_ms, wall_ms = timed_generated_block(
                decoder, initial_token, dynamic_token, start_position
            )
            sample_canary = compare_exact_prefix_digest(
                expected_prefix_digest,
                decoder,
                f"{mode}.sample.{sample_index}.sample",
            )
            prefix_canary_records.append(sample_canary)
            raw_samples.append(
                {
                    "sample_index": sample_index,
                    "chronological_index": sample_index,
                    "start_seconds": sample_started - experiment_started,
                    "end_seconds": time.perf_counter() - experiment_started,
                    "precondition": precondition,
                    "cuda_ms": cuda_ms,
                    "wall_ms": wall_ms,
                    "exact_prefix_canary": sample_canary,
                }
            )
        cuda_values = [float(sample["cuda_ms"]) for sample in raw_samples]
        wall_values = [float(sample["wall_ms"]) for sample in raw_samples]
        result[mode] = {
            "mode": mode,
            "raw_samples": raw_samples,
            "raw_cuda_ms": cuda_values,
            "raw_wall_ms": wall_values,
            "cuda_ms": cuda_values,
            "wall_ms": wall_values,
            "warmup_samples": warmup_samples,
            "cuda_summary": summarize_latency(
                cuda_values, PG.PAGE, decoder.batch_size
            ),
            "wall_summary": summarize_latency(
                wall_values, PG.PAGE, decoder.batch_size
            ),
            "restore_inside_timed_boundary": False,
            "precondition_inside_timed_boundary": False,
            "no_restore_between_hot_precondition_and_sample": True,
            "hot_precondition_steps": PG.PAGE,
            "planner_reset_without_cache_restore": True,
            "immutable_exact_prefix_canary": {
                "enabled": bool(expected_prefix_digest.get("enabled")),
                "passed": all(
                    record["passed"] for record in prefix_canary_records
                ),
                "check_count": len(prefix_canary_records),
                "checks": prefix_canary_records,
                "checked_after_terminal_synchronize_before_restore": True,
                "no_prefix_digest_between_hot_precondition_and_sample": True,
                "inside_timed_boundary": False,
            },
        }
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    return result


def build_generated_input_sequence(
    initial_token: torch.Tensor,
    generated_tokens: list[torch.Tensor],
) -> torch.Tensor:
    if len(generated_tokens) != PG.PAGE:
        raise ValueError("expected one generated page")
    rows = [initial_token.detach().cpu()]
    rows.extend(token.detach().cpu() for token in generated_tokens[:-1])
    return torch.stack(rows, dim=0).to(device="cuda", dtype=torch.long)


def residency_gate(
    profile: dict[str, Any],
    *,
    ratio_limit: float,
    absolute_limit_ms: float,
) -> dict[str, Any]:
    requested = [0, 15, 20, 23, 31]
    available = {
        int(record["layer"]): record for record in profile["per_layer_records"]
    }
    selected_indices = [index for index in requested if index in available]
    if selected_indices != requested:
        raise RuntimeError(
            f"residency gate requires layers {requested}, available={sorted(available)}"
        )
    all_attention = [
        float(record["attention_gpu_ms"])
        for record in profile["per_layer_records"]
    ]
    median_attention = float(statistics.median(all_attention))
    ceiling = max(absolute_limit_ms, ratio_limit * median_attention)
    selected = [
        {
            **available[index],
            "ratio_to_all_layer_median": (
                float(available[index]["attention_gpu_ms"]) / median_attention
                if median_attention > 0
                else math.inf
            ),
        }
        for index in requested
    ]
    finite_positive = all(
        math.isfinite(value) and value > 0.0 for value in all_attention
    )
    selected_max = max(float(row["attention_gpu_ms"]) for row in selected)
    passed = finite_positive and selected_max <= ceiling
    return {
        "passed": passed,
        "selected_layers": requested,
        "selected_layer_records": selected,
        "all_layer_attention_median_ms": median_attention,
        "selected_layer_maximum_attention_ms": selected_max,
        "ratio_limit": ratio_limit,
        "absolute_limit_ms": absolute_limit_ms,
        "effective_ceiling_ms": ceiling,
        "finite_positive_all_layers": finite_positive,
        "interpretation": (
            "fails when a selected early/middle/late attention graph exceeds "
            "both the absolute ceiling and the allowed multiple of the "
            "whole-model median; intended to detect the prior WSL/WDDM "
            "backing-placement cliff"
        ),
    }


def orchestration_environment() -> dict[str, str]:
    prefixes = ("PAGE_GAUGE_", "BACKEND_EXCLUSIVE_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    memory: dict[str, Any] = {"process_start": TOKEN.gpu_memory_state()}

    print(f"Loading local HF FP16 model for {args.backend}...", flush=True)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    layers, hq, hkv, hidden = PG.BASE_E2E.check_model(model)
    if (layers, hq, hkv, PG.DIM) != (32, 32, 8, 128):
        raise RuntimeError(
            "publication worker requires Mistral-7B geometry "
            f"L/Hq/Hkv/D=32/32/8/128, got {layers}/{hq}/{hkv}/{PG.DIM}"
        )
    config_payload = model.config.to_dict()
    parameter_sample = sampled_model_parameter_sha256(model)
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
    pages = math.ceil(max_context / PG.PAGE)
    initial_pages = args.context // PG.PAGE
    exact_pages = args.exact_tail // PG.PAGE
    cache = allocate_backend_cache(
        args.backend,
        layers,
        pages,
        initial_pages,
        exact_pages,
        args.batch_size,
        hkv,
        exact_sink_pages=(
            args.exact_sink_pages if args.backend == "page_gauge" else 0
        ),
    )
    torch.cuda.synchronize()
    memory["selected_backend_cache_allocated"] = TOKEN.gpu_memory_state()
    cache_manifest = cache_tensor_manifest(cache, args.backend)
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
        cache_manifest["exact_layout"] = {
            "tail_pages_per_request": exact_pages,
            "prefix_pages_per_request": args.exact_sink_pages,
            "storage_pages_per_request": exact_pages + args.exact_sink_pages,
            "prefix_logical_pages": list(range(args.exact_sink_pages)),
            "physical_order": (
                "[fixed contiguous prefix slots 0..S-1, modulo tail-ring "
                "slots S..S+T-1]"
            ),
            "prefix_gross_allocated_bytes": prefix_gross_bytes,
            "all_exact_prefix_int8_code_and_scale_storage_retained": bool(
                args.exact_sink_pages
            ),
            "compact_replacement_not_claimed": True,
        }

    print(
        "Generating coherent B1 prefixes and streaming them directly into "
        f"{args.backend} storage...",
        flush=True,
    )
    hf_logits, hf_generated, prefill_records, kv_sample_sha256 = (
        build_coherent_fixture(
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
            exact_sink_pages=(
                args.exact_sink_pages if args.backend == "page_gauge" else 0
            ),
        )
    )
    if not all(record["sampled_population_gate_passed"] for record in prefill_records):
        raise RuntimeError("sampled direct backend-cache population gate failed")
    memory["after_hf_dynamic_caches_released"] = TOKEN.gpu_memory_state()

    print("Packing common projections and constructing the selected decoder...", flush=True)
    PG.BASE_E2E.pack_model_projections(model)
    gc.collect()
    torch.cuda.empty_cache()
    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    with torch.inference_mode():
        position_ids = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        rope_probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(rope_probe, position_ids)
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
        max_context,
        args.exact_tail,
        args.baseline_split_pages,
        args.candidate_split_pages,
        rope_cos,
        rope_sin,
        "attention_add",
        args.tail_attention,
        args.batch_size,
        exact_sink_pages=(
            args.exact_sink_pages if args.backend == "page_gauge" else 0
        ),
    )
    del rope_probe, position_ids
    torch.cuda.synchronize()
    memory["decoder_constructed"] = TOKEN.gpu_memory_state()

    initial_token = tokens[:, args.context].to(device="cuda", dtype=torch.long)
    dynamic_token = torch.empty_like(initial_token)
    initial_cache = snapshot_mutated_page(decoder, args.context)
    initial_prefix_digest = immutable_exact_prefix_digest(decoder)
    prefix_canary_gates: dict[str, dict[str, Any]] = {}
    torch.cuda.synchronize()

    print(
        f"Validating eager then matched {args.cuda_graph_scope} graphs...",
        flush=True,
    )
    decoder.attention_graphs = None
    decoder.decoder_layer_graphs = None
    decoder.reset_attention_dispatch_counts()
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    eager_outputs, eager_generated = TOKEN.execute_direct_feedback(
        decoder,
        initial_token,
        args.context,
        dynamic_token,
        collect=True,
    )
    torch.cuda.synchronize()
    eager_cache = snapshot_mutated_page(decoder, args.context)
    prefix_canary_gates["eager"] = compare_exact_prefix_digest(
        initial_prefix_digest, decoder, "eager"
    )
    eager_attention_dispatch = decoder.attention_dispatch_counts()
    eager_layer_dispatch = decoder.decoder_layer_dispatch_counts()

    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    memory["before_cuda_graph_capture"] = TOKEN.gpu_memory_state()
    if args.cuda_graph_scope == "decoder_layer":
        decoder.capture_decoder_layer_graphs(args.context)
    else:
        decoder.capture_attention_graphs(args.context + 1)
    memory["after_cuda_graph_capture"] = TOKEN.gpu_memory_state()
    memory["cuda_graph_capture_delta"] = {
        field: (
            memory["after_cuda_graph_capture"][field]
            - memory["before_cuda_graph_capture"][field]
        )
        for field in (
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "torch_max_allocated_bytes",
            "torch_max_reserved_bytes",
            "cuda_mem_get_info_free_bytes",
        )
    }
    # Warmup and graph capture execute the append/finalization path. Restore
    # the entire mutated logical page before any equivalence observation.
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    decoder.reset_attention_dispatch_counts()
    graph_outputs, graph_generated = TOKEN.execute_direct_feedback(
        decoder,
        initial_token,
        args.context,
        dynamic_token,
        collect=True,
    )
    torch.cuda.synchronize()
    graph_cache = snapshot_mutated_page(decoder, args.context)
    prefix_canary_gates["graph_run_1"] = compare_exact_prefix_digest(
        initial_prefix_digest, decoder, "graph_run_1"
    )
    graph_attention_dispatch = decoder.attention_dispatch_counts()
    graph_layer_dispatch = decoder.decoder_layer_dispatch_counts()

    # A second replay without restoring the cache proves that the page-close
    # finalization and all in-page overwrites are deterministic/idempotent.
    decoder.reset_attention_dispatch_counts()
    repeated_graph_outputs, repeated_graph_generated = TOKEN.execute_direct_feedback(
        decoder,
        initial_token,
        args.context,
        dynamic_token,
        collect=True,
    )
    torch.cuda.synchronize()
    repeated_graph_cache = snapshot_mutated_page(decoder, args.context)
    prefix_canary_gates["graph_run_2_unrestored"] = compare_exact_prefix_digest(
        initial_prefix_digest, decoder, "graph_run_2_unrestored"
    )
    repeated_graph_attention_dispatch = decoder.attention_dispatch_counts()
    repeated_graph_layer_dispatch = decoder.decoder_layer_dispatch_counts()
    memory["attention_graphs_captured"] = TOKEN.gpu_memory_state()

    eager_vs_graph_logits = PG.compare_logit_sequences(eager_outputs, graph_outputs)
    eager_vs_graph_tokens = TOKEN.compare_token_sequences(
        [token.cpu() for token in eager_generated],
        [token.cpu() for token in graph_generated],
    )
    eager_vs_graph_cache = OUTER.compare_cache_snapshots(eager_cache, graph_cache)
    repeated_graph_logits = PG.compare_logit_sequences(
        graph_outputs, repeated_graph_outputs
    )
    repeated_graph_tokens = TOKEN.compare_token_sequences(
        [token.cpu() for token in graph_generated],
        [token.cpu() for token in repeated_graph_generated],
    )
    repeated_graph_cache_comparison = OUTER.compare_cache_snapshots(
        graph_cache, repeated_graph_cache
    )
    expected_graph_calls = layers * args.decode_steps
    if args.cuda_graph_scope == "decoder_layer":
        eager_selected_dispatch = eager_layer_dispatch
        graph_selected_dispatch = graph_layer_dispatch
        repeated_selected_dispatch = repeated_graph_layer_dispatch
        nested_attention_dispatch_passed = bool(
            graph_attention_dispatch["total_calls"] == 0
            and repeated_graph_attention_dispatch["total_calls"] == 0
        )
    else:
        eager_selected_dispatch = eager_attention_dispatch
        graph_selected_dispatch = graph_attention_dispatch
        repeated_selected_dispatch = repeated_graph_attention_dispatch
        nested_attention_dispatch_passed = True
    dispatch_passed = bool(
        eager_selected_dispatch["eager_calls"] == expected_graph_calls
        and eager_selected_dispatch["graph_replays"] == 0
        and graph_selected_dispatch["graph_replays"] == expected_graph_calls
        and graph_selected_dispatch["eager_calls"] == 0
        and repeated_selected_dispatch["graph_replays"] == expected_graph_calls
        and repeated_selected_dispatch["eager_calls"] == 0
        and nested_attention_dispatch_passed
    )
    graph_structure = (
        decoder.decoder_layer_graph_provenance()
        if args.cuda_graph_scope == "decoder_layer"
        else {
            "enabled": decoder.attention_graphs is not None,
            "scope": "attention",
            "total_graphs": len(decoder.attention_graphs or []),
        }
    )
    graph_structure_passed = bool(
        graph_structure["enabled"]
        and graph_structure["total_graphs"]
        == (
            expected_graph_calls
            if args.cuda_graph_scope == "decoder_layer"
            else layers
        )
    )
    within_backend_passed = bool(
        eager_vs_graph_logits["bitwise_identical"]
        and eager_vs_graph_tokens["bitwise_identical"]
        and eager_vs_graph_cache["passed"]
        and repeated_graph_logits["bitwise_identical"]
        and repeated_graph_tokens["bitwise_identical"]
        and repeated_graph_cache_comparison["passed"]
        and dispatch_passed
        and graph_structure_passed
    )

    hf_vs_backend = PREFILL.compare_logits(
        hf_logits,
        graph_outputs,
        reference_name="HF SDPA FP16 greedy",
        candidate_name=f"{args.backend} {args.cuda_graph_scope} graphs greedy",
        predicted_position_start=args.context + 1,
    )
    hf_vs_backend_tokens = TOKEN.compare_token_sequences(
        [token.cpu() for token in hf_generated],
        [token.cpu() for token in graph_generated],
    )
    cross_reference_passed = bool(
        hf_vs_backend["minimum_logits_cosine"] >= args.min_logits_cosine
        and hf_vs_backend["top1_agreement_fraction"] >= args.min_top1_agreement
        and hf_vs_backend_tokens["bitwise_identical"]
    )

    generated_inputs = build_generated_input_sequence(initial_token, hf_generated)
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    saved_decoder_layer_graphs = decoder.decoder_layer_graphs
    if args.cuda_graph_scope == "decoder_layer":
        # The component profiler monkeypatches decoder.attention to place
        # events around the real selected attention path. A whole-layer graph
        # bypasses that Python hook, so profile eagerly and then restore the
        # publication timing boundary without releasing its graph memory.
        decoder.decoder_layer_graphs = None
    try:
        profile = TOKEN.profile_representative_static_token(
            decoder,
            generated_inputs,
            args.context,
            initial_cache,
            representative_offset=PG.PAGE - 1,
        )
    finally:
        decoder.decoder_layer_graphs = saved_decoder_layer_graphs
    profile.pop("selected_real_queries", None)
    profile["cuda_graph_scope_during_profile"] = (
        "none" if args.cuda_graph_scope == "decoder_layer" else "attention"
    )
    profile["publication_cuda_graph_scope_restored_after_profile"] = (
        args.cuda_graph_scope
    )
    residency = residency_gate(
        profile,
        ratio_limit=args.residency_ratio_limit,
        absolute_limit_ms=args.residency_absolute_limit_ms,
    )
    prefix_canary_gates["component_profile"] = compare_exact_prefix_digest(
        initial_prefix_digest, decoder, "component_profile"
    )
    memory["after_residency_gate"] = TOKEN.gpu_memory_state()

    cache_scrub = torch.empty(
        args.cache_scrub_mib * 1024 * 1024,
        device="cuda",
        dtype=torch.uint8,
    )
    cache_scrub.zero_()
    torch.cuda.synchronize()
    memory["before_timing"] = TOKEN.gpu_memory_state()
    print(
        f"Timing {args.backend}: warmups={args.warmups}, repeats={args.repeats}...",
        flush=True,
    )
    timing_modes = measure_timing_modes(
        decoder=decoder,
        initial_token=initial_token,
        dynamic_token=dynamic_token,
        initial_cache=initial_cache,
        start_position=args.context,
        cache_scrub=cache_scrub,
        warmups=args.warmups,
        repeats=args.repeats,
        expected_prefix_digest=initial_prefix_digest,
    )
    timed_prefix_canary_gate = {
        "enabled": bool(initial_prefix_digest.get("enabled")),
        "passed": all(
            mode["immutable_exact_prefix_canary"]["passed"]
            for mode in timing_modes.values()
        ),
        "check_count": sum(
            mode["immutable_exact_prefix_canary"]["check_count"]
            for mode in timing_modes.values()
        ),
        "modes": {
            name: mode["immutable_exact_prefix_canary"]
            for name, mode in timing_modes.items()
        },
    }
    prefix_canary_gate = {
        "enabled": bool(initial_prefix_digest.get("enabled")),
        "initial": initial_prefix_digest,
        "phase_checks": prefix_canary_gates,
        "timed_samples": timed_prefix_canary_gate,
        "passed": bool(
            all(record["passed"] for record in prefix_canary_gates.values())
            and timed_prefix_canary_gate["passed"]
        ),
        "mutable_restore_excludes_prefix": True,
        "all_requests_and_prefix_slots_hashed": True,
    }
    within_backend_passed = bool(
        within_backend_passed and prefix_canary_gate["passed"]
    )
    memory["after_timing"] = TOKEN.gpu_memory_state()

    source_paths = (
        Path(__file__),
        PREFILL_PATH,
        TOKEN_PATH,
        ROOT / "diagnostics/benchmark_full_sequence_graph.py",
        ROOT / "scripts/benchmark_page_gauge_transformer.py",
        ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
        ROOT / "scripts/page_gauge_heterogeneous_fa2.py",
        ROOT / "scripts/prepare_flashinfer_page_gauge_heterogeneous.py",
        ROOT / "scripts/benchmark_page_gauge_overheads.py",
        ROOT / "scripts/benchmark_e2e_transformer.py",
        ROOT / "scripts/page_gauge_runtime.py",
        ROOT / "tests/page_gauge_append_extension.cu",
        ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch",
        ROOT / "patches/flashinfer-0.6.17-page-gauge-heterogeneous.patch",
        ROOT / "docs/decoder_layer_cuda_graphs.md",
    )
    model_revision = getattr(model.config, "_commit_hash", None)
    if model_revision is None:
        model_revision = token_provenance.get("tokenizer", {}).get(
            "resolved_snapshot_revision"
        )
    if model_revision is None:
        model_revision = f"sampled-parameters:{parameter_sample['sha256']}"
    config = {
        "backend": args.backend,
        "model": args.model,
        "model_revision": model_revision,
        "model_config_sha256": PREFILL.canonical_json_sha256(config_payload),
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail_tokens": args.exact_tail,
        "exact_sink_pages": args.exact_sink_pages,
        "exact_prefix_pages": args.exact_sink_pages,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "tail_attention": args.tail_attention,
        "cuda_graph_scope": args.cuda_graph_scope,
        "seed": args.seed,
        "token_offset": args.token_offset,
        "token_stride": args.token_stride,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cache_scrub_mib": args.cache_scrub_mib,
    }
    passed = bool(within_backend_passed and cross_reference_passed and residency["passed"])
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_backend_exclusive_generated_decode",
        "backend": args.backend,
        "passed": passed,
        "seed": args.seed,
        "model": args.model,
        "model_revision": config["model_revision"],
        "model_config_sha256": config["model_config_sha256"],
        "token_ids_sha256": token_provenance["token_ids_sha256"],
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail_tokens": args.exact_tail,
        "exact_sink_pages": args.exact_sink_pages,
        "exact_prefix_pages": args.exact_sink_pages,
        "config": config,
        "configuration": config,
        "claim_scope": (
            "backend-exclusive fixed-batch/fixed-context 16-step GPU-resident "
            "greedy decode; excludes model loading, prefill, graph capture, "
            "cache restoration, cache scrubbing, and hot preconditioning"
        ),
        "publication_orchestration_status": {
            "manual_worker_exact_prefix_supported": True,
            "williams_orchestrator_exact_prefix_integrated": False,
            "williams_analyzer_exact_prefix_attestation_integrated": False,
            "publication_pairing_deferred_until_manual_known_row_strict_gate": True,
        },
        "timed_work": {
            "seed_copies_d2d": 1,
            "decoder_steps": args.decode_steps,
            "decoder_plan_calls": args.decode_steps,
            "future_offset_preplanning_calls": 0,
            "wrapper_last_page_len_device_fills": (
                args.decode_steps
                if args.backend == "flashinfer_fp16"
                or args.tail_attention == "heterogeneous_fa2"
                else 2 * args.decode_steps
            ),
            "embedding_to_persistent_layer_input_d2d_copies": (
                args.decode_steps
                if args.cuda_graph_scope == "decoder_layer"
                else 0
            ),
            "captured_persistent_layer_output_writes": (
                expected_graph_calls
                if args.cuda_graph_scope == "decoder_layer"
                else 0
            ),
            "symmetric_internal_layer_output_copy_nodes": (
                expected_graph_calls
                if args.cuda_graph_scope == "decoder_layer"
                else 0
            ),
            "cuda_graph_scope": args.cuda_graph_scope,
            "cuda_graph_replays": expected_graph_calls,
            "attention_cuda_graph_replays": (
                expected_graph_calls if args.cuda_graph_scope == "attention" else 0
            ),
            "decoder_layer_cuda_graph_replays": (
                expected_graph_calls
                if args.cuda_graph_scope == "decoder_layer"
                else 0
            ),
            "gpu_argmax_operations": args.decode_steps,
            "host_synchronizations_between_tokens": 0,
            "synchronizations": "only immediately before and after each block",
            "generated_feedback": True,
            "prefill_included": False,
        },
        "attention_implementation": decoder.attention_implementation_provenance(),
        "cuda_graph_provenance": {
            **graph_structure,
            "structure_gate_passed": graph_structure_passed,
            **(
                {}
                if args.cuda_graph_scope == "decoder_layer"
                else {
                    "replays_per_decode_step": layers,
                    "included_operations": ["selected_attention_path"],
                }
            ),
        },
        "exclusivity": {
            "fresh_process_required": True,
            "selected_persistent_backend": args.backend,
            "opposite_backend_full_gpu_cache_allocated": False,
            "hf_dynamic_cache_scope": "one B1 request during fixture construction only",
            "hf_dynamic_caches_released_before_decoder_construction": True,
            "cache_serialization_or_reload": False,
            "direct_destination_construction": True,
            "page_gauge_derivation": (
                "per request/layer center, page quantization, fixed exact "
                "prefix, and exact-tail ring written directly from the live "
                "HF DynamicCache"
                if args.backend == "page_gauge"
                else None
            ),
            "flashinfer_derivation": (
                "post-RoPE FP16 K/V copied directly from the live HF DynamicCache"
                if args.backend == "flashinfer_fp16"
                else None
            ),
        },
        "token_source": token_provenance,
        "fixture_provenance": {
            "token_ids_sha256": token_provenance["token_ids_sha256"],
            "sampled_boundary_kv_sha256": kv_sample_sha256,
            "request_windows": quality_windows,
            "initial_seed_semantics": (
                "actual corpus successor at model position context for each request"
            ),
            "subsequent_input_semantics": "GPU greedy argmax feedback",
        },
        "model_provenance": {
            "requested_name_or_path": args.model,
            "resolved_revision": config["model_revision"],
            "config_sha256": config["model_config_sha256"],
            "sampled_parameters": parameter_sample,
            "parameter_count": parameter_count,
            "reference_dtype": "torch.float16",
        },
        "cache_build": {
            "layout": "request-major [L,B*P,16,Hkv,D]",
            "pages_per_request": pages,
            "initial_pages_per_request": initial_pages,
            "exact_ring_pages_per_request": exact_pages if args.backend == "page_gauge" else 0,
            "exact_tail_pages_per_request": exact_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_prefix_pages_per_request": args.exact_sink_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_storage_pages_per_request": exact_pages + args.exact_sink_pages
            if args.backend == "page_gauge"
            else 0,
            "exact_physical_layout": (
                "slots0..S-1=fixed logical prefix; slotsS..S+T-1=tail ring"
                if args.backend == "page_gauge" and args.exact_sink_pages
                else "slots0..T-1=tail ring"
            ),
            "prefix_storage_accounting": (
                "gross FP16 allocation; prefix INT8 codes/scales remain allocated"
                if args.backend == "page_gauge" and args.exact_sink_pages
                else "no exact prefix allocation"
            ),
            "selected_backend_cache": cache_manifest,
            "logical_fp16_kv_bytes": (
                layers
                * args.batch_size
                * pages
                * PG.PAGE
                * hkv
                * PG.DIM
                * 2
                * torch.tensor([], dtype=torch.float16).element_size()
            ),
            "request_records": prefill_records,
            "sampled_boundary_kv_sha256": kv_sample_sha256,
        },
        "correctness": {
            "passed": within_backend_passed and cross_reference_passed,
            "immutable_exact_prefix_canary": prefix_canary_gate,
            "same_backend_eager_vs_graph": {
                "passed": within_backend_passed,
                "cuda_graph_scope": args.cuda_graph_scope,
                "logits": eager_vs_graph_logits,
                "generated_tokens": eager_vs_graph_tokens,
                "mutated_page_cache": eager_vs_graph_cache,
                "repeated_graph_logits": repeated_graph_logits,
                "repeated_graph_generated_tokens": repeated_graph_tokens,
                "repeated_graph_mutated_page_cache": (
                    repeated_graph_cache_comparison
                ),
                "eager_attention_dispatch": eager_attention_dispatch,
                "graph_attention_dispatch": graph_attention_dispatch,
                "repeated_graph_attention_dispatch": (
                    repeated_graph_attention_dispatch
                ),
                "eager_decoder_layer_dispatch": eager_layer_dispatch,
                "graph_decoder_layer_dispatch": graph_layer_dispatch,
                "repeated_graph_decoder_layer_dispatch": (
                    repeated_graph_layer_dispatch
                ),
                "selected_eager_dispatch": eager_selected_dispatch,
                "selected_graph_dispatch": graph_selected_dispatch,
                "selected_repeated_graph_dispatch": repeated_selected_dispatch,
                "dispatch_passed": dispatch_passed,
                "graph_structure_passed": graph_structure_passed,
                "nested_attention_dispatch_passed": (
                    nested_attention_dispatch_passed
                ),
                "expected_graph_calls": expected_graph_calls,
                "checked_page_offsets": list(range(args.decode_steps)),
                "page_close_offset_checked": args.decode_steps - 1,
                "page_offset_coverage_fraction": 1.0,
            },
            "backend_vs_hf_sdpa_fp16_greedy": {
                "passed": cross_reference_passed,
                "logits": hf_vs_backend,
                "generated_tokens": hf_vs_backend_tokens,
                "thresholds": {
                    "minimum_logits_cosine": args.min_logits_cosine,
                    "minimum_top1_agreement": args.min_top1_agreement,
                    "exact_generated_tokens_required": True,
                },
            },
            "hashes": {
                "hf_logits_sha256": tensor_sequence_sha256(hf_logits),
                "eager_logits_sha256": tensor_sequence_sha256(eager_outputs),
                "graph_logits_sha256": tensor_sequence_sha256(graph_outputs),
                "hf_generated_tokens_sha256": tensor_sequence_sha256(hf_generated),
                "graph_generated_tokens_sha256": tensor_sequence_sha256(graph_generated),
                "repeated_graph_logits_sha256": tensor_sequence_sha256(
                    repeated_graph_outputs
                ),
                "repeated_graph_generated_tokens_sha256": tensor_sequence_sha256(
                    repeated_graph_generated
                ),
            },
        },
        "residency_gate": residency,
        "representative_layer_profile": profile,
        "timing_modes": timing_modes,
        "memory": memory,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "local_files_only": True,
            "orchestration_environment": orchestration_environment(),
        },
        "invocation": {
            "argv": sys.argv,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "cwd": str(Path.cwd()),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
    }
    return result


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "experiment": "page_gauge_backend_exclusive_generated_decode",
            "backend": args.backend,
            "passed": False,
            "seed": args.seed,
            "model": args.model,
            "batch_size": args.batch_size,
            "context": args.context,
            "decode_steps": args.decode_steps,
            "exact_tail_tokens": args.exact_tail,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "invocation": {
                "argv": sys.argv,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "cwd": str(Path.cwd()),
            },
            "worker_source_sha256": sha256_file(Path(__file__)),
        }
        args.output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "backend": result["backend"],
                "passed": result["passed"],
                "hf_minimum_logits_cosine": result["correctness"]
                ["backend_vs_hf_sdpa_fp16_greedy"]["logits"]
                ["minimum_logits_cosine"],
                "hf_top1_agreement": result["correctness"]
                ["backend_vs_hf_sdpa_fp16_greedy"]["logits"]
                ["top1_agreement_fraction"],
                "residency_passed": result["residency_gate"]["passed"],
                "cache_neutral_wall_ms": result["timing_modes"]["cache_neutral"]
                ["wall_summary"]["mean_ms_per_sequence"],
                "cache_hot_wall_ms": result["timing_modes"]["cache_hot"]
                ["wall_summary"]["mean_ms_per_sequence"],
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
