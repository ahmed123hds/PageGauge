#!/usr/bin/env python3
"""Full-transformer PageGauge decode versus an FP16 FlashInfer baseline.

The benchmark uses identical model weights, token IDs, projection code, and
logical prefix K/V values.  It includes every decoder layer, LM head, dynamic
paged-cache append, PageGauge completed-page finalization, and attention.  It
reports cache-neutral and cache-hot wall/GPU tokens/s plus per-layer latency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PAGE = 16
DIM = 128
TokenSequence = list[int] | list[list[int]]


def is_tvm_ffi_array(value: Any) -> bool:
    value_type = type(value)
    return (
        value_type.__module__ == "tvm_ffi.container"
        and value_type.__qualname__ == "Array"
    )


def freeze_structural_value(value: Any) -> Any:
    """Return a hashable, fail-closed representation of planner topology.

    FlashInfer's FA2 tensor-core decoder stores its launch topology in the
    private ``_plan_info`` payload. Public sequence lengths are insufficient:
    different lengths can share a launch topology, while a split-count change
    can change it without changing any Python tensor address. Only the small
    set of scalar/container/tensor forms used by the installed planner is
    accepted so an upstream ABI change cannot silently reuse a stale graph.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype),
            tuple(int(dimension) for dimension in value.shape),
            tuple(int(stride) for stride in value.stride()),
            int(value.data_ptr()),
        )
    if isinstance(value, (tuple, list)):
        return tuple(freeze_structural_value(item) for item in value)
    if is_tvm_ffi_array(value):
        # FlashInfer 0.6.17 may expose PrefillPlanInfo through TVM-FFI rather
        # than a Python list. Accept only this exact audited container type;
        # arbitrary Sequence implementations remain a fail-closed ABI error.
        return (
            "tvm_ffi.container.Array",
            tuple(freeze_structural_value(item) for item in value),
        )
    if isinstance(value, dict):
        return tuple(
            (str(key), freeze_structural_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    raise TypeError(
        "unsupported FlashInfer structural plan value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def tensor_pointer_signature(name: str, tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        name,
        str(tensor.dtype),
        tuple(int(dimension) for dimension in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        int(tensor.data_ptr()),
    )


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values after an explicit host copy.

    This helper is reserved for correctness/provenance outside timed regions;
    callers must not use it in a token-serving loop.
    """

    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAGE_KERNEL = load_local_module(
    "page_gauge_attention_kernel",
    ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
)
HETEROGENEOUS_FA2 = load_local_module(
    "page_gauge_heterogeneous_fa2_kernel",
    ROOT / "scripts/page_gauge_heterogeneous_fa2.py",
)
OVERHEAD = load_local_module(
    "page_gauge_overhead_kernel",
    ROOT / "scripts/benchmark_page_gauge_overheads.py",
)
RUNTIME = load_local_module(
    "page_gauge_runtime_helpers", ROOT / "scripts/page_gauge_runtime.py"
)
BASE_E2E = load_local_module(
    "page_gauge_base_e2e", ROOT / "scripts/benchmark_e2e_transformer.py"
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def folded_output_projection_bias(
    weight: torch.Tensor,
    original_bias: torch.Tensor | None,
    output_center: torch.Tensor,
) -> torch.Tensor:
    if output_center.dim() not in (2, 3):
        raise ValueError("output center must be [Hq,D] or [B,Hq,D]")
    batched = output_center.dim() == 3
    flattened = output_center.reshape(
        -1, output_center.shape[-2] * output_center.shape[-1]
    )
    center_bias = F.linear(flattened, weight, None)
    if original_bias is not None:
        center_bias = center_bias + original_bias
    return center_bias if batched else center_bias[0]


def bootstrap_geomean(values: list[float], seed: int, samples: int = 10000) -> list[float]:
    generator = random.Random(seed)
    draws = [
        geometric_mean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def hierarchical_bootstrap_geomean(
    groups: list[list[float]], seed: int, samples: int = 10000
) -> list[float]:
    """Bootstrap paired blocks while preserving their seed-level clustering."""
    if not groups or any(not group for group in groups):
        raise ValueError("bootstrap groups must be non-empty")
    generator = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected_groups = generator.choices(groups, k=len(groups))
        resampled = []
        for group in selected_groups:
            resampled.extend(generator.choices(group, k=len(group)))
        draws.append(geometric_mean(resampled))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument(
        "--exact-sink-pages",
        type=int,
        default=0,
        help=(
            "Keep a contiguous logical prefix of this many complete pages in "
            "fixed slots of the existing exact FP16 segment, in addition to "
            "the recent exact tail. The value must be nonnegative; zero "
            "preserves the original cache layout."
        ),
    )
    parser.add_argument("--baseline-split-pages", type=int, default=0)
    parser.add_argument("--candidate-split-pages", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--layer-profile-repeats", type=int, default=3)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument(
        "--disable-attention-cuda-graphs",
        action="store_true",
        help=(
            "Disable the matched per-layer attention CUDA graphs. This is a "
            "diagnostic option; the production path graphs both backends."
        ),
    )
    parser.add_argument(
        "--decoder-layer-cuda-graphs",
        action="store_true",
        help=(
            "Opt in to 16 offset-specific banks of complete decoder-layer "
            "graphs. Each replay contains normalization, QKV, RoPE/cache "
            "update, attention, projections, residuals, and MLP. The default "
            "remains the smaller matched attention-only graph boundary."
        ),
    )
    parser.add_argument(
        "--skip-integration-ablations",
        action="store_true",
        help="Skip the matched-eager and candidate-only-graph diagnostics.",
    )
    parser.add_argument(
        "--center-restore",
        choices=("attention_add", "projection_bias"),
        default="attention_add",
        help=(
            "Restore the PageGauge value center inside the captured attention "
            "graph or algebraically fold it into the output-projection bias."
        ),
    )
    parser.add_argument(
        "--tail-attention",
        choices=("fused_kernel", "flashinfer_merge", "heterogeneous_fa2"),
        default="flashinfer_merge",
        help=(
            "Use the validated two-wrapper FlashInfer exact-tail merge, or the "
            "slower experimental fused tail kernel as a negative ablation. "
            "heterogeneous_fa2 is the opt-in single-wrapper INT8/FP16 FA2 path."
        ),
    )
    parser.add_argument(
        "--old-value-scale-placement",
        choices=("probability", "value_fragment"),
        default="probability",
        help=(
            "For completed INT8 pages, apply the per-page V scale to the "
            "softmax probability fragment (legacy) or to the converted V "
            "fragment (algebraically identical and underflow-safe)."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260850, 20260851, 20260852)
    )
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    if args.model == "synthetic-mistral-page-gauge-smoke":
        from transformers import MistralConfig, MistralForCausalLM

        model = MistralForCausalLM(
            MistralConfig(
                vocab_size=256,
                hidden_size=4096,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=args.context + args.decode_steps + 1,
            )
        ).half().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        ).eval()
    layers, hq, hkv, _ = BASE_E2E.check_model(model)
    if (hq, hkv) != (32, 8):
        raise RuntimeError(
            f"publication runner currently requires Hq/Hkv=32/8; got {hq}/{hkv}"
        )
    BASE_E2E.pack_model_projections(model)
    return model.cuda()


class BaselineCache:
    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key = key
        self.value = value


class GaugeCache:
    def __init__(
        self,
        exact_key: torch.Tensor,
        exact_value: torch.Tensor,
        key_codes: torch.Tensor,
        value_codes: torch.Tensor,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        key_center: torch.Tensor,
        value_center: torch.Tensor,
        exact_tail_pages: int | None = None,
        exact_sink_pages: int = 0,
        exact_static_suffix_pages: int = 0,
        initial_context_pages: int | None = None,
    ) -> None:
        self.exact_key = exact_key
        self.exact_value = exact_value
        self.key_codes = key_codes
        self.value_codes = value_codes
        self.key_scales = key_scales
        self.value_scales = value_scales
        self.key_center = key_center
        self.value_center = value_center
        self.output_center = value_center.repeat_interleave(4, dim=2).contiguous()
        exact_prefix_pages = validate_exact_prefix_pages(exact_sink_pages)
        exact_static_suffix_pages = validate_exact_static_suffix_pages(
            exact_static_suffix_pages,
            exact_prefix_pages=exact_prefix_pages,
            initial_context_pages=initial_context_pages,
        )
        batch_size = int(key_center.shape[1])
        if batch_size <= 0 or int(exact_key.shape[1]) % batch_size:
            raise ValueError("exact cache pages must divide evenly across requests")
        storage_pages = int(exact_key.shape[1]) // batch_size
        exact_fixed_pages = exact_prefix_pages + exact_static_suffix_pages
        inferred_tail_pages = storage_pages - exact_fixed_pages
        if exact_tail_pages is None:
            exact_tail_pages = inferred_tail_pages
        if exact_tail_pages <= 0 or exact_tail_pages != inferred_tail_pages:
            raise ValueError(
                "exact cache storage must contain fixed prefix/suffix pages "
                "followed by tail-ring pages"
            )
        self.exact_tail_pages = int(exact_tail_pages)
        self.exact_prefix_pages = exact_prefix_pages
        self.exact_static_suffix_pages = exact_static_suffix_pages
        self.initial_context_pages = (
            int(initial_context_pages) if initial_context_pages is not None else None
        )
        self.exact_static_suffix_logical_begin = (
            self.initial_context_pages - self.exact_static_suffix_pages
            if self.exact_static_suffix_pages
            else None
        )
        self.exact_fixed_pages = exact_fixed_pages
        # Backward-compatible field/CLI spelling. Semantically this is now S,
        # the length of the contiguous exact prefix.
        self.exact_sink_pages = self.exact_prefix_pages
        self.exact_storage_pages_per_request = storage_pages


def request_major_page_table(
    batch_size: int,
    logical_page_capacity: int,
    physical_pages_per_request: int | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Map synchronized logical pages to request-major physical storage."""
    if batch_size <= 0 or logical_page_capacity <= 0:
        raise ValueError("batch size and page capacity must be positive")
    physical_pages = (
        logical_page_capacity
        if physical_pages_per_request is None
        else physical_pages_per_request
    )
    if physical_pages <= 0:
        raise ValueError("physical pages per request must be positive")
    logical = torch.arange(
        logical_page_capacity, device=device, dtype=torch.int32
    )
    physical = logical.remainder(physical_pages)
    request_offsets = (
        torch.arange(batch_size, device=device, dtype=torch.int32)
        * physical_pages
    )
    return request_offsets[:, None] + physical[None, :]


def validate_exact_prefix_pages(exact_prefix_pages: int) -> int:
    if not isinstance(exact_prefix_pages, int) or exact_prefix_pages < 0:
        raise ValueError("exact prefix pages must be a nonnegative integer")
    return int(exact_prefix_pages)


def validate_exact_static_suffix_pages(
    exact_static_suffix_pages: int,
    *,
    exact_prefix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> int:
    """Validate a fixed exact suffix of the original prefill cache."""

    if (
        not isinstance(exact_static_suffix_pages, int)
        or exact_static_suffix_pages < 0
    ):
        raise ValueError("exact static suffix pages must be a nonnegative integer")
    if exact_static_suffix_pages:
        if initial_context_pages is None or initial_context_pages <= 0:
            raise ValueError(
                "an exact static suffix requires positive initial context pages"
            )
        if exact_prefix_pages + exact_static_suffix_pages >= initial_context_pages:
            raise ValueError(
                "fixed exact prefix and suffix must leave an initial old-cache page"
            )
    return int(exact_static_suffix_pages)


def validate_exact_sink_pages(exact_sink_pages: int) -> int:
    """Legacy S0/S1 validator retained for frozen trace-protocol replay."""

    if exact_sink_pages not in (0, 1):
        raise ValueError("exact sink pages must be 0 or 1")
    return int(exact_sink_pages)


def validate_exact_prefix_attention_path(
    exact_prefix_pages: int, tail_attention: str
) -> int:
    exact_prefix_pages = validate_exact_prefix_pages(exact_prefix_pages)
    if exact_prefix_pages and tail_attention != "flashinfer_merge":
        raise ValueError(
            "exact prefix pages require the segmented flashinfer_merge path"
        )
    return exact_prefix_pages


def exact_physical_page_index(
    request: int,
    logical_page: int,
    exact_tail_pages: int,
    exact_sink_pages: int = 0,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> int:
    """Map one logical page into request-major fixed pages plus tail ring."""

    exact_sink_pages = validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=initial_context_pages,
    )
    if request < 0 or logical_page < 0 or exact_tail_pages <= 0:
        raise ValueError("exact page mapping inputs must be non-negative with a tail")
    exact_fixed_pages = exact_sink_pages + exact_static_suffix_pages
    storage_pages = exact_fixed_pages + exact_tail_pages
    suffix_begin = (
        int(initial_context_pages) - exact_static_suffix_pages
        if exact_static_suffix_pages
        else None
    )
    if logical_page < exact_sink_pages:
        local_page = logical_page
    elif (
        suffix_begin is not None
        and suffix_begin <= logical_page < int(initial_context_pages)
    ):
        local_page = exact_sink_pages + logical_page - suffix_begin
    else:
        local_page = exact_fixed_pages + logical_page % exact_tail_pages
    return request * storage_pages + local_page


def request_major_exact_page_table(
    batch_size: int,
    logical_page_capacity: int,
    exact_tail_pages: int,
    exact_sink_pages: int = 0,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Map all logical pages into fixed prefix/suffix slots plus a tail ring."""

    exact_sink_pages = validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=initial_context_pages,
    )
    if batch_size <= 0 or logical_page_capacity <= 0 or exact_tail_pages <= 0:
        raise ValueError("batch, logical capacity, and exact tail must be positive")
    if exact_sink_pages >= logical_page_capacity:
        raise ValueError("exact prefix must be smaller than logical page capacity")
    logical = torch.arange(logical_page_capacity, device=device, dtype=torch.int32)
    exact_fixed_pages = exact_sink_pages + exact_static_suffix_pages
    local = exact_fixed_pages + logical.remainder(exact_tail_pages)
    if exact_sink_pages:
        local = torch.where(logical < exact_sink_pages, logical, local)
    if exact_static_suffix_pages:
        assert initial_context_pages is not None
        suffix_begin = initial_context_pages - exact_static_suffix_pages
        suffix_mask = (logical >= suffix_begin) & (logical < initial_context_pages)
        suffix_local = exact_sink_pages + logical - suffix_begin
        local = torch.where(suffix_mask, suffix_local, local)
    storage_pages = exact_fixed_pages + exact_tail_pages
    request_offsets = (
        torch.arange(batch_size, device=device, dtype=torch.int32) * storage_pages
    )
    return request_offsets[:, None] + local[None, :]


def page_gauge_logical_partition(
    total_pages: int,
    exact_tail_pages: int,
    exact_sink_pages: int,
    last_page_len: int,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> dict[str, Any]:
    """Describe the exact-once prefix/suffix/old/tail logical partition."""

    exact_sink_pages = validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=initial_context_pages,
    )
    if total_pages <= 0 or exact_tail_pages <= 0:
        raise ValueError("total and exact-tail page counts must be positive")
    if not 1 <= last_page_len <= PAGE:
        raise ValueError("last page length lies outside one page")
    tail_logical_begin = total_pages - exact_tail_pages
    old_logical_begin = exact_sink_pages
    if tail_logical_begin <= old_logical_begin:
        raise ValueError("context must contain at least one quantized old page")
    prefix = tuple(range(exact_sink_pages))
    suffix_begin = (
        int(initial_context_pages) - exact_static_suffix_pages
        if exact_static_suffix_pages
        else tail_logical_begin
    )
    suffix = tuple(
        range(suffix_begin, int(initial_context_pages))
        if exact_static_suffix_pages
        else ()
    )
    tail = tuple(range(tail_logical_begin, total_pages))
    exact_set = set(prefix) | set(suffix) | set(tail)
    old = tuple(page for page in range(total_pages) if page not in exact_set)
    exact = tuple(page for page in range(total_pages) if page in exact_set)
    logical_union = old + exact
    coverage_disjoint = (
        len(set(logical_union)) == len(logical_union)
        and set(logical_union) == set(range(total_pages))
    )
    context_tokens = (total_pages - 1) * PAGE + last_page_len
    old_tokens = len(old) * PAGE
    exact_tokens = context_tokens - old_tokens
    if not coverage_disjoint or old_tokens + exact_tokens != context_tokens:
        raise RuntimeError("PageGauge logical partition is not exact and disjoint")
    return {
        "prefix_logical_pages": prefix,
        "sink_logical_pages": prefix,
        "static_suffix_logical_pages": suffix,
        "old_logical_pages": old,
        "tail_logical_pages": tail,
        "exact_logical_pages": exact,
        "tail_logical_begin": tail_logical_begin,
        "old_page_count": len(old),
        "exact_page_count": len(exact),
        "old_token_count": old_tokens,
        "exact_token_count": exact_tokens,
        "context_token_count": context_tokens,
        "coverage_disjoint": coverage_disjoint,
    }


def write_exact_prefix_tail_page_table(
    destination: torch.Tensor,
    logical_to_exact_physical: torch.Tensor,
    tail_logical_begin: int,
    total_page_count: int,
    exact_sink_pages: int,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> torch.Tensor:
    """Write the exact-once fixed prefix/suffix plus chronological tail."""

    exact_sink_pages = validate_exact_prefix_pages(exact_sink_pages)
    exact_static_suffix_pages = validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_sink_pages,
        initial_context_pages=initial_context_pages,
    )
    if destination.dtype != torch.int32 or logical_to_exact_physical.dtype != torch.int32:
        raise ValueError("exact page tables must use int32 indices")
    if destination.dim() != 2 or logical_to_exact_physical.dim() != 2:
        raise ValueError("exact page tables must have shape [B,P]")
    if int(destination.shape[0]) != int(logical_to_exact_physical.shape[0]):
        raise ValueError("exact page-table batch sizes differ")
    if not exact_sink_pages <= tail_logical_begin < total_page_count:
        raise ValueError("exact prefix and tail logical ranges overlap or are empty")
    if total_page_count > int(logical_to_exact_physical.shape[1]):
        raise ValueError("exact logical page table exceeds allocated capacity")
    suffix_begin = (
        int(initial_context_pages) - exact_static_suffix_pages
        if exact_static_suffix_pages
        else None
    )
    tail_logical_pages = [
        logical
        for logical in range(tail_logical_begin, total_page_count)
        if not (
            suffix_begin is not None
            and suffix_begin <= logical < int(initial_context_pages)
        )
    ]
    active_pages = (
        exact_sink_pages + exact_static_suffix_pages + len(tail_logical_pages)
    )
    if active_pages > int(destination.shape[1]):
        raise ValueError("exact active table exceeds destination capacity")
    if exact_sink_pages:
        destination[:, :exact_sink_pages].copy_(
            logical_to_exact_physical[:, :exact_sink_pages]
        )
    cursor = exact_sink_pages
    if exact_static_suffix_pages:
        assert suffix_begin is not None and initial_context_pages is not None
        destination[:, cursor : cursor + exact_static_suffix_pages].copy_(
            logical_to_exact_physical[:, suffix_begin:initial_context_pages]
        )
        cursor += exact_static_suffix_pages
    if tail_logical_pages:
        logical_indices = torch.tensor(
            tail_logical_pages,
            device=logical_to_exact_physical.device,
            dtype=torch.long,
        )
        destination[:, cursor:active_pages].copy_(
            logical_to_exact_physical.index_select(1, logical_indices)
        )
    return destination[:, :active_pages]


def write_page_gauge_old_page_table(
    destination: torch.Tensor,
    logical_to_physical: torch.Tensor,
    old_logical_pages: tuple[int, ...],
) -> torch.Tensor:
    """Write an arbitrary exact-once old-page set in logical order."""

    if destination.dtype != torch.int32 or logical_to_physical.dtype != torch.int32:
        raise ValueError("old page tables must use int32 indices")
    if destination.dim() != 2 or logical_to_physical.dim() != 2:
        raise ValueError("old page tables must have shape [B,P]")
    if destination.shape[0] != logical_to_physical.shape[0]:
        raise ValueError("old page-table batch sizes differ")
    if not old_logical_pages:
        raise ValueError("old page set cannot be empty")
    if tuple(sorted(set(old_logical_pages))) != old_logical_pages:
        raise ValueError("old logical pages must be unique and increasing")
    if old_logical_pages[-1] >= int(logical_to_physical.shape[1]):
        raise ValueError("old logical page exceeds allocated capacity")
    if len(old_logical_pages) > int(destination.shape[1]):
        raise ValueError("old page set exceeds destination capacity")
    logical_indices = torch.tensor(
        old_logical_pages, device=logical_to_physical.device, dtype=torch.long
    )
    destination[:, : len(old_logical_pages)].copy_(
        logical_to_physical.index_select(1, logical_indices)
    )
    return destination[:, : len(old_logical_pages)]


def write_exact_sink_tail_page_table(
    destination: torch.Tensor,
    logical_to_exact_physical: torch.Tensor,
    tail_logical_begin: int,
    total_page_count: int,
    exact_sink_pages: int,
) -> torch.Tensor:
    """Backward-compatible name for the generic exact-prefix page-table writer."""

    return write_exact_prefix_tail_page_table(
        destination,
        logical_to_exact_physical,
        tail_logical_begin,
        total_page_count,
        exact_sink_pages,
    )


def write_heterogeneous_page_table(
    destination: torch.Tensor,
    old_pages: torch.Tensor,
    exact_ring_pages: torch.Tensor,
    old_page_count: int,
    total_page_count: int,
) -> torch.Tensor:
    """Write one logical table spanning old-code and exact-ring storage.

    The custom kernel interprets entries before ``old_page_count`` as physical
    identifiers in the INT8 cache and later entries as physical identifiers in
    the centered FP16 ring.  ``destination`` is persistent so attention graph
    capture never depends on a temporary concatenation allocation.
    """
    if destination.dtype != torch.int32:
        raise ValueError("heterogeneous page table must use int32 indices")
    if (
        destination.dim() != 2
        or old_pages.shape != destination.shape
        or exact_ring_pages.shape != destination.shape
    ):
        raise ValueError("heterogeneous page tables must have matching [B,P] shapes")
    if not 0 <= old_page_count <= total_page_count <= int(destination.shape[1]):
        raise ValueError("heterogeneous page counts lie outside table capacity")
    if old_page_count:
        destination[:, :old_page_count].copy_(old_pages[:, :old_page_count])
    if old_page_count < total_page_count:
        destination[:, old_page_count:total_page_count].copy_(
            exact_ring_pages[:, old_page_count:total_page_count]
        )
    return destination[:, :total_page_count]


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def cache_storage(baseline: BaselineCache, gauge: GaugeCache) -> dict[str, Any]:
    baseline_bytes = tensor_bytes(baseline.key, baseline.value)
    gauge_bytes = tensor_bytes(
        gauge.exact_key,
        gauge.exact_value,
        gauge.key_codes,
        gauge.value_codes,
        gauge.key_scales,
        gauge.value_scales,
        gauge.key_center,
        gauge.value_center,
        gauge.output_center,
    )
    batch_size = int(gauge.key_center.shape[1])
    exact_storage_pages = int(gauge.exact_key.shape[1]) // batch_size
    exact_page_bytes_all_layers = (
        gauge.exact_key[:, 0].numel() * gauge.exact_key.element_size()
        + gauge.exact_value[:, 0].numel() * gauge.exact_value.element_size()
    )
    return {
        "flashinfer_fp16_bytes": baseline_bytes,
        "page_gauge_bytes": gauge_bytes,
        "page_gauge_fraction_of_fp16": gauge_bytes / baseline_bytes,
        "page_gauge_compression_ratio": baseline_bytes / gauge_bytes,
        "batch_size": batch_size,
        "page_gauge_exact_ring_pages": gauge.exact_tail_pages,
        "page_gauge_exact_tail_pages_per_request": gauge.exact_tail_pages,
        "page_gauge_exact_sink_pages_per_request": gauge.exact_sink_pages,
        "page_gauge_exact_prefix_pages_per_request": gauge.exact_prefix_pages,
        "page_gauge_exact_static_suffix_pages_per_request": (
            gauge.exact_static_suffix_pages
        ),
        "page_gauge_exact_fixed_pages_per_request": gauge.exact_fixed_pages,
        "page_gauge_initial_context_pages": gauge.initial_context_pages,
        "page_gauge_exact_storage_pages_per_request": exact_storage_pages,
        "page_gauge_total_exact_storage_pages": int(gauge.exact_key.shape[1]),
        "page_gauge_exact_sink_gross_allocated_bytes": (
            exact_page_bytes_all_layers
            * batch_size
            * gauge.exact_sink_pages
        ),
        "page_gauge_exact_prefix_gross_allocated_bytes": (
            exact_page_bytes_all_layers
            * batch_size
            * gauge.exact_prefix_pages
        ),
        "page_gauge_exact_static_suffix_gross_allocated_bytes": (
            exact_page_bytes_all_layers
            * batch_size
            * gauge.exact_static_suffix_pages
        ),
        "physical_page_layout": (
            "request-major [fixed contiguous prefix slots, fixed original-"
            "prefill suffix slots, modulo tail-ring slots]"
        ),
        "prefix_storage_accounting": (
            "gross FP16 prefix allocation; prefix INT8 codes/scales remain allocated"
            if gauge.exact_prefix_pages
            else "no exact prefix allocation"
        ),
        "sink_storage_accounting": (
            "legacy alias: gross FP16 prefix allocation; prefix INT8 codes/scales "
            "remain allocated"
            if gauge.exact_sink_pages
            else "legacy alias: no exact prefix allocation"
        ),
        "note": "backend cache tensors only; excludes shared model weights and workspaces",
    }


def snapshot_append_slot(
    decoder: "TransformerDecoder", position: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    logical_page, offset = divmod(position, PAGE)
    if decoder.backend == "flashinfer_fp16":
        assert isinstance(decoder.cache, BaselineCache)
        cache_pages = tuple(
            request * decoder.max_pages + logical_page
            for request in range(decoder.batch_size)
        )
        key = torch.stack(
            [decoder.cache.key[:, page, offset] for page in cache_pages], dim=1
        ).clone()
        value = torch.stack(
            [decoder.cache.value[:, page, offset] for page in cache_pages], dim=1
        ).clone()
    else:
        assert isinstance(decoder.cache, GaugeCache)
        cache_pages = tuple(
            exact_physical_page_index(
                request,
                logical_page,
                decoder.exact_tail_pages,
                decoder.exact_sink_pages,
            )
            for request in range(decoder.batch_size)
        )
        key = torch.stack(
            [decoder.cache.exact_key[:, page, offset] for page in cache_pages],
            dim=1,
        ).clone()
        value = torch.stack(
            [decoder.cache.exact_value[:, page, offset] for page in cache_pages],
            dim=1,
        ).clone()
    return key, value, cache_pages, offset


def restore_append_slot(
    decoder: "TransformerDecoder",
    snapshot: tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int],
) -> None:
    key, value, cache_pages, offset = snapshot
    if len(cache_pages) != decoder.batch_size:
        raise ValueError("append snapshot batch size mismatch")
    if decoder.backend == "flashinfer_fp16":
        assert isinstance(decoder.cache, BaselineCache)
        for request, cache_page in enumerate(cache_pages):
            decoder.cache.key[:, cache_page, offset].copy_(key[:, request])
            decoder.cache.value[:, cache_page, offset].copy_(value[:, request])
    else:
        assert isinstance(decoder.cache, GaugeCache)
        for request, cache_page in enumerate(cache_pages):
            decoder.cache.exact_key[:, cache_page, offset].copy_(key[:, request])
            decoder.cache.exact_value[:, cache_page, offset].copy_(value[:, request])


def mutated_cache_page_indices(
    decoder: "TransformerDecoder", start_position: int, decode_steps: int
) -> dict[str, tuple[int, ...]]:
    """Return every request-major physical page a decode range can mutate."""

    if start_position < 0 or decode_steps <= 0:
        raise ValueError("cache snapshot range must be non-negative and non-empty")
    first_logical = start_position // PAGE
    last_logical = (start_position + decode_steps - 1) // PAGE
    if last_logical >= decoder.max_pages:
        raise ValueError("cache snapshot range exceeds decoder capacity")
    logical_pages = tuple(range(first_logical, last_logical + 1))
    full_pages = tuple(
        request * decoder.max_pages + logical_page
        for request in range(decoder.batch_size)
        for logical_page in logical_pages
    )
    result = {"full": full_pages}
    if decoder.backend == "page_gauge":
        exact_prefix_pages = getattr(
            decoder,
            "exact_prefix_pages",
            getattr(decoder, "exact_sink_pages", 0),
        )
        if any(logical_page < exact_prefix_pages for logical_page in logical_pages):
            raise ValueError(
                "mutable decode range overlaps immutable exact-prefix pages"
            )
        exact_static_suffix_pages = int(
            getattr(decoder, "exact_static_suffix_pages", 0)
        )
        exact_fixed_pages = exact_prefix_pages + exact_static_suffix_pages
        storage_pages = decoder.exact_tail_pages + exact_fixed_pages
        local_pages = {
            exact_physical_page_index(
                0,
                logical_page,
                decoder.exact_tail_pages,
                exact_prefix_pages,
                exact_static_suffix_pages,
                getattr(decoder, "initial_context_pages", None),
            )
            for logical_page in logical_pages
        }
        if any(local_page < exact_fixed_pages for local_page in local_pages):
            raise RuntimeError("mutable exact-tail mapping entered a fixed slot")
        ring_pages = tuple(
            request * storage_pages + local_page
            for request in range(decoder.batch_size)
            for local_page in sorted(local_pages)
        )
        result["exact_ring"] = ring_pages
    return result


def cache_range_mapping_policy(decoder: "TransformerDecoder") -> dict[str, int]:
    """Freeze the request-major cache ABI used by a mutable-range snapshot."""

    prefix_pages = (
        int(
            getattr(
                decoder,
                "exact_prefix_pages",
                getattr(decoder, "exact_sink_pages", 0),
            )
        )
        if decoder.backend == "page_gauge"
        else 0
    )
    tail_pages = (
        int(decoder.exact_tail_pages) if decoder.backend == "page_gauge" else 0
    )
    suffix_pages = (
        int(getattr(decoder, "exact_static_suffix_pages", 0))
        if decoder.backend == "page_gauge"
        else 0
    )
    return {
        "batch_size": int(decoder.batch_size),
        "full_pages_per_request": int(decoder.max_pages),
        "exact_prefix_pages_per_request": prefix_pages,
        "exact_static_suffix_pages_per_request": suffix_pages,
        "exact_fixed_pages_per_request": prefix_pages + suffix_pages,
        "exact_tail_pages_per_request": tail_pages,
        "exact_storage_pages_per_request": prefix_pages + suffix_pages + tail_pages,
    }


def validated_cache_range_snapshot_pages(
    decoder: "TransformerDecoder", snapshot: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    """Recompute snapshot indices before any restore can write cache storage."""

    if snapshot.get("backend") != decoder.backend:
        raise ValueError("cache range snapshot backend mismatch")
    if snapshot.get("mapping_policy") != cache_range_mapping_policy(decoder):
        raise ValueError("cache range snapshot mapping policy is stale")
    start_position = snapshot.get("start_position")
    decode_steps = snapshot.get("decode_steps")
    if not isinstance(start_position, int) or not isinstance(decode_steps, int):
        raise ValueError("cache range snapshot decode window is invalid")
    expected = mutated_cache_page_indices(decoder, start_position, decode_steps)
    observed = snapshot.get("page_indices")
    if observed != expected:
        raise ValueError("cache range snapshot page indices are stale or unsafe")
    return expected


def snapshot_mutated_cache_range(
    decoder: "TransformerDecoder", start_position: int, decode_steps: int
) -> dict[str, Any]:
    """Clone all KV tensors touched by a sustained decode window."""

    pages = mutated_cache_page_indices(decoder, start_position, decode_steps)
    snapshot: dict[str, Any] = {
        "backend": decoder.backend,
        "start_position": int(start_position),
        "decode_steps": int(decode_steps),
        "mapping_policy": cache_range_mapping_policy(decoder),
        "page_indices": pages,
        "device_position": (
            decoder.device_position.clone()
            if getattr(decoder, "device_position", None) is not None
            else None
        ),
    }
    full = torch.tensor(pages["full"], device="cuda", dtype=torch.long)
    if decoder.backend == "flashinfer_fp16":
        assert isinstance(decoder.cache, BaselineCache)
        snapshot["tensors"] = {
            "key": decoder.cache.key.index_select(1, full).clone(),
            "value": decoder.cache.value.index_select(1, full).clone(),
        }
        return snapshot

    assert isinstance(decoder.cache, GaugeCache)
    exact = torch.tensor(pages["exact_ring"], device="cuda", dtype=torch.long)
    snapshot["tensors"] = {
        "exact_key": decoder.cache.exact_key.index_select(1, exact).clone(),
        "exact_value": decoder.cache.exact_value.index_select(1, exact).clone(),
        "key_codes": decoder.cache.key_codes.index_select(1, full).clone(),
        "value_codes": decoder.cache.value_codes.index_select(1, full).clone(),
        "key_scales": decoder.cache.key_scales.index_select(1, full).clone(),
        "value_scales": decoder.cache.value_scales.index_select(1, full).clone(),
    }
    return snapshot


def restore_mutated_cache_range(
    decoder: "TransformerDecoder", snapshot: dict[str, Any]
) -> None:
    """Restore a range snapshot without allocating the opposite backend."""

    pages = validated_cache_range_snapshot_pages(decoder, snapshot)
    tensors = snapshot["tensors"]
    full = torch.tensor(pages["full"], device="cuda", dtype=torch.long)

    def restore_pages(
        destination: torch.Tensor,
        indices: torch.Tensor,
        source: torch.Tensor,
    ) -> None:
        if int(source.shape[1]) != int(indices.numel()):
            raise ValueError("cache snapshot tensor length does not match page indices")
        if source.device == destination.device:
            destination.index_copy_(1, indices, source)
            return
        if source.device.type != "cpu":
            raise ValueError("cache snapshot tensors must reside on CUDA or CPU")
        for source_offset, physical_page in enumerate(indices.tolist()):
            destination[:, physical_page].copy_(
                source[:, source_offset], non_blocking=source.is_pinned()
            )

    if decoder.backend == "flashinfer_fp16":
        assert isinstance(decoder.cache, BaselineCache)
        restore_pages(decoder.cache.key, full, tensors["key"])
        restore_pages(decoder.cache.value, full, tensors["value"])
    else:
        assert isinstance(decoder.cache, GaugeCache)
        exact = torch.tensor(pages["exact_ring"], device="cuda", dtype=torch.long)
        restore_pages(decoder.cache.exact_key, exact, tensors["exact_key"])
        restore_pages(decoder.cache.exact_value, exact, tensors["exact_value"])
        restore_pages(decoder.cache.key_codes, full, tensors["key_codes"])
        restore_pages(decoder.cache.value_codes, full, tensors["value_codes"])
        restore_pages(decoder.cache.key_scales, full, tensors["key_scales"])
        restore_pages(decoder.cache.value_scales, full, tensors["value_scales"])
    if snapshot.get("device_position") is not None:
        if getattr(decoder, "device_position", None) is None:
            raise ValueError("snapshot contains a device position but decoder does not")
        decoder.device_position.copy_(snapshot["device_position"])


def offload_cache_range_snapshot(
    snapshot: dict[str, Any], *, pin_memory: bool = False
) -> dict[str, Any]:
    """Move a validation/restore snapshot off GPU before graph capture/timing."""

    def host_copy(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        result = tensor.detach().cpu()
        return result.pin_memory() if pin_memory else result

    return {
        "backend": snapshot["backend"],
        "start_position": snapshot["start_position"],
        "decode_steps": snapshot["decode_steps"],
        "mapping_policy": snapshot["mapping_policy"],
        "page_indices": snapshot["page_indices"],
        "device_position": host_copy(snapshot.get("device_position")),
        "tensors": {
            name: host_copy(tensor) for name, tensor in snapshot["tensors"].items()
        },
    }


def compare_mutated_cache_ranges(
    expected: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    if (
        expected.get("backend") != observed.get("backend")
        or expected.get("start_position") != observed.get("start_position")
        or expected.get("decode_steps") != observed.get("decode_steps")
        or expected.get("mapping_policy") != observed.get("mapping_policy")
        or expected.get("page_indices") != observed.get("page_indices")
    ):
        raise ValueError("cache range snapshots describe different decode windows")
    expected_tensors = expected["tensors"]
    observed_tensors = observed["tensors"]
    if set(expected_tensors) != set(observed_tensors):
        raise ValueError("cache range snapshots contain different tensors")
    records = {}
    for name in sorted(expected_tensors):
        reference = expected_tensors[name]
        result = observed_tensors[name]
        if reference.shape != result.shape or reference.dtype != result.dtype:
            raise ValueError(f"cache tensor {name} shape or dtype mismatch")
        identical = bool(torch.equal(reference, result))
        maximum_absolute_error = (
            0.0
            if identical
            else float((reference.float() - result.float()).abs().max().item())
        )
        records[name] = {
            "bitwise_identical": identical,
            "maximum_absolute_error": maximum_absolute_error,
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
        }
    expected_position = expected.get("device_position")
    observed_position = observed.get("device_position")
    position_identical = (
        expected_position is None
        and observed_position is None
        or isinstance(expected_position, torch.Tensor)
        and isinstance(observed_position, torch.Tensor)
        and bool(torch.equal(expected_position, observed_position))
    )
    return {
        "passed": position_identical
        and all(record["bitwise_identical"] for record in records.values()),
        "tensors": records,
        "device_position_bitwise_identical": position_identical,
        "full_pages_checked": len(expected["page_indices"]["full"]),
        "exact_ring_pages_checked": len(
            expected["page_indices"].get("exact_ring", ())
        ),
        "exact_tail_ring_pages_checked": len(
            expected["page_indices"].get("exact_ring", ())
        ),
    }


def build_caches(
    layers: int,
    pages: int,
    initial_pages: int,
    exact_pages: int,
    hkv: int,
    seed: int,
    batch_size: int = 1,
    exact_sink_pages: int = 0,
) -> tuple[BaselineCache, GaugeCache]:
    exact_sink_pages = validate_exact_prefix_pages(exact_sink_pages)
    if not 0 < exact_pages + exact_sink_pages < initial_pages <= pages:
        raise ValueError(
            "cache page counts must leave at least one quantized old page"
        )
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (layers, batch_size * pages, PAGE, hkv, DIM)
    key = torch.randn(
        shape, generator=generator, device="cuda", dtype=torch.float16
    )
    key.mul_(0.35)
    value = torch.randn(
        shape, generator=generator, device="cuda", dtype=torch.float16
    )
    value.mul_(0.35)
    key_center = torch.empty(
        layers, batch_size, hkv, DIM, device="cuda", dtype=torch.float16
    )
    value_center = torch.empty_like(key_center)
    exact_storage_pages = exact_pages + exact_sink_pages
    exact_shape = (layers, batch_size * exact_storage_pages, PAGE, hkv, DIM)
    exact_key = torch.empty(exact_shape, device="cuda", dtype=torch.float16)
    exact_value = torch.empty_like(exact_key)
    key_codes = torch.empty_like(key, dtype=torch.int8)
    value_codes = torch.empty_like(value, dtype=torch.int8)
    key_scales = torch.empty(
        layers, batch_size * pages, hkv, device="cuda", dtype=torch.float16
    )
    value_scales = torch.empty_like(key_scales)
    for layer in range(layers):
        for request in range(batch_size):
            page_start = request * pages
            page_end = page_start + pages
            prefix_end = page_start + initial_pages
            key_center[layer, request].copy_(
                key[layer, page_start:prefix_end]
                .float()
                .mean(dim=(0, 1))
                .half()
            )
            value_center[layer, request].copy_(
                value[layer, page_start:prefix_end]
                .float()
                .mean(dim=(0, 1))
                .half()
            )
            OVERHEAD.quantize_completed_kv_pages(
                key[layer, page_start:page_end],
                value[layer, page_start:page_end],
                key_center[layer, request],
                value_center[layer, request],
                key_codes[layer, page_start:page_end],
                value_codes[layer, page_start:page_end],
                key_scales[layer, page_start:page_end],
                value_scales[layer, page_start:page_end],
            )
            exact_logical_pages = tuple(range(exact_sink_pages)) + tuple(
                range(initial_pages - exact_pages, initial_pages)
            )
            for logical_page in exact_logical_pages:
                source_page = page_start + logical_page
                ring_page = exact_physical_page_index(
                    request,
                    logical_page,
                    exact_pages,
                    exact_sink_pages,
                )
                exact_key[layer, ring_page].copy_(
                    (
                        key[layer, source_page].float()
                        - key_center[layer, request][None].float()
                    ).half()
                )
                exact_value[layer, ring_page].copy_(
                    (
                        value[layer, source_page].float()
                        - value_center[layer, request][None].float()
                    ).half()
                )
    torch.cuda.synchronize()
    return BaselineCache(key, value), GaugeCache(
        exact_key,
        exact_value,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        key_center,
        value_center,
        exact_tail_pages=exact_pages,
        exact_sink_pages=exact_sink_pages,
    )


class GraphDecodeWrapper:
    def __init__(
        self,
        flashinfer,
        pages: int,
        kv_dtype: torch.dtype,
        custom_page_gauge: bool,
        batch_size: int = 1,
        heterogeneous_page_gauge: bool = False,
        v_fragment_page_gauge: bool = False,
    ) -> None:
        if pages <= 0 or batch_size <= 0:
            raise ValueError("page capacity and batch size must be positive")
        if sum(
            bool(value)
            for value in (
                custom_page_gauge,
                heterogeneous_page_gauge,
                v_fragment_page_gauge,
            )
        ) > 1:
            raise ValueError("select exactly one custom PageGauge wrapper")
        self.batch_size = batch_size
        # FlashInfer receives these graph buffers during wrapper construction.
        # Keep the initial state valid even though plan() overwrites it before
        # the first run, so constructors never observe uninitialized metadata.
        self.indptr = torch.arange(
            batch_size + 1, device="cuda", dtype=torch.int32
        )
        self.indices = torch.empty(
            batch_size * pages, device="cuda", dtype=torch.int32
        )
        self.last_len = torch.full(
            (batch_size,), PAGE, device="cuda", dtype=torch.int32
        )
        workspace = torch.empty(
            128 * 1024 * 1024, device="cuda", dtype=torch.uint8
        )
        self.workspace = workspace
        kwargs = {
            "use_cuda_graph": True,
            "paged_kv_indptr_buffer": self.indptr,
            "paged_kv_indices_buffer": self.indices,
            "paged_kv_last_page_len_buffer": self.last_len,
        }
        if heterogeneous_page_gauge:
            self.wrapper = HETEROGENEOUS_FA2.make_heterogeneous_page_gauge_wrapper(
                flashinfer, workspace, **kwargs
            )
        elif v_fragment_page_gauge:
            self.wrapper = HETEROGENEOUS_FA2.make_v_fragment_page_gauge_wrapper(
                flashinfer, workspace, **kwargs
            )
        elif custom_page_gauge:
            self.wrapper = PAGE_KERNEL.make_page_gauge_wrapper(
                flashinfer, workspace, **kwargs
            )
        else:
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_tensor_cores=True,
                backend="fa2",
                **kwargs,
            )
        self.kv_dtype = kv_dtype
        self.run_semantics = (
            "NHD",
            "fa2_tensor_core_decode",
            "NONE",
            -1,
            0.0,
            str(torch.float16),
            str(kv_dtype),
            str(torch.float16),
            1.0 / math.sqrt(DIM),
            32,
            8,
            DIM,
            PAGE,
        )
        self.signature: tuple[int, int, int, int, int, int | None] | None = None
        self.structural_signature: tuple[Any, ...] | None = None
        self.planned_split_pages: int | None = None
        self.plan_invocations = 0
        self.plan_rebuilds = 0
        self.last_len_fills = 0

    def reset_operation_counts(self) -> None:
        self.plan_invocations = 0
        self.plan_rebuilds = 0
        self.last_len_fills = 0

    def operation_counts(self) -> dict[str, int]:
        return {
            "plan_invocations": self.plan_invocations,
            "plan_rebuilds": self.plan_rebuilds,
            "last_page_len_device_fills": self.last_len_fills,
        }

    def serving_metadata_snapshot(self) -> dict[str, Any]:
        """Snapshot active page tables and semantic planner workspace bytes.

        FlashInfer's CUDA-graph planner stores launch arrays in a byte
        workspace.  Hash the allocated semantic ranges (not alignment padding)
        so eager and graph validation can prove that the supported plan API
        materialized identical runtime scheduling state.
        """

        if self.signature is None:
            raise RuntimeError("serving metadata requested before plan()")
        raw_plan_info = getattr(self.wrapper, "_plan_info", None)
        supported_plan_container = isinstance(
            raw_plan_info, (tuple, list)
        ) or is_tvm_ffi_array(raw_plan_info)
        if not supported_plan_container or len(raw_plan_info) != 15:
            raise RuntimeError("expected FlashInfer PrefillPlanInfo with 15 fields")
        plan_info = tuple(int(value) for value in raw_plan_info)
        pages_per_request = int(self.signature[1])
        active_indices = self.batch_size * pages_per_request
        indices = self.indices[:active_indices]

        int_workspace = getattr(self.wrapper, "_int_workspace_buffer", None)
        if not isinstance(int_workspace, torch.Tensor):
            raise RuntimeError("FlashInfer planner has no integer workspace buffer")
        workspace_bytes = int_workspace.contiguous().view(torch.uint8).flatten()
        padded_batch_size = plan_info[0]
        total_num_rows = plan_info[1]
        enable_cuda_graph = bool(plan_info[13])
        split_kv = bool(plan_info[14])
        valid_tiles = None
        if split_kv:
            mask = workspace_bytes[
                plan_info[12] : plan_info[12] + padded_batch_size
            ].detach().cpu()
            valid_tiles = int(mask.count_nonzero().item())
        initialized_tile_count = (
            valid_tiles if valid_tiles is not None else total_num_rows
        )
        ranges: list[tuple[str, int, int]] = [
            ("request_indices", plan_info[4], 4 * initialized_tile_count),
            ("qo_tile_indices", plan_info[5], 4 * initialized_tile_count),
            ("kv_tile_indices", plan_info[6], 4 * initialized_tile_count),
            ("o_indptr", plan_info[8], 4 * (self.batch_size + 1)),
            ("kv_chunk_size", plan_info[9], 4),
        ]
        if enable_cuda_graph:
            ranges.append(("total_num_rows", plan_info[2], 4))
        if split_kv:
            ranges.extend(
                (
                    ("merge_indptr", plan_info[7], 4 * (total_num_rows + 1)),
                    ("block_valid_mask", plan_info[12], padded_batch_size),
                )
            )

        workspace_components: dict[str, dict[str, Any]] = {}
        combined = hashlib.sha256()
        for name, offset, size in ranges:
            if offset < 0 or size < 0 or offset + size > workspace_bytes.numel():
                raise RuntimeError(
                    f"FlashInfer planner workspace range {name} lies out of bounds"
                )
            host = workspace_bytes[offset : offset + size].detach().cpu()
            payload = host.numpy().tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            combined.update(name.encode("utf-8"))
            combined.update(payload)
            workspace_components[name] = {
                "offset_bytes": int(offset),
                "size_bytes": int(size),
                "sha256": digest,
            }

        kv_chunk_host = workspace_bytes[
            plan_info[9] : plan_info[9] + 4
        ].detach().cpu()
        kv_chunk_size_tokens = int.from_bytes(
            kv_chunk_host.numpy().tobytes(), "little", signed=True
        )
        return {
            "pages_per_request": pages_per_request,
            "active_index_count": active_indices,
            "indptr": self.indptr[: self.batch_size + 1].detach().cpu().tolist(),
            "active_indices_sha256": tensor_content_sha256(indices),
            "active_indices_first_last": (
                [int(indices[0].item()), int(indices[-1].item())]
                if active_indices
                else []
            ),
            "last_page_len": self.last_len.detach().cpu().tolist(),
            "plan_info": list(plan_info),
            "kv_chunk_size_tokens": kv_chunk_size_tokens,
            "initialized_tile_count": initialized_tile_count,
            "effective_valid_split_tiles": valid_tiles,
            "semantic_int_workspace_sha256": combined.hexdigest(),
            "semantic_int_workspace_components": workspace_components,
        }

    def _structural_signature(self) -> tuple[Any, ...]:
        plan_info = getattr(self.wrapper, "_plan_info", None)
        if plan_info is None:
            raise RuntimeError(
                "FlashInfer wrapper did not expose _plan_info after plan(); "
                "refusing to construct a CUDA-graph topology guard"
            )
        pointer_records = [
            tensor_pointer_signature("owned_indptr", self.indptr),
            tensor_pointer_signature("owned_indices", self.indices),
            tensor_pointer_signature("owned_last_len", self.last_len),
        ]
        workspace = getattr(self, "workspace", None)
        if isinstance(workspace, torch.Tensor):
            pointer_records.append(tensor_pointer_signature("owned_workspace", workspace))
        for attribute in (
            "_paged_kv_indptr_buf",
            "_paged_kv_indices_buf",
            "_paged_kv_last_page_len_buf",
            "_qo_indptr_buf",
            "_float_workspace_buffer",
            "_int_workspace_buffer",
        ):
            tensor = getattr(self.wrapper, attribute, None)
            if isinstance(tensor, torch.Tensor):
                pointer_records.append(tensor_pointer_signature(attribute, tensor))
        module_uri = getattr(self.wrapper, "page_gauge_module_uri", None)
        cached_module = getattr(self.wrapper, "_cached_module", None)
        module_identity = (
            (module_uri, id(cached_module) if cached_module is not None else None)
            if module_uri is not None
            else (
                (
                    f"{type(cached_module).__module__}.{type(cached_module).__qualname__}",
                    id(cached_module),
                )
                if cached_module is not None
                else f"{type(self.wrapper).__module__}.{type(self.wrapper).__qualname__}"
            )
        )
        return (
            "flashinfer_fa2_decode_topology_v1",
            freeze_structural_value(plan_info),
            module_identity,
            self.batch_size,
            32,
            8,
            DIM,
            PAGE,
            str(self.kv_dtype),
            freeze_structural_value(self.run_semantics),
            freeze_structural_value(getattr(self.wrapper, "_backend", "fa2")),
            self.planned_split_pages,
            tuple(pointer_records),
        )

    def plan(
        self,
        physical_indices: torch.Tensor,
        logical_tokens: int,
        last_page_len: int,
        split_pages: int,
        page_table_epoch: int | None = None,
    ) -> None:
        self.plan_invocations += 1
        if physical_indices.dim() == 1:
            if self.batch_size != 1:
                raise ValueError("batched page indices must be [B,pages]")
            physical_indices = physical_indices.view(1, -1)
        if (
            physical_indices.dim() != 2
            or int(physical_indices.shape[0]) != self.batch_size
        ):
            raise ValueError("physical page indices must have shape [B,pages]")
        pages_per_request = int(physical_indices.shape[1])
        if pages_per_request <= 0:
            raise ValueError("decode segment must contain at least one page")
        total_pages = self.batch_size * pages_per_request
        self.last_len.fill_(last_page_len)
        self.last_len_fills += 1
        self.planned_split_pages = int(split_pages)
        signature = (
            self.batch_size,
            pages_per_request,
            math.ceil(logical_tokens / 32),
            split_pages,
            physical_indices.data_ptr(),
            page_table_epoch,
        )
        if signature == self.signature:
            return
        self.plan_rebuilds += 1
        torch.arange(
            0,
            total_pages + 1,
            pages_per_request,
            device=self.indptr.device,
            dtype=torch.int32,
            out=self.indptr,
        )
        self.indices[:total_pages].view(
            self.batch_size, pages_per_request
        ).copy_(physical_indices)
        self.wrapper.plan(
            self.indptr,
            self.indices[:total_pages],
            self.last_len,
            32,
            8,
            DIM,
            PAGE,
            pos_encoding_mode="NONE",
            q_data_type=torch.float16,
            kv_data_type=self.kv_dtype,
            o_data_type=torch.float16,
            sm_scale=1.0 / math.sqrt(DIM),
            fixed_split_size=split_pages or None,
        )
        self.signature = signature
        self.structural_signature = self._structural_signature()


class TransformerDecoder:
    def __init__(
        self,
        model,
        flashinfer,
        append_extension,
        backend: str,
        cache: BaselineCache | GaugeCache,
        max_context: int,
        exact_tail: int,
        baseline_split_pages: int,
        candidate_split_pages: int,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        center_restore: str,
        tail_attention: str = "flashinfer_merge",
        batch_size: int = 1,
        device_dynamic_decoder_layer_graphs: bool = False,
        old_value_scale_placement: str = "probability",
        exact_sink_pages: int = 0,
        exact_static_suffix_pages: int = 0,
        initial_context_pages: int | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.cache = cache
        self.append_extension = append_extension
        self.layers, self.hq, self.hkv, self.hidden = BASE_E2E.check_model(model)
        self.intermediate = int(model.config.intermediate_size)
        self.max_pages = math.ceil(max_context / PAGE)
        self.exact_tail_pages = exact_tail // PAGE
        self.exact_prefix_pages = validate_exact_prefix_attention_path(
            exact_sink_pages, tail_attention
        )
        self.exact_static_suffix_pages = validate_exact_static_suffix_pages(
            exact_static_suffix_pages,
            exact_prefix_pages=self.exact_prefix_pages,
            initial_context_pages=initial_context_pages,
        )
        if self.exact_static_suffix_pages and tail_attention != "flashinfer_merge":
            raise ValueError(
                "an exact static suffix requires the segmented flashinfer_merge path"
            )
        self.initial_context_pages = (
            int(initial_context_pages) if initial_context_pages is not None else None
        )
        self.exact_static_suffix_logical_begin = (
            self.initial_context_pages - self.exact_static_suffix_pages
            if self.exact_static_suffix_pages
            else None
        )
        # Retain the serialized/API key while treating it as generic prefix S.
        self.exact_sink_pages = self.exact_prefix_pages
        self.exact_fixed_pages = (
            self.exact_prefix_pages + self.exact_static_suffix_pages
        )
        self.exact_storage_pages = self.exact_tail_pages + self.exact_fixed_pages
        self.baseline_split_pages = baseline_split_pages
        self.candidate_split_pages = candidate_split_pages
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        if (
            rope_cos.device.type != "cuda"
            or rope_sin.device != rope_cos.device
            or rope_cos.dtype != torch.float16
            or rope_sin.dtype != torch.float16
            or rope_cos.dim() != 2
            or rope_cos.shape != rope_sin.shape
            or int(rope_cos.shape[1]) != DIM
            or not rope_cos.is_contiguous()
            or not rope_sin.is_contiguous()
        ):
            raise ValueError("RoPE tables must be matching contiguous CUDA FP16 [positions,128]")
        self.center_restore = center_restore
        self.tail_attention = tail_attention
        self.old_value_scale_placement = old_value_scale_placement
        if old_value_scale_placement not in ("probability", "value_fragment"):
            raise ValueError(
                "old INT8 V scale placement must be probability or value_fragment"
            )
        if tail_attention not in (
            "flashinfer_merge",
            "fused_kernel",
            "heterogeneous_fa2",
        ):
            raise ValueError(f"unknown exact-tail attention path {tail_attention}")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.batch_size = batch_size
        self.device_dynamic_decoder_layer_graphs = bool(
            device_dynamic_decoder_layer_graphs
        )
        self.device_position = (
            torch.zeros(1, device=rope_cos.device, dtype=torch.int32)
            if self.device_dynamic_decoder_layer_graphs
            else None
        )
        if self.device_dynamic_decoder_layer_graphs:
            required_entrypoints = (
                "rope_append_fp16_dynamic",
                "rope_append_page_gauge_dynamic",
            )
            missing = [
                name for name in required_entrypoints if not hasattr(append_extension, name)
            ]
            if missing:
                raise RuntimeError(
                    "device-dynamic decoder graphs require append extension entrypoints: "
                    + ", ".join(missing)
                )
        if (
            backend == "page_gauge"
            and tail_attention in ("fused_kernel", "heterogeneous_fa2")
            and center_restore != "attention_add"
        ):
            raise ValueError(
                f"the {tail_attention} path restores the center in its epilogue; "
                "use --center-restore attention_add"
            )
        if (
            backend == "page_gauge"
            and batch_size > 1
            and center_restore == "projection_bias"
        ):
            raise ValueError(
                "projection_bias cannot encode request-specific value centers; "
                "use --center-restore attention_add for batch size greater than one"
            )
        all_pages = request_major_page_table(batch_size, self.max_pages)
        self.all_pages = all_pages
        self.exact_ring_pages = None
        self.exact_plan_pages = None
        self.old_plan_pages = None
        self.old_layout_signature: tuple[int, int, int] | None = None
        self.exact_layout_signature: tuple[int, int, int] | None = None
        self.exact_page_table_updates = 0
        self.baseline_wrapper = None
        self.old_wrapper = None
        self.exact_wrapper = None
        self.heterogeneous_wrapper = None
        self.heterogeneous_pages = None
        self.heterogeneous_old_kv_len = None
        self.heterogeneous_layout_signature: tuple[int, int] | None = None
        if backend == "flashinfer_fp16":
            assert isinstance(cache, BaselineCache)
            if (
                int(cache.key.shape[1]) != batch_size * self.max_pages
                or cache.key.shape != cache.value.shape
            ):
                raise ValueError(
                    "baseline cache must use request-major [L,B*P,16,Hkv,D] storage"
                )
            self.baseline_wrapper = GraphDecodeWrapper(
                flashinfer, self.max_pages, torch.float16, False, batch_size
            )
        elif backend == "page_gauge":
            assert isinstance(cache, GaugeCache)
            if (
                int(cache.key_codes.shape[1]) != batch_size * self.max_pages
                or cache.key_codes.shape != cache.value_codes.shape
                or tuple(cache.key_center.shape[:2]) != (self.layers, batch_size)
                or cache.key_center.shape != cache.value_center.shape
            ):
                raise ValueError(
                    "PageGauge cache must use request-major pages and "
                    "per-request [L,B,Hkv,D] centers"
                )
            exact_ring_size = int(cache.exact_key.shape[1]) // batch_size
            if (
                int(cache.exact_key.shape[1])
                != batch_size * self.exact_storage_pages
                or cache.exact_tail_pages != self.exact_tail_pages
                or cache.exact_prefix_pages != self.exact_prefix_pages
                or cache.exact_static_suffix_pages
                != self.exact_static_suffix_pages
                or cache.initial_context_pages != self.initial_context_pages
            ):
                raise ValueError(
                    "exact cache must contain the configured fixed prefix/suffix "
                    "pages followed by tail-ring pages"
                )
            self.exact_ring_pages = request_major_exact_page_table(
                batch_size,
                self.max_pages,
                self.exact_tail_pages,
                self.exact_sink_pages,
                self.exact_static_suffix_pages,
                self.initial_context_pages,
            )
            if self.exact_fixed_pages:
                self.exact_plan_pages = torch.empty(
                    batch_size,
                    exact_ring_size,
                    device=self.exact_ring_pages.device,
                    dtype=torch.int32,
                )
            if self.exact_static_suffix_pages:
                self.old_plan_pages = torch.empty_like(self.all_pages)
            if tail_attention == "heterogeneous_fa2":
                self.heterogeneous_pages = torch.empty_like(self.all_pages)
                self.heterogeneous_old_kv_len = torch.empty(
                    batch_size, device="cuda", dtype=torch.int32
                )
                self.heterogeneous_wrapper = GraphDecodeWrapper(
                    flashinfer,
                    self.max_pages,
                    torch.int8,
                    False,
                    batch_size,
                    heterogeneous_page_gauge=True,
                )
            else:
                self.old_wrapper = GraphDecodeWrapper(
                    flashinfer,
                    self.max_pages,
                    torch.int8,
                    old_value_scale_placement == "probability",
                    batch_size,
                    v_fragment_page_gauge=(
                        old_value_scale_placement == "value_fragment"
                    ),
                )
                self.exact_wrapper = GraphDecodeWrapper(
                    flashinfer, exact_ring_size, torch.float16, False, batch_size
                )
        else:
            raise ValueError(f"unknown backend {backend}")
        self.rotated_query = torch.empty(
            batch_size, self.hq, DIM, device="cuda", dtype=torch.float16
        )
        # Give independently captured layer graphs distinct mutable outputs and
        # LSE scratch so graph ownership never depends on cross-layer aliasing.
        self.attention_output = torch.empty(
            self.layers,
            batch_size,
            self.hq,
            DIM,
            device="cuda",
            dtype=torch.float16,
        )
        self.old_lse = torch.empty(
            self.layers,
            batch_size,
            self.hq,
            device="cuda",
            dtype=torch.float32,
        )
        self.exact_output = torch.empty_like(self.attention_output)
        self.exact_lse = torch.empty_like(self.old_lse)
        self.current_pages = 1
        self.old_pages = 0
        self.exact_pages = 1
        self.old_logical_begin = self.exact_sink_pages
        self.old_logical_end = self.exact_sink_pages
        self.old_logical_pages: tuple[int, ...] = ()
        self.tail_logical_begin = self.exact_sink_pages
        self.attention_graphs: list[torch.cuda.CUDAGraph] | None = None
        self.attention_graph_plan_signature = None
        self.attention_graph_replays = 0
        self.attention_eager_calls = 0
        self.decoder_layer_graphs: list[list[torch.cuda.CUDAGraph]] | None = None
        self.decoder_layer_graph_buffers: torch.Tensor | None = None
        self.decoder_layer_graph_start_position: int | None = None
        self.decoder_layer_graph_plan_signature = None
        self.decoder_layer_graph_replays = 0
        self.decoder_layer_eager_calls = 0
        self.decoder_layer_graph_bank_misses = 0
        self.decoder_layer_graph_banks: dict[
            tuple[Any, ...], list[torch.cuda.CUDAGraph]
        ] = {}
        self.decoder_layer_graph_bank_buffers: dict[
            tuple[Any, ...], torch.Tensor
        ] = {}
        self.decoder_layer_graph_bank_capture_positions: dict[
            tuple[Any, ...], int
        ] = {}
        self.decoder_layer_graph_preflight_positions: dict[
            tuple[Any, ...], tuple[int, int]
        ] = {}
        self.decoder_layer_graph_preflight_counts: dict[tuple[Any, ...], int] = {}
        self.decoder_layer_graph_preflight_plan_info: dict[
            tuple[Any, ...], dict[str, Any]
        ] = {}
        self.decoder_layer_graph_preflight_runtime_metadata: dict[
            tuple[Any, ...], list[dict[str, Any]]
        ] = {}
        self.decoder_layer_graph_preflight_position_count = 0
        self.decoder_layer_graph_preflight_range: tuple[int, int] | None = None
        self.decoder_layer_graph_preflight_positions_sha256: str | None = None
        self.decoder_layer_graph_bank_replay_enabled = True
        self.decoder_layer_graph_bank_strict = False
        self.decoder_layer_graph_bank_memory: dict[str, int] = {}
        self.decoder_plan_calls = 0
        self.device_position_fills = 0
        self.heterogeneous_page_table_updates = 0
        self.output_projection_biases: list[torch.Tensor | None] = []
        for layer_index, layer in enumerate(self.model.model.layers):
            original_bias = getattr(layer.self_attn.o_proj, "bias", None)
            if self.backend == "flashinfer_fp16":
                self.output_projection_biases.append(original_bias)
                continue
            if self.center_restore == "projection_bias":
                assert isinstance(self.cache, GaugeCache)
                with torch.no_grad():
                    center_bias = folded_output_projection_bias(
                        layer.self_attn.o_proj.weight,
                        original_bias,
                        self.cache.output_center[layer_index, 0],
                    )
                self.output_projection_biases.append(center_bias)
            else:
                self.output_projection_biases.append(original_bias)

    def exact_physical_page(self, request: int, logical_page: int) -> int:
        if self.backend != "page_gauge":
            raise ValueError("exact physical pages exist only for PageGauge")
        if request >= self.batch_size:
            raise ValueError("request index lies outside the decoder batch")
        return exact_physical_page_index(
            request,
            logical_page,
            self.exact_tail_pages,
            self.exact_sink_pages,
            self.exact_static_suffix_pages,
            self.initial_context_pages,
        )

    def plan_signature(self):
        if self.backend == "flashinfer_fp16":
            assert self.baseline_wrapper is not None
            return self.baseline_wrapper.signature
        if self.tail_attention == "heterogeneous_fa2":
            assert self.heterogeneous_wrapper is not None
            return self.heterogeneous_wrapper.signature
        assert self.old_wrapper is not None and self.exact_wrapper is not None
        return (self.old_wrapper.signature, self.exact_wrapper.signature)

    def structural_plan_signature(self) -> tuple[Any, ...]:
        """Topology and pointer guard for reusable complete-layer graphs."""

        if self.backend == "flashinfer_fp16":
            assert self.baseline_wrapper is not None
            wrappers = (self.baseline_wrapper.structural_signature,)
        elif self.tail_attention == "heterogeneous_fa2":
            assert self.heterogeneous_wrapper is not None
            wrappers = (self.heterogeneous_wrapper.structural_signature,)
        else:
            assert self.old_wrapper is not None and self.exact_wrapper is not None
            wrappers = (
                self.old_wrapper.structural_signature,
                self.exact_wrapper.structural_signature,
            )
        if any(signature is None for signature in wrappers):
            raise RuntimeError("decoder structural signature requested before plan()")
        cache_tensors = (
            ("key", self.cache.key, "value", self.cache.value)
            if isinstance(self.cache, BaselineCache)
            else (
                "exact_key",
                self.cache.exact_key,
                "exact_value",
                self.cache.exact_value,
                "key_codes",
                self.cache.key_codes,
                "value_codes",
                self.cache.value_codes,
                "key_scales",
                self.cache.key_scales,
                "value_scales",
                self.cache.value_scales,
                "key_center",
                self.cache.key_center,
                "value_center",
                self.cache.value_center,
                "output_center",
                self.cache.output_center,
            )
        )
        cache_pointer_records = tuple(
            tensor_pointer_signature(str(cache_tensors[index]), cache_tensors[index + 1])
            for index in range(0, len(cache_tensors), 2)
        )
        dynamic_position_record = (
            tensor_pointer_signature("device_position", self.device_position)
            if self.device_position is not None
            else None
        )
        return (
            "page_gauge_decoder_layer_topology_v3_exact_prefix",
            self.backend,
            self.tail_attention,
            self.center_restore,
            self.old_value_scale_placement,
            self.exact_tail_pages,
            self.exact_prefix_pages,
            self.exact_static_suffix_pages,
            self.initial_context_pages,
            self.exact_pages if self.tail_attention == "fused_kernel" else None,
            tuple(wrappers),
            tensor_pointer_signature("rope_cos", self.rope_cos),
            tensor_pointer_signature("rope_sin", self.rope_sin),
            dynamic_position_record,
            tensor_pointer_signature("rotated_query", self.rotated_query),
            tensor_pointer_signature("attention_output", self.attention_output),
            tensor_pointer_signature("old_lse", self.old_lse),
            tensor_pointer_signature("exact_output", self.exact_output),
            tensor_pointer_signature("exact_lse", self.exact_lse),
            (
                tensor_pointer_signature(
                    "heterogeneous_old_kv_len", self.heterogeneous_old_kv_len
                )
                if self.heterogeneous_old_kv_len is not None
                else None
            ),
            (
                tensor_pointer_signature("exact_plan_pages", self.exact_plan_pages)
                if self.exact_plan_pages is not None
                else None
            ),
            (
                tensor_pointer_signature("old_plan_pages", self.old_plan_pages)
                if self.old_plan_pages is not None
                else None
            ),
            cache_pointer_records,
        )

    def structural_plan_info(self) -> dict[str, Any]:
        wrappers = (
            ("baseline", self.baseline_wrapper),
            ("old_int8", self.old_wrapper),
            ("exact_fp16", self.exact_wrapper),
            ("heterogeneous", self.heterogeneous_wrapper),
        )
        return {
            name: {
                "plan_info": freeze_structural_value(
                    getattr(wrapper.wrapper, "_plan_info", None)
                ),
                "resolved_backend": freeze_structural_value(
                    getattr(wrapper.wrapper, "_backend", "fa2")
                ),
                "fixed_split_pages": wrapper.planned_split_pages,
            }
            for name, wrapper in wrappers
            if wrapper is not None
        }

    def plan(self, context: int) -> None:
        self.decoder_plan_calls += 1
        pages = math.ceil(context / PAGE)
        if context <= 0 or pages > self.max_pages:
            raise ValueError("context lies outside the allocated cache capacity")
        last_len = (context - 1) % PAGE + 1
        self.current_pages = pages
        if self.backend == "flashinfer_fp16":
            assert self.baseline_wrapper is not None
            self.baseline_wrapper.plan(
                self.all_pages[:, :pages],
                context,
                last_len,
                self.baseline_split_pages,
            )
            return
        exact_static_suffix_pages = getattr(self, "exact_static_suffix_pages", 0)
        initial_context_pages = getattr(self, "initial_context_pages", None)
        exact_fixed_pages = getattr(
            self,
            "exact_fixed_pages",
            self.exact_sink_pages + exact_static_suffix_pages,
        )
        partition = page_gauge_logical_partition(
            pages,
            self.exact_tail_pages,
            self.exact_sink_pages,
            last_len,
            exact_static_suffix_pages,
            initial_context_pages,
        )
        self.tail_logical_begin = int(partition["tail_logical_begin"])
        self.old_logical_pages = tuple(partition["old_logical_pages"])
        self.old_pages = int(partition["old_page_count"])
        self.exact_pages = int(partition["exact_page_count"])
        self.old_logical_begin = self.old_logical_pages[0]
        self.old_logical_end = self.old_logical_pages[-1] + 1
        old_tokens = int(partition["old_token_count"])
        exact_tokens = int(partition["exact_token_count"])
        if not partition["coverage_disjoint"] or old_tokens + exact_tokens != context:
            raise RuntimeError("PageGauge page sets do not cover context exactly once")
        assert self.exact_ring_pages is not None
        if self.tail_attention == "heterogeneous_fa2":
            assert (
                self.heterogeneous_wrapper is not None
                and self.heterogeneous_pages is not None
                and self.heterogeneous_old_kv_len is not None
            )
            layout_signature = (self.old_pages, pages)
            if layout_signature != self.heterogeneous_layout_signature:
                write_heterogeneous_page_table(
                    self.heterogeneous_pages,
                    self.all_pages,
                    self.exact_ring_pages,
                    self.old_pages,
                    pages,
                )
                self.heterogeneous_old_kv_len.fill_(old_tokens)
                self.heterogeneous_layout_signature = layout_signature
                self.heterogeneous_page_table_updates += 1
            self.heterogeneous_wrapper.plan(
                self.heterogeneous_pages[:, :pages],
                context,
                last_len,
                self.candidate_split_pages,
            )
            return
        assert self.old_wrapper is not None and self.exact_wrapper is not None
        old_indices = self.all_pages[
            :, self.old_logical_begin : self.old_logical_end
        ]
        if exact_static_suffix_pages:
            assert self.old_plan_pages is not None
            old_layout_signature = (
                self.tail_logical_begin,
                pages,
                self.old_pages,
            )
            if old_layout_signature != self.old_layout_signature:
                old_indices = write_page_gauge_old_page_table(
                    self.old_plan_pages,
                    self.all_pages,
                    self.old_logical_pages,
                )
                self.old_layout_signature = old_layout_signature
            else:
                old_indices = self.old_plan_pages[:, : self.old_pages]
        self.old_wrapper.plan(
            old_indices,
            old_tokens,
            PAGE,
            self.candidate_split_pages,
        )
        exact_indices = self.exact_ring_pages[
            :, self.tail_logical_begin : pages
        ]
        exact_page_table_epoch = None
        if exact_fixed_pages:
            assert self.exact_plan_pages is not None
            layout_signature = (self.tail_logical_begin, pages, self.exact_pages)
            if layout_signature != self.exact_layout_signature:
                exact_indices = write_exact_prefix_tail_page_table(
                    self.exact_plan_pages,
                    self.exact_ring_pages,
                    self.tail_logical_begin,
                    pages,
                    self.exact_sink_pages,
                    exact_static_suffix_pages,
                    initial_context_pages,
                )
                self.exact_layout_signature = layout_signature
                self.exact_page_table_updates += 1
            else:
                exact_indices = self.exact_plan_pages[:, : self.exact_pages]
            exact_page_table_epoch = pages
        self.exact_wrapper.plan(
            exact_indices,
            exact_tokens,
            last_len,
            self.candidate_split_pages,
            page_table_epoch=exact_page_table_epoch,
        )

    def append(
        self,
        layer: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ) -> None:
        if self.device_dynamic_decoder_layer_graphs:
            if self.device_position is None:
                raise RuntimeError("device-dynamic append has no persistent position scalar")
            if self.backend == "flashinfer_fp16":
                assert isinstance(self.cache, BaselineCache)
                self.append_extension.rope_append_fp16_dynamic(
                    query,
                    key,
                    value,
                    self.rope_cos,
                    self.rope_sin,
                    self.device_position,
                    self.rotated_query,
                    self.cache.key[layer],
                    self.cache.value[layer],
                )
            else:
                assert isinstance(self.cache, GaugeCache)
                self.append_extension.rope_append_page_gauge_dynamic(
                    query,
                    key,
                    value,
                    self.rope_cos,
                    self.rope_sin,
                    self.device_position,
                    self.cache.key_center[layer],
                    self.cache.value_center[layer],
                    self.rotated_query,
                    self.cache.exact_key[layer],
                    self.cache.exact_value[layer],
                    self.cache.key_codes[layer],
                    self.cache.value_codes[layer],
                    self.cache.key_scales[layer],
                    self.cache.value_scales[layer],
                    self.exact_fixed_pages,
                )
            return
        cosine = self.rope_cos[position]
        sine = self.rope_sin[position]
        if self.backend == "flashinfer_fp16":
            assert isinstance(self.cache, BaselineCache)
            self.append_extension.rope_append_fp16(
                query,
                key,
                value,
                cosine,
                sine,
                self.rotated_query,
                self.cache.key[layer],
                self.cache.value[layer],
                position,
            )
        else:
            assert isinstance(self.cache, GaugeCache)
            self.append_extension.rope_append_page_gauge(
                query,
                key,
                value,
                cosine,
                sine,
                self.cache.key_center[layer],
                self.cache.value_center[layer],
                self.rotated_query,
                self.cache.exact_key[layer],
                self.cache.exact_value[layer],
                self.cache.key_codes[layer],
                self.cache.value_codes[layer],
                self.cache.key_scales[layer],
                self.cache.value_scales[layer],
                self.exact_fixed_pages,
                position,
            )

    def eager_attention(self, layer: int) -> torch.Tensor:
        if self.backend == "flashinfer_fp16":
            assert isinstance(self.cache, BaselineCache)
            assert self.baseline_wrapper is not None
            return self.baseline_wrapper.wrapper.run(
                self.rotated_query,
                (self.cache.key[layer], self.cache.value[layer]),
                out=self.attention_output[layer],
            )
        assert isinstance(self.cache, GaugeCache)
        if self.tail_attention == "heterogeneous_fa2":
            assert (
                self.heterogeneous_wrapper is not None
                and self.heterogeneous_old_kv_len is not None
            )
            return self.heterogeneous_wrapper.wrapper.run(
                self.rotated_query,
                (self.cache.key_codes[layer], self.cache.value_codes[layer]),
                self.cache.key_scales[layer],
                self.cache.value_scales[layer],
                self.cache.exact_key[layer],
                self.cache.exact_value[layer],
                self.heterogeneous_old_kv_len,
                self.cache.value_center[layer],
                1.0 / math.sqrt(DIM),
                out=self.attention_output[layer],
            )
        assert self.old_wrapper is not None and self.exact_wrapper is not None
        self.old_wrapper.wrapper.run(
            self.rotated_query,
            (self.cache.key_codes[layer], self.cache.value_codes[layer]),
            self.cache.key_scales[layer],
            self.cache.value_scales[layer],
            1.0 / math.sqrt(DIM),
            out=self.attention_output[layer],
            lse=self.old_lse[layer],
            return_lse=True,
        )
        if self.tail_attention == "fused_kernel":
            self.append_extension.exact_tail_merge_center(
                self.rotated_query,
                self.cache.exact_key[layer],
                self.cache.exact_value[layer],
                self.exact_wrapper.indices[
                    : self.batch_size * self.exact_pages
                ],
                self.exact_wrapper.last_len,
                self.attention_output[layer],
                self.old_lse[layer],
                self.cache.value_center[layer],
                1.0 / math.sqrt(DIM),
            )
            return self.attention_output[layer]
        self.exact_wrapper.wrapper.run(
            self.rotated_query,
            (self.cache.exact_key[layer], self.cache.exact_value[layer]),
            out=self.exact_output[layer],
            lse=self.exact_lse[layer],
            return_lse=True,
        )
        import flashinfer

        flashinfer.merge_state_in_place(
            self.attention_output[layer],
            self.old_lse[layer],
            self.exact_output[layer],
            self.exact_lse[layer],
        )
        if self.center_restore == "attention_add":
            self.attention_output[layer].add_(self.cache.output_center[layer])
        return self.attention_output[layer]

    def graph_attention(self, layer: int) -> torch.Tensor:
        """Attention body captured by CUDA Graph."""
        return self.eager_attention(layer)

    def attention(self, layer: int) -> torch.Tensor:
        if (
            self.attention_graphs is not None
            and self.plan_signature() == self.attention_graph_plan_signature
        ):
            self.attention_graph_replays += 1
            self.attention_graphs[layer].replay()
            return self.attention_output[layer]
        self.attention_eager_calls += 1
        return self.eager_attention(layer)

    def reset_attention_dispatch_counts(self) -> None:
        self.attention_graph_replays = 0
        self.attention_eager_calls = 0
        self.decoder_layer_graph_replays = 0
        self.decoder_layer_eager_calls = 0
        self.decoder_layer_graph_bank_misses = 0

    def reset_runtime_operation_counts(self) -> None:
        self.decoder_plan_calls = 0
        self.device_position_fills = 0
        self.heterogeneous_page_table_updates = 0
        self.exact_page_table_updates = 0
        for wrapper in (
            self.baseline_wrapper,
            self.old_wrapper,
            self.exact_wrapper,
            self.heterogeneous_wrapper,
        ):
            if wrapper is not None:
                wrapper.reset_operation_counts()

    def runtime_operation_counts(self) -> dict[str, Any]:
        wrapper_counts = {}
        for name, wrapper in (
            ("baseline", self.baseline_wrapper),
            ("old_int8", self.old_wrapper),
            ("exact_fp16", self.exact_wrapper),
            ("heterogeneous", self.heterogeneous_wrapper),
        ):
            if wrapper is not None:
                wrapper_counts[name] = wrapper.operation_counts()
        return {
            "decoder_plan_calls": self.decoder_plan_calls,
            "device_position_fills": self.device_position_fills,
            "heterogeneous_page_table_updates": self.heterogeneous_page_table_updates,
            "exact_page_table_updates": self.exact_page_table_updates,
            "wrappers": wrapper_counts,
        }

    def serving_metadata_snapshot(self) -> dict[str, Any]:
        """Return final active page-table/planner state for validation only."""

        wrappers = {}
        for name, wrapper in (
            ("baseline", self.baseline_wrapper),
            ("old_int8", self.old_wrapper),
            ("exact_fp16", self.exact_wrapper),
            ("heterogeneous", self.heterogeneous_wrapper),
        ):
            if wrapper is not None:
                wrappers[name] = wrapper.serving_metadata_snapshot()
        heterogeneous_old_kv_len = None
        if self.heterogeneous_old_kv_len is not None:
            heterogeneous_old_kv_len = (
                self.heterogeneous_old_kv_len.detach().cpu().tolist()
            )
        return {
            "current_pages": int(self.current_pages),
            "old_pages": int(self.old_pages),
            "exact_pages": int(self.exact_pages),
            "exact_tail_pages": int(self.exact_tail_pages),
            "exact_sink_pages": int(self.exact_sink_pages),
            "exact_prefix_pages": int(self.exact_prefix_pages),
            "exact_static_suffix_pages": int(self.exact_static_suffix_pages),
            "exact_fixed_pages": int(self.exact_fixed_pages),
            "initial_context_pages": self.initial_context_pages,
            "old_logical_range_start_inclusive_end_exclusive": [
                int(self.old_logical_begin),
                int(self.old_logical_end),
            ],
            "exact_sink_logical_pages": list(range(self.exact_sink_pages)),
            "exact_prefix_logical_pages": list(range(self.exact_prefix_pages)),
            "exact_static_suffix_logical_pages": (
                list(
                    range(
                        int(self.exact_static_suffix_logical_begin),
                        int(self.initial_context_pages),
                    )
                )
                if self.exact_static_suffix_pages
                else []
            ),
            "old_logical_pages": list(self.old_logical_pages),
            "exact_tail_logical_range_start_inclusive_end_exclusive": [
                int(self.tail_logical_begin),
                int(self.current_pages),
            ],
            "logical_page_partition_disjoint": bool(
                len(self.old_logical_pages) == self.old_pages
                and len(set(self.old_logical_pages)) == self.old_pages
                and self.exact_pages + self.old_pages == self.current_pages
            ),
            "heterogeneous_layout_signature": (
                list(self.heterogeneous_layout_signature)
                if self.heterogeneous_layout_signature is not None
                else None
            ),
            "heterogeneous_old_kv_len": heterogeneous_old_kv_len,
            "wrappers": wrappers,
        }

    def attention_dispatch_counts(self) -> dict[str, Any]:
        total = self.attention_graph_replays + self.attention_eager_calls
        return {
            "graph_replays": self.attention_graph_replays,
            "eager_calls": self.attention_eager_calls,
            "total_calls": total,
            "graph_coverage_fraction": (
                self.attention_graph_replays / total if total else 0.0
            ),
        }

    def decoder_layer_dispatch_counts(self) -> dict[str, Any]:
        total = self.decoder_layer_graph_replays + self.decoder_layer_eager_calls
        return {
            "graph_replays": self.decoder_layer_graph_replays,
            "eager_calls": self.decoder_layer_eager_calls,
            "graph_bank_misses": getattr(self, "decoder_layer_graph_bank_misses", 0),
            "total_calls": total,
            "graph_coverage_fraction": (
                self.decoder_layer_graph_replays / total if total else 0.0
            ),
        }

    def decoder_layer_graph_provenance(self) -> dict[str, Any]:
        dynamic_banks = getattr(self, "decoder_layer_graph_banks", {})
        if getattr(self, "device_dynamic_decoder_layer_graphs", False):
            graph_count = sum(len(bank) for bank in dynamic_banks.values())
            buffers = getattr(self, "decoder_layer_graph_bank_buffers", {})
            buffer_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in buffers.values()
            )
            capture_positions = getattr(
                self, "decoder_layer_graph_bank_capture_positions", {}
            )
            preflight = getattr(
                self, "decoder_layer_graph_preflight_positions", {}
            )
            preflight_counts = getattr(
                self, "decoder_layer_graph_preflight_counts", {}
            )
            preflight_plan_info = getattr(
                self, "decoder_layer_graph_preflight_plan_info", {}
            )
            preflight_runtime_metadata = getattr(
                self, "decoder_layer_graph_preflight_runtime_metadata", {}
            )

            def summarize_runtime_states(
                states: list[dict[str, Any]],
            ) -> dict[str, Any]:
                names = sorted(
                    {
                        name
                        for state in states
                        for name in state.get("wrappers", {})
                    }
                )
                summaries = {}
                for name in names:
                    records = [
                        state["wrappers"][name]
                        for state in states
                        if name in state.get("wrappers", {})
                    ]
                    valid_tiles = [
                        record["effective_valid_split_tiles"]
                        for record in records
                        if record["effective_valid_split_tiles"] is not None
                    ]
                    summaries[name] = {
                        "sampled_boundary_state_count": len(records),
                        "active_pages_per_request_min": min(
                            record["pages_per_request"] for record in records
                        ),
                        "active_pages_per_request_max": max(
                            record["pages_per_request"] for record in records
                        ),
                        "kv_chunk_size_tokens_min": min(
                            record["kv_chunk_size_tokens"] for record in records
                        ),
                        "kv_chunk_size_tokens_max": max(
                            record["kv_chunk_size_tokens"] for record in records
                        ),
                        "effective_valid_split_tiles_min": (
                            min(valid_tiles) if valid_tiles else None
                        ),
                        "effective_valid_split_tiles_max": (
                            max(valid_tiles) if valid_tiles else None
                        ),
                    }
                return summaries
            return {
                "enabled": bool(dynamic_banks),
                "scope": "complete_decoder_layer_device_dynamic_position",
                "device_dynamic_position": True,
                "graph_bank_count": len(dynamic_banks),
                "graphs_per_bank": self.layers,
                "total_graphs": graph_count,
                "graph_pools": len(dynamic_banks),
                "replays_per_decode_step": self.layers,
                "nested_attention_graphs": False,
                "graph_pool_policy": "one shared serial-replay pool per structural plan signature",
                "raw_cuda_graph_retained_after_instantiation": False,
                "persistent_hidden_buffer_bytes": int(buffer_bytes),
                "capture_memory": dict(
                    getattr(self, "decoder_layer_graph_bank_memory", {})
                ),
                "capture_memory_scope": (
                    "graph-bank-only delta after exhaustive preflight; worker-level "
                    "before/after records are authoritative for total startup cost"
                ),
                "capture_positions": sorted(int(value) for value in capture_positions.values()),
                "preflight_bucket_ranges": sorted(
                    (
                        {
                            "first_position": int(bounds[0]),
                            "last_position": int(bounds[1]),
                            "position_count": int(preflight_counts.get(signature, 0)),
                            "capture_position": int(capture_positions[signature]),
                            "structural_signature_sha256": hashlib.sha256(
                                repr(signature).encode("utf-8")
                            ).hexdigest(),
                            "plan": preflight_plan_info.get(signature),
                            "planner_runtime_states": preflight_runtime_metadata.get(
                                signature, []
                            ),
                            "planner_runtime_summary": summarize_runtime_states(
                                preflight_runtime_metadata.get(signature, [])
                            ),
                        }
                        for signature, bounds in preflight.items()
                    ),
                    key=lambda record: record["first_position"],
                ),
                "preflight_position_count": int(
                    getattr(self, "decoder_layer_graph_preflight_position_count", 0)
                ),
                "preflight_range_start_inclusive_end_exclusive": list(
                    getattr(self, "decoder_layer_graph_preflight_range", ()) or ()
                ),
                "preflight_positions_sha256": getattr(
                    self, "decoder_layer_graph_preflight_positions_sha256", None
                ),
                "strict_missing_bucket_failure": bool(
                    getattr(self, "decoder_layer_graph_bank_strict", False)
                ),
                "graph_bank_misses": int(
                    getattr(self, "decoder_layer_graph_bank_misses", 0)
                ),
                "position_guard": (
                    "actual FlashInfer _plan_info topology plus stable wrapper, "
                    "cache, scratch, RoPE-table, and device-position pointers"
                ),
                "device_position_updates": (
                    "one persistent CUDA int32 scalar fill inside each decoder step"
                ),
                "captured_page_offsets": "all offsets selected on device",
                "included_operations": [
                    "input_layernorm",
                    "packed_qkv_projection",
                    "device_position_rope_append_and_page_finalization",
                    "selected_attention_path",
                    "output_projection_and_residual",
                    "post_attention_layernorm",
                    "packed_gate_up_projection",
                    "silu_times_up",
                    "down_projection_and_residual",
                    "persistent_layer_output_write",
                ],
                "excluded_operations": [
                    "device_position_fill",
                    "current_position_flashinfer_plan_update",
                    "runtime_page_table_updates",
                    "token_embedding",
                    "final_model_norm",
                    "lm_head",
                    "gpu_argmax",
                    "one_time_graph_bank_preflight_and_capture",
                ],
            }
        captured = self.decoder_layer_graphs is not None
        graph_count = (
            sum(len(bank) for bank in self.decoder_layer_graphs)
            if captured
            else 0
        )
        buffer_bytes = (
            self.decoder_layer_graph_buffers.numel()
            * self.decoder_layer_graph_buffers.element_size()
            if self.decoder_layer_graph_buffers is not None
            else 0
        )
        return {
            "enabled": captured,
            "scope": "complete_decoder_layer",
            "captured_page_offsets": list(range(PAGE)) if captured else [],
            "graphs_per_offset": self.layers if captured else 0,
            "total_graphs": graph_count,
            "graph_pools": PAGE if captured else 0,
            "replays_per_decode_step": self.layers if captured else 0,
            "nested_attention_graphs": False,
            "graph_pool_policy": "one shared serial-replay pool per page offset",
            "raw_cuda_graph_retained_after_instantiation": False,
            "persistent_hidden_buffer_bytes": int(buffer_bytes),
            "captured_start_position": self.decoder_layer_graph_start_position,
            "position_guard": (
                "exact captured logical page and invariant FlashInfer plan signature"
            ),
            "included_operations": [
                "input_layernorm",
                "packed_qkv_projection",
                "rope_append_and_page_finalization",
                "selected_attention_path",
                "output_projection_and_residual",
                "post_attention_layernorm",
                "packed_gate_up_projection",
                "silu_times_up",
                "down_projection_and_residual",
                "persistent_layer_output_write",
            ],
            "excluded_operations": [
                "token_embedding",
                "final_model_norm",
                "lm_head",
                "gpu_argmax",
            ],
        }

    def attention_implementation_provenance(self) -> dict[str, Any]:
        if self.backend == "flashinfer_fp16":
            return {
                "implementation": "flashinfer_fp16_fa2",
                "logical_attention_segments": 1,
                "custom_module_uri": None,
                "custom_module_source_hashes": None,
                "old_int8_value_scale_placement": None,
                "exact_sink_pages": 0,
                "exact_prefix_pages": 0,
            }
        if self.tail_attention == "heterogeneous_fa2":
            assert self.heterogeneous_wrapper is not None
            wrapper = self.heterogeneous_wrapper.wrapper
            return {
                "implementation": "page_gauge_heterogeneous_int8_fp16_fa2",
                "logical_attention_segments": 1,
                "custom_module_uri": wrapper.page_gauge_module_uri,
                "custom_module_source_hashes": wrapper.page_gauge_source_hashes,
                "value_center_restoration": "inside_split_output_transform",
                "old_int8_value_scale_placement": "value_fragment",
                "exact_sink_pages": 0,
                "exact_prefix_pages": 0,
            }
        return {
            "implementation": f"page_gauge_segmented_{self.tail_attention}",
            "logical_attention_segments": 2,
            "custom_module_uri": getattr(
                self.old_wrapper.wrapper, "page_gauge_module_uri", None
            ),
            "custom_module_source_hashes": getattr(
                self.old_wrapper.wrapper, "page_gauge_source_hashes", None
            ),
            "value_center_restoration": self.center_restore,
            "old_int8_value_scale_placement": self.old_value_scale_placement,
            "exact_tail_pages": self.exact_tail_pages,
            "exact_sink_pages": self.exact_sink_pages,
            "exact_prefix_pages": self.exact_prefix_pages,
            "exact_segment_logical_order": (
                "[contiguous exact prefix logical pages 0..S-1, chronological "
                "recent exact tail]"
                if self.exact_prefix_pages
                else "chronological recent exact tail"
            ),
            "old_segment_logical_range": (
                "[S, tail_start) with every exact-prefix page excluded"
                if self.exact_prefix_pages
                else "[0, tail_start)"
            ),
            "exact_physical_layout": (
                "fixed slots 0..S-1 followed by modulo tail-ring slots S..S+T-1"
            ),
            "prefix_uses_existing_exact_wrapper": True,
            "logical_attention_segments_unchanged_by_prefix": True,
            "sink_uses_existing_exact_wrapper": True,
            "logical_attention_segments_unchanged_by_sink": True,
        }

    def capture_attention_graphs(self, context: int) -> None:
        """Capture the matched attention subgraph once per model layer.

        The segmented PageGauge paths contain old and exact attention followed
        by an online-state merge.  The heterogeneous path instead captures one
        FA2 wrapper spanning both representations and restores the value center
        in its output transform.
        Eagerly dispatching the attention operations from Python creates
        host-launch bubbles that are absent from standalone CUDA timing.
        Capturing both the FP16 and PageGauge attention paths preserves every
        operation while reducing each path to one host graph launch.

        Graph replay is guarded by the FlashInfer plan signature. If a future
        context changes the padded kernel plan, attention safely falls back to
        eager execution until graphs are explicitly recaptured.
        """
        # Attention-only and complete-layer graphs are mutually exclusive.
        self.decoder_layer_graphs = None
        self.decoder_layer_graph_buffers = None
        self.decoder_layer_graph_start_position = None
        self.decoder_layer_graph_plan_signature = None
        self.plan(context)
        self.rotated_query.zero_()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for layer in range(self.layers):
                self.graph_attention(layer)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graphs: list[torch.cuda.CUDAGraph] = []
        for layer in range(self.layers):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                self.graph_attention(layer)
            graphs.append(graph)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self.attention_graphs = graphs
        self.attention_graph_plan_signature = self.plan_signature()

    def _execute_decoder_layer(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        position: int,
        destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute the complete mathematical body of one Mistral decoder layer."""
        layer = self.model.model.layers[layer_index]
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
        query = q_flat.reshape(self.batch_size, self.hq, DIM)
        key = k_flat.reshape(self.batch_size, self.hkv, DIM)
        value = v_flat.reshape(self.batch_size, self.hkv, DIM)
        self.append(layer_index, query, key, value, position)
        attended = self.attention(layer_index)
        projected = F.linear(
            attended.reshape(self.batch_size, -1),
            attention.o_proj.weight,
            self.output_projection_biases[layer_index],
        )
        hidden = residual + projected
        mlp_input = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        gate_up = F.linear(
            mlp_input,
            mlp.gate_up_weight,
            getattr(mlp, "gate_up_bias", None),
        )
        gate, up = gate_up.split(mlp._pkv_intermediate, dim=-1)
        activated = F.silu(gate) * up
        hidden = hidden + mlp.down_proj(activated)
        if destination is not None:
            destination.copy_(hidden)
            return destination
        return hidden

    def _decoder_layer_graph_offset(self, position: int) -> int | None:
        if (
            self.decoder_layer_graphs is None
            or self.decoder_layer_graph_start_position is None
            or self.plan_signature() != self.decoder_layer_graph_plan_signature
        ):
            return None
        offset = position - self.decoder_layer_graph_start_position
        return offset if 0 <= offset < PAGE else None

    @torch.inference_mode()
    def capture_decoder_layer_graphs(
        self, start_position: int, capture_warmups: int = 1
    ) -> None:
        """Capture 16 offset-specific banks of complete decoder-layer graphs.

        A bank contains one graph per layer and is replayed serially. Stable
        [B,H] buffers connect adjacent layer graphs, so generated token IDs may
        change on every replay while all tensor addresses remain invariant.
        The exact logical page is part of the replay guard because append and
        page-finalization destinations are position-specific graph arguments.
        """
        if start_position % PAGE:
            raise ValueError("decoder-layer graph capture must start page aligned")
        if capture_warmups <= 0:
            raise ValueError("decoder-layer graph capture warmups must be positive")
        self.attention_graphs = None
        self.attention_graph_plan_signature = None

        signatures = []
        for offset in range(PAGE):
            self.plan(start_position + offset + 1)
            signatures.append(self.plan_signature())
        torch.cuda.synchronize()
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise RuntimeError(
                "decoder-layer graphs require one invariant FlashInfer plan bucket"
            )

        buffers = torch.zeros(
            PAGE,
            self.layers + 1,
            self.batch_size,
            self.hidden,
            device="cuda",
            dtype=torch.float16,
        )
        current_stream = torch.cuda.current_stream()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(capture_warmups):
                for offset in range(PAGE):
                    position = start_position + offset
                    self.plan(position + 1)
                    for layer_index in range(self.layers):
                        self._execute_decoder_layer(
                            layer_index,
                            buffers[offset, layer_index],
                            position,
                            buffers[offset, layer_index + 1],
                        )
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph_banks: list[list[torch.cuda.CUDAGraph]] = []
        for offset in range(PAGE):
            position = start_position + offset
            with torch.cuda.stream(capture_stream):
                self.plan(position + 1)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize()
            pool = torch.cuda.graph_pool_handle()
            bank: list[torch.cuda.CUDAGraph] = []
            for layer_index in range(self.layers):
                # Default keep_graph=False instantiates the executable at
                # capture end and releases the raw cudaGraph_t. With 512
                # graphs, retaining both forms would be material overhead.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=pool,
                    stream=capture_stream,
                ):
                    self._execute_decoder_layer(
                        layer_index,
                        buffers[offset, layer_index],
                        position,
                        buffers[offset, layer_index + 1],
                    )
                bank.append(graph)
            graph_banks.append(bank)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize()
        if (
            len(graph_banks) != PAGE
            or any(len(bank) != self.layers for bank in graph_banks)
        ):
            raise RuntimeError("decoder-layer graph capture did not cover every layer")
        self.decoder_layer_graphs = graph_banks
        self.decoder_layer_graph_buffers = buffers
        self.decoder_layer_graph_start_position = start_position
        self.decoder_layer_graph_plan_signature = signatures[0]
        self.reset_attention_dispatch_counts()

    def preflight_dynamic_decoder_layer_graph_banks(
        self,
        start_position: int,
        decode_steps: int,
        *,
        maximum_graph_banks: int = 64,
    ) -> dict[tuple[Any, ...], int]:
        """Plan every served position and enumerate all launch topologies.

        This is deliberately exhaustive and runs before timing. A page-count
        transition invokes FlashInfer's supported ``plan`` API; positions
        inside the page still enqueue the real persistent ``last_len`` fill.
        Any capacity failure, unknown plan-info value, or excessive topology
        count aborts before capture rather than permitting an eager fallback.
        """

        if not self.device_dynamic_decoder_layer_graphs:
            raise RuntimeError("device-dynamic decoder-layer graphs are not enabled")
        if start_position < 0 or decode_steps <= 0:
            raise ValueError("preflight position and decode length must be positive")
        if start_position + decode_steps > int(self.rope_cos.shape[0]):
            raise ValueError("preflight range exceeds the allocated RoPE table")
        if maximum_graph_banks <= 0:
            raise ValueError("maximum graph-bank count must be positive")

        representatives: dict[tuple[Any, ...], int] = {}
        bounds: dict[tuple[Any, ...], list[int]] = {}
        counts: dict[tuple[Any, ...], int] = {}
        plan_info: dict[tuple[Any, ...], dict[str, Any]] = {}
        runtime_metadata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        previous_runtime_state = None
        position_hasher = hashlib.sha256()
        for position in range(start_position, start_position + decode_steps):
            position_hasher.update(int(position).to_bytes(8, "little", signed=True))
            try:
                self.plan(position + 1)
                signature = self.structural_plan_signature()
            except BaseException as error:
                raise RuntimeError(
                    "decoder-layer graph-bank structural preflight failed at "
                    f"position {position} (context {position + 1})"
                ) from error
            representatives.setdefault(signature, position)
            current_plan_info = self.structural_plan_info()
            plan_info.setdefault(signature, current_plan_info)
            runtime_state = freeze_structural_value(self.plan_signature())
            if runtime_state != previous_runtime_state:
                serving = self.serving_metadata_snapshot()
                runtime_metadata.setdefault(signature, []).append(
                    {
                        "position": int(position),
                        "context": int(position + 1),
                        "current_pages": int(self.current_pages),
                        "old_pages": int(self.old_pages),
                        "exact_pages": int(self.exact_pages),
                        "wrappers": {
                            name: {
                                "pages_per_request": record[
                                    "pages_per_request"
                                ],
                                "kv_chunk_size_tokens": record[
                                    "kv_chunk_size_tokens"
                                ],
                                "effective_valid_split_tiles": record[
                                    "effective_valid_split_tiles"
                                ],
                                "initialized_tile_count": record[
                                    "initialized_tile_count"
                                ],
                                "semantic_int_workspace_sha256": record[
                                    "semantic_int_workspace_sha256"
                                ],
                            }
                            for name, record in serving["wrappers"].items()
                        },
                    }
                )
                previous_runtime_state = runtime_state
            position_hasher.update(
                hashlib.sha256(
                    repr((position, signature, current_plan_info)).encode("utf-8")
                ).digest()
            )
            counts[signature] = counts.get(signature, 0) + 1
            if signature not in bounds:
                bounds[signature] = [position, position]
            else:
                bounds[signature][1] = position
            if len(representatives) > maximum_graph_banks:
                raise RuntimeError(
                    "decoder-layer graph-bank preflight discovered more than "
                    f"{maximum_graph_banks} structural FlashInfer topologies"
                )

        # Leave the wrapper in the first served state. This replan is also a
        # capacity check, but remains outside the eventual timing boundary.
        self.plan(start_position + 1)
        self.decoder_layer_graph_preflight_positions = {
            signature: (limits[0], limits[1])
            for signature, limits in bounds.items()
        }
        self.decoder_layer_graph_preflight_counts = counts
        self.decoder_layer_graph_preflight_plan_info = plan_info
        self.decoder_layer_graph_preflight_runtime_metadata = runtime_metadata
        self.decoder_layer_graph_preflight_position_count = sum(counts.values())
        self.decoder_layer_graph_preflight_range = (
            start_position,
            start_position + decode_steps,
        )
        self.decoder_layer_graph_preflight_positions_sha256 = position_hasher.hexdigest()
        if self.decoder_layer_graph_preflight_position_count != decode_steps:
            raise RuntimeError("decoder-layer graph-bank preflight did not cover every position")
        return representatives

    @torch.inference_mode()
    def capture_dynamic_decoder_layer_graph_banks(
        self,
        start_position: int,
        decode_steps: int,
        *,
        capture_warmups: int = 1,
        maximum_graph_banks: int = 64,
        strict: bool = True,
    ) -> None:
        """Pre-capture one 32-layer graph bank per actual planner topology."""

        if capture_warmups <= 0:
            raise ValueError("decoder-layer graph capture warmups must be positive")
        representatives = self.preflight_dynamic_decoder_layer_graph_banks(
            start_position,
            decode_steps,
            maximum_graph_banks=maximum_graph_banks,
        )
        self.attention_graphs = None
        self.attention_graph_plan_signature = None
        self.decoder_layer_graphs = None
        self.decoder_layer_graph_buffers = None
        self.decoder_layer_graph_start_position = None
        self.decoder_layer_graph_plan_signature = None
        self.decoder_layer_graph_banks = {}
        self.decoder_layer_graph_bank_buffers = {}
        self.decoder_layer_graph_bank_capture_positions = {}

        torch.cuda.synchronize()
        free_before, total_bytes = torch.cuda.mem_get_info()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        capture_started = time.perf_counter()
        current_stream = torch.cuda.current_stream()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(current_stream)

        for signature, capture_position in representatives.items():
            self.plan(capture_position + 1)
            observed = self.structural_plan_signature()
            if observed != signature:
                raise RuntimeError(
                    "FlashInfer structural plan changed between preflight and capture"
                )
            buffers = torch.zeros(
                self.layers + 1,
                self.batch_size,
                self.hidden,
                device="cuda",
                dtype=torch.float16,
            )
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                assert self.device_position is not None
                self.device_position.fill_(capture_position)
                for _ in range(capture_warmups):
                    for layer_index in range(self.layers):
                        self._execute_decoder_layer(
                            layer_index,
                            buffers[layer_index],
                            capture_position,
                            buffers[layer_index + 1],
                        )
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize()

            pool = torch.cuda.graph_pool_handle()
            bank: list[torch.cuda.CUDAGraph] = []
            for layer_index in range(self.layers):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=pool,
                    stream=capture_stream,
                ):
                    self._execute_decoder_layer(
                        layer_index,
                        buffers[layer_index],
                        capture_position,
                        buffers[layer_index + 1],
                    )
                bank.append(graph)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize()
            if len(bank) != self.layers:
                raise RuntimeError("decoder-layer graph-bank capture is incomplete")
            self.decoder_layer_graph_banks[signature] = bank
            self.decoder_layer_graph_bank_buffers[signature] = buffers
            self.decoder_layer_graph_bank_capture_positions[signature] = capture_position

        if set(self.decoder_layer_graph_banks) != set(representatives):
            raise RuntimeError("not every preflight topology received a graph bank")
        self.plan(start_position + 1)
        assert self.device_position is not None
        self.device_position.fill_(start_position)
        torch.cuda.synchronize()
        capture_wall_seconds = time.perf_counter() - capture_started
        free_after, total_after = torch.cuda.mem_get_info()
        if total_after != total_bytes:
            raise RuntimeError("CUDA device capacity changed during graph-bank capture")
        self.decoder_layer_graph_bank_memory = {
            "persistent_hidden_buffer_bytes": int(
                sum(
                    buffer.numel() * buffer.element_size()
                    for buffer in self.decoder_layer_graph_bank_buffers.values()
                )
            ),
            "torch_allocated_delta_bytes": int(
                torch.cuda.memory_allocated() - allocated_before
            ),
            "torch_reserved_delta_bytes": int(
                torch.cuda.memory_reserved() - reserved_before
            ),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "torch_peak_allocated_increment_over_capture_start_bytes": int(
                torch.cuda.max_memory_allocated() - allocated_before
            ),
            "torch_peak_reserved_increment_over_capture_start_bytes": int(
                torch.cuda.max_memory_reserved() - reserved_before
            ),
            "cuda_free_delta_bytes": int(free_after - free_before),
            "cuda_consumed_delta_bytes": int(free_before - free_after),
            "one_time_graph_bank_capture_wall_seconds": float(
                capture_wall_seconds
            ),
        }
        self.decoder_layer_graph_bank_strict = bool(strict)
        self.decoder_layer_graph_bank_replay_enabled = True
        self.reset_attention_dispatch_counts()

    @torch.inference_mode()
    def step(
        self, token: torch.Tensor, position: int, profile_layers: bool = False
    ) -> tuple[torch.Tensor, list[tuple[torch.cuda.Event, torch.cuda.Event]]]:
        if token.numel() != self.batch_size:
            raise ValueError(
                f"expected {self.batch_size} synchronized tokens, got {token.numel()}"
            )
        if not 0 <= position < int(self.rope_cos.shape[0]):
            raise ValueError("decode position lies outside the allocated RoPE table")
        self.plan(position + 1)
        if self.device_dynamic_decoder_layer_graphs:
            assert self.device_position is not None
            # This device fill intentionally remains outside the captured
            # layer graphs and inside every timed decoder step. All 32 layer
            # replays consume the same stable scalar pointer on this stream.
            self.device_position.fill_(position)
            self.device_position_fills += 1
        hidden = self.model.model.embed_tokens(
            token.reshape(self.batch_size, 1)
        )[:, 0]
        layer_events = []
        if (
            self.device_dynamic_decoder_layer_graphs
            and self.decoder_layer_graph_bank_replay_enabled
            and self.decoder_layer_graph_bank_strict
            and not self.decoder_layer_graph_banks
        ):
            self.decoder_layer_graph_bank_misses += 1
            raise RuntimeError(
                "strict device-dynamic decoder-layer replay was enabled before "
                "any structural graph bank was captured"
            )
        if (
            self.device_dynamic_decoder_layer_graphs
            and self.decoder_layer_graph_bank_replay_enabled
            and self.decoder_layer_graph_banks
        ):
            structural_signature = self.structural_plan_signature()
            graph_bank = self.decoder_layer_graph_banks.get(structural_signature)
            graph_buffers = self.decoder_layer_graph_bank_buffers.get(
                structural_signature
            )
            if graph_bank is None or graph_buffers is None:
                self.decoder_layer_graph_bank_misses += 1
                if self.decoder_layer_graph_bank_strict:
                    raise RuntimeError(
                        "no pre-captured decoder-layer graph bank for the current "
                        "FlashInfer structural plan signature"
                    )
            else:
                if len(graph_bank) != self.layers:
                    raise RuntimeError("device-dynamic decoder-layer graph bank is incomplete")
                graph_buffers[0].copy_(hidden)
                for graph in graph_bank:
                    start = (
                        torch.cuda.Event(enable_timing=True) if profile_layers else None
                    )
                    end = (
                        torch.cuda.Event(enable_timing=True) if profile_layers else None
                    )
                    if start is not None:
                        start.record()
                    graph.replay()
                    self.decoder_layer_graph_replays += 1
                    if end is not None and start is not None:
                        end.record()
                        layer_events.append((start, end))
                hidden = graph_buffers[-1]
                normalized = self.model.model.norm(hidden)
                return self.model.lm_head(normalized).float(), layer_events
        graph_offset = self._decoder_layer_graph_offset(position)
        if graph_offset is not None:
            assert (
                self.decoder_layer_graphs is not None
                and self.decoder_layer_graph_buffers is not None
            )
            graph_buffers = self.decoder_layer_graph_buffers[graph_offset]
            if len(self.decoder_layer_graphs[graph_offset]) != self.layers:
                raise RuntimeError("decoder-layer graph bank is incomplete")
            graph_buffers[0].copy_(hidden)
            for layer_index, graph in enumerate(
                self.decoder_layer_graphs[graph_offset]
            ):
                start = (
                    torch.cuda.Event(enable_timing=True) if profile_layers else None
                )
                end = (
                    torch.cuda.Event(enable_timing=True) if profile_layers else None
                )
                if start is not None:
                    start.record()
                graph.replay()
                self.decoder_layer_graph_replays += 1
                if end is not None and start is not None:
                    end.record()
                    layer_events.append((start, end))
            hidden = graph_buffers[-1]
        else:
            for layer_index in range(self.layers):
                start = (
                    torch.cuda.Event(enable_timing=True) if profile_layers else None
                )
                end = (
                    torch.cuda.Event(enable_timing=True) if profile_layers else None
                )
                if start is not None:
                    start.record()
                hidden = self._execute_decoder_layer(
                    layer_index, hidden, position
                )
                self.decoder_layer_eager_calls += 1
                if end is not None and start is not None:
                    end.record()
                    layer_events.append((start, end))
        normalized = self.model.model.norm(hidden)
        return self.model.lm_head(normalized).float(), layer_events

def token_tensor_from_sequence(
    tokens: TokenSequence,
    batch_size: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Return step-major synchronized tokens with shape [steps,B]."""
    if batch_size <= 0 or not tokens:
        raise ValueError("batch size and token sequence must be non-zero")
    token_tensor = torch.tensor(tokens, device=device, dtype=torch.long)
    if token_tensor.dim() == 1:
        if batch_size != 1:
            raise ValueError("batched tokens must use step-major shape [steps,B]")
        token_tensor = token_tensor[:, None]
    if token_tensor.dim() != 2 or int(token_tensor.shape[1]) != batch_size:
        raise ValueError("tokens must have step-major shape [steps,B]")
    return token_tensor


def token_tensor_for_step(
    token: int | list[int],
    batch_size: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    values = [token] if isinstance(token, int) else token
    tensor = torch.tensor(values, device=device, dtype=torch.long)
    if tensor.dim() != 1 or tensor.numel() != batch_size:
        raise ValueError("one token per synchronized request is required")
    return tensor


def timed_sequence(
    decoder: TransformerDecoder, tokens: TokenSequence, start_position: int
) -> tuple[list[torch.Tensor], float, float]:
    token_tensor = token_tensor_from_sequence(tokens, decoder.batch_size)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    logits = [
        decoder.step(token_tensor[index], start_position + index)[0]
        for index in range(int(token_tensor.shape[0]))
    ]
    end.record()
    torch.cuda.synchronize()
    return logits, float(begin.elapsed_time(end)), (time.perf_counter() - wall_start) * 1e3


def summarize_sequence(
    gpu_ms: list[float],
    wall_ms: list[float],
    steps: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    output_tokens = steps * batch_size
    return {
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "batch_size": batch_size,
        "decode_steps": steps,
        "output_tokens_per_sequence": output_tokens,
        "gpu_p50_ms_per_decode_step": statistics.median(gpu_ms) / steps,
        "wall_p50_ms_per_decode_step": statistics.median(wall_ms) / steps,
        "wall_mean_ms_per_decode_step": statistics.mean(wall_ms) / steps,
        "wall_p95_ms_per_decode_step": percentile(wall_ms, 0.95) / steps,
        "gpu_p50_ms_per_token": statistics.median(gpu_ms) / output_tokens,
        "wall_p50_ms_per_token": statistics.median(wall_ms) / output_tokens,
        "wall_mean_ms_per_token": statistics.mean(wall_ms) / output_tokens,
        "wall_tokens_per_second": output_tokens
        / (statistics.mean(wall_ms) / 1e3),
        "wall_decode_steps_per_second": steps
        / (statistics.mean(wall_ms) / 1e3),
        "wall_p95_ms_per_token": percentile(wall_ms, 0.95) / output_tokens,
    }


def compare_logit_sequences(
    expected: list[torch.Tensor], observed: list[torch.Tensor]
) -> dict[str, Any]:
    if len(expected) != len(observed) or not expected:
        raise ValueError("logit sequences must have the same non-zero length")
    errors = [
        float((reference - result).abs().max())
        for reference, result in zip(expected, observed)
    ]
    exact = [torch.equal(reference, result) for reference, result in zip(expected, observed)]
    diagnostic_offsets = sorted(
        offset
        for offset in {0, len(errors) - 3, len(errors) - 2, len(errors) - 1}
        if 0 <= offset < len(errors)
    )
    return {
        "checked_steps": len(errors),
        "checked_batch_rows": sum(
            int(reference.shape[0]) if reference.dim() > 1 else 1
            for reference in expected
        ),
        "bitwise_identical": all(exact),
        "maximum_absolute_error": max(errors),
        "per_offset_maximum_absolute_error": {
            str(offset): errors[offset] for offset in diagnostic_offsets
        },
    }


def measure_modes(
    baseline: TransformerDecoder,
    candidate: TransformerDecoder,
    tokens: TokenSequence,
    start_position: int,
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    decoders = {"flashinfer_fp16": baseline, "page_gauge": candidate}
    result = {}
    for mode_index, mode in enumerate(("cache_neutral", "cache_hot")):
        for _ in range(warmups):
            for decoder in decoders.values():
                if mode == "cache_neutral":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    timed_sequence(decoder, tokens, start_position)
                timed_sequence(decoder, tokens, start_position)
        samples = {name: {"gpu": [], "wall": []} for name in decoders}
        names = list(decoders)
        for repeat in range(repeats):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                decoder = decoders[name]
                if mode == "cache_neutral":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    timed_sequence(decoder, tokens, start_position)
                _, gpu_ms, wall_ms = timed_sequence(decoder, tokens, start_position)
                samples[name]["gpu"].append(gpu_ms)
                samples[name]["wall"].append(wall_ms)
        summaries = {
            name: summarize_sequence(
                values["gpu"],
                values["wall"],
                len(tokens),
                baseline.batch_size,
            )
            for name, values in samples.items()
        }
        ratio_of_means = (
            summaries["page_gauge"]["wall_tokens_per_second"]
            / summaries["flashinfer_fp16"]["wall_tokens_per_second"]
        )
        paired_wall_speedups = [
            baseline_ms / candidate_ms
            for baseline_ms, candidate_ms in zip(
                samples["flashinfer_fp16"]["wall"],
                samples["page_gauge"]["wall"],
            )
        ]
        paired_gpu_speedups = [
            baseline_ms / candidate_ms
            for baseline_ms, candidate_ms in zip(
                samples["flashinfer_fp16"]["gpu"],
                samples["page_gauge"]["gpu"],
            )
        ]
        summaries["page_gauge_speedup_wall_tokens_per_second"] = geometric_mean(
            paired_wall_speedups
        )
        summaries["ratio_of_mean_wall_tokens_per_second"] = ratio_of_means
        summaries["paired_speedups"] = {
            "wall": paired_wall_speedups,
            "gpu": paired_gpu_speedups,
            "wall_geomean": geometric_mean(paired_wall_speedups),
            "wall_median": statistics.median(paired_wall_speedups),
            "wall_range": [min(paired_wall_speedups), max(paired_wall_speedups)],
            "wall_paired_bootstrap_95_ci": bootstrap_geomean(
                paired_wall_speedups, bootstrap_seed + mode_index
            ),
            "gpu_geomean": geometric_mean(paired_gpu_speedups),
            "gpu_median": statistics.median(paired_gpu_speedups),
            "gpu_range": [min(paired_gpu_speedups), max(paired_gpu_speedups)],
            "gpu_paired_bootstrap_95_ci": bootstrap_geomean(
                paired_gpu_speedups, bootstrap_seed + 100 + mode_index
            ),
        }
        result[mode] = summaries
    return result


def summarize_layer_samples(samples: list[list[float]]) -> dict[str, Any]:
    per_layer = [
        statistics.median(sample[layer] for sample in samples)
        for layer in range(len(samples[0]))
    ]
    return {
        "samples_ms": samples,
        "per_layer_p50_ms": per_layer,
        "median_layer_p50_ms": statistics.median(per_layer),
        "sum_layer_p50_ms": sum(per_layer),
    }


def profile_layer_pair(
    baseline: TransformerDecoder,
    candidate: TransformerDecoder,
    token: int | list[int],
    position: int,
    repeats: int,
) -> dict[str, dict[str, Any]]:
    if baseline.batch_size != candidate.batch_size:
        raise ValueError("profiled decoders must have the same batch size")
    token_tensor = token_tensor_for_step(token, baseline.batch_size)
    decoders = {"flashinfer_fp16": baseline, "page_gauge": candidate}
    for _ in range(min(10, max(2, repeats // 3))):
        for decoder in decoders.values():
            decoder.step(token_tensor, position)
        torch.cuda.synchronize()
    samples: dict[str, list[list[float]]] = {name: [] for name in decoders}
    names = list(decoders)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            _, events = decoders[name].step(
                token_tensor, position, profile_layers=True
            )
            torch.cuda.synchronize()
            samples[name].append(
                [float(start.elapsed_time(end)) for start, end in events]
            )
    return {
        name: summarize_layer_samples(backend_samples)
        for name, backend_samples in samples.items()
    }


def selected_cuda_graph_scope(args: argparse.Namespace) -> str:
    if args.disable_attention_cuda_graphs and args.decoder_layer_cuda_graphs:
        raise ValueError(
            "--disable-attention-cuda-graphs and --decoder-layer-cuda-graphs "
            "are mutually exclusive"
        )
    if args.disable_attention_cuda_graphs:
        return "none"
    return "decoder_layer" if args.decoder_layer_cuda_graphs else "attention"


def selected_cuda_graphs(decoder: TransformerDecoder, scope: str):
    if scope == "attention":
        return decoder.attention_graphs
    if scope == "decoder_layer":
        return decoder.decoder_layer_graphs
    if scope == "none":
        return None
    raise ValueError(f"unknown CUDA graph scope {scope}")


def assign_selected_cuda_graphs(
    decoder: TransformerDecoder, scope: str, graphs
) -> None:
    if scope == "attention":
        decoder.attention_graphs = graphs
        return
    if scope == "decoder_layer":
        decoder.decoder_layer_graphs = graphs
        return
    if scope != "none":
        raise ValueError(f"unknown CUDA graph scope {scope}")


def selected_cuda_graph_dispatch_counts(
    decoder: TransformerDecoder, scope: str
) -> dict[str, Any]:
    if scope == "attention":
        return decoder.attention_dispatch_counts()
    if scope == "decoder_layer":
        return decoder.decoder_layer_dispatch_counts()
    if scope == "none":
        return decoder.decoder_layer_dispatch_counts()
    raise ValueError(f"unknown CUDA graph scope {scope}")


def capture_selected_cuda_graphs(
    decoder: TransformerDecoder, scope: str, start_position: int
) -> None:
    if scope == "attention":
        decoder.capture_attention_graphs(start_position + 1)
    elif scope == "decoder_layer":
        decoder.capture_decoder_layer_graphs(start_position)
    elif scope != "none":
        raise ValueError(f"unknown CUDA graph scope {scope}")


def main() -> None:
    args = parse_args()
    try:
        cuda_graph_scope = selected_cuda_graph_scope(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if (
        args.context <= args.exact_tail
        or args.context % PAGE
        or args.exact_tail % PAGE
    ):
        raise SystemExit("context and exact tail must be page aligned; context must be larger")
    try:
        exact_prefix_pages = validate_exact_prefix_attention_path(
            args.exact_sink_pages, args.tail_attention
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if (
        args.context // PAGE
        <= args.exact_tail // PAGE + exact_prefix_pages
    ):
        raise SystemExit("context must contain at least one quantized old page")
    if (
        args.batch_size <= 0
        or args.decode_steps <= 0
        or args.repeats <= 0
        or args.layer_profile_repeats <= 0
    ):
        raise SystemExit("decode steps and repeat counts must be positive")
    if args.repeats % 2:
        raise SystemExit("repeats must be even for balanced A/B and B/A blocks")
    if args.warmups < 0:
        raise SystemExit("warmups cannot be negative")
    if cuda_graph_scope != "none" and args.decode_steps != PAGE:
        raise SystemExit(
            "CUDA-graph validation requires one complete 16-token page; "
            "pass --disable-attention-cuda-graphs for other decode lengths"
        )
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    print("Loading model and kernels...", flush=True)
    model = load_model(args)
    import flashinfer

    append_extension = RUNTIME.load_append_extension()
    layers, hq, hkv, hidden = BASE_E2E.check_model(model)
    # One additional page-transition token is allocated and checked outside
    # the measured window to prove that stale graph plans are never replayed.
    max_context = args.context + args.decode_steps + 1
    pages = math.ceil(max_context / PAGE)
    initial_pages = args.context // PAGE
    exact_pages = args.exact_tail // PAGE
    with torch.inference_mode():
        position_ids = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(probe, position_ids)
        if rope_cos.dim() == 3:
            rope_cos, rope_sin = rope_cos[0], rope_sin[0]
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()
    vocab = int(model.config.vocab_size)
    seed_results = []
    for seed in args.seeds:
        print(f"seed={seed}: constructing logically identical caches", flush=True)
        torch.cuda.reset_peak_memory_stats()
        baseline_cache, gauge_cache = build_caches(
            layers,
            pages,
            initial_pages,
            exact_pages,
            hkv,
            seed + 1000,
            args.batch_size,
            args.exact_sink_pages,
        )
        storage = cache_storage(baseline_cache, gauge_cache)
        baseline = TransformerDecoder(
            model,
            flashinfer,
            append_extension,
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
            args.batch_size,
        )
        candidate = TransformerDecoder(
            model,
            flashinfer,
            append_extension,
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
            args.batch_size,
            old_value_scale_placement=args.old_value_scale_placement,
            exact_sink_pages=args.exact_sink_pages,
        )
        generator = torch.Generator().manual_seed(seed)
        validation_token_matrix = torch.randint(
            0,
            vocab,
            (args.decode_steps + 1, args.batch_size),
            generator=generator,
        )
        if args.batch_size == 1:
            validation_tokens: TokenSequence = [
                int(value) for value in validation_token_matrix[:, 0]
            ]
        else:
            validation_tokens = [
                [int(value) for value in step]
                for step in validation_token_matrix
            ]
        tokens = validation_tokens[: args.decode_steps]

        # Establish backend-local eager references first.  This catches stale
        # graph metadata independently of the approximate-vs-FP16 comparison.
        eager_references, _, _ = timed_sequence(baseline, tokens, args.context)
        eager_approximations, _, _ = timed_sequence(candidate, tokens, args.context)
        graph_replay_correctness: dict[str, Any] = {
            "enabled": cuda_graph_scope != "none",
            "cuda_graph_scope": cuda_graph_scope,
            "scope": (
                "one page of steady-state graph replay followed by a guarded "
                "eager page-transition token"
            ),
        }
        if cuda_graph_scope != "none":
            # The measured window starts immediately after a page-aligned
            # prefix. Its padded FlashInfer plan is stable for all 16 tokens.
            capture_selected_cuda_graphs(baseline, cuda_graph_scope, args.context)
            capture_selected_cuda_graphs(candidate, cuda_graph_scope, args.context)
            baseline.reset_attention_dispatch_counts()
            candidate.reset_attention_dispatch_counts()
            references, _, _ = timed_sequence(baseline, tokens, args.context)
            approximations, _, _ = timed_sequence(candidate, tokens, args.context)
            first_dispatch = {
                "flashinfer_fp16": selected_cuda_graph_dispatch_counts(
                    baseline, cuda_graph_scope
                ),
                "page_gauge": selected_cuda_graph_dispatch_counts(
                    candidate, cuda_graph_scope
                ),
            }
            baseline.reset_attention_dispatch_counts()
            candidate.reset_attention_dispatch_counts()
            repeated_references, _, _ = timed_sequence(
                baseline, tokens, args.context
            )
            repeated_approximations, _, _ = timed_sequence(
                candidate, tokens, args.context
            )

            # The next token changes the page-table/plan signature. Compare an
            # explicitly eager transition with the guarded graph path, then
            # restore the single overwritten cache slot so timed sequences can
            # safely replay the preceding page.
            transition_position = args.context + args.decode_steps
            transition_token = token_tensor_for_step(
                validation_tokens[-1], args.batch_size
            )
            baseline_snapshot = snapshot_append_slot(baseline, transition_position)
            candidate_snapshot = snapshot_append_slot(candidate, transition_position)
            saved_baseline_graphs = selected_cuda_graphs(
                baseline, cuda_graph_scope
            )
            saved_candidate_graphs = selected_cuda_graphs(
                candidate, cuda_graph_scope
            )
            assign_selected_cuda_graphs(baseline, cuda_graph_scope, None)
            assign_selected_cuda_graphs(candidate, cuda_graph_scope, None)
            baseline.reset_attention_dispatch_counts()
            candidate.reset_attention_dispatch_counts()
            eager_transition_reference = baseline.step(
                transition_token, transition_position
            )[0]
            eager_transition_approximation = candidate.step(
                transition_token, transition_position
            )[0]
            torch.cuda.synchronize()
            eager_transition_dispatch = {
                "flashinfer_fp16": selected_cuda_graph_dispatch_counts(
                    baseline, cuda_graph_scope
                ),
                "page_gauge": selected_cuda_graph_dispatch_counts(
                    candidate, cuda_graph_scope
                ),
            }
            restore_append_slot(baseline, baseline_snapshot)
            restore_append_slot(candidate, candidate_snapshot)
            assign_selected_cuda_graphs(
                baseline, cuda_graph_scope, saved_baseline_graphs
            )
            assign_selected_cuda_graphs(
                candidate, cuda_graph_scope, saved_candidate_graphs
            )
            baseline.reset_attention_dispatch_counts()
            candidate.reset_attention_dispatch_counts()
            guarded_transition_reference = baseline.step(
                transition_token, transition_position
            )[0]
            guarded_transition_approximation = candidate.step(
                transition_token, transition_position
            )[0]
            torch.cuda.synchronize()
            guarded_transition_dispatch = {
                "flashinfer_fp16": selected_cuda_graph_dispatch_counts(
                    baseline, cuda_graph_scope
                ),
                "page_gauge": selected_cuda_graph_dispatch_counts(
                    candidate, cuda_graph_scope
                ),
            }
            restore_append_slot(baseline, baseline_snapshot)
            restore_append_slot(candidate, candidate_snapshot)
            expected_graph_calls = layers * args.decode_steps
            expected_transition_calls = layers
            backend_checks = {
                "flashinfer_fp16_eager_vs_graph": compare_logit_sequences(
                    eager_references, references
                ),
                "page_gauge_eager_vs_graph": compare_logit_sequences(
                    eager_approximations, approximations
                ),
                "flashinfer_fp16_repeat_replay": compare_logit_sequences(
                    references, repeated_references
                ),
                "page_gauge_repeat_replay": compare_logit_sequences(
                    approximations, repeated_approximations
                ),
                "flashinfer_fp16_guarded_transition": compare_logit_sequences(
                    [eager_transition_reference], [guarded_transition_reference]
                ),
                "page_gauge_guarded_transition": compare_logit_sequences(
                    [eager_transition_approximation],
                    [guarded_transition_approximation],
                ),
            }
            steady_state_dispatch_passed = all(
                counts["graph_replays"] == expected_graph_calls
                and counts["eager_calls"] == 0
                for counts in first_dispatch.values()
            )
            transition_dispatch_passed = all(
                eager_transition_dispatch[name]["graph_replays"] == 0
                and eager_transition_dispatch[name]["eager_calls"]
                == expected_transition_calls
                and guarded_transition_dispatch[name]["graph_replays"] == 0
                and guarded_transition_dispatch[name]["eager_calls"]
                == expected_transition_calls
                for name in first_dispatch
            )
            dispatch_passed = (
                steady_state_dispatch_passed and transition_dispatch_passed
            )
            graph_replay_correctness.update(
                {
                    "backend_checks": backend_checks,
                    "steady_state_dispatch": first_dispatch,
                    "explicit_eager_transition_dispatch": eager_transition_dispatch,
                    "guarded_transition_dispatch": guarded_transition_dispatch,
                    "expected_graph_replays_per_backend": expected_graph_calls,
                    "expected_guarded_eager_calls_per_backend": expected_transition_calls,
                    "dispatch_guard_passed": dispatch_passed,
                    "passed": dispatch_passed
                    and all(
                        check["bitwise_identical"]
                        for check in backend_checks.values()
                    ),
                }
            )
        else:
            references = eager_references
            approximations = eager_approximations
            graph_replay_correctness["passed"] = True
        cosine_by_step_request = []
        top1_by_step_request = []
        maximum_absolute_by_step_request = []
        for reference, approximation in zip(references, approximations):
            cosine_by_step_request.append(
                [
                    float(value)
                    for value in F.cosine_similarity(
                        reference, approximation, dim=-1
                    ).tolist()
                ]
            )
            top1_by_step_request.append(
                [
                    bool(value)
                    for value in reference.argmax(dim=-1)
                    .eq(approximation.argmax(dim=-1))
                    .tolist()
                ]
            )
            maximum_absolute_by_step_request.append(
                [
                    float(value)
                    for value in (reference - approximation)
                    .abs()
                    .amax(dim=-1)
                    .tolist()
                ]
            )
        cosines = [value for row in cosine_by_step_request for value in row]
        top1 = [value for row in top1_by_step_request for value in row]
        maximum_absolute = max(
            value for row in maximum_absolute_by_step_request for value in row
        )
        correctness_passed = (
            min(cosines) >= args.min_logits_cosine
            and sum(top1) / len(top1) >= args.min_top1_agreement
            and graph_replay_correctness["passed"]
        )
        scrub = torch.zeros(
            args.cache_scrub_mib * 1024 * 1024 // 4,
            device="cuda",
            dtype=torch.int32,
        )
        saved_baseline_graphs = selected_cuda_graphs(baseline, cuda_graph_scope)
        saved_candidate_graphs = selected_cuda_graphs(candidate, cuda_graph_scope)
        if cuda_graph_scope == "none":
            primary_integration = "matched_eager"
            variant_configurations = [("matched_eager", None, None)]
        else:
            primary_integration = (
                "matched_graph"
                if cuda_graph_scope == "attention"
                else "matched_decoder_layer_graph"
            )
            variant_configurations = [
                (
                    primary_integration,
                    saved_baseline_graphs,
                    saved_candidate_graphs,
                )
            ]
            if not args.skip_integration_ablations:
                variant_configurations.extend(
                    [
                        (
                            (
                                "candidate_graph_baseline_eager"
                                if cuda_graph_scope == "attention"
                                else "candidate_decoder_layer_graph_baseline_eager"
                            ),
                            None,
                            saved_candidate_graphs,
                        ),
                        ("matched_eager", None, None),
                    ]
                )
        integration_variants = {}
        for variant_index, (
            variant_name,
            baseline_graphs,
            candidate_graphs,
        ) in enumerate(variant_configurations):
            assign_selected_cuda_graphs(
                baseline, cuda_graph_scope, baseline_graphs
            )
            assign_selected_cuda_graphs(
                candidate, cuda_graph_scope, candidate_graphs
            )
            baseline.reset_attention_dispatch_counts()
            candidate.reset_attention_dispatch_counts()
            variant_modes = measure_modes(
                baseline,
                candidate,
                tokens,
                args.context,
                scrub,
                args.warmups,
                args.repeats,
                seed + 2000 + variant_index * 200,
            )
            integration_variants[variant_name] = {
                "baseline_attention_mode": (
                    f"{cuda_graph_scope}_cuda_graph"
                    if baseline_graphs is not None
                    else "eager"
                ),
                "page_gauge_attention_mode": (
                    f"{cuda_graph_scope}_cuda_graph"
                    if candidate_graphs is not None
                    else "eager"
                ),
                "timing_modes": variant_modes,
                "attention_dispatch": {
                    "flashinfer_fp16": selected_cuda_graph_dispatch_counts(
                        baseline, cuda_graph_scope
                    ),
                    "page_gauge": selected_cuda_graph_dispatch_counts(
                        candidate, cuda_graph_scope
                    ),
                },
            }
        modes = integration_variants[primary_integration]["timing_modes"]
        # Layer profiles use the same symmetric graph boundary as the primary
        # timing comparison.
        assign_selected_cuda_graphs(
            baseline, cuda_graph_scope, saved_baseline_graphs
        )
        assign_selected_cuda_graphs(
            candidate, cuda_graph_scope, saved_candidate_graphs
        )
        profile_positions = {"first_decode_token": args.context}
        page_close_position = args.context + (
            PAGE - 1 - (args.context % PAGE)
        ) % PAGE
        if page_close_position < args.context + args.decode_steps:
            profile_positions["page_close"] = page_close_position
        layer_profiles = {}
        for phase, layer_position in profile_positions.items():
            layer_token = tokens[layer_position - args.context]
            paired_profile = profile_layer_pair(
                baseline,
                candidate,
                layer_token,
                layer_position,
                args.layer_profile_repeats,
            )
            baseline_profile = paired_profile["flashinfer_fp16"]
            candidate_profile = paired_profile["page_gauge"]
            layer_profiles[phase] = {
                "position": layer_position,
                "batch_size": args.batch_size,
                "latency_scope": "one synchronized batched decode step",
                "finalizes_page": layer_position % PAGE == PAGE - 1,
                "flashinfer_fp16": baseline_profile,
                "page_gauge": candidate_profile,
                "page_gauge_speedup_sum_layer_latency": baseline_profile[
                    "sum_layer_p50_ms"
                ]
                / candidate_profile["sum_layer_p50_ms"],
            }
        attention_implementations = {
            "flashinfer_fp16": baseline.attention_implementation_provenance(),
            "page_gauge": candidate.attention_implementation_provenance(),
        }
        cuda_graph_provenance = {
            "flashinfer_fp16": (
                baseline.decoder_layer_graph_provenance()
                if cuda_graph_scope == "decoder_layer"
                else {"enabled": cuda_graph_scope == "attention", "scope": cuda_graph_scope}
            ),
            "page_gauge": (
                candidate.decoder_layer_graph_provenance()
                if cuda_graph_scope == "decoder_layer"
                else {"enabled": cuda_graph_scope == "attention", "scope": cuda_graph_scope}
            ),
        }
        seed_results.append(
            {
                "seed": seed,
                "timing_modes": modes,
                "primary_integration": primary_integration,
                "integration_variants": integration_variants,
                "layer_latency": layer_profiles,
                "cache_storage": storage,
                "attention_implementations": attention_implementations,
                "cuda_graph_provenance": cuda_graph_provenance,
                "correctness": {
                    "batch_size": args.batch_size,
                    "checked_decode_steps": args.decode_steps,
                    "checked_request_steps": args.decode_steps * args.batch_size,
                    "measured_decode_steps": args.decode_steps,
                    "measured_output_tokens": args.decode_steps * args.batch_size,
                    "guarded_page_transition_checked": (
                        cuda_graph_scope != "none"
                    ),
                    "minimum_logits_cosine": min(cosines),
                    "logits_cosine_by_step_request": cosine_by_step_request,
                    "top1_agreement_fraction": sum(top1) / len(top1),
                    "top1_agreement_by_step_request": top1_by_step_request,
                    "maximum_logits_absolute_error": maximum_absolute,
                    "maximum_logits_absolute_error_by_step_request": (
                        maximum_absolute_by_step_request
                    ),
                "graph_replay": graph_replay_correctness,
                    "cuda_graph_provenance": cuda_graph_provenance,
                    "passed": correctness_passed,
                },
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        print(
            f"  steady-state full-model speedup="
            f"{modes['cache_hot']['page_gauge_speedup_wall_tokens_per_second']:.4f}x "
            f"cosine={min(cosines):.7f} top1={sum(top1)}/{len(top1)}",
            flush=True,
        )
        assign_selected_cuda_graphs(baseline, cuda_graph_scope, None)
        assign_selected_cuda_graphs(candidate, cuda_graph_scope, None)
        saved_baseline_graphs = None
        saved_candidate_graphs = None
        variant_configurations.clear()
        del baseline, candidate, baseline_cache, gauge_cache, scrub
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def aggregate_timing_payloads(
        timing_payloads: list[dict[str, Any]], bootstrap_seed: int
    ) -> dict[str, Any]:
        aggregate = {}
        for mode_index, mode in enumerate(("cache_neutral", "cache_hot")):
            ratios = [
                payload[mode]["page_gauge_speedup_wall_tokens_per_second"]
                for payload in timing_payloads
            ]
            paired_groups = [
                payload[mode]["paired_speedups"]["wall"]
                for payload in timing_payloads
            ]
            aggregate[mode] = {
                "page_gauge_speedup_wall_tokens_per_second_geomean": geometric_mean(
                    ratios
                ),
                "seed_bootstrap_95_ci": bootstrap_geomean(
                    ratios, bootstrap_seed + mode_index
                ),
                "hierarchical_seed_and_paired_block_bootstrap_95_ci": (
                    hierarchical_bootstrap_geomean(
                        paired_groups, bootstrap_seed + 100 + mode_index
                    )
                ),
                "flashinfer_wall_tokens_per_second_median": statistics.median(
                    payload[mode]["flashinfer_fp16"]["wall_tokens_per_second"]
                    for payload in timing_payloads
                ),
                "page_gauge_wall_tokens_per_second_median": statistics.median(
                    payload[mode]["page_gauge"]["wall_tokens_per_second"]
                    for payload in timing_payloads
                ),
            }
        return aggregate

    mode_aggregates = aggregate_timing_payloads(
        [seed["timing_modes"] for seed in seed_results], args.seeds[0] + 700
    )
    common_variants = set.intersection(
        *(set(seed["integration_variants"]) for seed in seed_results)
    )
    integration_variant_aggregates = {
        variant: aggregate_timing_payloads(
            [
                seed["integration_variants"][variant]["timing_modes"]
                for seed in seed_results
            ],
            args.seeds[0] + 1200 + variant_index * 200,
        )
        for variant_index, variant in enumerate(sorted(common_variants))
    }
    common_layer_phases = set.intersection(
        *(set(seed["layer_latency"]) for seed in seed_results)
    )
    layer_aggregates = {}
    for phase in sorted(common_layer_phases):
        ratios = [
            seed["layer_latency"][phase][
                "page_gauge_speedup_sum_layer_latency"
            ]
            for seed in seed_results
        ]
        layer_aggregates[phase] = {
            "finalizes_page": seed_results[0]["layer_latency"][phase][
                "finalizes_page"
            ],
            "page_gauge_speedup_sum_layer_latency_geomean": geometric_mean(
                ratios
            ),
            "seed_bootstrap_95_ci": bootstrap_geomean(
                ratios, args.seeds[0] + 900 + len(layer_aggregates)
            ),
            "flashinfer_sum_layer_p50_ms_median": statistics.median(
                seed["layer_latency"][phase]["flashinfer_fp16"][
                    "sum_layer_p50_ms"
                ]
                for seed in seed_results
            ),
            "page_gauge_sum_layer_p50_ms_median": statistics.median(
                seed["layer_latency"][phase]["page_gauge"][
                    "sum_layer_p50_ms"
                ]
                for seed in seed_results
            ),
        }
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
            ROOT / "scripts/page_gauge_heterogeneous_fa2.py",
            ROOT / "scripts/prepare_flashinfer_page_gauge_heterogeneous.py",
            ROOT / "scripts/benchmark_page_gauge_overheads.py",
            ROOT / "scripts/benchmark_e2e_transformer.py",
            ROOT / "scripts/page_gauge_runtime.py",
            ROOT / "tests/page_gauge_append_extension.cu",
            ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch",
            ROOT / "patches/flashinfer-0.6.17-page-gauge-heterogeneous.patch",
        )
    }
    result = {
        "schema_version": 5,
        "experiment": "page_gauge_full_transformer_decode",
        "claim_scope": (
            "functional smoke only; randomly initialized one-layer model"
            if args.model.startswith("synthetic-")
            else (
                f"full pretrained decoder synchronized batch-{args.batch_size} "
                "one-page steady-state decode "
                "with a deterministically synthesized long-prefix KV state; "
                "excludes prefill"
            )
        ),
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": DIM,
        "context": args.context,
        "batch_size": args.batch_size,
        "synchronized_batch": True,
        "decode_steps": args.decode_steps,
        "output_tokens_per_measured_sequence": (
            args.batch_size * args.decode_steps
        ),
        "exact_tail_tokens": args.exact_tail,
        "exact_sink_pages": args.exact_sink_pages,
        "exact_prefix_pages": args.exact_sink_pages,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "fused_rope_append_finalize": True,
        "value_center_restore": args.center_restore,
        "exact_tail_attention": args.tail_attention,
        "old_int8_value_scale_placement": args.old_value_scale_placement,
        "attention_implementations": {
            **seed_results[0]["attention_implementations"],
        },
        "cuda_graph_provenance": seed_results[0]["cuda_graph_provenance"],
        "attention_cuda_graphs": cuda_graph_scope == "attention",
        "decoder_layer_cuda_graphs": cuda_graph_scope == "decoder_layer",
        "cuda_graph_scope": cuda_graph_scope,
        "attention_cuda_graph_scope": (
            "the complete attention path of both backends in the primary matched "
            "comparison; eager and candidate-only graph variants are ablations; "
            "replay is guarded by the FlashInfer plan signature"
            if cuda_graph_scope == "attention"
            else None
        ),
        "decoder_layer_cuda_graph_scope": (
            "16 exact-position banks times every decoder layer; embedding, final "
            "norm, LM head, and argmax remain outside; each layer graph includes "
            "both attention and MLP; the exact logical page and FlashInfer plan "
            "signature guard replay, and the next-page token falls back eager"
            if cuda_graph_scope == "decoder_layer"
            else None
        ),
        "primary_integration": primary_integration,
        "cache_storage": seed_results[0]["cache_storage"],
        "physical_page_layout": "request-major",
        "value_centers_are_per_request": True,
        "timing_protocol": {
            "primary_mode": "cache_hot",
            "primary_statistic": (
                "geometric mean of within-block paired baseline/candidate wall "
                "latency ratios"
            ),
            "pair_order": "AB, BA repeated; repeats must be even",
            "cache_scrub_bytes": args.cache_scrub_mib * 1024 * 1024,
            "cache_neutral_definition": (
                "one L2-sized scrub before each full decode sequence; this "
                "conditions sequence start and is not per-layer cache neutral"
            ),
            "cache_hot_definition": (
                "one untimed same-backend full sequence immediately before each "
                "timed full sequence"
            ),
            "wall_clock_includes_every_decoder_layer_lm_head_and_cache_update": True,
            "throughput_unit": (
                "aggregate output tokens across all synchronized requests"
            ),
            "latency_unit": (
                "per synchronized decode step, with per-output-token fields "
                "reported separately"
            ),
        },
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seeds": list(args.seeds),
        "seed_results": seed_results,
        "aggregate": {
            "primary_mode": "cache_hot",
            "primary_integration": primary_integration,
            "timing_modes": mode_aggregates,
            "integration_variants": integration_variant_aggregates,
            "layer_latency": layer_aggregates,
            "correctness_passed": all(
                seed["correctness"]["passed"] for seed in seed_results
            ),
            "minimum_logits_cosine": min(
                seed["correctness"]["minimum_logits_cosine"]
                for seed in seed_results
            ),
            "top1_agreement_fraction": statistics.mean(
                seed["correctness"]["top1_agreement_fraction"]
                for seed in seed_results
            ),
            "thresholds": {
                "minimum_logits_cosine": args.min_logits_cosine,
                "minimum_top1_agreement": args.min_top1_agreement,
            },
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
        },
        "source_sha256": source_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")
    if not result["aggregate"]["correctness_passed"]:
        raise SystemExit("full-model correctness thresholds failed")


if __name__ == "__main__":
    main()
