#!/usr/bin/env python3
"""Ampere-only launch-shape variants for the frozen PageGauge FA2 kernel.

The RTX 5090 path is intentionally untouched.  On an A100, FlashInfer's
shared-memory heuristic selects ``NUM_MMA_KV=4`` for the INT8 paged kernel.
That specialization uses 255 registers and an 88-byte stack frame per thread
in the audited SM80 cubin.  The otherwise identical ``NUM_MMA_KV=2``
specialization uses 206 registers and no stack frame.  This module derives a
content-addressed include tree from the frozen PageGauge header and caps only
the paged-kernel dispatch.  Quantization, contractions, scale placement,
softmax, center restoration, and cache layout are unchanged.

This is an opt-in A100 diagnostic until an actual SM80 timing/correctness run
selects a variant.  It must not silently replace the publication kernel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_legacy_module():
    existing = sys.modules.get("page_gauge_attention_kernel")
    if existing is not None:
        return existing
    path = ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py"
    spec = importlib.util.spec_from_file_location("page_gauge_attention_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_module()
BASE_INCLUDE = ROOT / "build/gauge_affine_flashinfer/include"
BASE_HEADER = BASE_INCLUDE / "flashinfer/attention/prefill.cuh"
EXPECTED_BASE_HEADER_SHA256 = (
    "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"
)
SUPPORTED_CAPS = (1, 2)
DISPATCH_EXPRESSION = (
    "min(static_cast<uint32_t>(max_num_mma_kv_smem), max_num_mma_kv_reg)"
)
TRANSFORM_VERSION = "page_gauge_sm80_paged_mma_kv_cap_v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_cap(cap: int) -> int:
    if not isinstance(cap, int) or cap not in SUPPORTED_CAPS:
        raise ValueError(f"SM80 NUM_MMA_KV cap must be one of {SUPPORTED_CAPS}")
    return cap


def _derived_include(cap: int) -> Path:
    return ROOT / f"build/gauge_affine_flashinfer_sm80_mmakv{cap}/include"


def _transform_header(payload: bytes, cap: int) -> bytes:
    cap = _require_cap(cap)
    text = payload.decode("utf-8")
    positions: list[int] = []
    cursor = 0
    while True:
        found = text.find(DISPATCH_EXPRESSION, cursor)
        if found < 0:
            break
        positions.append(found)
        cursor = found + len(DISPATCH_EXPRESSION)
    if len(positions) != 3:
        raise RuntimeError(
            "frozen FlashInfer header dispatch structure changed: expected "
            f"three NUM_MMA_KV dispatch expressions, found {len(positions)}"
        )
    # The final occurrence is BatchPrefillWithPagedKVCacheDispatched.  The
    # earlier two are non-paged paths and remain byte-for-byte unchanged.
    position = positions[-1]
    replacement = (
        f"min({DISPATCH_EXPRESSION}, static_cast<uint32_t>({cap}))"
    )
    text = (
        text[:position]
        + replacement
        + text[position + len(DISPATCH_EXPRESSION) :]
    )
    if text.count(replacement) != 1:
        raise RuntimeError("SM80 paged dispatch cap was not inserted exactly once")
    return text.encode("utf-8")


def prepare_include(cap: int) -> Path:
    cap = _require_cap(cap)
    if not BASE_HEADER.is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/prepare_flashinfer_page_gauge.py")],
            check=True,
        )
    base_payload = BASE_HEADER.read_bytes()
    observed_base = _sha256_bytes(base_payload)
    if observed_base != EXPECTED_BASE_HEADER_SHA256:
        raise RuntimeError(
            "frozen PageGauge header SHA256 mismatch: expected "
            f"{EXPECTED_BASE_HEADER_SHA256}, got {observed_base}"
        )
    derived_payload = _transform_header(base_payload, cap)
    include = _derived_include(cap)
    header = include / "flashinfer/attention/prefill.cuh"
    if header.is_file():
        if header.read_bytes() != derived_payload:
            raise RuntimeError(
                f"stale or modified SM80 derived header at {header}; use a clean build tree"
            )
        return include
    if include.exists():
        raise RuntimeError(f"incomplete SM80 derived include tree at {include}")
    include.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE_INCLUDE, include)
    header.write_bytes(derived_payload)
    return include


def source_hashes(cap: int) -> dict[str, Any]:
    cap = _require_cap(cap)
    include = prepare_include(cap)
    derived_payload = (include / "flashinfer/attention/prefill.cuh").read_bytes()
    variant_payload = LEGACY.AFFINE_VARIANT.encode("utf-8")
    combined = derived_payload + b"\0" + variant_payload
    return {
        "header_sha256": _sha256_bytes(derived_payload),
        "expected_header_sha256": _sha256_bytes(
            _transform_header(BASE_HEADER.read_bytes(), cap)
        ),
        "header_matches_expected": True,
        "base_header_sha256": EXPECTED_BASE_HEADER_SHA256,
        "variant_sha256": _sha256_bytes(variant_payload),
        "module_source_sha256": _sha256_bytes(combined),
        "transform_version": TRANSFORM_VERSION,
        "sm80_paged_num_mma_kv_cap": cap,
        "math_changed": False,
    }


def module_uri(cap: int) -> str:
    hashes = source_hashes(cap)
    return (
        f"page_gauge_int8_fa2_sm80_mmakv{cap}_v1_"
        f"{hashes['module_source_sha256'][:16]}"
    )


def install_vendor_generator_hook(cap: int) -> None:
    cap = _require_cap(cap)
    include = prepare_include(cap).resolve()
    import flashinfer.decode as decode
    from flashinfer.jit.attention import modules

    modules.dtype_map_kv[torch.int8] = "int8_t"
    marker = str(include)
    if getattr(decode.gen_customize_batch_prefill_module, "_page_gauge_vendor", None) == marker:
        return
    original = decode.gen_customize_batch_prefill_module
    all_variant_roots = {_derived_include(value).resolve() for value in SUPPORTED_CAPS}
    base = BASE_INCLUDE.resolve()

    def with_vendor_include(*args: Any, **kwargs: Any):
        spec = original(*args, **kwargs)
        existing = []
        for path in list(spec.extra_include_dirs or []):
            resolved = Path(path).resolve()
            if resolved == base or resolved in all_variant_roots:
                continue
            existing.append(path)
        spec.extra_include_dirs = [include, *existing]
        return spec

    with_vendor_include._page_gauge_vendor = marker  # type: ignore[attr-defined]
    decode.gen_customize_batch_prefill_module = with_vendor_include


def make_page_gauge_wrapper(
    flashinfer: Any,
    workspace: torch.Tensor,
    *,
    sm80_num_mma_kv_cap: int = 2,
    **wrapper_kwargs: Any,
):
    """Build the frozen PageGauge math with an Ampere paged-tile cap."""

    cap = _require_cap(sm80_num_mma_kv_cap)
    if workspace.device.type != "cuda" or workspace.dtype != torch.uint8:
        raise ValueError("workspace must be a CUDA uint8 tensor")
    if tuple(torch.cuda.get_device_capability(workspace.device)) != (8, 0):
        raise RuntimeError("the SM80 NUM_MMA_KV cap is authorized only on compute capability 8.0")
    install_vendor_generator_hook(cap)
    hashes = source_hashes(cap)
    jit_args = [
        module_uri(cap),
        torch.float16,
        torch.int8,
        torch.float16,
        torch.int32,
        LEGACY.DIM,
        LEGACY.DIM,
        ["k_page_scale", "v_page_scale"],
        ["half", "half"],
        ["sm_scale"],
        ["double"],
        "PageAffineInt8Attention",
        LEGACY.AFFINE_VARIANT,
    ]
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
        jit_args=jit_args,
        **wrapper_kwargs,
    )
    wrapper.page_gauge_module_uri = jit_args[0]
    wrapper.page_gauge_source_hashes = hashes
    return wrapper


__all__ = [
    "SUPPORTED_CAPS",
    "TRANSFORM_VERSION",
    "make_page_gauge_wrapper",
    "module_uri",
    "prepare_include",
    "source_hashes",
]
