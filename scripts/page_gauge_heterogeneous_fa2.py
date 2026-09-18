#!/usr/bin/env python3
"""FlashInfer FA2 wrapper for PageGauge's heterogeneous INT8/FP16 cache.

This module is intentionally separate from the production PageGauge wrapper.
It exposes one decode-only FA2 launch whose logical page table addresses old
INT8 affine pages before ``old_kv_len`` and exact centered FP16 ring pages
after it.  The legacy two-segment implementation remains the fallback.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR_INCLUDE = ROOT / "build/gauge_heterogeneous_flashinfer/include"
VENDOR_HEADER = VENDOR_INCLUDE / "flashinfer/attention/prefill.cuh"
LEGACY_VENDOR_INCLUDE = ROOT / "build/gauge_affine_flashinfer/include"
EXPECTED_VENDOR_HEADER_SHA256 = (
    "2a3f3018576d04697477e3030507379ccc2d126036255ed81a238b63e9572e56"
)
PAGE = 16
DIM = 128
HQ = 32
HKV = 8


VARIANT_DECL = r"""
namespace flashinfer {

template <>
struct vec_cast<half, int8_t> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const int8_t* src) {
    static_assert(vec_size % 4 == 0);
    const half2 bias = __float2half2_rn(1152.0f);
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      uint32_t packed = reinterpret_cast<const uint32_t*>(src)[i] ^ 0x80808080u;
      uint32_t lo = 0x64006400u | (packed & 0x000000ffu) |
                    ((packed & 0x0000ff00u) << 8);
      uint32_t hi = 0x64006400u | ((packed & 0x00ff0000u) >> 16) |
                    ((packed & 0xff000000u) >> 8);
      reinterpret_cast<half2*>(dst)[2 * i] =
          __hsub2(*reinterpret_cast<half2*>(&lo), bias);
      reinterpret_cast<half2*>(dst)[2 * i + 1] =
          __hsub2(*reinterpret_cast<half2*>(&hi), bias);
    }
  }
};

}  // namespace flashinfer

struct PageGaugeHeterogeneousAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ PageGaugeHeterogeneousAttention(const Params& params,
                                                       uint32_t batch_idx,
                                                       uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }
};
"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_prepared() -> None:
    if not VENDOR_HEADER.is_file():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_flashinfer_page_gauge_heterogeneous.py"),
            ],
            check=True,
        )
    observed_header_sha256 = _sha256_bytes(VENDOR_HEADER.read_bytes())
    if observed_header_sha256 != EXPECTED_VENDOR_HEADER_SHA256:
        raise RuntimeError(
            "heterogeneous PageGauge vendor header SHA256 mismatch: expected "
            f"{EXPECTED_VENDOR_HEADER_SHA256}, got {observed_header_sha256}"
        )


def source_hashes() -> dict[str, str]:
    ensure_prepared()
    header_hash = _sha256_bytes(VENDOR_HEADER.read_bytes())
    variant_hash = _sha256_bytes(VARIANT_DECL.encode("utf-8"))
    combined_hash = _sha256_bytes(
        VENDOR_HEADER.read_bytes() + b"\0" + VARIANT_DECL.encode("utf-8")
    )
    return {
        "header_sha256": header_hash,
        "expected_header_sha256": EXPECTED_VENDOR_HEADER_SHA256,
        "header_matches_expected": header_hash == EXPECTED_VENDOR_HEADER_SHA256,
        "variant_sha256": variant_hash,
        "module_source_sha256": combined_hash,
    }


def module_uri() -> str:
    return (
        "page_gauge_heterogeneous_int8_fp16_fa2_sm120_v1_"
        f"{source_hashes()['module_source_sha256'][:16]}"
    )


def v_fragment_module_uri() -> str:
    """URI for the isolated old-cache V-fragment scaling A/B gate."""
    return (
        "page_gauge_int8_v_fragment_scale_fa2_sm120_v1_"
        f"{source_hashes()['module_source_sha256'][:16]}"
    )


def install_vendor_generator_hook() -> None:
    """Put the heterogeneous include tree first for this custom JIT module."""
    ensure_prepared()
    import flashinfer.decode as decode
    from flashinfer.jit.attention import modules

    modules.dtype_map_kv[torch.int8] = "int8_t"
    marker = str(VENDOR_INCLUDE.resolve())
    if getattr(decode.gen_customize_batch_prefill_module, "_page_gauge_vendor", None) == marker:
        return
    original = decode.gen_customize_batch_prefill_module

    def with_vendor_include(*args: Any, **kwargs: Any):
        spec = original(*args, **kwargs)
        vendor = VENDOR_INCLUDE.resolve()
        legacy = LEGACY_VENDOR_INCLUDE.resolve()
        existing = [
            Path(path).resolve()
            for path in list(spec.extra_include_dirs or [])
            if Path(path).resolve() not in (vendor, legacy)
        ]
        spec.extra_include_dirs = [vendor, *existing]
        return spec

    with_vendor_include._page_gauge_vendor = marker  # type: ignore[attr-defined]
    decode.gen_customize_batch_prefill_module = with_vendor_include


def make_heterogeneous_page_gauge_wrapper(
    flashinfer: Any, workspace: torch.Tensor, **wrapper_kwargs: Any
):
    """Return the decode-only custom wrapper.

    ``BatchDecodeWithPagedKVCacheWrapper`` has exactly one query row per
    request, so FlashInfer selects CTA_TILE_Q=16.  The heterogeneous code is
    compile-time disabled in the otherwise generated CTA64/CTA128 fallbacks.
    """
    if workspace.device.type != "cuda" or workspace.dtype != torch.uint8:
        raise ValueError("workspace must be a CUDA uint8 tensor")
    install_vendor_generator_hook()
    hashes = source_hashes()
    jit_args = [
        module_uri(),
        torch.float16,
        torch.int8,
        torch.float16,
        torch.int32,
        DIM,
        DIM,
        [
            "k_page_scale",
            "v_page_scale",
            "exact_k_cache",
            "exact_v_cache",
            "old_kv_len",
            "value_center",
        ],
        ["half", "half", "half", "half", "int32_t", "half"],
        ["sm_scale"],
        ["double"],
        "PageGaugeHeterogeneousAttention",
        VARIANT_DECL,
    ]
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
        jit_args=jit_args,
        **wrapper_kwargs,
    )
    # Provenance is kept on the wrapper without changing FlashInfer's API.
    wrapper.page_gauge_module_uri = jit_args[0]
    wrapper.page_gauge_source_hashes = hashes
    return wrapper


def make_v_fragment_page_gauge_wrapper(
    flashinfer: Any, workspace: torch.Tensor, **wrapper_kwargs: Any
):
    """Return the isolated old-INT8 module with scale applied to V fragments.

    This A/B gate has the legacy two-tensor ABI and does not instantiate the
    heterogeneous path.  It exists to distinguish probability underflow from
    exact-tail scheduling effects using a fresh, hash-qualified JIT module.
    """
    if workspace.device.type != "cuda" or workspace.dtype != torch.uint8:
        raise ValueError("workspace must be a CUDA uint8 tensor")
    install_vendor_generator_hook()
    hashes = source_hashes()
    jit_args = [
        v_fragment_module_uri(),
        torch.float16,
        torch.int8,
        torch.float16,
        torch.int32,
        DIM,
        DIM,
        ["k_page_scale", "v_page_scale"],
        ["half", "half"],
        ["sm_scale"],
        ["double"],
        "PageGaugeHeterogeneousAttention",
        VARIANT_DECL,
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


def plan_decode(
    wrapper: Any,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    *,
    fixed_split_pages: int = 0,
) -> None:
    """Plan the supported Hq/Hkv=32/8, D128, page16 decode shape."""
    for name, tensor in (
        ("indptr", indptr),
        ("indices", indices),
        ("last_page_len", last_page_len),
    ):
        if tensor.device.type != "cuda" or tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA int32")
    if indptr.ndim != 1 or last_page_len.shape != (indptr.numel() - 1,):
        raise ValueError("indptr/last_page_len batch dimensions disagree")
    if fixed_split_pages < 0:
        raise ValueError("fixed_split_pages must be non-negative")
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        HQ,
        HKV,
        DIM,
        PAGE,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=torch.int8,
        o_data_type=torch.float16,
        sm_scale=1.0 / math.sqrt(DIM),
        fixed_split_size=fixed_split_pages or None,
    )


def validate_run_inputs(
    query: torch.Tensor,
    old_k_cache: torch.Tensor,
    old_v_cache: torch.Tensor,
    k_page_scale: torch.Tensor,
    v_page_scale: torch.Tensor,
    exact_k_cache: torch.Tensor,
    exact_v_cache: torch.Tensor,
    old_kv_len: torch.Tensor,
    value_center: torch.Tensor,
) -> None:
    """Reject any accidental expansion beyond the first supported ABI."""
    batch = int(query.shape[0]) if query.ndim == 3 else -1
    expected_query = (batch, HQ, DIM)
    if tuple(query.shape) != expected_query or query.dtype != torch.float16:
        raise ValueError(f"query must have decode-only shape [B,{HQ},{DIM}] and dtype float16")
    if query.device.type != "cuda" or not query.is_contiguous():
        raise ValueError("query must be contiguous CUDA storage")
    for name, cache, dtype in (
        ("old_k_cache", old_k_cache, torch.int8),
        ("old_v_cache", old_v_cache, torch.int8),
        ("exact_k_cache", exact_k_cache, torch.float16),
        ("exact_v_cache", exact_v_cache, torch.float16),
    ):
        if (
            cache.device.type != "cuda"
            or cache.dtype != dtype
            or cache.ndim != 4
            or tuple(cache.shape[1:]) != (PAGE, HKV, DIM)
            or not cache.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA [{name.split('_')[0]}_pages,16,8,128] {dtype}")
    if old_k_cache.shape != old_v_cache.shape or exact_k_cache.shape != exact_v_cache.shape:
        raise ValueError("K/V cache shapes must match within each representation")
    for name, scale in (("k_page_scale", k_page_scale), ("v_page_scale", v_page_scale)):
        if (
            scale.device.type != "cuda"
            or scale.dtype != torch.float16
            or tuple(scale.shape) != (old_k_cache.shape[0], HKV)
            or not scale.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA [old_pages,{HKV}] float16")
    if (
        old_kv_len.device.type != "cuda"
        or old_kv_len.dtype != torch.int32
        or tuple(old_kv_len.shape) != (batch,)
        or not old_kv_len.is_contiguous()
    ):
        raise ValueError("old_kv_len must be contiguous CUDA [B] int32")
    if bool(torch.any(old_kv_len % PAGE).item()):
        raise ValueError(f"old_kv_len must be page-aligned ({PAGE} tokens)")
    if (
        value_center.device.type != "cuda"
        or value_center.dtype != torch.float16
        or tuple(value_center.shape) != (batch, HKV, DIM)
        or not value_center.is_contiguous()
    ):
        raise ValueError(f"value_center must be contiguous CUDA [B,{HKV},{DIM}] float16")


def run_decode(
    wrapper: Any,
    query: torch.Tensor,
    old_k_cache: torch.Tensor,
    old_v_cache: torch.Tensor,
    k_page_scale: torch.Tensor,
    v_page_scale: torch.Tensor,
    exact_k_cache: torch.Tensor,
    exact_v_cache: torch.Tensor,
    old_kv_len: torch.Tensor,
    value_center: torch.Tensor,
    **run_kwargs: Any,
):
    validate_run_inputs(
        query,
        old_k_cache,
        old_v_cache,
        k_page_scale,
        v_page_scale,
        exact_k_cache,
        exact_v_cache,
        old_kv_len,
        value_center,
    )
    return wrapper.run(
        query,
        (old_k_cache, old_v_cache),
        k_page_scale,
        v_page_scale,
        exact_k_cache,
        exact_v_cache,
        old_kv_len,
        value_center,
        1.0 / math.sqrt(DIM),
        **run_kwargs,
    )


__all__ = [
    "DIM",
    "HKV",
    "HQ",
    "PAGE",
    "make_heterogeneous_page_gauge_wrapper",
    "make_v_fragment_page_gauge_wrapper",
    "module_uri",
    "plan_decode",
    "run_decode",
    "source_hashes",
    "validate_run_inputs",
    "v_fragment_module_uri",
]
