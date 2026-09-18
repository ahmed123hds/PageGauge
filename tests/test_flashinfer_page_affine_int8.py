from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_page_affine",
    ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_signed_page_quantization_reconstructs_with_bias() -> None:
    torch.manual_seed(4)
    values = torch.randn(3, 16, 8, 128, device="cuda", dtype=torch.float16)
    codes, bias, scale, reconstructed = MODULE.quantize_pages(values)
    manual = (
        codes.permute(0, 2, 1, 3)
        .reshape(3, 8, 16, 8, 16)
        .float()
        * scale[:, :, None, :, None].float()
        + bias[:, :, None, :, None].float()
    ).reshape(3, 8, 16, 128).permute(0, 2, 1, 3)
    assert torch.allclose(manual.half(), reconstructed)
    assert float((reconstructed.float() - values.float()).norm() / values.float().norm()) < 0.01


def test_page_gauge_uses_one_scale_per_page_head() -> None:
    torch.manual_seed(7)
    values = torch.randn(3, 16, 8, 128, device="cuda", dtype=torch.float16)
    codes, center, scale, reconstructed, centered = MODULE.quantize_page_gauge(values)
    assert codes.shape == values.shape
    assert center.shape == (8, 128)
    assert scale.shape == (3, 8)
    manual = (
        codes.permute(0, 2, 1, 3).float() * scale[:, :, None, None].float()
    ).permute(0, 2, 1, 3)
    assert torch.allclose(manual.half(), reconstructed)
    assert float((reconstructed.float() - centered.float()).norm() / centered.float().norm()) < 0.02


def test_segment_page_table_keeps_disjoint_sink_and_tail_exact() -> None:
    old_indptr, old_indices, exact_indptr, exact_indices = MODULE.segment_page_table(
        [8, 6], tail_pages=2, sink_pages=1
    )
    assert old_indptr.cpu().tolist() == [0, 5, 8]
    assert old_indices.cpu().tolist() == [1, 2, 3, 4, 5, 9, 10, 11]
    assert exact_indptr.cpu().tolist() == [0, 3, 6]
    assert exact_indices.cpu().tolist() == [0, 6, 7, 8, 12, 13]


def test_factorized_page_gauge_kernel_matches_reconstructed_cache() -> None:
    import flashinfer

    torch.manual_seed(11)
    pages = 32
    values_k = torch.randn(
        pages, 16, 8, 128, device="cuda", dtype=torch.float16
    ) * 0.3
    values_v = torch.randn_like(values_k) * 0.3
    k_codes, _, k_scale, reconstructed_k, _ = MODULE.quantize_page_gauge(values_k)
    v_codes, _, v_scale, reconstructed_v, _ = MODULE.quantize_page_gauge(values_v)
    query = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16) * 0.3
    indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([16], device="cuda", dtype=torch.int32)
    baseline = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
    )
    candidate = MODULE.make_page_gauge_wrapper(
        flashinfer,
        torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    MODULE.plan(baseline, indptr, indices, last_page_len, torch.float16)
    MODULE.plan(candidate, indptr, indices, last_page_len, torch.int8)
    reference = baseline.run(query, (reconstructed_k, reconstructed_v))
    actual = candidate.run(
        query,
        (k_codes, v_codes),
        k_scale,
        v_scale,
        1.0 / (128**0.5),
    )
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert float(relative) < 0.01
