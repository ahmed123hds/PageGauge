from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "page_gauge_overheads", ROOT / "scripts/benchmark_page_gauge_overheads.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fused_completed_page_quantizer_matches_stored_scale_algebra() -> None:
    torch.manual_seed(21)
    shape = (3, 16, 2, 128)
    key = torch.randn(shape, device="cuda", dtype=torch.float16)
    value = torch.randn(shape, device="cuda", dtype=torch.float16)
    key_center = torch.randn(2, 128, device="cuda", dtype=torch.float16) * 0.1
    value_center = torch.randn(2, 128, device="cuda", dtype=torch.float16) * 0.1
    key_codes = torch.empty_like(key, dtype=torch.int8)
    value_codes = torch.empty_like(value, dtype=torch.int8)
    key_scales = torch.empty(3, 2, device="cuda", dtype=torch.float16)
    value_scales = torch.empty_like(key_scales)

    MODULE.quantize_completed_kv_pages(
        key,
        value,
        key_center,
        value_center,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
    )
    torch.cuda.synchronize()

    for original, center, codes, scales in (
        (key, key_center, key_codes, key_scales),
        (value, value_center, value_codes, value_scales),
    ):
        reconstructed = (
            codes.float() * scales[:, None, :, None].float()
            + center[None, None].float()
        )
        relative = (reconstructed - original.float()).norm() / original.float().norm()
        assert float(relative) < 0.02
        expected_scale = (
            (original.float() - center[None, None].float())
            .permute(0, 2, 1, 3)
            .abs()
            .amax(dim=(2, 3))
            .div(127.0)
            .clamp_min(2.0**-20)
            .half()
        )
        assert torch.equal(scales, expected_scale)


def test_append_finalize_fuses_exact_write_and_completed_page_quantization() -> None:
    torch.manual_seed(31)
    batch, heads = 3, 2
    incoming_k = torch.randn(
        16, batch, heads, 128, device="cuda", dtype=torch.float16
    )
    incoming_v = torch.randn_like(incoming_k)
    key_center = torch.randn(heads, 128, device="cuda", dtype=torch.float16) * 0.1
    value_center = torch.randn_like(key_center) * 0.1
    page_ids = torch.arange(batch, device="cuda", dtype=torch.int32)
    baseline_k = torch.zeros(
        batch, 16, heads, 128, device="cuda", dtype=torch.float16
    )
    baseline_v = torch.zeros_like(baseline_k)
    exact_k = torch.zeros_like(baseline_k)
    exact_v = torch.zeros_like(baseline_v)
    codes_k = torch.empty_like(baseline_k, dtype=torch.int8)
    codes_v = torch.empty_like(baseline_v, dtype=torch.int8)
    scales_k = torch.empty(batch, heads, device="cuda", dtype=torch.float16)
    scales_v = torch.empty_like(scales_k)
    for token in range(16):
        offsets = torch.full(
            (batch,), token, device="cuda", dtype=torch.int32
        )
        MODULE.append_kv(
            incoming_k[token], incoming_v[token], baseline_k, baseline_v,
            page_ids, offsets,
        )
        MODULE.append_finalize_page_gauge(
            incoming_k[token], incoming_v[token], key_center, value_center,
            exact_k, exact_v, codes_k, codes_v, scales_k, scales_v,
            page_ids, offsets,
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        exact_k.float() + key_center[None, None].float(),
        baseline_k.float(),
        atol=1.5e-3,
        rtol=0.0,
    )
    reconstructed = (
        codes_k.float() * scales_k[:, None, :, None].float()
        + key_center[None, None].float()
    )
    relative = (reconstructed - baseline_k.float()).norm() / baseline_k.float().norm()
    assert float(relative) < 0.02
