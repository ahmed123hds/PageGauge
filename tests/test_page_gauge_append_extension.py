from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "page_gauge_runtime", ROOT / "scripts/page_gauge_runtime.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)


def test_device_dynamic_append_matches_static_position_abi() -> None:
    """Root GPU smoke: static and graph-safe device-position ABIs are bitwise."""

    extension = RUNTIME.load_append_extension()
    torch.manual_seed(20261003)
    batch = 2
    pages_per_request = 3
    ring_pages_per_request = 2
    # Three complete pages exercise page-close finalization before and after
    # the two-page exact ring wraps (page 2 reuses ring page 0).
    positions = list(range(48))
    rope_cos = torch.randn(48, 128, device="cuda", dtype=torch.float16)
    rope_sin = torch.randn_like(rope_cos)
    query = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(48, batch, 8, 128, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    device_position = torch.zeros(1, device="cuda", dtype=torch.int32)

    static_q = torch.empty_like(query)
    dynamic_q = torch.empty_like(query)
    static_fp16_k = torch.zeros(
        batch * pages_per_request, 16, 8, 128, device="cuda", dtype=torch.float16
    )
    static_fp16_v = torch.zeros_like(static_fp16_k)
    dynamic_fp16_k = torch.zeros_like(static_fp16_k)
    dynamic_fp16_v = torch.zeros_like(static_fp16_k)
    for position in positions:
        extension.rope_append_fp16(
            query,
            keys[position],
            values[position],
            rope_cos[position],
            rope_sin[position],
            static_q,
            static_fp16_k,
            static_fp16_v,
            position,
        )
        device_position.fill_(position)
        extension.rope_append_fp16_dynamic(
            query,
            keys[position],
            values[position],
            rope_cos,
            rope_sin,
            device_position,
            dynamic_q,
            dynamic_fp16_k,
            dynamic_fp16_v,
        )
    torch.cuda.synchronize()
    assert torch.equal(static_q, dynamic_q)
    assert torch.equal(static_fp16_k, dynamic_fp16_k)
    assert torch.equal(static_fp16_v, dynamic_fp16_v)
    final_cos = rope_cos[positions[-1]]
    final_sin = rope_sin[positions[-1]]
    expected_query = query * final_cos + torch.cat(
        (-query[..., 64:], query[..., :64]), dim=-1
    ) * final_sin
    final_key = keys[positions[-1]]
    expected_key = final_key * final_cos + torch.cat(
        (-final_key[..., 64:], final_key[..., :64]), dim=-1
    ) * final_sin
    assert torch.equal(static_q, expected_query)
    for request in range(batch):
        assert torch.equal(
            static_fp16_k[
                request * pages_per_request + positions[-1] // 16,
                positions[-1] % 16,
            ],
            expected_key[request],
        )

    key_center = torch.randn(
        batch, 8, 128, device="cuda", dtype=torch.float16
    )
    value_center = torch.randn_like(key_center)
    exact_shape = (batch * ring_pages_per_request, 16, 8, 128)
    code_shape = (batch * pages_per_request, 16, 8, 128)
    static_exact_k = torch.zeros(
        exact_shape, device="cuda", dtype=torch.float16
    )
    static_exact_v = torch.zeros_like(static_exact_k)
    dynamic_exact_k = torch.zeros_like(static_exact_k)
    dynamic_exact_v = torch.zeros_like(static_exact_k)
    static_codes_k = torch.zeros(code_shape, device="cuda", dtype=torch.int8)
    static_codes_v = torch.zeros_like(static_codes_k)
    dynamic_codes_k = torch.zeros_like(static_codes_k)
    dynamic_codes_v = torch.zeros_like(static_codes_k)
    static_scales_k = torch.zeros(
        batch * pages_per_request, 8, device="cuda", dtype=torch.float16
    )
    static_scales_v = torch.zeros_like(static_scales_k)
    dynamic_scales_k = torch.zeros_like(static_scales_k)
    dynamic_scales_v = torch.zeros_like(static_scales_k)
    for position in positions:
        extension.rope_append_page_gauge(
            query,
            keys[position],
            values[position],
            rope_cos[position],
            rope_sin[position],
            key_center,
            value_center,
            static_q,
            static_exact_k,
            static_exact_v,
            static_codes_k,
            static_codes_v,
            static_scales_k,
            static_scales_v,
            0,
            position,
        )
        device_position.fill_(position)
        extension.rope_append_page_gauge_dynamic(
            query,
            keys[position],
            values[position],
            rope_cos,
            rope_sin,
            device_position,
            key_center,
            value_center,
            dynamic_q,
            dynamic_exact_k,
            dynamic_exact_v,
            dynamic_codes_k,
            dynamic_codes_v,
            dynamic_scales_k,
            dynamic_scales_v,
            0,
        )
    torch.cuda.synchronize()
    assert torch.equal(static_q, dynamic_q)
    assert torch.equal(static_exact_k, dynamic_exact_k)
    assert torch.equal(static_exact_v, dynamic_exact_v)
    assert torch.equal(static_codes_k, dynamic_codes_k)
    assert torch.equal(static_codes_v, dynamic_codes_v)
    assert torch.equal(static_scales_k, dynamic_scales_k)
    assert torch.equal(static_scales_v, dynamic_scales_v)


def test_fused_rope_append_finalizes_exact_page_in_same_launch() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(41)
    query = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(16, 1, 8, 128, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    cosine = torch.ones(128, device="cuda", dtype=torch.float16)
    sine = torch.zeros_like(cosine)
    key_center = torch.randn(8, 128, device="cuda", dtype=torch.float16) * 0.1
    value_center = torch.randn_like(key_center) * 0.1
    rotated_query = torch.empty_like(query)
    exact_k = torch.zeros(1, 16, 8, 128, device="cuda", dtype=torch.float16)
    exact_v = torch.zeros_like(exact_k)
    codes_k = torch.empty_like(exact_k, dtype=torch.int8)
    codes_v = torch.empty_like(exact_v, dtype=torch.int8)
    scales_k = torch.empty(1, 8, device="cuda", dtype=torch.float16)
    scales_v = torch.empty_like(scales_k)
    for position in range(16):
        extension.rope_append_page_gauge(
            query,
            keys[position],
            values[position],
            cosine,
            sine,
            key_center,
            value_center,
            rotated_query,
            exact_k,
            exact_v,
            codes_k,
            codes_v,
            scales_k,
            scales_v,
            0,
            position,
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(rotated_query, query)
    torch.testing.assert_close(
        exact_k.float() + key_center[None, None].float(),
        keys[:, 0].float().unsqueeze(0),
        atol=1.5e-3,
        rtol=0.0,
    )
    reconstructed = (
        codes_k.float() * scales_k[:, None, :, None].float()
        + key_center[None, None].float()
    )
    relative = (
        reconstructed - keys[:, 0].float().unsqueeze(0)
    ).norm() / keys[:, 0].float().norm()
    assert float(relative) < 0.02


def test_fused_append_reuses_fp16_ring_after_writing_full_int8_cache() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(73)
    pages = 4
    ring_pages = 2
    query = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(
        pages * 16, 1, 8, 128, device="cuda", dtype=torch.float16
    )
    values = torch.randn_like(keys)
    cosine = torch.ones(128, device="cuda", dtype=torch.float16)
    sine = torch.zeros_like(cosine)
    key_center = torch.randn(8, 128, device="cuda", dtype=torch.float16) * 0.1
    value_center = torch.randn_like(key_center) * 0.1
    rotated_query = torch.empty_like(query)
    exact_k = torch.zeros(
        ring_pages, 16, 8, 128, device="cuda", dtype=torch.float16
    )
    exact_v = torch.zeros_like(exact_k)
    codes_k = torch.empty(
        pages, 16, 8, 128, device="cuda", dtype=torch.int8
    )
    codes_v = torch.empty_like(codes_k)
    scales_k = torch.empty(pages, 8, device="cuda", dtype=torch.float16)
    scales_v = torch.empty_like(scales_k)

    for position in range(pages * 16):
        extension.rope_append_page_gauge(
            query,
            keys[position],
            values[position],
            cosine,
            sine,
            key_center,
            value_center,
            rotated_query,
            exact_k,
            exact_v,
            codes_k,
            codes_v,
            scales_k,
            scales_v,
            0,
            position,
        )
    torch.cuda.synchronize()

    reconstructed = (
        codes_k.float() * scales_k[:, None, :, None].float()
        + key_center[None, None].float()
    )
    reference = keys[:, 0].reshape(pages, 16, 8, 128).float()
    relative = (reconstructed - reference).norm() / reference.norm()
    assert float(relative) < 0.02
    for logical_page in (2, 3):
        ring_page = logical_page % ring_pages
        torch.testing.assert_close(
            exact_k[ring_page].float() + key_center[None].float(),
            reference[logical_page],
            atol=2.0e-3,
            rtol=0.0,
        )


def test_exact_sink_slot_survives_multiple_tail_ring_wraps() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(20260820)
    pages = 5
    tail_pages = 2
    sink_pages = 1
    query = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(pages * 16, 1, 8, 128, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    cosine = torch.ones(128, device="cuda", dtype=torch.float16)
    sine = torch.zeros_like(cosine)
    key_center = torch.randn(8, 128, device="cuda", dtype=torch.float16) * 0.1
    value_center = torch.randn_like(key_center) * 0.1
    rotated_query = torch.empty_like(query)
    exact_k = torch.zeros(
        sink_pages + tail_pages, 16, 8, 128, device="cuda", dtype=torch.float16
    )
    exact_v = torch.zeros_like(exact_k)
    codes_k = torch.empty(
        pages, 16, 8, 128, device="cuda", dtype=torch.int8
    )
    codes_v = torch.empty_like(codes_k)
    scales_k = torch.empty(pages, 8, device="cuda", dtype=torch.float16)
    scales_v = torch.empty_like(scales_k)

    for position in range(pages * 16):
        extension.rope_append_page_gauge(
            query,
            keys[position],
            values[position],
            cosine,
            sine,
            key_center,
            value_center,
            rotated_query,
            exact_k,
            exact_v,
            codes_k,
            codes_v,
            scales_k,
            scales_v,
            sink_pages,
            position,
        )
    torch.cuda.synchronize()

    reference_key = keys[:, 0].reshape(pages, 16, 8, 128).float()
    reference_value = values[:, 0].reshape(pages, 16, 8, 128).float()
    torch.testing.assert_close(
        exact_k[0].float() + key_center[None].float(),
        reference_key[0],
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        exact_v[0].float() + value_center[None].float(),
        reference_value[0],
        atol=2.0e-3,
        rtol=0.0,
    )
    for logical_page in (3, 4):
        physical_page = sink_pages + logical_page % tail_pages
        torch.testing.assert_close(
            exact_k[physical_page].float() + key_center[None].float(),
            reference_key[logical_page],
            atol=2.0e-3,
            rtol=0.0,
        )


def test_exact_prefix_s3_slots_survive_static_and_dynamic_tail_wraps() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(20260821)
    batch = 2
    logical_pages = 8
    prefix_pages = 3
    tail_pages = 2
    positions = list(range(logical_pages * 16))
    query = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(
        len(positions), batch, 8, 128, device="cuda", dtype=torch.float16
    )
    values = torch.randn_like(keys)
    rope_cos = torch.ones(len(positions), 128, device="cuda", dtype=torch.float16)
    rope_sin = torch.zeros_like(rope_cos)
    key_center = torch.randn(
        batch, 8, 128, device="cuda", dtype=torch.float16
    ) * 0.1
    value_center = torch.randn_like(key_center) * 0.1
    static_q = torch.empty_like(query)
    dynamic_q = torch.empty_like(query)
    exact_shape = (batch * (prefix_pages + tail_pages), 16, 8, 128)
    code_shape = (batch * logical_pages, 16, 8, 128)
    static_exact_k = torch.zeros(exact_shape, device="cuda", dtype=torch.float16)
    static_exact_v = torch.zeros_like(static_exact_k)
    dynamic_exact_k = torch.zeros_like(static_exact_k)
    dynamic_exact_v = torch.zeros_like(static_exact_k)
    static_codes_k = torch.zeros(code_shape, device="cuda", dtype=torch.int8)
    static_codes_v = torch.zeros_like(static_codes_k)
    dynamic_codes_k = torch.zeros_like(static_codes_k)
    dynamic_codes_v = torch.zeros_like(static_codes_k)
    static_scales_k = torch.zeros(
        batch * logical_pages, 8, device="cuda", dtype=torch.float16
    )
    static_scales_v = torch.zeros_like(static_scales_k)
    dynamic_scales_k = torch.zeros_like(static_scales_k)
    dynamic_scales_v = torch.zeros_like(static_scales_k)
    device_position = torch.zeros(1, device="cuda", dtype=torch.int32)
    static_prefix_k = None
    static_prefix_v = None
    dynamic_prefix_k = None
    dynamic_prefix_v = None

    for position in positions:
        extension.rope_append_page_gauge(
            query,
            keys[position],
            values[position],
            rope_cos[position],
            rope_sin[position],
            key_center,
            value_center,
            static_q,
            static_exact_k,
            static_exact_v,
            static_codes_k,
            static_codes_v,
            static_scales_k,
            static_scales_v,
            prefix_pages,
            position,
        )
        device_position.fill_(position)
        extension.rope_append_page_gauge_dynamic(
            query,
            keys[position],
            values[position],
            rope_cos,
            rope_sin,
            device_position,
            key_center,
            value_center,
            dynamic_q,
            dynamic_exact_k,
            dynamic_exact_v,
            dynamic_codes_k,
            dynamic_codes_v,
            dynamic_scales_k,
            dynamic_scales_v,
            prefix_pages,
        )
        if position == prefix_pages * 16 - 1:
            prefix_physical = torch.tensor(
                [
                    request * (prefix_pages + tail_pages) + prefix_page
                    for request in range(batch)
                    for prefix_page in range(prefix_pages)
                ],
                device="cuda",
                dtype=torch.long,
            )
            static_prefix_k = static_exact_k.index_select(0, prefix_physical).clone()
            static_prefix_v = static_exact_v.index_select(0, prefix_physical).clone()
            dynamic_prefix_k = dynamic_exact_k.index_select(0, prefix_physical).clone()
            dynamic_prefix_v = dynamic_exact_v.index_select(0, prefix_physical).clone()

    torch.cuda.synchronize()
    assert static_prefix_k is not None and static_prefix_v is not None
    assert dynamic_prefix_k is not None and dynamic_prefix_v is not None
    assert torch.equal(static_q, dynamic_q)
    assert torch.equal(static_exact_k, dynamic_exact_k)
    assert torch.equal(static_exact_v, dynamic_exact_v)
    assert torch.equal(static_codes_k, dynamic_codes_k)
    assert torch.equal(static_codes_v, dynamic_codes_v)
    assert torch.equal(static_scales_k, dynamic_scales_k)
    assert torch.equal(static_scales_v, dynamic_scales_v)
    assert torch.equal(static_exact_k.index_select(0, prefix_physical), static_prefix_k)
    assert torch.equal(static_exact_v.index_select(0, prefix_physical), static_prefix_v)
    assert torch.equal(dynamic_exact_k.index_select(0, prefix_physical), dynamic_prefix_k)
    assert torch.equal(dynamic_exact_v.index_select(0, prefix_physical), dynamic_prefix_v)

    assert bool(torch.all(static_scales_k > 0))
    assert bool(torch.all(static_scales_v > 0))
    request_major_key = keys.permute(1, 0, 2, 3).reshape_as(static_codes_k).float()
    request_major_value = values.permute(1, 0, 2, 3).reshape_as(static_codes_v).float()
    expanded_key_center = (
        key_center[:, None, None]
        .expand(batch, logical_pages, 16, 8, 128)
        .reshape_as(request_major_key)
        .float()
    )
    expanded_value_center = (
        value_center[:, None, None]
        .expand(batch, logical_pages, 16, 8, 128)
        .reshape_as(request_major_value)
        .float()
    )
    reconstructed_key = (
        static_codes_k.float()
        * static_scales_k[:, None, :, None].float()
        + expanded_key_center
    )
    reconstructed_value = (
        static_codes_v.float()
        * static_scales_v[:, None, :, None].float()
        + expanded_value_center
    )
    assert float((reconstructed_key - request_major_key).norm() / request_major_key.norm()) < 0.02
    assert float(
        (reconstructed_value - request_major_value).norm()
        / request_major_value.norm()
    ) < 0.02

    for request in range(batch):
        source_key_pages = keys[:, request].reshape(logical_pages, 16, 8, 128)
        source_value_pages = values[:, request].reshape(logical_pages, 16, 8, 128)
        for logical_page in (*range(prefix_pages), 6, 7):
            local_page = (
                logical_page
                if logical_page < prefix_pages
                else prefix_pages + logical_page % tail_pages
            )
            physical_page = request * (prefix_pages + tail_pages) + local_page
            torch.testing.assert_close(
                static_exact_k[physical_page].float()
                + key_center[request][None].float(),
                source_key_pages[logical_page].float(),
                atol=2.0e-3,
                rtol=0.0,
            )
            torch.testing.assert_close(
                static_exact_v[physical_page].float()
                + value_center[request][None].float(),
                source_value_pages[logical_page].float(),
                atol=2.0e-3,
                rtol=0.0,
            )

    with pytest.raises(RuntimeError, match="must be nonnegative"):
        extension.rope_append_page_gauge(
            query,
            keys[0],
            values[0],
            rope_cos[0],
            rope_sin[0],
            key_center,
            value_center,
            static_q,
            static_exact_k,
            static_exact_v,
            static_codes_k,
            static_codes_v,
            static_scales_k,
            static_scales_v,
            -1,
            0,
        )
    with pytest.raises(RuntimeError, match="at least one tail-ring page"):
        extension.rope_append_page_gauge_dynamic(
            query,
            keys[0],
            values[0],
            rope_cos,
            rope_sin,
            device_position,
            key_center,
            value_center,
            dynamic_q,
            dynamic_exact_k,
            dynamic_exact_v,
            dynamic_codes_k,
            dynamic_codes_v,
            dynamic_scales_k,
            dynamic_scales_v,
            prefix_pages + tail_pages,
        )
    oversized_exact_k = torch.zeros(
        batch * (logical_pages + tail_pages),
        16,
        8,
        128,
        device="cuda",
        dtype=torch.float16,
    )
    oversized_exact_v = torch.zeros_like(oversized_exact_k)
    with pytest.raises(RuntimeError, match="logical cache capacity"):
        extension.rope_append_page_gauge(
            query,
            keys[0],
            values[0],
            rope_cos[0],
            rope_sin[0],
            key_center,
            value_center,
            static_q,
            oversized_exact_k,
            oversized_exact_v,
            static_codes_k,
            static_codes_v,
            static_scales_k,
            static_scales_v,
            logical_pages,
            0,
        )


def test_fused_exact_tail_merge_center_matches_online_softmax() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(97)
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    ring_pages = 16
    sm_scale = head_dim**-0.5
    log2_e = 1.4426950408889634

    query = torch.randn(
        1, query_heads, head_dim, device="cuda", dtype=torch.float16
    )
    exact_key = torch.randn(
        ring_pages, 16, kv_heads, head_dim,
        device="cuda", dtype=torch.float16,
    ) * 0.35
    exact_value = torch.randn_like(exact_key) * 0.35
    value_center = torch.randn(
        kv_heads, head_dim, device="cuda", dtype=torch.float16
    ) * 0.1

    for tail_tokens in (1, 15, 16, 17, 241, 255, 256):
        num_pages = (tail_tokens + 15) // 16
        # Exercise wrapped/non-monotonic ring addressing, not just identity.
        indices = torch.arange(
            7, 7 + num_pages, device="cuda", dtype=torch.int32
        ).remainder(ring_pages)
        last_len = torch.tensor(
            [(tail_tokens - 1) % 16 + 1], device="cuda", dtype=torch.int32
        )
        old_output = torch.randn(
            1, query_heads, head_dim, device="cuda", dtype=torch.float16
        ) * 0.2
        old_lse = torch.randn(
            1, query_heads, device="cuda", dtype=torch.float32
        ) * 0.5 + 8.0
        output_before = old_output.clone()
        lse_before = old_lse.clone()

        logical_key = exact_key[indices.long()].reshape(
            -1, kv_heads, head_dim
        )[:tail_tokens].float()
        logical_value = exact_value[indices.long()].reshape(
            -1, kv_heads, head_dim
        )[:tail_tokens].float()
        kv_for_query = torch.arange(query_heads, device="cuda") // (
            query_heads // kv_heads
        )
        tail_scores_log2 = torch.einsum(
            "qd,qtd->qt",
            query[0].float(),
            logical_key[:, kv_for_query].permute(1, 0, 2),
        ) * (sm_scale * log2_e)
        merged_max = torch.maximum(
            lse_before[0], tail_scores_log2.amax(dim=1)
        )
        old_weight = torch.exp2(lse_before[0] - merged_max)
        tail_weight = torch.exp2(tail_scores_log2 - merged_max[:, None])
        denominator = old_weight + tail_weight.sum(dim=1)
        reference = (
            old_weight[:, None] * output_before[0].float()
            + torch.einsum(
                "qt,qtd->qd",
                tail_weight,
                logical_value[:, kv_for_query].permute(1, 0, 2),
            )
        ) / denominator[:, None]
        reference = reference + value_center[kv_for_query].float()
        reference_lse = torch.log2(denominator) + merged_max

        extension.exact_tail_merge_center(
            query,
            exact_key,
            exact_value,
            indices,
            last_len,
            old_output,
            old_lse,
            value_center,
            sm_scale,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            old_output.float(), reference[None], atol=3.0e-3, rtol=2.0e-3
        )
        torch.testing.assert_close(
            old_lse[0], reference_lse, atol=2.0e-4, rtol=2.0e-5
        )


def test_batched_append_uses_request_major_physical_pages() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(131)
    batch = 3
    pages_per_request = 3
    ring_pages_per_request = 2
    query = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(
        pages_per_request * 16,
        batch,
        8,
        128,
        device="cuda",
        dtype=torch.float16,
    )
    values = torch.randn_like(keys)
    cosine = torch.ones(128, device="cuda", dtype=torch.float16)
    sine = torch.zeros_like(cosine)
    key_center = (
        torch.randn(batch, 8, 128, device="cuda", dtype=torch.float16) * 0.1
    )
    value_center = torch.randn_like(key_center) * 0.1
    rotated_query = torch.empty_like(query)
    exact_k = torch.zeros(
        batch * ring_pages_per_request,
        16,
        8,
        128,
        device="cuda",
        dtype=torch.float16,
    )
    exact_v = torch.zeros_like(exact_k)
    codes_k = torch.empty(
        batch * pages_per_request,
        16,
        8,
        128,
        device="cuda",
        dtype=torch.int8,
    )
    codes_v = torch.empty_like(codes_k)
    scales_k = torch.empty(
        batch * pages_per_request, 8, device="cuda", dtype=torch.float16
    )
    scales_v = torch.empty_like(scales_k)
    fp16_k = torch.zeros(
        batch * pages_per_request,
        16,
        8,
        128,
        device="cuda",
        dtype=torch.float16,
    )
    fp16_v = torch.zeros_like(fp16_k)

    for position in range(pages_per_request * 16):
        extension.rope_append_fp16(
            query,
            keys[position],
            values[position],
            cosine,
            sine,
            rotated_query,
            fp16_k,
            fp16_v,
            position,
        )
        extension.rope_append_page_gauge(
            query,
            keys[position],
            values[position],
            cosine,
            sine,
            key_center,
            value_center,
            rotated_query,
            exact_k,
            exact_v,
            codes_k,
            codes_v,
            scales_k,
            scales_v,
            0,
            position,
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(rotated_query, query)
    for request in range(batch):
        physical_page = request * pages_per_request
        reference_key = keys[:, request].reshape(
            pages_per_request, 16, 8, 128
        ).float()
        reference_value = values[:, request].reshape(
            pages_per_request, 16, 8, 128
        ).float()
        torch.testing.assert_close(
            fp16_k[physical_page : physical_page + pages_per_request].float(),
            reference_key,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            fp16_v[physical_page : physical_page + pages_per_request].float(),
            reference_value,
            atol=0.0,
            rtol=0.0,
        )
        for logical_page in (1, 2):
            ring_page = (
                request * ring_pages_per_request
                + logical_page % ring_pages_per_request
            )
            torch.testing.assert_close(
                exact_k[ring_page].float() + key_center[request, None].float(),
                reference_key[logical_page],
                atol=2.0e-3,
                rtol=0.0,
            )
            torch.testing.assert_close(
                exact_v[ring_page].float() + value_center[request, None].float(),
                reference_value[logical_page],
                atol=2.0e-3,
                rtol=0.0,
            )
        reconstructed = (
            codes_k[physical_page : physical_page + pages_per_request].float()
            * scales_k[
                physical_page : physical_page + pages_per_request,
                None,
                :,
                None,
            ].float()
            + key_center[request, None].float()
        )
        relative = (reconstructed - reference_key).norm() / reference_key.norm()
        assert float(relative) < 0.02
        reconstructed_value = (
            codes_v[physical_page : physical_page + pages_per_request].float()
            * scales_v[
                physical_page : physical_page + pages_per_request,
                None,
                :,
                None,
            ].float()
            + value_center[request, None].float()
        )
        value_relative = (
            reconstructed_value - reference_value
        ).norm() / reference_value.norm()
        assert float(value_relative) < 0.02


def test_batched_exact_tail_merge_center_matches_per_request_oracle() -> None:
    extension = RUNTIME.load_append_extension()
    torch.manual_seed(173)
    batch = 3
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    ring_pages = 4
    tail_tokens = [33, 37, 47]
    num_pages = 3
    sm_scale = head_dim**-0.5
    log2_e = 1.4426950408889634
    query = torch.randn(
        batch, query_heads, head_dim, device="cuda", dtype=torch.float16
    )
    exact_key = torch.randn(
        batch * ring_pages,
        16,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    ) * 0.35
    exact_value = torch.randn_like(exact_key) * 0.35
    value_center = torch.randn(
        batch, kv_heads, head_dim, device="cuda", dtype=torch.float16
    ) * 0.1
    indices = torch.stack(
        [
            request * ring_pages
            + torch.tensor([2, 3, 0], device="cuda", dtype=torch.int32)
            for request in range(batch)
        ]
    ).reshape(-1)
    last_len = torch.tensor(
        [(tokens - 1) % 16 + 1 for tokens in tail_tokens],
        device="cuda",
        dtype=torch.int32,
    )
    old_output = torch.randn(
        batch, query_heads, head_dim, device="cuda", dtype=torch.float16
    ) * 0.2
    old_lse = torch.randn(
        batch, query_heads, device="cuda", dtype=torch.float32
    ) * 0.5 + 8.0
    output_before = old_output.clone()
    lse_before = old_lse.clone()
    kv_for_query = torch.arange(query_heads, device="cuda") // (
        query_heads // kv_heads
    )
    references = []
    reference_lses = []
    for request in range(batch):
        request_tail_tokens = tail_tokens[request]
        request_indices = indices.view(batch, num_pages)[request].long()
        logical_key = exact_key[request_indices].reshape(
            -1, kv_heads, head_dim
        )[:request_tail_tokens].float()
        logical_value = exact_value[request_indices].reshape(
            -1, kv_heads, head_dim
        )[:request_tail_tokens].float()
        scores = torch.einsum(
            "qd,qtd->qt",
            query[request].float(),
            logical_key[:, kv_for_query].permute(1, 0, 2),
        ) * (sm_scale * log2_e)
        merged_max = torch.maximum(lse_before[request], scores.amax(dim=1))
        old_weight = torch.exp2(lse_before[request] - merged_max)
        tail_weight = torch.exp2(scores - merged_max[:, None])
        denominator = old_weight + tail_weight.sum(dim=1)
        reference = (
            old_weight[:, None] * output_before[request].float()
            + torch.einsum(
                "qt,qtd->qd",
                tail_weight,
                logical_value[:, kv_for_query].permute(1, 0, 2),
            )
        ) / denominator[:, None]
        references.append(reference + value_center[request, kv_for_query].float())
        reference_lses.append(torch.log2(denominator) + merged_max)

    extension.exact_tail_merge_center(
        query,
        exact_key,
        exact_value,
        indices,
        last_len,
        old_output,
        old_lse,
        value_center,
        sm_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        old_output.float(), torch.stack(references), atol=3.0e-3, rtol=2.0e-3
    )
    torch.testing.assert_close(
        old_lse, torch.stack(reference_lses), atol=2.0e-4, rtol=2.0e-5
    )
