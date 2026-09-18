from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gauge_int8_model", ROOT / "scripts/evaluate_gauge_int8_model.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_key_center_is_softmax_gauge() -> None:
    torch.manual_seed(1)
    query = torch.randn(3, 8)
    key = torch.randn(11, 8)
    center = torch.randn(8)
    exact = torch.softmax(query @ key.T, dim=-1)
    centered = torch.softmax(query @ (key - center).T, dim=-1)
    assert torch.allclose(exact, centered, atol=1e-6)


def test_value_affine_folds_after_probability_sum() -> None:
    torch.manual_seed(2)
    probability = torch.softmax(torch.randn(3, 11), dim=-1)
    codes = torch.randint(-10, 11, (11, 8)).float()
    scale = torch.rand(8)
    center = torch.randn(8)
    dense = probability @ (codes * scale + center)
    folded = (probability @ codes) * scale + center
    assert torch.allclose(dense, folded, atol=1e-6)


def test_page_affine_int8_has_small_reconstruction_error() -> None:
    torch.manual_seed(3)
    values = torch.randn(1, 2, 32, 128)
    reconstructed = MODULE.page_affine_reconstruct(values, 16, 16)
    relative = (reconstructed - values).norm() / values.norm()
    assert relative < 0.01


def test_page_scale_moves_across_qk_and_pv_contractions() -> None:
    torch.manual_seed(5)
    query = torch.randn(4, 8)
    key_codes = torch.randint(-127, 128, (3, 6, 8)).float()
    value_codes = torch.randint(-127, 128, (3, 6, 8)).float()
    key_scale = torch.rand(3)
    value_scale = torch.rand(3)

    dense_key = key_codes * key_scale[:, None, None]
    direct_scores = torch.einsum("qd,ptd->qpt", query, dense_key)
    factorized_scores = (
        torch.einsum("qd,ptd->qpt", query, key_codes) * key_scale[None, :, None]
    )
    assert torch.allclose(direct_scores, factorized_scores, atol=1e-4)

    probability = torch.softmax(factorized_scores.flatten(1), dim=-1).reshape(4, 3, 6)
    dense_value = value_codes * value_scale[:, None, None]
    direct_output = torch.einsum("qpt,ptd->qd", probability, dense_value)
    factorized_output = torch.einsum(
        "qpt,ptd,p->qd", probability, value_codes, value_scale
    )
    assert torch.allclose(direct_output, factorized_output, atol=1e-4)


def test_diagonal_and_additive_attention_gauges_are_exact() -> None:
    torch.manual_seed(9)
    query = torch.randn(3, 8)
    key = torch.randn(11, 8)
    value = torch.randn(11, 8)
    key_center = torch.randn(8)
    value_center = torch.randn(8)
    key_normalizer = torch.rand(8) + 0.2
    value_normalizer = torch.rand(8) + 0.2

    probability = torch.softmax(query @ key.T, dim=-1)
    reference = probability @ value
    transformed_query = query * key_normalizer
    transformed_key = (key - key_center) / key_normalizer
    transformed_value = (value - value_center) / value_normalizer
    transformed_probability = torch.softmax(
        transformed_query @ transformed_key.T, dim=-1
    )
    restored = (transformed_probability @ transformed_value) * value_normalizer + value_center
    assert torch.allclose(reference, restored, atol=2e-6)


def test_randomized_hadamard_is_an_exact_attention_gauge() -> None:
    torch.manual_seed(12)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 11, 8)
    value = torch.randn(2, 11, 8)
    key_rotation = MODULE.make_orthogonal_gauge(2, 8, 4, 17, 0, query.device)
    value_rotation = MODULE.make_orthogonal_gauge(2, 8, 4, 17, 1, query.device)

    identity = torch.eye(8).expand(2, -1, -1)
    assert torch.allclose(
        key_rotation @ key_rotation.transpose(-1, -2), identity, atol=1e-6
    )
    reference_probability = torch.softmax(
        torch.einsum("hqd,htd->hqt", query, key), dim=-1
    )
    reference = torch.einsum("hqt,htd->hqd", reference_probability, value)

    rotated_query = torch.einsum("hqd,hde->hqe", query, key_rotation)
    rotated_key = torch.einsum("htd,hde->hte", key, key_rotation)
    rotated_value = torch.einsum("htd,hde->hte", value, value_rotation)
    rotated_probability = torch.softmax(
        torch.einsum("hqd,htd->hqt", rotated_query, rotated_key), dim=-1
    )
    restored = torch.einsum(
        "hqd,hde->hqe",
        torch.einsum("hqt,htd->hqd", rotated_probability, rotated_value),
        value_rotation.transpose(-1, -2),
    )
    assert torch.allclose(reference, restored, atol=2e-6)
