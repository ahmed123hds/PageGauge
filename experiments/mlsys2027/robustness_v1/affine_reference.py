"""CPU/NumPy reference mathematics, never a production kernel or speed baseline."""
from __future__ import annotations
import math
import numpy as np


def array(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Reference inputs must be finite")
    return x


def softmax(x):
    x = array(x)
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def attention(q, k, v):
    q, k, v = array(q), array(k), array(v)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Expected q[queries,d], k[tokens,d], v[tokens,dv]")
    return softmax(q @ k.T / math.sqrt(k.shape[1])) @ v


def generalized_reconstruct(z, row_scales, transform, center):
    z, scales, transform, center = map(array, (z, row_scales, transform, center))
    if z.ndim != 2 or scales.shape != (z.shape[0],) or transform.shape != (z.shape[1], z.shape[1]) or center.shape != (z.shape[1],):
        raise ValueError("Invalid generalized affine dimensions")
    if np.any(scales <= 0) or np.linalg.matrix_rank(transform) != transform.shape[0]:
        raise ValueError("Positive scales and invertible transform required")
    return (scales[:, None] * z) @ transform + center


def generalized_factorized(q, zk, zv, sk, sv, ak, av, ck, cv):
    # Call shape validation; no performance claim for this deliberately simple oracle.
    generalized_reconstruct(zk, sk, ak, ck)
    generalized_reconstruct(zv, sv, av, cv)
    q, zk, zv, sk, sv, ak, av, cv = map(array, (q, zk, zv, sk, sv, ak, av, cv))
    scores = ((q @ ak.T) @ zk.T) * sk[None, :] / math.sqrt(q.shape[1])
    probabilities = softmax(scores)
    return ((probabilities * sv[None, :]) @ zv) @ av + cv


def attention_error_bound(q, k, v, khat, vhat):
    """Fixed-query bound; finite precision and multi-layer propagation are separate."""
    q, k, v, khat, vhat = map(array, (q, k, v, khat, vhat))
    error = q @ (khat - k).T / math.sqrt(k.shape[1])
    epsilon_k = (error.max(axis=-1) - error.min(axis=-1)) / 2
    center = v.mean(axis=0)
    radius = np.linalg.norm(v - center, axis=-1).max()
    epsilon_v = np.linalg.norm(vhat - v, axis=-1).max()
    actual = np.linalg.norm(attention(q, khat, vhat) - attention(q, k, v), axis=-1)
    bound = radius * np.minimum(2.0, epsilon_k) + epsilon_v
    return {"epsilon_score_modulo_shift": epsilon_k.tolist(), "value_radius": float(radius),
            "epsilon_value": float(epsilon_v), "output_error_l2": actual.tolist(), "bound_l2": bound.tolist(),
            "bound_satisfied_with_fp64_tolerance": bool(np.all(actual <= bound + 1e-10 * (1 + bound)))}


def round_away(x):
    x = np.asarray(x)
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def exact_mask(tokens, initial, policy):
    page = policy["page_size"]
    pages = (tokens + page - 1) // page
    logical = np.arange(tokens) // page
    initial_pages = initial // page
    return ((logical < policy["exact_prefix_pages"]) |
            ((logical >= initial_pages - policy["exact_static_suffix_pages"]) & (logical < initial_pages)) |
            (logical >= pages - policy["exact_tail_tokens"] // page))


def mixed_reconstructions(x, center, initial, policy):
    """x[tokens,heads,d]; NumPy emulation, NOT evidence of production finalizer correctness.

    FP32 division/round-half-away, stored FP16 max-absolute scale. Only complete
    historical pages are quantized; incomplete recent pages remain centered FP16.
    """
    x = np.asarray(x, dtype=np.float16)
    center = np.asarray(center, dtype=np.float16)
    if x.ndim != 3 or center.shape != x.shape[1:] or not np.isfinite(x).all() or not np.isfinite(center).all():
        raise ValueError("Invalid capture tensor/center")
    residual = x.astype(np.float32) - center.astype(np.float32)
    centered_half = residual.astype(np.float16)
    stored_exact = centered_half.astype(np.float64) + center.astype(np.float64)
    reconstructed_real = stored_exact.copy()
    reconstructed_half = stored_exact.copy()
    mask = exact_mask(len(x), initial, policy)
    page = policy["page_size"]
    scale_values = []
    clipping = 0
    for start in range(0, len(x), page):
        end = min(start + page, len(x))
        if mask[start:end].all():
            continue
        if end - start != page or mask[start:end].any():
            raise ValueError("Quantized page must be complete and outside exact regions")
        tile = residual[start:end]
        scales = np.maximum(np.max(np.abs(tile), axis=(0, 2)) / np.float32(127), np.float32(2.0**-20)).astype(np.float16)
        unbounded = round_away(tile / scales.astype(np.float32)[None, :, None])
        clipping += int(np.count_nonzero(np.abs(unbounded) > 127))
        codes = np.clip(unbounded, -127, 127).astype(np.int8)
        real = codes.astype(np.float64) * scales.astype(np.float64)[None, :, None]
        reconstructed_real[start:end] = real + center.astype(np.float64)
        reconstructed_half[start:end] = real.astype(np.float16).astype(np.float64) + center.astype(np.float64)
        scale_values.extend(scales.astype(float).tolist())
    return {"centered": stored_exact, "real": reconstructed_real, "half": reconstructed_half,
            "exact_mask": mask, "clipped_values": clipping,
            "scale_min": min(scale_values) if scale_values else None,
            "scale_max": max(scale_values) if scale_values else None}


def output_metrics(observed, expected):
    observed, expected = array(observed), array(expected)
    delta = observed - expected
    denom = np.linalg.norm(observed, axis=-1) * np.linalg.norm(expected, axis=-1)
    cosine = np.divide((observed * expected).sum(axis=-1), denom,
                       out=np.full(denom.shape, np.nan), where=denom > 0)
    return {"max_abs": float(np.abs(delta).max()), "l2_per_query": np.linalg.norm(delta, axis=-1).tolist(),
            "cosine_per_query": [float(x) if np.isfinite(x) else None for x in cosine]}
