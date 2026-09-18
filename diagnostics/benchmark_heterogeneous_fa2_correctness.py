#!/usr/bin/env python3
"""Correctness oracle for PageGauge's single heterogeneous FlashInfer FA2 path.

The candidate consumes one logical page table whose old-page entries index the
INT8 code cache and whose tail entries index a centered FP16 ring.  Every case
is checked against stock FP16 FA2 over the reconstructed logical cache and an
independent dense PyTorch attention calculation.  This is a correctness gate,
not a performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import page_gauge_heterogeneous_fa2 as HETERO  # noqa: E402


PAGE = HETERO.PAGE
DIM = HETERO.DIM
HQ = HETERO.HQ
HKV = HETERO.HKV
GROUPS = HQ // HKV


@dataclass(frozen=True)
class Case:
    name: str
    old_tokens: int
    total_tokens: int
    ring_pages: int
    zero_center: bool = False

    @property
    def total_pages(self) -> int:
        return math.ceil(self.total_tokens / PAGE)

    @property
    def old_pages(self) -> int:
        return self.old_tokens // PAGE

    @property
    def exact_pages(self) -> int:
        return self.total_pages - self.old_pages

    @property
    def last_page_len(self) -> int:
        return (self.total_tokens - 1) % PAGE + 1


CASES = (
    Case("old_only", old_tokens=256, total_tokens=256, ring_pages=1),
    Case("exact_only_last1", old_tokens=0, total_tokens=1, ring_pages=2, zero_center=True),
    Case("exact_only_last15", old_tokens=0, total_tokens=15, ring_pages=2, zero_center=True),
    Case("exact_only_last16", old_tokens=0, total_tokens=16, ring_pages=2, zero_center=True),
    Case("boundary_last1", old_tokens=16, total_tokens=17, ring_pages=4),
    Case("boundary_last15", old_tokens=16, total_tokens=31, ring_pages=4),
    Case("boundary_last16", old_tokens=16, total_tokens=32, ring_pages=4),
    Case("boundary_after_full_cta", old_tokens=144, total_tokens=176, ring_pages=4),
    Case("ring_wrap_last15", old_tokens=144, total_tokens=191, ring_pages=4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--splits", default="auto,2,8")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-abs", type=float, default=0.025)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-lse-abs", type=float, default=0.025)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/page_gauge_int8/heterogeneous_fa2_correctness.json",
    )
    return parser.parse_args()


def parse_splits(value: str) -> list[int]:
    splits: list[int] = []
    for item in value.split(","):
        item = item.strip().lower()
        split = 0 if item == "auto" else int(item)
        if split < 0:
            raise ValueError("split sizes must be non-negative")
        if split not in splits:
            splits.append(split)
    if not splits:
        raise ValueError("at least one split is required")
    return splits


def select_cases(value: str) -> list[Case]:
    if value.strip().lower() == "all":
        return list(CASES)
    requested = {item.strip() for item in value.split(",") if item.strip()}
    known = {case.name: case for case in CASES}
    missing = requested - known.keys()
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    return [case for case in CASES if case.name in requested]


def tensor_sha256(tensor: torch.Tensor) -> str:
    host = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(host).hexdigest()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = actual_f - expected_f
    cosine = F.cosine_similarity(actual_f, expected_f, dim=-1)
    relative = difference.norm(dim=-1) / expected_f.norm(dim=-1).clamp_min(1e-8)
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2_max": float(relative.max()),
        "relative_l2_mean": float(relative.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
    }


def make_indptr(batch_size: int, pages: int) -> torch.Tensor:
    return torch.arange(
        0,
        (batch_size + 1) * pages,
        pages,
        device="cuda",
        dtype=torch.int32,
    )


def quantize_old_pages(
    pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One signed-INT8 scale per [request,page,KV-head]."""
    scale = (
        pages.float().abs().amax(dim=(2, 4)) / 127.0
    ).clamp_min(2.0**-20).half()
    codes = (
        pages.float() / scale[:, :, None, :, None].float()
    ).round().clamp(-127, 127).to(torch.int8).contiguous()
    # Match the kernel's half conversion followed by half2 multiplication.
    reconstructed = (
        codes.half() * scale[:, :, None, :, None]
    ).half().contiguous()
    return codes, scale.contiguous(), reconstructed


def build_fixture(case: Case, batch_size: int, seed: int) -> dict[str, torch.Tensor]:
    if case.old_tokens % PAGE:
        raise ValueError("old_tokens must be page aligned")
    if case.exact_pages > case.ring_pages:
        raise ValueError("the exact segment cannot exceed ring capacity")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch_size, case.total_pages, PAGE, HKV, DIM)
    centered_k = (
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.08
    ).contiguous()
    centered_v = (
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.08
    ).contiguous()
    query = (
        torch.randn(
            (batch_size, HQ, DIM),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.35
    ).contiguous()
    if case.zero_center:
        value_center = torch.zeros(
            (batch_size, HKV, DIM), device="cuda", dtype=torch.float16
        )
    else:
        value_center = (
            torch.randn(
                (batch_size, HKV, DIM),
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            * 0.025
        ).contiguous()

    old_storage_pages = max(case.old_pages, 1)
    if case.old_pages:
        old_codes_k, old_scale_k, old_reconstructed_k = quantize_old_pages(
            centered_k[:, : case.old_pages]
        )
        old_codes_v, old_scale_v, old_reconstructed_v = quantize_old_pages(
            centered_v[:, : case.old_pages]
        )
    else:
        old_codes_k = torch.full(
            (batch_size, 1, PAGE, HKV, DIM),
            -91,
            device="cuda",
            dtype=torch.int8,
        )
        old_codes_v = torch.full_like(old_codes_k, 73)
        old_scale_k = torch.full(
            (batch_size, 1, HKV), 0.0031, device="cuda", dtype=torch.float16
        )
        old_scale_v = torch.full_like(old_scale_k, 0.0047)
        old_reconstructed_k = torch.empty(
            (batch_size, 0, PAGE, HKV, DIM), device="cuda", dtype=torch.float16
        )
        old_reconstructed_v = torch.empty_like(old_reconstructed_k)
    assert int(old_codes_k.shape[1]) == old_storage_pages

    exact_k = torch.full(
        (batch_size, case.ring_pages, PAGE, HKV, DIM),
        float("nan"),
        device="cuda",
        dtype=torch.float16,
    )
    exact_v = torch.full_like(exact_k, float("nan"))
    unified_indices: list[int] = []
    baseline_indices: list[int] = []
    for request in range(batch_size):
        for logical_page in range(case.total_pages):
            baseline_indices.append(request * case.total_pages + logical_page)
            if logical_page < case.old_pages:
                unified_indices.append(request * old_storage_pages + logical_page)
            else:
                ring_page = logical_page % case.ring_pages
                exact_k[request, ring_page].copy_(centered_k[request, logical_page])
                exact_v[request, ring_page].copy_(centered_v[request, logical_page])
                unified_indices.append(request * case.ring_pages + ring_page)

    logical_k = torch.cat(
        (old_reconstructed_k, centered_k[:, case.old_pages :]), dim=1
    ).contiguous()
    logical_centered_v = torch.cat(
        (old_reconstructed_v, centered_v[:, case.old_pages :]), dim=1
    ).contiguous()
    logical_v = (
        logical_centered_v.float() + value_center[:, None, None].float()
    ).half().contiguous()
    baseline_k = logical_k.reshape(-1, PAGE, HKV, DIM).contiguous()
    baseline_v = logical_v.reshape(-1, PAGE, HKV, DIM).contiguous()

    return {
        "query": query,
        "old_k": old_codes_k.reshape(-1, PAGE, HKV, DIM).contiguous(),
        "old_v": old_codes_v.reshape(-1, PAGE, HKV, DIM).contiguous(),
        "k_scale": old_scale_k.reshape(-1, HKV).contiguous(),
        "v_scale": old_scale_v.reshape(-1, HKV).contiguous(),
        "exact_k": exact_k.reshape(-1, PAGE, HKV, DIM).contiguous(),
        "exact_v": exact_v.reshape(-1, PAGE, HKV, DIM).contiguous(),
        "old_len": torch.full(
            (batch_size,), case.old_tokens, device="cuda", dtype=torch.int32
        ),
        "value_center": value_center.contiguous(),
        "indptr": make_indptr(batch_size, case.total_pages),
        "unified_indices": torch.tensor(
            unified_indices, device="cuda", dtype=torch.int32
        ),
        "baseline_indices": torch.tensor(
            baseline_indices, device="cuda", dtype=torch.int32
        ),
        "last_len": torch.full(
            (batch_size,), case.last_page_len, device="cuda", dtype=torch.int32
        ),
        "baseline_k": baseline_k,
        "baseline_v": baseline_v,
        "logical_k": logical_k,
        "logical_v": logical_v,
    }


def dense_reference(fixture: dict[str, torch.Tensor], total_tokens: int) -> torch.Tensor:
    query = fixture["query"].float()
    key = fixture["logical_k"].flatten(1, 2)[:, :total_tokens].float()
    value = fixture["logical_v"].flatten(1, 2)[:, :total_tokens].float()
    key = key.repeat_interleave(GROUPS, dim=2)
    value = value.repeat_interleave(GROUPS, dim=2)
    score = torch.einsum("bhd,bthd->bht", query, key) / math.sqrt(DIM)
    probability = torch.softmax(score, dim=-1)
    return torch.einsum("bht,bthd->bhd", probability, value).half()


def plan_baseline(
    wrapper: Any,
    fixture: dict[str, torch.Tensor],
    fixed_split_pages: int,
) -> None:
    wrapper.plan(
        fixture["indptr"],
        fixture["baseline_indices"],
        fixture["last_len"],
        HQ,
        HKV,
        DIM,
        PAGE,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        o_data_type=torch.float16,
        sm_scale=1.0 / math.sqrt(DIM),
        fixed_split_size=fixed_split_pages or None,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    splits = parse_splits(args.splits)
    cases = select_cases(args.cases)

    import flashinfer

    baseline_workspace = torch.empty(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    candidate_workspace = torch.empty_like(baseline_workspace)
    baseline = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        baseline_workspace, "NHD", use_tensor_cores=True, backend="fa2"
    )
    candidate = HETERO.make_heterogeneous_page_gauge_wrapper(
        flashinfer, candidate_workspace
    )

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    split_outputs: dict[str, list[tuple[int, torch.Tensor]]] = {}
    torch.cuda.reset_peak_memory_stats()
    for case_index, case in enumerate(cases):
        fixture = build_fixture(case, args.batch_size, args.seed + case_index * 97)
        reference = dense_reference(fixture, case.total_tokens)
        split_outputs[case.name] = []
        for split in splits:
            plan_baseline(baseline, fixture, split)
            HETERO.plan_decode(
                candidate,
                fixture["indptr"],
                fixture["unified_indices"],
                fixture["last_len"],
                fixed_split_pages=split,
            )
            baseline_output = torch.empty_like(fixture["query"])
            candidate_output = torch.empty_like(fixture["query"])
            baseline_lse = torch.empty(
                (args.batch_size, HQ), device="cuda", dtype=torch.float32
            )
            candidate_lse = torch.empty_like(baseline_lse)
            baseline.run(
                fixture["query"],
                (fixture["baseline_k"], fixture["baseline_v"]),
                out=baseline_output,
                lse=baseline_lse,
                return_lse=True,
            )
            HETERO.run_decode(
                candidate,
                fixture["query"],
                fixture["old_k"],
                fixture["old_v"],
                fixture["k_scale"],
                fixture["v_scale"],
                fixture["exact_k"],
                fixture["exact_v"],
                fixture["old_len"],
                fixture["value_center"],
                out=candidate_output,
                lse=candidate_lse,
                return_lse=True,
            )
            torch.cuda.synchronize()
            against_fa2 = metrics(candidate_output, baseline_output)
            against_dense = metrics(candidate_output, reference)
            lse_max_abs = float((candidate_lse - baseline_lse).abs().max())
            passed = (
                against_fa2["max_abs"] <= args.max_abs
                and against_fa2["cosine_min"] >= args.min_cosine
                and lse_max_abs <= args.max_lse_abs
                and bool(torch.isfinite(candidate_output).all())
                and bool(torch.isfinite(candidate_lse).all())
            )
            if not passed:
                failures.append(f"{case.name}/split={split or 'auto'}")
            record = {
                "case": asdict(case),
                "last_page_len": case.last_page_len,
                "fixed_split_pages": split,
                "candidate_vs_reconstructed_fp16_fa2": against_fa2,
                "candidate_vs_dense": against_dense,
                "candidate_vs_fp16_lse_max_abs": lse_max_abs,
                "candidate_output_sha256": tensor_sha256(candidate_output),
                "unified_indices": fixture["unified_indices"].cpu().tolist(),
                "unified_indices_sha256": tensor_sha256(fixture["unified_indices"]),
                "passed": passed,
            }
            records.append(record)
            split_outputs[case.name].append((split, candidate_output.clone()))
            print(
                f"{case.name:25s} split={split or 'auto':>4} "
                f"max={against_fa2['max_abs']:.6f} "
                f"cos={against_fa2['cosine_min']:.8f} "
                f"lse={lse_max_abs:.6f} {'PASS' if passed else 'FAIL'}",
                flush=True,
            )

    split_invariance: list[dict[str, Any]] = []
    for case_name, outputs in split_outputs.items():
        reference_split, reference_output = outputs[0]
        for split, output in outputs[1:]:
            comparison = metrics(output, reference_output)
            passed = (
                comparison["max_abs"] <= args.max_abs
                and comparison["cosine_min"] >= args.min_cosine
            )
            if not passed:
                failures.append(
                    f"{case_name}/split-invariance={reference_split or 'auto'}:{split or 'auto'}"
                )
            split_invariance.append(
                {
                    "case": case_name,
                    "reference_split_pages": reference_split,
                    "candidate_split_pages": split,
                    "metrics": comparison,
                    "passed": passed,
                }
            )

    source_paths = [
        Path(__file__).resolve(),
        Path(HETERO.__file__).resolve(),
        HETERO.VENDOR_HEADER.resolve(),
    ]
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_heterogeneous_fa2_correctness",
        "passed": not failures,
        "failures": failures,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "module": {
            "uri": HETERO.module_uri(),
            **HETERO.source_hashes(),
        },
        "protocol": {
            "batch_size": args.batch_size,
            "splits": splits,
            "num_qo_heads": HQ,
            "num_kv_heads": HKV,
            "head_dim": DIM,
            "page_size": PAGE,
            "layout": "NHD",
            "position_encoding": "NONE",
            "old_kv_len_page_aligned": True,
            "exact_ring_values_centered": True,
            "value_center_restored_inside_split_output_transform": True,
            "thresholds": {
                "max_abs": args.max_abs,
                "min_cosine": args.min_cosine,
                "max_lse_abs": args.max_lse_abs,
            },
        },
        "records": records,
        "split_invariance": split_invariance,
        "memory": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    if failures:
        raise SystemExit(f"correctness gate failed: {failures}")


if __name__ == "__main__":
    main()
