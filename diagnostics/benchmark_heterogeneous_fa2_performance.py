#!/usr/bin/env python3
"""Resident B4x20K gate for PageGauge's heterogeneous FlashInfer FA2 path.

This diagnostic compares three complete, matched attention boundaries over one
coherent synthetic layer:

1. stock FP16 FlashInfer FA2 over the reconstructed logical cache;
2. legacy PageGauge INT8 old-cache attention plus the exact FP16 ring, online
   state merge, and output-center restoration; and
3. the single heterogeneous PageGauge FA2 wrapper.

Every method is captured as one CUDA graph.  Timing therefore measures one
graph replay of the entire method boundary, and the graph inspected for node
counts is exactly the graph being timed.  This is an isolated component gate,
not a transformer or publication result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PAGE = 16
DIM = 128
HQ = 32
HKV = 8
GROUPS = HQ // HKV
METHODS = (
    "flashinfer_fp16",
    "page_gauge_legacy_segmented",
    "page_gauge_heterogeneous",
)
LATIN_ORDERS = (
    METHODS,
    (METHODS[1], METHODS[2], METHODS[0]),
    (METHODS[2], METHODS[0], METHODS[1]),
    (METHODS[0], METHODS[2], METHODS[1]),
    (METHODS[2], METHODS[1], METHODS[0]),
    (METHODS[1], METHODS[0], METHODS[2]),
)


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORACLE = load_local_module(
    "page_gauge_heterogeneous_correctness_fixture_for_performance",
    ROOT / "diagnostics/benchmark_heterogeneous_fa2_correctness.py",
)
HETERO = ORACLE.HETERO
LEGACY = load_local_module(
    "page_gauge_legacy_affine_for_heterogeneous_performance",
    ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=20481)
    parser.add_argument("--exact-tail-capacity", type=int, default=256)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-abs", type=float, default=0.025)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-lse-abs", type=float, default=0.025)
    parser.add_argument("--required-neutral-speedup", type=float, default=1.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/page_gauge_int8/heterogeneous_fa2_b4_20k_performance.json"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.context <= 0:
        raise SystemExit("--context must be positive")
    if args.exact_tail_capacity <= 0 or args.exact_tail_capacity % PAGE:
        raise SystemExit("--exact-tail-capacity must be a positive multiple of 16")
    if args.context <= args.exact_tail_capacity:
        raise SystemExit("the resident gate requires a non-empty old INT8 segment")
    if args.baseline_split_pages <= 0 or args.candidate_split_pages <= 0:
        raise SystemExit("fixed split sizes must be positive")
    if args.warmups < 20 or args.warmups % len(LATIN_ORDERS):
        raise SystemExit("--warmups must be >=20 and divisible by 6")
    if args.repeats < 120 or args.repeats % len(LATIN_ORDERS):
        raise SystemExit("--repeats must be >=120 and divisible by 6")
    if args.cache_scrub_mib <= 0 or args.workspace_mib <= 0:
        raise SystemExit("scrub and workspace sizes must be positive")
    if args.required_neutral_speedup <= 0:
        raise SystemExit("--required-neutral-speedup must be positive")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def sampled_tensor_sha256(tensor: torch.Tensor, samples: int = 4096) -> str:
    """Hash deterministic evenly spaced values without copying a full cache."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() > samples:
        index = torch.linspace(
            0,
            flat.numel() - 1,
            samples,
            device=flat.device,
            dtype=torch.float64,
        ).long()
        flat = flat[index]
    payload = flat.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    metadata = f"{tuple(tensor.shape)}|{tensor.dtype}|{tensor.numel()}".encode()
    return hashlib.sha256(metadata + b"\0" + payload).hexdigest()


def tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "data_ptr": tensor.data_ptr(),
        "bytes": tensor_nbytes(tensor),
        "sampled_sha256": sampled_tensor_sha256(tensor),
    }


def cuda_memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "minimum_ms": min(values),
        "p05_ms": percentile(values, 0.05),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "maximum_ms": max(values),
        "mean_ms": statistics.mean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_ratios(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "arithmetic_mean": statistics.mean(values),
        "geometric_mean": math.exp(statistics.mean(math.log(value) for value in values)),
    }


def output_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = actual_f - expected_f
    relative = difference.norm(dim=-1) / expected_f.norm(dim=-1).clamp_min(1e-8)
    cosine = F.cosine_similarity(actual_f, expected_f, dim=-1)
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2_max": float(relative.max()),
        "relative_l2_mean": float(relative.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
    }


def lse_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = actual.float() - expected.float()
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
    }


def make_indptr(batch_size: int, pages_per_request: int) -> torch.Tensor:
    return torch.arange(
        0,
        (batch_size + 1) * pages_per_request,
        pages_per_request,
        device="cuda",
        dtype=torch.int32,
    )


def make_segment_tables(
    batch_size: int,
    old_pages: int,
    total_pages: int,
    ring_pages: int,
    last_page_len: int,
) -> dict[str, torch.Tensor]:
    exact_pages = total_pages - old_pages
    old_indices = torch.arange(
        batch_size * old_pages, device="cuda", dtype=torch.int32
    )
    exact_indices = torch.tensor(
        [
            request * ring_pages + logical_page % ring_pages
            for request in range(batch_size)
            for logical_page in range(old_pages, total_pages)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    return {
        "old_indptr": make_indptr(batch_size, old_pages),
        "old_indices": old_indices,
        "old_last_len": torch.full(
            (batch_size,), PAGE, device="cuda", dtype=torch.int32
        ),
        "exact_indptr": make_indptr(batch_size, exact_pages),
        "exact_indices": exact_indices,
        "exact_last_len": torch.full(
            (batch_size,), last_page_len, device="cuda", dtype=torch.int32
        ),
    }


def plan_standard(
    wrapper: Any,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_len: torch.Tensor,
    kv_dtype: torch.dtype,
    fixed_split_pages: int,
) -> None:
    wrapper.plan(
        indptr,
        indices,
        last_len,
        HQ,
        HKV,
        DIM,
        PAGE,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=kv_dtype,
        o_data_type=torch.float16,
        sm_scale=1.0 / math.sqrt(DIM),
        fixed_split_size=fixed_split_pages,
    )


def cuda_graph_node_summary(graph: torch.cuda.CUDAGraph) -> dict[str, Any]:
    """Return CUDA Runtime graph-node counts through cuda-python."""
    try:
        from cuda.bindings import runtime

        raw_graph = graph.raw_cuda_graph()
        handle = runtime.cudaGraph_t(int(raw_graph))
        status, _, node_count = runtime.cudaGraphGetNodes(handle, 0)
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphGetNodes(size) returned {status}")
        status, nodes, returned_count = runtime.cudaGraphGetNodes(
            handle, int(node_count)
        )
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphGetNodes(nodes) returned {status}")
        type_counts: dict[str, int] = {}
        for node in nodes[: int(returned_count)]:
            type_status, node_type = runtime.cudaGraphNodeGetType(node)
            if type_status != runtime.cudaError_t.cudaSuccess:
                label = f"query_error_{int(type_status)}"
            else:
                label = getattr(node_type, "name", str(node_type))
            type_counts[label] = type_counts.get(label, 0) + 1
        kernel_count = sum(
            count for label, count in type_counts.items() if "kernel" in label.lower()
        )
        return {
            "available": True,
            "raw_graph_node_count": int(returned_count),
            "kernel_node_count": kernel_count,
            "node_type_counts": type_counts,
        }
    except Exception as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def capture_operations(
    operations: dict[str, Callable[[], None]],
) -> tuple[dict[str, torch.cuda.CUDAGraph], dict[str, dict[str, Any]]]:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            for name in METHODS:
                operations[name]()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name in METHODS:
        # ``keep_graph`` is required to expose the raw cudaGraph_t for the
        # launch/node-parity audit after capture.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=capture_stream):
            operations[name]()
        graphs[name] = graph
        summaries[name] = cuda_graph_node_summary(graph)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graphs, summaries


def paired_speedups(raw: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    baseline = raw["flashinfer_fp16"]
    legacy = raw["page_gauge_legacy_segmented"]
    heterogeneous = raw["page_gauge_heterogeneous"]
    return {
        "legacy_over_flashinfer": summarize_ratios(
            [base / candidate for base, candidate in zip(baseline, legacy)]
        ),
        "heterogeneous_over_flashinfer": summarize_ratios(
            [base / candidate for base, candidate in zip(baseline, heterogeneous)]
        ),
        "heterogeneous_over_legacy": summarize_ratios(
            [old / new for old, new in zip(legacy, heterogeneous)]
        ),
    }


def time_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph],
    scrub: torch.Tensor,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("cache_neutral", "cache_hot"):
        for repeat in range(warmups):
            order = LATIN_ORDERS[repeat % len(LATIN_ORDERS)]
            for name in order:
                if mode == "cache_neutral":
                    scrub.add_(1)
                else:
                    graphs[name].replay()
                graphs[name].replay()
        torch.cuda.synchronize()

        pending: list[
            tuple[int, int, int, str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        for repeat in range(repeats):
            latin_index = repeat % len(LATIN_ORDERS)
            order = LATIN_ORDERS[latin_index]
            for position, name in enumerate(order):
                if mode == "cache_neutral":
                    scrub.add_(1)
                else:
                    graphs[name].replay()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                pending.append((repeat, latin_index, position, name, start, end))
        torch.cuda.synchronize()

        raw = {name: [] for name in METHODS}
        records = []
        for repeat, latin_index, position, name, start, end in pending:
            elapsed = float(start.elapsed_time(end))
            raw[name].append(elapsed)
            records.append(
                {
                    "repeat": repeat,
                    "latin_order_index": latin_index,
                    "position": position,
                    "method": name,
                    "elapsed_ms": elapsed,
                }
            )
        result[mode] = {
            "summaries": {name: summarize_ms(raw[name]) for name in METHODS},
            "paired_speedups": paired_speedups(raw),
            "raw_event_timings_ms_by_method": raw,
            "raw_event_records_in_execution_order": records,
        }
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_grad_enabled(False)

    import flashinfer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory: dict[str, dict[str, int]] = {"entry": cuda_memory_snapshot()}

    total_pages = math.ceil(args.context / PAGE)
    ring_pages = args.exact_tail_capacity // PAGE
    exact_pages = min(total_pages, ring_pages)
    old_pages = total_pages - exact_pages
    old_tokens = old_pages * PAGE
    exact_tokens = args.context - old_tokens
    last_page_len = (args.context - 1) % PAGE + 1
    case = ORACLE.Case(
        "resident_b4_20k",
        old_tokens=old_tokens,
        total_tokens=args.context,
        ring_pages=ring_pages,
    )
    fixture = ORACLE.build_fixture(case, args.batch_size, args.seed)
    tables = make_segment_tables(
        args.batch_size,
        old_pages,
        total_pages,
        ring_pages,
        last_page_len,
    )
    memory["after_coherent_fixture"] = cuda_memory_snapshot()

    workspace_bytes = args.workspace_mib * 1024 * 1024
    baseline_workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    legacy_workspace = torch.empty_like(baseline_workspace)
    exact_workspace = torch.empty_like(baseline_workspace)
    heterogeneous_workspace = torch.empty_like(baseline_workspace)

    # Compile and plan the legacy URI before installing the heterogeneous include
    # hook.  This preserves the old module byte-for-byte as the fallback control.
    baseline = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        baseline_workspace, "NHD", use_tensor_cores=True, backend="fa2"
    )
    legacy = LEGACY.make_page_gauge_wrapper(flashinfer, legacy_workspace)
    exact = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        exact_workspace, "NHD", use_tensor_cores=True, backend="fa2"
    )
    ORACLE.plan_baseline(baseline, fixture, args.baseline_split_pages)
    plan_standard(
        legacy,
        tables["old_indptr"],
        tables["old_indices"],
        tables["old_last_len"],
        torch.int8,
        args.candidate_split_pages,
    )
    plan_standard(
        exact,
        tables["exact_indptr"],
        tables["exact_indices"],
        tables["exact_last_len"],
        torch.float16,
        args.candidate_split_pages,
    )

    query = fixture["query"]
    output_center = fixture["value_center"].repeat_interleave(GROUPS, dim=1).contiguous()
    baseline_output = torch.empty_like(query)
    baseline_lse = torch.empty(
        (args.batch_size, HQ), device="cuda", dtype=torch.float32
    )
    legacy_output = torch.empty_like(query)
    legacy_lse = torch.empty_like(baseline_lse)
    exact_output = torch.empty_like(query)
    exact_lse = torch.empty_like(baseline_lse)

    sm_scale = 1.0 / math.sqrt(DIM)

    def run_baseline() -> None:
        baseline.run(
            query,
            (fixture["baseline_k"], fixture["baseline_v"]),
            out=baseline_output,
            lse=baseline_lse,
            return_lse=True,
        )

    def run_legacy() -> None:
        legacy.run(
            query,
            (fixture["old_k"], fixture["old_v"]),
            fixture["k_scale"],
            fixture["v_scale"],
            sm_scale,
            out=legacy_output,
            lse=legacy_lse,
            return_lse=True,
        )
        exact.run(
            query,
            (fixture["exact_k"], fixture["exact_v"]),
            out=exact_output,
            lse=exact_lse,
            return_lse=True,
        )
        flashinfer.merge_state_in_place(
            legacy_output, legacy_lse, exact_output, exact_lse
        )
        legacy_output.add_(output_center)

    # Force the old module to load before the heterogeneous include hook exists.
    run_baseline()
    run_legacy()
    torch.cuda.synchronize()

    heterogeneous = HETERO.make_heterogeneous_page_gauge_wrapper(
        flashinfer, heterogeneous_workspace
    )
    HETERO.plan_decode(
        heterogeneous,
        fixture["indptr"],
        fixture["unified_indices"],
        fixture["last_len"],
        fixed_split_pages=args.candidate_split_pages,
    )
    heterogeneous_output = torch.empty_like(query)
    heterogeneous_lse = torch.empty_like(baseline_lse)

    def run_heterogeneous() -> None:
        # Raw wrapper call avoids validation synchronizations inside the timed graph.
        heterogeneous.run(
            query,
            (fixture["old_k"], fixture["old_v"]),
            fixture["k_scale"],
            fixture["v_scale"],
            fixture["exact_k"],
            fixture["exact_v"],
            fixture["old_len"],
            fixture["value_center"],
            sm_scale,
            out=heterogeneous_output,
            lse=heterogeneous_lse,
            return_lse=True,
        )

    operations = {
        "flashinfer_fp16": run_baseline,
        "page_gauge_legacy_segmented": run_legacy,
        "page_gauge_heterogeneous": run_heterogeneous,
    }
    for name in METHODS:
        operations[name]()
    torch.cuda.synchronize()
    memory["after_wrappers_and_jit"] = cuda_memory_snapshot()

    eager_correctness = {
        "legacy_vs_flashinfer": {
            "output": output_metrics(legacy_output, baseline_output),
            "lse": lse_metrics(legacy_lse, baseline_lse),
        },
        "heterogeneous_vs_flashinfer": {
            "output": output_metrics(heterogeneous_output, baseline_output),
            "lse": lse_metrics(heterogeneous_lse, baseline_lse),
        },
        "heterogeneous_vs_legacy": {
            "output": output_metrics(heterogeneous_output, legacy_output),
            "lse": lse_metrics(heterogeneous_lse, legacy_lse),
        },
    }

    graphs, node_summaries = capture_operations(operations)
    for name in METHODS:
        graphs[name].replay()
    torch.cuda.synchronize()
    memory["after_cuda_graph_capture"] = cuda_memory_snapshot()

    graph_correctness = {
        "legacy_vs_flashinfer": {
            "output": output_metrics(legacy_output, baseline_output),
            "lse": lse_metrics(legacy_lse, baseline_lse),
        },
        "heterogeneous_vs_flashinfer": {
            "output": output_metrics(heterogeneous_output, baseline_output),
            "lse": lse_metrics(heterogeneous_lse, baseline_lse),
        },
        "heterogeneous_vs_legacy": {
            "output": output_metrics(heterogeneous_output, legacy_output),
            "lse": lse_metrics(heterogeneous_lse, legacy_lse),
        },
        "eager_vs_graph_bitwise": {
            # The eager tensors are overwritten by graph replay.  Determinism is
            # instead gated by a second replay and cloned graph outputs below.
        },
    }
    first_graph_outputs = {
        "flashinfer_fp16": (baseline_output.clone(), baseline_lse.clone()),
        "page_gauge_legacy_segmented": (legacy_output.clone(), legacy_lse.clone()),
        "page_gauge_heterogeneous": (
            heterogeneous_output.clone(),
            heterogeneous_lse.clone(),
        ),
    }
    for name in METHODS:
        graphs[name].replay()
    torch.cuda.synchronize()
    current_outputs = {
        "flashinfer_fp16": (baseline_output, baseline_lse),
        "page_gauge_legacy_segmented": (legacy_output, legacy_lse),
        "page_gauge_heterogeneous": (heterogeneous_output, heterogeneous_lse),
    }
    graph_correctness["graph_replay_determinism"] = {
        name: {
            "output_bitwise_equal": bool(
                torch.equal(first_graph_outputs[name][0], current_outputs[name][0])
            ),
            "lse_bitwise_equal": bool(
                torch.equal(first_graph_outputs[name][1], current_outputs[name][1])
            ),
        }
        for name in METHODS
    }
    del graph_correctness["eager_vs_graph_bitwise"]

    scrub = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024 // 4,
        device="cuda",
        dtype=torch.int32,
    )
    memory["before_timing"] = cuda_memory_snapshot()
    timings = time_graphs(graphs, scrub, args.warmups, args.repeats)
    memory["after_timing"] = cuda_memory_snapshot()

    cache_payload = {
        "flashinfer_fp16_bytes": tensor_nbytes(fixture["baseline_k"])
        + tensor_nbytes(fixture["baseline_v"]),
        "page_gauge_bytes": sum(
            tensor_nbytes(fixture[name])
            for name in (
                "old_k",
                "old_v",
                "k_scale",
                "v_scale",
                "exact_k",
                "exact_v",
                "value_center",
            )
        ),
    }
    cache_payload["page_gauge_fraction_of_fp16"] = (
        cache_payload["page_gauge_bytes"] / cache_payload["flashinfer_fp16_bytes"]
    )
    cache_payload["page_gauge_reduction_percent"] = 100.0 * (
        1.0 - cache_payload["page_gauge_fraction_of_fp16"]
    )

    descriptors = {
        name: tensor_descriptor(fixture[name])
        for name in (
            "query",
            "old_k",
            "old_v",
            "k_scale",
            "v_scale",
            "exact_k",
            "exact_v",
            "old_len",
            "value_center",
            "unified_indices",
            "baseline_indices",
        )
    }
    descriptors["shared_candidate_storage"] = {
        "legacy_and_heterogeneous_use_the_same_tensor_objects": True,
        "legacy_old_k_data_ptr": fixture["old_k"].data_ptr(),
        "heterogeneous_old_k_data_ptr": fixture["old_k"].data_ptr(),
        "legacy_exact_k_data_ptr": fixture["exact_k"].data_ptr(),
        "heterogeneous_exact_k_data_ptr": fixture["exact_k"].data_ptr(),
    }

    correctness_failures: list[str] = []
    for boundary, comparisons in (
        ("eager", eager_correctness),
        ("graph", graph_correctness),
    ):
        for comparison_name in ("legacy_vs_flashinfer", "heterogeneous_vs_flashinfer"):
            comparison = comparisons[comparison_name]
            if comparison["output"]["max_abs"] > args.max_abs:
                correctness_failures.append(f"{boundary}/{comparison_name}/max_abs")
            if comparison["output"]["cosine_min"] < args.min_cosine:
                correctness_failures.append(f"{boundary}/{comparison_name}/cosine")
            if comparison["lse"]["max_abs"] > args.max_lse_abs:
                correctness_failures.append(f"{boundary}/{comparison_name}/lse")
    for name, gate in graph_correctness["graph_replay_determinism"].items():
        if not gate["output_bitwise_equal"] or not gate["lse_bitwise_equal"]:
            correctness_failures.append(f"graph_replay_determinism/{name}")

    node_failures = [
        name for name, summary in node_summaries.items() if not summary["available"]
    ]
    node_parity = False
    if not node_failures:
        node_parity = (
            node_summaries["page_gauge_heterogeneous"]["kernel_node_count"]
            == node_summaries["flashinfer_fp16"]["kernel_node_count"]
        )
        if not node_parity:
            node_failures.append("heterogeneous_kernel_node_parity")

    neutral_speedup = timings["cache_neutral"]["paired_speedups"][
        "heterogeneous_over_flashinfer"
    ]["geometric_mean"]
    hot_speedup = timings["cache_hot"]["paired_speedups"][
        "heterogeneous_over_flashinfer"
    ]["geometric_mean"]
    performance_passed = neutral_speedup >= args.required_neutral_speedup
    failures = [
        *correctness_failures,
        *node_failures,
        *(
            []
            if performance_passed
            else [
                f"cache_neutral_speedup={neutral_speedup:.6f}"
                f"<{args.required_neutral_speedup:.6f}"
            ]
        ),
    ]

    source_paths = {
        "performance_diagnostic": Path(__file__).resolve(),
        "correctness_fixture": Path(ORACLE.__file__).resolve(),
        "heterogeneous_wrapper": Path(HETERO.__file__).resolve(),
        "legacy_wrapper": Path(LEGACY.__file__).resolve(),
        "heterogeneous_header": HETERO.VENDOR_HEADER.resolve(),
        "legacy_header": (
            ROOT
            / "build/gauge_affine_flashinfer/include/flashinfer/attention/prefill.cuh"
        ).resolve(),
    }
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_heterogeneous_fa2_resident_performance_gate",
        "scope": (
            "single coherent synthetic layer; one complete CUDA graph replay per "
            "method; throughput-oriented component localization, not transformer latency"
        ),
        "passed": not failures,
        "failures": failures,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "workload": {
            "batch_size": args.batch_size,
            "context_tokens_per_request": args.context,
            "total_pages_per_request": total_pages,
            "old_int8_pages_per_request": old_pages,
            "old_int8_tokens_per_request": old_tokens,
            "exact_ring_pages_per_request": ring_pages,
            "exact_valid_pages_per_request": exact_pages,
            "exact_valid_tokens_per_request": exact_tokens,
            "last_page_len": last_page_len,
            "num_qo_heads": HQ,
            "num_kv_heads": HKV,
            "head_dim": DIM,
            "page_size": PAGE,
            "layout": "NHD",
            "position_encoding": "NONE",
            "baseline_split_pages": args.baseline_split_pages,
            "candidate_split_pages": args.candidate_split_pages,
            "seed": args.seed,
        },
        "method_boundaries": {
            "flashinfer_fp16": "FP16 FA2 main attention plus normal split-K reduction",
            "page_gauge_legacy_segmented": (
                "old INT8 FA2 plus exact FP16 FA2 plus merge_state_in_place plus center add"
            ),
            "page_gauge_heterogeneous": (
                "one heterogeneous INT8/FP16 FA2 main attention plus normal split-K reduction; "
                "center restored inside split output transform"
            ),
            "timed_execution": "one replay of the captured complete-method CUDA graph",
            "python_dispatches_per_boundary": {
                "flashinfer_fp16": 1,
                "page_gauge_legacy_segmented": 4,
                "page_gauge_heterogeneous": 1,
            },
        },
        "correctness_thresholds": {
            "max_abs": args.max_abs,
            "min_cosine": args.min_cosine,
            "max_lse_abs": args.max_lse_abs,
        },
        "eager_correctness": eager_correctness,
        "graph_correctness": graph_correctness,
        "cuda_graph_nodes": node_summaries,
        "heterogeneous_kernel_node_parity_with_flashinfer": node_parity,
        "timing_protocol": {
            "warmups_per_mode_per_method": args.warmups,
            "timed_repeats_per_mode_per_method": args.repeats,
            "method_order": "balanced six-order three-way Latin schedule",
            "latin_orders": [list(order) for order in LATIN_ORDERS],
            "cache_neutral": (
                "touch unrelated GPU buffer before each timed graph; scrub is outside event"
            ),
            "cache_hot": "replay the same method immediately before its timed graph",
            "cache_scrub_bytes": tensor_nbytes(scrub),
            "primary_gate": "cache-neutral paired geometric-mean speedup",
            "required_neutral_speedup": args.required_neutral_speedup,
        },
        "timings": timings,
        "performance_gate": {
            "required_cache_neutral_speedup": args.required_neutral_speedup,
            "observed_cache_neutral_speedup": neutral_speedup,
            "observed_cache_hot_speedup": hot_speedup,
            "passed": performance_passed,
        },
        "cache_payload": cache_payload,
        "memory": memory,
        "workspace_bytes_per_wrapper": workspace_bytes,
        "input_provenance": descriptors,
        "module": {
            "legacy_uri": "page_gauge_int8_fa2_v3",
            "heterogeneous_uri": HETERO.module_uri(),
            **HETERO.source_hashes(),
        },
        "source_sha256": {
            name: file_sha256(path) for name, path in source_paths.items()
        },
        "command": [sys.executable, *sys.argv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}", flush=True)
    for mode in ("cache_neutral", "cache_hot"):
        print(mode, flush=True)
        for name in METHODS:
            p50 = timings[mode]["summaries"][name]["p50_ms"]
            print(f"  {name:31s} p50={p50:.6f} ms", flush=True)
        speedup = timings[mode]["paired_speedups"][
            "heterogeneous_over_flashinfer"
        ]["geometric_mean"]
        print(f"  heterogeneous/FI paired geomean={speedup:.6f}x", flush=True)
    print(f"graph nodes: {json.dumps(node_summaries, sort_keys=True)}", flush=True)
    if failures:
        raise SystemExit(f"performance gate failed: {failures}")


if __name__ == "__main__":
    main()
