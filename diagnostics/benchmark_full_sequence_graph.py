#!/usr/bin/env python3
"""Matched whole-sequence CUDA-graph diagnostic for PageGauge.

This is deliberately a static-token, one-page launch-overhead experiment.  It
captures the same complete 16-step decoder boundary for the FP16 baseline and
PageGauge: embedding, every decoder layer, cache append/finalization,
attention, projections, MLPs, final norm, LM head, and FP32 logits conversion.
It is not a production autoregressive-throughput claim because token feedback,
sampling, and dynamic graph-input updates are outside the captured boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/benchmark_page_gauge_transformer.py"
APPEND_PATH = ROOT / "tests/page_gauge_append_extension.cu"
DEPENDENCY_PATHS = (
    RUNNER_PATH,
    APPEND_PATH,
    ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
    ROOT / "scripts/benchmark_page_gauge_overheads.py",
    ROOT / "scripts/benchmark_e2e_transformer.py",
    ROOT / "scripts/page_gauge_runtime.py",
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "page_gauge_full_sequence_graph_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PG = load_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--baseline-split-pages", type=int, default=64)
    parser.add_argument("--candidate-split-pages", type=int, default=0)
    parser.add_argument(
        "--center-restore",
        choices=("attention_add", "projection_bias"),
        default="attention_add",
    )
    parser.add_argument(
        "--tail-attention",
        choices=("fused_kernel", "flashinfer_merge"),
        default="flashinfer_merge",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--capture-warmups", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jsonable_signature(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable_signature(item) for item in value]
    if isinstance(value, list):
        return [jsonable_signature(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def snapshot_mutated_page(decoder, start_position: int) -> dict[str, Any]:
    """Clone every persistent cache tensor touched by one aligned page."""
    logical_page, offset = divmod(start_position, PG.PAGE)
    if offset:
        raise ValueError("the full-sequence graph must begin on a page boundary")
    if decoder.backend == "flashinfer_fp16":
        physical_pages = tuple(
            request * decoder.max_pages + logical_page
            for request in range(decoder.batch_size)
        )
        return {
            "backend": decoder.backend,
            "logical_page": logical_page,
            "ring_page": None,
            "physical_pages": {"cache": physical_pages},
            "tensors": {
                "key": torch.stack(
                    [decoder.cache.key[:, page] for page in physical_pages],
                    dim=1,
                ).clone(),
                "value": torch.stack(
                    [decoder.cache.value[:, page] for page in physical_pages],
                    dim=1,
                ).clone(),
            },
        }
    ring_page = logical_page % decoder.exact_tail_pages
    exact_physical_pages = tuple(
        request * decoder.exact_tail_pages + ring_page
        for request in range(decoder.batch_size)
    )
    code_physical_pages = tuple(
        request * decoder.max_pages + logical_page
        for request in range(decoder.batch_size)
    )
    return {
        "backend": decoder.backend,
        "logical_page": logical_page,
        "ring_page": ring_page,
        "physical_pages": {
            "exact": exact_physical_pages,
            "codes": code_physical_pages,
        },
        "tensors": {
            "exact_key": torch.stack(
                [decoder.cache.exact_key[:, page] for page in exact_physical_pages],
                dim=1,
            ).clone(),
            "exact_value": torch.stack(
                [decoder.cache.exact_value[:, page] for page in exact_physical_pages],
                dim=1,
            ).clone(),
            "key_codes": torch.stack(
                [decoder.cache.key_codes[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
            "value_codes": torch.stack(
                [decoder.cache.value_codes[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
            "key_scales": torch.stack(
                [decoder.cache.key_scales[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
            "value_scales": torch.stack(
                [decoder.cache.value_scales[:, page] for page in code_physical_pages],
                dim=1,
            ).clone(),
        },
    }


def restore_mutated_page(decoder, snapshot: dict[str, Any]) -> None:
    if snapshot["backend"] != decoder.backend:
        raise ValueError("cache snapshot belongs to a different backend")
    physical_pages = snapshot["physical_pages"]
    tensors = snapshot["tensors"]
    if decoder.backend == "flashinfer_fp16":
        for request, page in enumerate(physical_pages["cache"]):
            decoder.cache.key[:, page].copy_(tensors["key"][:, request])
            decoder.cache.value[:, page].copy_(tensors["value"][:, request])
        return
    for request, page in enumerate(physical_pages["exact"]):
        decoder.cache.exact_key[:, page].copy_(tensors["exact_key"][:, request])
        decoder.cache.exact_value[:, page].copy_(tensors["exact_value"][:, request])
    for request, page in enumerate(physical_pages["codes"]):
        decoder.cache.key_codes[:, page].copy_(tensors["key_codes"][:, request])
        decoder.cache.value_codes[:, page].copy_(
            tensors["value_codes"][:, request]
        )
        decoder.cache.key_scales[:, page].copy_(
            tensors["key_scales"][:, request]
        )
        decoder.cache.value_scales[:, page].copy_(
            tensors["value_scales"][:, request]
        )


def compare_cache_snapshots(
    expected: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    if (
        expected["backend"] != observed["backend"]
        or expected["logical_page"] != observed["logical_page"]
        or expected["ring_page"] != observed["ring_page"]
        or expected["physical_pages"] != observed["physical_pages"]
    ):
        return {
            "passed": False,
            "metadata_match": False,
            "tensor_results": {},
        }
    expected_tensors = expected["tensors"]
    observed_tensors = observed["tensors"]
    if set(expected_tensors) != set(observed_tensors):
        return {
            "passed": False,
            "metadata_match": True,
            "tensor_key_match": False,
            "tensor_results": {},
        }
    tensor_results = {}
    for name in sorted(expected_tensors):
        reference = expected_tensors[name]
        result = observed_tensors[name]
        exact = bool(torch.equal(reference, result))
        maximum_absolute_error = float(
            (reference.float() - result.float()).abs().max().item()
        )
        tensor_results[name] = {
            "bitwise_identical": exact,
            "maximum_absolute_error": maximum_absolute_error,
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
        }
    return {
        "passed": all(
            value["bitwise_identical"] for value in tensor_results.values()
        ),
        "metadata_match": True,
        "tensor_key_match": True,
        "tensor_results": tensor_results,
    }


@torch.inference_mode()
def execute_static_sequence(
    decoder, token_tensor: torch.Tensor, start_position: int
) -> list[torch.Tensor]:
    if (
        token_tensor.dim() != 2
        or int(token_tensor.shape[1]) != decoder.batch_size
    ):
        raise ValueError("static tokens must have synchronized shape [steps,B]")
    return [
        decoder.step(token_tensor[offset], start_position + offset)[0]
        for offset in range(int(token_tensor.shape[0]))
    ]


def clone_outputs(outputs: list[torch.Tensor]) -> list[torch.Tensor]:
    clones = [output.detach().clone() for output in outputs]
    torch.cuda.synchronize()
    return clones


def assert_plan_bucket_is_invariant(decoder, start_position: int, steps: int) -> Any:
    signatures = []
    for offset in range(steps):
        decoder.plan(start_position + offset + 1)
        signatures.append(decoder.plan_signature())
    torch.cuda.synchronize()
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(
            f"{decoder.backend} changes FlashInfer plan bucket inside the page"
        )
    return signatures[0]


class CapturedSequence:
    def __init__(
        self,
        decoder,
        graph: torch.cuda.CUDAGraph,
        replay_stream: torch.cuda.Stream,
        token_tensor: torch.Tensor,
        outputs: list[torch.Tensor],
        start_position: int,
        plan_signature: Any,
    ) -> None:
        self.decoder = decoder
        self.graph = graph
        self.replay_stream = replay_stream
        self.token_tensor = token_tensor
        self.outputs = outputs
        self.start_position = start_position
        self.plan_signature = plan_signature

    def assert_replay_safe(self) -> None:
        if self.decoder.attention_graphs is not None:
            raise RuntimeError("nested per-layer attention graphs must stay disabled")
        if self.decoder.plan_signature() != self.plan_signature:
            raise RuntimeError("full-sequence graph plan signature is stale")

    def replay(self) -> None:
        self.assert_replay_safe()
        # Keep event ordering unambiguous even on runtimes that retain the
        # original capture stream as the CUDA graph's launch stream.
        with torch.cuda.stream(self.replay_stream):
            self.graph.replay()


def capture_full_sequence(
    decoder,
    token_tensor: torch.Tensor,
    start_position: int,
    initial_cache: dict[str, Any],
    capture_warmups: int,
) -> CapturedSequence:
    # The outer graph must contain the real attention kernels. Capturing calls
    # to the already-captured per-layer graphs would change the graph boundary.
    decoder.attention_graphs = None
    plan_signature = assert_plan_bucket_is_invariant(
        decoder, start_position, int(token_tensor.shape[0])
    )

    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)
    warm_outputs = None
    with torch.cuda.stream(capture_stream):
        for _ in range(capture_warmups):
            warm_outputs = execute_static_sequence(
                decoder, token_tensor, start_position
            )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    del warm_outputs

    # Capture and eager validation must start from the same logical cache.
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    capture_stream.wait_stream(current_stream)
    graph = torch.cuda.CUDAGraph()
    graph_outputs: list[torch.Tensor] = []
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_outputs = execute_static_sequence(
            decoder, token_tensor, start_position
        )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    return CapturedSequence(
        decoder,
        graph,
        capture_stream,
        token_tensor,
        graph_outputs,
        start_position,
        plan_signature,
    )


def validate_backend_graph(
    captured: CapturedSequence, initial_cache: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[torch.Tensor]]]:
    decoder = captured.decoder
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    eager_outputs = clone_outputs(
        execute_static_sequence(
            decoder, captured.token_tensor, captured.start_position
        )
    )
    eager_cache = snapshot_mutated_page(decoder, captured.start_position)
    torch.cuda.synchronize()

    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    captured.replay()
    torch.cuda.synchronize()
    first_graph_outputs = clone_outputs(captured.outputs)
    first_graph_cache = snapshot_mutated_page(decoder, captured.start_position)
    torch.cuda.synchronize()

    # Replaying without a restore exercises the same steady-state behavior as
    # the repeated timing loop. Causal last-page lengths make future slots
    # invisible until their corresponding graph nodes overwrite them.
    captured.replay()
    torch.cuda.synchronize()
    repeated_graph_outputs = clone_outputs(captured.outputs)
    repeated_graph_cache = snapshot_mutated_page(
        decoder, captured.start_position
    )
    torch.cuda.synchronize()

    eager_vs_graph_logits = PG.compare_logit_sequences(
        eager_outputs, first_graph_outputs
    )
    repeated_graph_logits = PG.compare_logit_sequences(
        first_graph_outputs, repeated_graph_outputs
    )
    eager_vs_graph_cache = compare_cache_snapshots(
        eager_cache, first_graph_cache
    )
    repeated_graph_cache_result = compare_cache_snapshots(
        first_graph_cache, repeated_graph_cache
    )
    passed = (
        eager_vs_graph_logits["bitwise_identical"]
        and repeated_graph_logits["bitwise_identical"]
        and eager_vs_graph_cache["passed"]
        and repeated_graph_cache_result["passed"]
    )
    restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    return (
        {
            "passed": passed,
            "eager_vs_graph_logits": eager_vs_graph_logits,
            "repeated_graph_logits": repeated_graph_logits,
            "eager_vs_graph_cache": eager_vs_graph_cache,
            "repeated_graph_cache": repeated_graph_cache_result,
        },
        {
            "eager": eager_outputs,
            "graph": first_graph_outputs,
        },
    )


def compare_backends(
    baseline: list[torch.Tensor], candidate: list[torch.Tensor]
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("backend output sequences must have equal non-zero length")
    batch_size = int(baseline[0].shape[0])
    if any(
        reference.shape != approximation.shape
        or reference.dim() != 2
        or int(reference.shape[0]) != batch_size
        for reference, approximation in zip(baseline, candidate)
    ):
        raise ValueError("backend logits must have matching [B,V] shapes")
    per_step_cosines = [
        [
            float(value)
            for value in F.cosine_similarity(
                reference, approximation, dim=-1
            ).tolist()
        ]
        for reference, approximation in zip(baseline, candidate)
    ]
    per_step_top1 = [
        [
            int(value)
            for value in reference.argmax(dim=-1)
            .eq(approximation.argmax(dim=-1))
            .tolist()
        ]
        for reference, approximation in zip(baseline, candidate)
    ]
    cosines = [value for step in per_step_cosines for value in step]
    top1 = [value for step in per_step_top1 for value in step]
    max_errors = [
        float((reference - approximation).abs().max().item())
        for reference, approximation in zip(baseline, candidate)
    ]
    return {
        "checked_steps": len(baseline),
        "batch_size": batch_size,
        "checked_request_steps": len(baseline) * batch_size,
        "minimum_logits_cosine": min(cosines),
        "mean_logits_cosine": statistics.mean(cosines),
        "top1_agreement_fraction": sum(top1) / len(top1),
        "maximum_absolute_logit_error": max(max_errors),
        "per_step_per_request_logits_cosine": per_step_cosines,
        "per_step_per_request_top1_agreement": per_step_top1,
    }


def replay_and_synchronize(captured: CapturedSequence) -> None:
    captured.replay()
    torch.cuda.synchronize()


def timed_replay(captured: CapturedSequence) -> tuple[float, float]:
    captured.assert_replay_safe()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    with torch.cuda.stream(captured.replay_stream):
        begin.record()
        # The signature guard is intentionally outside the timed static-graph
        # boundary. A production dynamic-input path must time its metadata
        # updates separately.
        captured.graph.replay()
        end.record()
    torch.cuda.synchronize()
    return (
        float(begin.elapsed_time(end)),
        (time.perf_counter() - wall_start) * 1e3,
    )


def summarize_mode(
    samples: dict[str, dict[str, list[float]]],
    blocks: list[dict[str, Any]],
    steps: int,
    batch_size: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summaries = {
        backend: PG.summarize_sequence(
            values["gpu"], values["wall"], steps, batch_size
        )
        for backend, values in samples.items()
    }
    wall_speedups = [
        baseline / candidate
        for baseline, candidate in zip(
            samples["flashinfer_fp16"]["wall"],
            samples["page_gauge"]["wall"],
        )
    ]
    gpu_speedups = [
        baseline / candidate
        for baseline, candidate in zip(
            samples["flashinfer_fp16"]["gpu"],
            samples["page_gauge"]["gpu"],
        )
    ]
    ab_wall = wall_speedups[0::2]
    ba_wall = wall_speedups[1::2]
    midpoint = len(wall_speedups) // 2
    summaries.update(
        {
            "page_gauge_speedup_wall_geomean": PG.geometric_mean(
                wall_speedups
            ),
            "page_gauge_speedup_gpu_geomean": PG.geometric_mean(gpu_speedups),
            "ratio_of_mean_wall_tokens_per_second": (
                summaries["page_gauge"]["wall_tokens_per_second"]
                / summaries["flashinfer_fp16"]["wall_tokens_per_second"]
            ),
            "paired_speedups": {
                "wall": wall_speedups,
                "gpu": gpu_speedups,
                "wall_paired_bootstrap_95_ci": PG.bootstrap_geomean(
                    wall_speedups, bootstrap_seed
                ),
                "gpu_paired_bootstrap_95_ci": PG.bootstrap_geomean(
                    gpu_speedups, bootstrap_seed + 1
                ),
                "order_stratified_wall_geomean": {
                    "AB": PG.geometric_mean(ab_wall),
                    "BA": PG.geometric_mean(ba_wall),
                },
                "time_stratified_wall_geomean": {
                    "first_half": PG.geometric_mean(wall_speedups[:midpoint]),
                    "second_half": PG.geometric_mean(wall_speedups[midpoint:]),
                },
            },
            "balanced_blocks": blocks,
        }
    )
    return summaries


def measure_graph_modes(
    captured: dict[str, CapturedSequence],
    initial_caches: dict[str, dict[str, Any]],
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    names = ["flashinfer_fp16", "page_gauge"]
    for mode_index, mode in enumerate(("cache_neutral_start", "cache_hot")):
        for name in names:
            restore_mutated_page(captured[name].decoder, initial_caches[name])
        torch.cuda.synchronize()

        for warmup in range(warmups):
            order = names if warmup % 2 == 0 else list(reversed(names))
            for name in order:
                if mode == "cache_neutral_start":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    replay_and_synchronize(captured[name])
                replay_and_synchronize(captured[name])

        samples = {
            name: {"gpu": [], "wall": []}
            for name in names
        }
        blocks = []
        measurement_start = time.perf_counter()
        for repeat in range(repeats):
            order = names if repeat % 2 == 0 else list(reversed(names))
            block: dict[str, Any] = {
                "repeat": repeat,
                "order": "AB" if repeat % 2 == 0 else "BA",
                "start_seconds": time.perf_counter() - measurement_start,
                "measurements": {},
            }
            for name in order:
                if mode == "cache_neutral_start":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    replay_and_synchronize(captured[name])
                gpu_ms, wall_ms = timed_replay(captured[name])
                samples[name]["gpu"].append(gpu_ms)
                samples[name]["wall"].append(wall_ms)
                block["measurements"][name] = {
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                }
            block["end_seconds"] = time.perf_counter() - measurement_start
            blocks.append(block)
        result[mode] = summarize_mode(
            samples,
            blocks,
            steps,
            batch_size,
            seed + mode_index * 100,
        )
    return result


def make_decoders(args: argparse.Namespace):
    model_args = SimpleNamespace(
        model=args.model,
        context=args.context,
        decode_steps=args.decode_steps,
        local_files_only=args.local_files_only,
    )
    model = PG.load_model(model_args)
    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    layers, _, hkv, hidden = PG.BASE_E2E.check_model(model)
    max_context = args.context + args.decode_steps + 1
    pages = math.ceil(max_context / PG.PAGE)
    initial_pages = args.context // PG.PAGE
    exact_pages = args.exact_tail // PG.PAGE
    with torch.inference_mode():
        positions = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(probe, positions)
        if rope_cos.dim() == 3:
            rope_cos, rope_sin = rope_cos[0], rope_sin[0]
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()
    baseline_cache, gauge_cache = PG.build_caches(
        layers,
        pages,
        initial_pages,
        exact_pages,
        hkv,
        args.seed + 1000,
        args.batch_size,
    )
    common = (model, flashinfer, append_extension)
    baseline = PG.TransformerDecoder(
        *common,
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
    candidate = PG.TransformerDecoder(
        *common,
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
    )
    return model, baseline, candidate, PG.cache_storage(baseline_cache, gauge_cache)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.decode_steps != PG.PAGE:
        raise SystemExit("the whole-sequence diagnostic requires exactly 16 steps")
    if (
        args.context <= args.exact_tail
        or args.context % PG.PAGE
        or args.exact_tail % PG.PAGE
    ):
        raise SystemExit(
            "context and exact tail must be page aligned; context must exceed tail"
        )
    if args.repeats <= 0 or args.repeats % 2:
        raise SystemExit("repeats must be a positive even number for AB/BA balance")
    if args.warmups < 0 or args.capture_warmups <= 0:
        raise SystemExit("warmups are invalid")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    if (
        args.tail_attention == "fused_kernel"
        and args.center_restore != "attention_add"
    ):
        raise SystemExit(
            "the fused tail kernel requires --center-restore attention_add"
        )
    if args.batch_size > 1 and args.center_restore == "projection_bias":
        raise SystemExit(
            "request-specific value centers require --center-restore "
            "attention_add for batch size greater than one"
        )

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    source_paths = (*DEPENDENCY_PATHS, Path(__file__))
    source_hashes = {
        str(path.relative_to(ROOT)): source_sha256(path)
        for path in source_paths
    }
    torch.cuda.reset_peak_memory_stats()
    print("Loading model, caches, and current attention kernels...", flush=True)
    model, baseline, candidate, cache_storage = make_decoders(args)
    decoders = {
        "flashinfer_fp16": baseline,
        "page_gauge": candidate,
    }

    generator = torch.Generator().manual_seed(args.seed)
    token_tensor = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.decode_steps, args.batch_size),
        generator=generator,
        dtype=torch.long,
    ).to(device="cuda")
    tokens = token_tensor.cpu().tolist()
    initial_caches = {
        name: snapshot_mutated_page(decoder, args.context)
        for name, decoder in decoders.items()
    }
    torch.cuda.synchronize()

    capture_memory = {}
    captured = {}
    for name, decoder in decoders.items():
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        print(f"Capturing matched full sequence: {name}", flush=True)
        captured[name] = capture_full_sequence(
            decoder,
            token_tensor,
            args.context,
            initial_caches[name],
            args.capture_warmups,
        )
        capture_memory[name] = {
            "allocated_before": before_allocated,
            "allocated_after": int(torch.cuda.memory_allocated()),
            "allocated_delta": int(torch.cuda.memory_allocated()) - before_allocated,
            "reserved_before": before_reserved,
            "reserved_after": int(torch.cuda.memory_reserved()),
            "reserved_delta": int(torch.cuda.memory_reserved()) - before_reserved,
        }

    same_backend_validation = {}
    validation_outputs = {}
    for name in decoders:
        validation, outputs = validate_backend_graph(
            captured[name], initial_caches[name]
        )
        same_backend_validation[name] = validation
        validation_outputs[name] = outputs

    cross_backend_validation = {
        mode: compare_backends(
            validation_outputs["flashinfer_fp16"][mode],
            validation_outputs["page_gauge"][mode],
        )
        for mode in ("eager", "graph")
    }
    cross_backend_passed = all(
        result["minimum_logits_cosine"] >= args.min_logits_cosine
        and result["top1_agreement_fraction"] >= args.min_top1_agreement
        for result in cross_backend_validation.values()
    )
    graph_equivalence_passed = all(
        result["passed"] for result in same_backend_validation.values()
    )

    timing_modes: dict[str, Any] = {}
    timing_skipped_reason = None
    if graph_equivalence_passed:
        cache_scrub = torch.zeros(
            args.cache_scrub_mib * 1024 * 1024 // 4,
            device="cuda",
            dtype=torch.int32,
        )
        timing_modes = measure_graph_modes(
            captured,
            initial_caches,
            cache_scrub,
            args.warmups,
            args.repeats,
            args.decode_steps,
            args.batch_size,
            args.seed + 3000,
        )
    else:
        timing_skipped_reason = "same-backend eager/graph equivalence failed"

    import flashinfer

    result = {
        "schema_version": 2,
        "experiment": "page_gauge_static_token_one_page_full_decoder_cuda_graph",
        "claim_scope": (
            f"matched synchronized batch-{args.batch_size} static-token one-page "
            "full-decoder CUDA-graph lower bound; "
            "not production autoregressive throughput"
        ),
        "captured_boundary": (
            f"16 fixed-token synchronized batch-{args.batch_size} steps including "
            "embedding, all decoder layers, "
            "cache append/page finalization, attention, projections, MLP, "
            "final norm, LM head, and FP32 logits conversion"
        ),
        "excluded_or_static": [
            "token feedback and sampling",
            "dynamic token-buffer update",
            "dynamic position/page-table update outside the invariant plan bucket",
            "prefill",
            "ragged request lengths or positions",
        ],
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
        "output_tokens_per_sequence": args.batch_size * args.decode_steps,
        "configuration": {
            **{
                key: value
                for key, value in vars(args).items()
                if key != "output"
            },
            "output": str(args.output),
            "tokens": tokens,
        },
        "graph_protocol": {
            "matched_boundary": True,
            "one_outer_graph_launch_per_16_token_sequence": True,
            "throughput_counts_aggregate_output_tokens": True,
            "synchronized_equal_length_requests": True,
            "timed_boundary_excludes_python_plan_signature_guard": True,
            "nested_attention_graphs_disabled": True,
            "capture_warmups_on_side_stream": args.capture_warmups,
            "plan_bucket_invariant_over_page": True,
            "plan_signatures": {
                name: jsonable_signature(graph.plan_signature)
                for name, graph in captured.items()
            },
            "cache_state_restored_around_capture_and_validation": True,
            "mutated_pages_restored_for_every_request": True,
            "cache_conditioning": {
                "cache_neutral_start": (
                    "one scrub before each timed whole-sequence graph replay"
                ),
                "cache_hot": (
                    "one synchronized same-backend whole-sequence graph replay "
                    "immediately before each timed replay"
                ),
            },
            "pair_order": "AB, BA repeated; raw balanced blocks retained",
        },
        "correctness": {
            "same_backend_eager_vs_graph": same_backend_validation,
            "cross_backend": cross_backend_validation,
            "thresholds": {
                "minimum_logits_cosine": args.min_logits_cosine,
                "minimum_top1_agreement": args.min_top1_agreement,
            },
            "graph_equivalence_passed": graph_equivalence_passed,
            "cross_backend_passed": cross_backend_passed,
            "passed": graph_equivalence_passed and cross_backend_passed,
        },
        "timing_modes": timing_modes,
        "timing_skipped_reason": timing_skipped_reason,
        "cache_storage": cache_storage,
        "physical_page_layout": "request-major",
        "cuda_memory": {
            "capture_deltas": capture_memory,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "final_allocated_bytes": int(torch.cuda.memory_allocated()),
            "final_reserved_bytes": int(torch.cuda.memory_reserved()),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "total_gpu_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "flashinfer": str(flashinfer.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_sha256_at_process_start": source_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    if not result["correctness"]["passed"]:
        raise SystemExit("full-sequence graph correctness validation failed")


if __name__ == "__main__":
    main()
