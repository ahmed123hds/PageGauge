#!/usr/bin/env python3
"""Production-like per-token whole-decoder CUDA-graph diagnostic.

The diagnostic captures one complete decoder-step graph for each of the 16
offsets in a fixed, page-aligned decode window.  Each replay consumes a shared
static token buffer.  Greedy argmax and every device-to-device graph-input copy
remain outside the graph and inside the timed sequence boundary.  The reusable
fixture entrypoint prefers generated feedback and permits a clearly labelled
teacher-forced fallback only when the two backends' generated tokens diverge.

This is more production-like than a single static 16-step graph, but it is
still a fixed-batch, fixed-context, one-page experiment.  It excludes tokenizer
I/O, request scheduling, ragged batches, and the next-page replan.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTER_GRAPH_PATH = ROOT / "diagnostics/benchmark_full_sequence_graph.py"
HETEROGENEOUS_FA2_PATH = ROOT / "scripts/page_gauge_heterogeneous_fa2.py"


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OUTER = load_local_module(
    "page_gauge_outer_graph_helpers_for_token_step", OUTER_GRAPH_PATH
)
PG = OUTER.PG

GENERATED_FEEDBACK = "generated_feedback"
TEACHER_FORCED = "teacher_forced"


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
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--capture-warmups", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class CapturedTokenStep:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        logits: torch.Tensor,
        position: int,
        plan_signature: Any,
    ) -> None:
        self.graph = graph
        self.logits = logits
        self.position = position
        self.plan_signature = plan_signature


class CapturedTokenPage:
    def __init__(
        self,
        decoder,
        steps: list[CapturedTokenStep],
        replay_stream: torch.cuda.Stream,
        static_token: torch.Tensor,
        dynamic_token: torch.Tensor,
        start_position: int,
        invariant_plan_signature: Any,
    ) -> None:
        self.decoder = decoder
        self.steps = steps
        self.replay_stream = replay_stream
        self.static_token = static_token
        self.dynamic_token = dynamic_token
        self.start_position = start_position
        self.invariant_plan_signature = invariant_plan_signature

    @property
    def batch_size(self) -> int:
        return int(self.static_token.numel())

    def assert_replay_safe(self) -> None:
        if self.decoder.attention_graphs is not None:
            raise RuntimeError("nested attention graphs must remain disabled")
        if len(self.steps) != PG.PAGE:
            raise RuntimeError("a token-step page must contain 16 graphs")
        if self.decoder.plan_signature() != self.invariant_plan_signature:
            raise RuntimeError("token-step graph plan signature is stale")
        if any(
            step.plan_signature != self.invariant_plan_signature
            for step in self.steps
        ):
            raise RuntimeError("page-offset graphs do not share one plan bucket")

    def enqueue(
        self,
        token_inputs: torch.Tensor,
        input_mode: str,
        collect: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        validate_token_inputs(token_inputs, input_mode, self.batch_size)
        outputs: list[torch.Tensor] = []
        generated: list[torch.Tensor] = []
        seed = token_inputs if input_mode == GENERATED_FEEDBACK else token_inputs[0]
        # The seed copy and all 15 between-token copies are explicitly inside
        # the timed boundary: 16 D2D graph-input updates for a 16-token page.
        self.static_token.copy_(seed)
        for offset, step in enumerate(self.steps):
            step.graph.replay()
            if collect:
                outputs.append(step.logits.detach().clone())
            torch.argmax(step.logits, dim=-1, out=self.dynamic_token)
            if collect:
                generated.append(self.dynamic_token.detach().clone())
            if offset + 1 < len(self.steps):
                next_token = (
                    self.dynamic_token
                    if input_mode == GENERATED_FEEDBACK
                    else token_inputs[offset + 1]
                )
                self.static_token.copy_(next_token)
        return outputs, generated

    def replay_and_synchronize(
        self, token_inputs: torch.Tensor, input_mode: str
    ) -> None:
        self.assert_replay_safe()
        with torch.cuda.stream(self.replay_stream):
            self.enqueue(token_inputs, input_mode)
        torch.cuda.synchronize()

    def timed_replay(
        self, token_inputs: torch.Tensor, input_mode: str
    ) -> tuple[float, float]:
        self.assert_replay_safe()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        with torch.cuda.stream(self.replay_stream):
            begin.record()
            self.enqueue(token_inputs, input_mode)
            end.record()
        torch.cuda.synchronize()
        return (
            float(begin.elapsed_time(end)),
            (time.perf_counter() - wall_start) * 1e3,
        )


def validate_token_inputs(
    token_inputs: torch.Tensor, input_mode: str, batch_size: int
) -> None:
    if token_inputs.device.type != "cuda" or token_inputs.dtype != torch.long:
        raise ValueError("token inputs must be CUDA int64 tensors")
    if input_mode == GENERATED_FEEDBACK:
        if tuple(token_inputs.shape) != (batch_size,):
            raise ValueError("generated-feedback seed must have shape [B]")
        return
    if input_mode == TEACHER_FORCED:
        if tuple(token_inputs.shape) != (PG.PAGE, batch_size):
            raise ValueError("teacher-forced tokens must have shape [16,B]")
        return
    raise ValueError(f"unknown token input mode: {input_mode}")


@torch.inference_mode()
def execute_eager_page(
    decoder,
    token_inputs: torch.Tensor,
    input_mode: str,
    start_position: int,
    collect: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    validate_token_inputs(token_inputs, input_mode, decoder.batch_size)
    token = (
        token_inputs.detach().clone()
        if input_mode == GENERATED_FEEDBACK
        else token_inputs[0].detach().clone()
    )
    outputs: list[torch.Tensor] = []
    generated: list[torch.Tensor] = []
    for offset in range(PG.PAGE):
        if input_mode == TEACHER_FORCED:
            token.copy_(token_inputs[offset])
        logits = decoder.step(token, start_position + offset)[0]
        if collect:
            outputs.append(logits.detach().clone())
        token = logits.argmax(dim=-1)
        if collect:
            generated.append(token.detach().clone())
    return outputs, generated


def capture_token_page(
    decoder,
    initial_token: torch.Tensor,
    start_position: int,
    initial_cache: dict[str, Any],
    capture_warmups: int,
) -> CapturedTokenPage:
    decoder.attention_graphs = None
    signatures = []
    for offset in range(PG.PAGE):
        decoder.plan(start_position + offset + 1)
        signatures.append(decoder.plan_signature())
    torch.cuda.synchronize()
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(
            "per-token graphs require one invariant FlashInfer plan bucket"
        )

    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(capture_warmups):
            execute_eager_page(
                decoder,
                initial_token,
                GENERATED_FEEDBACK,
                start_position,
                collect=False,
            )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()

    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    static_token = initial_token.detach().clone()
    dynamic_token = torch.empty_like(static_token)
    capture_stream.wait_stream(current_stream)
    shared_pool = torch.cuda.graph_pool_handle()
    captured_steps: list[CapturedTokenStep] = []
    for offset in range(PG.PAGE):
        # Keep the raw cudaGraph_t solely so the diagnostic can enumerate its
        # node types. Explicit instantiation keeps first-replay latency out of
        # validation and timing; the resulting cudaGraphExec_t is otherwise
        # equivalent to the default keep_graph=False path.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        logits = None
        with torch.cuda.graph(
            graph,
            pool=shared_pool,
            stream=capture_stream,
        ):
            logits = decoder.step(static_token, start_position + offset)[0]
        assert logits is not None
        graph.instantiate()
        captured_steps.append(
            CapturedTokenStep(
                graph,
                logits,
                start_position + offset,
                signatures[offset],
            )
        )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    return CapturedTokenPage(
        decoder,
        captured_steps,
        capture_stream,
        static_token,
        dynamic_token,
        start_position,
        signatures[0],
    )


def compare_token_sequences(
    expected: list[torch.Tensor], observed: list[torch.Tensor]
) -> dict[str, Any]:
    if len(expected) != len(observed) or not expected:
        raise ValueError("token sequences must have equal non-zero length")
    per_step = [
        reference.eq(result).tolist()
        for reference, result in zip(expected, observed)
    ]
    flattened = [bool(value) for step in per_step for value in step]
    return {
        "checked_steps": len(expected),
        "checked_request_steps": len(flattened),
        "bitwise_identical": all(flattened),
        "agreement_fraction": sum(flattened) / len(flattened),
        "per_step_per_request_agreement": per_step,
        "expected_tokens": [tensor.tolist() for tensor in expected],
        "observed_tokens": [tensor.tolist() for tensor in observed],
    }


def validate_backend(
    captured: CapturedTokenPage,
    token_inputs: torch.Tensor,
    input_mode: str,
    initial_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decoder = captured.decoder
    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    eager_outputs, eager_tokens = execute_eager_page(
        decoder,
        token_inputs,
        input_mode,
        captured.start_position,
        collect=True,
    )
    torch.cuda.synchronize()
    eager_cache = OUTER.snapshot_mutated_page(
        decoder, captured.start_position
    )

    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    with torch.cuda.stream(captured.replay_stream):
        graph_outputs, graph_tokens = captured.enqueue(
            token_inputs, input_mode, collect=True
        )
    torch.cuda.synchronize()
    graph_cache = OUTER.snapshot_mutated_page(
        decoder, captured.start_position
    )

    with torch.cuda.stream(captured.replay_stream):
        repeated_outputs, repeated_tokens = captured.enqueue(
            token_inputs, input_mode, collect=True
        )
    torch.cuda.synchronize()
    repeated_cache = OUTER.snapshot_mutated_page(
        decoder, captured.start_position
    )

    eager_vs_graph_logits = PG.compare_logit_sequences(
        eager_outputs, graph_outputs
    )
    repeated_graph_logits = PG.compare_logit_sequences(
        graph_outputs, repeated_outputs
    )
    eager_vs_graph_tokens = compare_token_sequences(
        eager_tokens, graph_tokens
    )
    repeated_graph_tokens = compare_token_sequences(
        graph_tokens, repeated_tokens
    )
    eager_vs_graph_cache = OUTER.compare_cache_snapshots(
        eager_cache, graph_cache
    )
    repeated_graph_cache = OUTER.compare_cache_snapshots(
        graph_cache, repeated_cache
    )
    passed = (
        eager_vs_graph_logits["bitwise_identical"]
        and repeated_graph_logits["bitwise_identical"]
        and eager_vs_graph_tokens["bitwise_identical"]
        and repeated_graph_tokens["bitwise_identical"]
        and eager_vs_graph_cache["passed"]
        and repeated_graph_cache["passed"]
    )
    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    return (
        {
            "passed": passed,
            "eager_vs_graph_logits": eager_vs_graph_logits,
            "repeated_graph_logits": repeated_graph_logits,
            "eager_vs_graph_generated_tokens": eager_vs_graph_tokens,
            "repeated_graph_generated_tokens": repeated_graph_tokens,
            "eager_vs_graph_cache": eager_vs_graph_cache,
            "repeated_graph_cache": repeated_graph_cache,
        },
        {
            "eager_outputs": eager_outputs,
            "graph_outputs": graph_outputs,
            "eager_tokens": eager_tokens,
            "graph_tokens": graph_tokens,
        },
    )


def measure_modes(
    captured: dict[str, CapturedTokenPage],
    token_inputs: dict[str, torch.Tensor],
    input_mode: str,
    initial_caches: dict[str, dict[str, Any]],
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    names = ["flashinfer_fp16", "page_gauge"]
    result = {}
    for mode_index, mode in enumerate(("cache_neutral_start", "cache_hot")):
        for name in names:
            OUTER.restore_mutated_page(
                captured[name].decoder, initial_caches[name]
            )
        torch.cuda.synchronize()
        for warmup in range(warmups):
            order = names if warmup % 2 == 0 else list(reversed(names))
            for name in order:
                if mode == "cache_neutral_start":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    captured[name].replay_and_synchronize(
                        token_inputs[name], input_mode
                    )
                captured[name].replay_and_synchronize(
                    token_inputs[name], input_mode
                )

        samples = {name: {"gpu": [], "wall": []} for name in names}
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
                    captured[name].replay_and_synchronize(
                        token_inputs[name], input_mode
                    )
                gpu_ms, wall_ms = captured[name].timed_replay(
                    token_inputs[name], input_mode
                )
                samples[name]["gpu"].append(gpu_ms)
                samples[name]["wall"].append(wall_ms)
                block["measurements"][name] = {
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                }
            block["end_seconds"] = time.perf_counter() - measurement_start
            blocks.append(block)
        result[mode] = OUTER.summarize_mode(
            samples,
            blocks,
            PG.PAGE,
            captured["flashinfer_fp16"].batch_size,
            seed + mode_index * 100,
        )
    return result


@torch.inference_mode()
def execute_direct_feedback(
    decoder,
    initial_token: torch.Tensor,
    start_position: int,
    dynamic_token: torch.Tensor,
    collect: bool = False,
    synchronize_each_token: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Execute one generated page without a whole-decoder CUDA graph.

    ``synchronize_each_token`` is the honest token-synchronous serving control:
    the seed copy, decoder step, GPU argmax, and resulting stream completion are
    all inside each token's end-to-end boundary.  When false, all 16 steps are
    deep-queued and synchronized only at the page boundary.
    """
    if tuple(initial_token.shape) != (decoder.batch_size,):
        raise ValueError("direct feedback seed must have shape [B]")
    if tuple(dynamic_token.shape) != (decoder.batch_size,):
        raise ValueError("direct feedback buffer must have shape [B]")
    outputs: list[torch.Tensor] = []
    generated: list[torch.Tensor] = []
    # Seed D2D copy, all decoder kernels/attention graph replays, and all
    # argmax writes remain on one stream.  The optional synchronization occurs
    # only after argmax has produced the actual next-token input.
    dynamic_token.copy_(initial_token)
    for offset in range(PG.PAGE):
        logits = decoder.step(dynamic_token, start_position + offset)[0]
        if collect:
            outputs.append(logits.detach().clone())
        torch.argmax(logits, dim=-1, out=dynamic_token)
        if collect:
            generated.append(dynamic_token.detach().clone())
        if synchronize_each_token:
            torch.cuda.synchronize()
    return outputs, generated


def _timed_direct_feedback(
    decoder,
    initial_token: torch.Tensor,
    start_position: int,
    dynamic_token: torch.Tensor,
    synchronize_each_token: bool = False,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    execute_direct_feedback(
        decoder,
        initial_token,
        start_position,
        dynamic_token,
        synchronize_each_token=synchronize_each_token,
    )
    end.record()
    torch.cuda.synchronize()
    return (
        float(begin.elapsed_time(end)),
        (time.perf_counter() - wall_start) * 1e3,
    )


def measure_direct_feedback_modes(
    decoders: dict[str, Any],
    initial_tokens: dict[str, torch.Tensor],
    dynamic_tokens: dict[str, torch.Tensor],
    initial_caches: dict[str, dict[str, Any]],
    start_position: int,
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    seed: int,
    synchronize_each_token: bool = False,
) -> dict[str, Any]:
    names = ["flashinfer_fp16", "page_gauge"]
    result = {}
    for mode_index, mode in enumerate(("cache_neutral_start", "cache_hot")):
        _restore_fixture_caches(decoders, initial_caches)
        warmup_blocks = []
        warmup_start = time.perf_counter()
        for warmup in range(warmups):
            order = names if warmup % 2 == 0 else list(reversed(names))
            warmup_block: dict[str, Any] = {
                "warmup": warmup,
                "order": "AB" if warmup % 2 == 0 else "BA",
                "start_seconds": time.perf_counter() - warmup_start,
                "measurements": {},
            }
            for name in order:
                precondition = None
                if mode == "cache_neutral_start":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    precondition_gpu_ms, precondition_wall_ms = (
                        _timed_direct_feedback(
                            decoders[name],
                            initial_tokens[name],
                            start_position,
                            dynamic_tokens[name],
                            synchronize_each_token=synchronize_each_token,
                        )
                    )
                    precondition = {
                        "gpu_ms": precondition_gpu_ms,
                        "wall_ms": precondition_wall_ms,
                    }
                gpu_ms, wall_ms = _timed_direct_feedback(
                    decoders[name],
                    initial_tokens[name],
                    start_position,
                    dynamic_tokens[name],
                    synchronize_each_token=synchronize_each_token,
                )
                warmup_block["measurements"][name] = {
                    "precondition": precondition,
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                }
            warmup_block["end_seconds"] = time.perf_counter() - warmup_start
            warmup_blocks.append(warmup_block)

        samples = {name: {"gpu": [], "wall": []} for name in names}
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
                    _timed_direct_feedback(
                        decoders[name],
                        initial_tokens[name],
                        start_position,
                        dynamic_tokens[name],
                        synchronize_each_token=synchronize_each_token,
                    )
                gpu_ms, wall_ms = _timed_direct_feedback(
                    decoders[name],
                    initial_tokens[name],
                    start_position,
                    dynamic_tokens[name],
                    synchronize_each_token=synchronize_each_token,
                )
                samples[name]["gpu"].append(gpu_ms)
                samples[name]["wall"].append(wall_ms)
                block["measurements"][name] = {
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                }
            block["end_seconds"] = time.perf_counter() - measurement_start
            blocks.append(block)
        result[mode] = OUTER.summarize_mode(
            samples,
            blocks,
            PG.PAGE,
            decoders["flashinfer_fp16"].batch_size,
            seed + mode_index * 100,
        )
        result[mode]["warmup_blocks"] = warmup_blocks
        result[mode]["token_synchronization"] = {
            "enabled": synchronize_each_token,
            "synchronizations_per_measured_block": (
                PG.PAGE + 2 if synchronize_each_token else 2
            ),
            "boundary": (
                "after each GPU argmax, plus before and after the 16-token block"
                if synchronize_each_token
                else "before and after the 16-token block only"
            ),
        }
    return result


@torch.inference_mode()
def execute_static_teacher_forced_page(
    decoder,
    teacher_forced_tokens: torch.Tensor,
    start_position: int,
    retain_logits: bool,
) -> list[torch.Tensor]:
    """Run the coherent corpus tokens, optionally retaining every logits tensor."""
    validate_token_inputs(
        teacher_forced_tokens, TEACHER_FORCED, decoder.batch_size
    )
    retained: list[torch.Tensor] = []
    for offset in range(PG.PAGE):
        logits = decoder.step(
            teacher_forced_tokens[offset], start_position + offset
        )[0]
        if retain_logits:
            retained.append(logits)
        else:
            # Make output lifetime, rather than a subsequent argmax or D2D
            # dependency, the only difference between the two controls.
            del logits
    return retained


def _timed_static_teacher_forced_page(
    decoder,
    teacher_forced_tokens: torch.Tensor,
    start_position: int,
    retain_logits: bool,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    retained = execute_static_teacher_forced_page(
        decoder,
        teacher_forced_tokens,
        start_position,
        retain_logits,
    )
    end.record()
    torch.cuda.synchronize()
    gpu_ms = float(begin.elapsed_time(end))
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    # Keep all requested logits live through both the GPU event and host wall
    # boundary.  Releasing them earlier would collapse the retain control.
    del retained
    return gpu_ms, wall_ms


def measure_static_teacher_forced_modes(
    decoders: dict[str, Any],
    teacher_forced_tokens: torch.Tensor,
    initial_caches: dict[str, dict[str, Any]],
    start_position: int,
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Balanced timing for static inputs with two explicit logits lifetimes."""
    names = ["flashinfer_fp16", "page_gauge"]
    result: dict[str, Any] = {}
    for lifetime_index, retain_logits in enumerate((False, True)):
        lifetime = "retain_logits" if retain_logits else "discard_logits"
        lifetime_result: dict[str, Any] = {}
        for mode_index, mode in enumerate(("cache_neutral_start", "cache_hot")):
            _restore_fixture_caches(decoders, initial_caches)
            warmup_blocks = []
            warmup_start = time.perf_counter()
            for warmup in range(warmups):
                order = names if warmup % 2 == 0 else list(reversed(names))
                block: dict[str, Any] = {
                    "warmup": warmup,
                    "order": "AB" if warmup % 2 == 0 else "BA",
                    "start_seconds": time.perf_counter() - warmup_start,
                    "measurements": {},
                }
                for name in order:
                    precondition = None
                    if mode == "cache_neutral_start":
                        cache_scrub.add_(1)
                        torch.cuda.synchronize()
                    else:
                        pre_gpu, pre_wall = _timed_static_teacher_forced_page(
                            decoders[name],
                            teacher_forced_tokens,
                            start_position,
                            retain_logits,
                        )
                        precondition = {
                            "gpu_ms": pre_gpu,
                            "wall_ms": pre_wall,
                        }
                    gpu_ms, wall_ms = _timed_static_teacher_forced_page(
                        decoders[name],
                        teacher_forced_tokens,
                        start_position,
                        retain_logits,
                    )
                    block["measurements"][name] = {
                        "precondition": precondition,
                        "gpu_ms": gpu_ms,
                        "wall_ms": wall_ms,
                    }
                block["end_seconds"] = time.perf_counter() - warmup_start
                warmup_blocks.append(block)

            samples = {name: {"gpu": [], "wall": []} for name in names}
            blocks = []
            measurement_start = time.perf_counter()
            for repeat in range(repeats):
                order = names if repeat % 2 == 0 else list(reversed(names))
                block = {
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
                        _timed_static_teacher_forced_page(
                            decoders[name],
                            teacher_forced_tokens,
                            start_position,
                            retain_logits,
                        )
                    gpu_ms, wall_ms = _timed_static_teacher_forced_page(
                        decoders[name],
                        teacher_forced_tokens,
                        start_position,
                        retain_logits,
                    )
                    samples[name]["gpu"].append(gpu_ms)
                    samples[name]["wall"].append(wall_ms)
                    block["measurements"][name] = {
                        "gpu_ms": gpu_ms,
                        "wall_ms": wall_ms,
                    }
                block["end_seconds"] = time.perf_counter() - measurement_start
                blocks.append(block)
            summary = OUTER.summarize_mode(
                samples,
                blocks,
                PG.PAGE,
                decoders["flashinfer_fp16"].batch_size,
                seed + lifetime_index * 1000 + mode_index * 100,
            )
            summary["warmup_blocks"] = warmup_blocks
            summary["logits_lifetime"] = {
                "retain_logits": retain_logits,
                "full_logits_tensors_live_through_block_end": (
                    PG.PAGE if retain_logits else 0
                ),
                "argmax_calls": 0,
                "token_input": "predeclared CUDA [16,B] corpus successors",
            }
            lifetime_result[mode] = summary
        result[lifetime] = lifetime_result
    return result


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def tensor_systematic_sample_sha256(
    tensor: torch.Tensor, maximum_values: int = 4096
) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    stride = max(1, (int(flat.numel()) + maximum_values - 1) // maximum_values)
    sample = flat[::stride][:maximum_values].contiguous().cpu()
    return {
        "sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
        "sample_values": int(sample.numel()),
        "flat_stride": stride,
    }


def _event_sample_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("event sample list must not be empty")
    ordered = sorted(values)
    return {
        "raw_gpu_ms": values,
        "minimum_gpu_ms": ordered[0],
        "median_gpu_ms": statistics.median(values),
        "mean_gpu_ms": statistics.mean(values),
        "p95_gpu_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "maximum_gpu_ms": ordered[-1],
        "repeats": len(values),
    }


def gpu_memory_state() -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    state: dict[str, Any] = {
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cuda_mem_get_info_free_bytes": int(free_bytes),
        "cuda_mem_get_info_total_bytes": int(total_bytes),
    }
    executable = "/usr/lib/wsl/lib/nvidia-smi"
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=memory.total,memory.reserved,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        state["nvidia_smi_query"] = {
            "fields": [
                "memory.total_mib",
                "memory.reserved_mib",
                "memory.used_mib",
                "memory.free_mib",
            ],
            "raw": completed.stdout.strip(),
        }
    except Exception as error:
        state["nvidia_smi_query"] = {
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return state


@torch.inference_mode()
def relocate_page_gauge_cache_tensors(decoder) -> dict[str, Any]:
    """Terminally clone/swap every GaugeCache tensor after memory is freed."""
    if decoder.backend != "page_gauge" or decoder.attention_graphs is not None:
        raise ValueError("relocation requires eager PageGauge with graphs disabled")
    cache = decoder.cache
    attributes = (
        "key_codes",
        "value_codes",
        "key_scales",
        "value_scales",
        "key_center",
        "value_center",
        "exact_key",
        "exact_value",
        "output_center",
    )
    result: dict[str, Any] = {
        "memory_before": gpu_memory_state(),
        "tensor_records": {},
        "clone_order": list(attributes),
    }
    for attribute in attributes:
        source = getattr(cache, attribute)
        source_sample = tensor_systematic_sample_sha256(source)
        record: dict[str, Any] = {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "bytes": int(source.numel() * source.element_size()),
            "source_data_ptr": int(source.data_ptr()),
            "source_systematic_sample": source_sample,
            "memory_before_clone": gpu_memory_state(),
        }
        destination = source.clone()
        torch.cuda.synchronize()
        destination_sample = tensor_systematic_sample_sha256(destination)
        record.update(
            {
                "destination_data_ptr": int(destination.data_ptr()),
                "destination_systematic_sample": destination_sample,
                "sample_hash_bitwise_identical": (
                    source_sample["sha256"] == destination_sample["sha256"]
                ),
            }
        )
        setattr(cache, attribute, destination)
        del source, destination
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        record["memory_after_swap"] = gpu_memory_state()
        result["tensor_records"][attribute] = record
    result["all_sample_hashes_bitwise_identical"] = all(
        record["sample_hash_bitwise_identical"]
        for record in result["tensor_records"].values()
    )
    result["all_pointers_changed"] = all(
        record["source_data_ptr"] != record["destination_data_ptr"]
        for record in result["tensor_records"].values()
    )
    result["memory_after"] = gpu_memory_state()
    return result


def _scale_distribution(scales: torch.Tensor) -> dict[str, Any]:
    values = scales.detach().float().reshape(-1)
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        raise RuntimeError("live PageGauge scales contain non-finite values")
    absolute = values.abs()
    nonzero = absolute[absolute != 0]
    quantiles = torch.quantile(
        absolute,
        torch.tensor([0.01, 0.50, 0.99], device=absolute.device),
    )
    fp16_min_normal = 2.0**-14
    return {
        "count": int(values.numel()),
        "source_dtype": str(scales.dtype),
        "zero_count": int((absolute == 0).sum().item()),
        "nonzero_count": int(nonzero.numel()),
        "fp16_subnormal_nonzero_count": int(
            ((absolute != 0) & (absolute < fp16_min_normal)).sum().item()
        ),
        "fp16_min_normal_threshold": fp16_min_normal,
        "minimum_nonzero": (
            float(nonzero.min().item()) if nonzero.numel() else None
        ),
        "p01_absolute": float(quantiles[0].item()),
        "p50_absolute": float(quantiles[1].item()),
        "p99_absolute": float(quantiles[2].item()),
        "maximum_absolute": float(absolute.max().item()),
    }


def _diagnostic_layer_indices(layer_count: int) -> list[int]:
    indices = {0, layer_count // 2 - 1, layer_count - 1}
    if layer_count > 20:
        indices.update((19, 20))
    return sorted(index for index in indices if 0 <= index < layer_count)


def _code_distribution(
    codes: torch.Tensor, maximum_sample_values: int = 1_048_576
) -> dict[str, Any]:
    flat = codes.detach().reshape(-1)
    stride = max(1, (int(flat.numel()) + maximum_sample_values - 1) // maximum_sample_values)
    sample = flat[::stride][:maximum_sample_values].float()
    quantiles = torch.quantile(
        sample,
        torch.tensor([0.01, 0.50, 0.99], device=sample.device),
    )
    return {
        "count": int(flat.numel()),
        "source_dtype": str(codes.dtype),
        "minimum": int(flat.min().item()),
        "maximum": int(flat.max().item()),
        "zero_count": int((flat == 0).sum().item()),
        "negative_128_count": int((flat == -128).sum().item()),
        "positive_127_count": int((flat == 127).sum().item()),
        "systematic_sample": {
            "count": int(sample.numel()),
            "flat_stride": stride,
            "mean": float(sample.mean().item()),
            "standard_deviation": float(sample.std(unbiased=False).item()),
            "mean_absolute": float(sample.abs().mean().item()),
            "p01": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
        },
    }


def live_page_gauge_scale_distributions(
    decoder, start_position: int, include_code_distributions: bool = True
) -> dict[str, Any]:
    if decoder.backend != "page_gauge":
        raise ValueError("scale distributions require the PageGauge decoder")
    decoder.plan(start_position + 1)
    page_indices = decoder.all_pages[:, : decoder.old_pages].reshape(-1)
    cache = decoder.cache
    key_scales = cache.key_scales[:, page_indices]
    value_scales = cache.value_scales[:, page_indices]
    per_layer = {
        str(layer): {
            "key_scales": _scale_distribution(key_scales[layer]),
            "value_scales": _scale_distribution(value_scales[layer]),
        }
        for layer in range(decoder.layers)
    }
    selected_code_layers = {}
    if include_code_distributions:
        for layer in _diagnostic_layer_indices(decoder.layers):
            key_codes = cache.key_codes[layer, page_indices]
            value_codes = cache.value_codes[layer, page_indices]
            selected_code_layers[str(layer)] = {
                "key_codes": _code_distribution(key_codes),
                "value_codes": _code_distribution(value_codes),
            }
            del key_codes, value_codes
    result = {
        "scope": (
            "all initialized request-major INT8 old pages attended at the "
            "first decode position"
        ),
        "old_pages_per_request": int(decoder.old_pages),
        "selected_physical_pages": int(page_indices.numel()),
        "key_scales": _scale_distribution(key_scales),
        "value_scales": _scale_distribution(value_scales),
        "per_layer_scale_distributions": per_layer,
        "selected_layer_code_distributions": selected_code_layers,
        "code_distributions_skipped_for_residency_control": (
            not include_code_distributions
        ),
    }
    torch.cuda.synchronize()
    return result


@torch.inference_mode()
def profile_representative_static_token(
    decoder,
    teacher_forced_tokens: torch.Tensor,
    start_position: int,
    initial_cache: dict[str, Any],
    representative_offset: int = PG.PAGE - 1,
) -> dict[str, Any]:
    """Profile full layers and attention for one valid coherent token."""
    if not 0 <= representative_offset < PG.PAGE:
        raise ValueError("representative offset must lie in the decode page")
    selected_layers = set(_diagnostic_layer_indices(decoder.layers))
    real_queries: dict[int, torch.Tensor] = {}
    # Collect model-produced queries in a separate untimed pass.  Cloning them
    # inside the event-instrumented pass would contaminate three layer sums.
    OUTER.restore_mutated_page(decoder, initial_cache)
    for offset in range(representative_offset):
        logits = decoder.step(
            teacher_forced_tokens[offset], start_position + offset
        )[0]
        del logits
    torch.cuda.synchronize()
    query_collection_attention = decoder.attention

    def collect_query(layer_index: int) -> torch.Tensor:
        if layer_index in selected_layers:
            real_queries[layer_index] = decoder.rotated_query.detach().clone()
        return query_collection_attention(layer_index)

    decoder.attention = collect_query
    try:
        collection_logits = decoder.step(
            teacher_forced_tokens[representative_offset],
            start_position + representative_offset,
        )[0]
    finally:
        decoder.__dict__.pop("attention", None)
    torch.cuda.synchronize()
    del collection_logits

    OUTER.restore_mutated_page(decoder, initial_cache)
    for offset in range(representative_offset):
        logits = decoder.step(
            teacher_forced_tokens[offset], start_position + offset
        )[0]
        del logits
    torch.cuda.synchronize()

    attention_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
    original_attention = decoder.attention

    def profiled_attention(layer_index: int) -> torch.Tensor:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = original_attention(layer_index)
        end.record()
        attention_events.append((layer_index, begin, end))
        return output

    full_begin = torch.cuda.Event(enable_timing=True)
    full_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    full_begin.record()
    decoder.attention = profiled_attention
    try:
        logits, layer_events = decoder.step(
            teacher_forced_tokens[representative_offset],
            start_position + representative_offset,
            profile_layers=True,
        )
    finally:
        # Remove the instance shadow so subsequent paths use the class method.
        decoder.__dict__.pop("attention", None)
    full_end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    layer_ms = [float(begin.elapsed_time(end)) for begin, end in layer_events]
    attention_ms = [
        float(begin.elapsed_time(end))
        for _, begin, end in attention_events
    ]
    attention_layer_indices = [layer for layer, _, _ in attention_events]
    if (
        len(layer_ms) != decoder.layers
        or len(attention_ms) != decoder.layers
        or attention_layer_indices != list(range(decoder.layers))
    ):
        raise RuntimeError("representative profiling did not cover every layer")
    full_gpu_ms = float(full_begin.elapsed_time(full_end))
    del logits
    return {
        "backend": decoder.backend,
        "representative_offset": representative_offset,
        "absolute_position": start_position + representative_offset,
        "selection_reason": "page-closing token includes steady-state work and finalization",
        "attention_dispatch": (
            "per_layer_cuda_graph"
            if decoder.attention_graphs is not None
            else "fully_eager"
        ),
        "full_step_gpu_ms": full_gpu_ms,
        "full_step_wall_ms": wall_ms,
        "per_layer_gpu_ms": layer_ms,
        "per_attention_gpu_ms": attention_ms,
        "per_layer_records": [
            {
                "layer": layer_index,
                "full_layer_gpu_ms": layer_ms[layer_index],
                "attention_gpu_ms": attention_ms[layer_index],
                "non_attention_gpu_ms": (
                    layer_ms[layer_index] - attention_ms[layer_index]
                ),
            }
            for layer_index in range(decoder.layers)
        ],
        "sum_layer_gpu_ms": float(sum(layer_ms)),
        "sum_attention_gpu_ms": float(sum(attention_ms)),
        "sum_non_attention_layer_gpu_ms": float(
            sum(layer_ms) - sum(attention_ms)
        ),
        "outside_layer_gpu_ms": float(full_gpu_ms - sum(layer_ms)),
        "selected_real_queries": real_queries,
        "selected_real_query_sha256": {
            str(layer): tensor_sha256(query)
            for layer, query in sorted(real_queries.items())
        },
        "profiling_note": (
            "CUDA event instrumentation is diagnostic and may perturb absolute "
            "latency; sums localize work within the same instrumented step"
        ),
    }


def _matched_random_query(
    real_query: torch.Tensor, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=real_query.device)
    generator.manual_seed(seed)
    random_query = torch.randn(
        real_query.shape,
        device=real_query.device,
        dtype=real_query.dtype,
        generator=generator,
    )
    real_rms = real_query.float().square().mean().sqrt()
    random_rms = random_query.float().square().mean().sqrt()
    return (random_query * (real_rms / random_rms)).to(real_query.dtype)


@torch.inference_mode()
def profile_live_attention_components(
    decoder,
    real_queries: dict[int, torch.Tensor],
    repeats: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Decompose attention on live caches and valid model-produced queries."""
    if repeats <= 0:
        raise ValueError("component profiling repeats must be positive")
    selected_layers = sorted(real_queries)
    expected_layers = _diagnostic_layer_indices(decoder.layers)
    if selected_layers != expected_layers:
        raise ValueError(
            f"expected real queries for layers {expected_layers}, got {selected_layers}"
        )
    result: dict[str, Any] = {
        "backend": decoder.backend,
        "layers": {},
        "repeats": repeats,
        "pair_order": "real/random query AB, BA alternating",
        "query_control": (
            "random FP16 query independently sampled per layer and rescaled to "
            "the real query RMS; cache, plan, output buffers, and kernel fixed"
        ),
    }
    v_fragment_wrapper = None
    v_fragment_output = None
    v_fragment_lse = None
    if decoder.backend == "page_gauge":
        import flashinfer

        heterogeneous = load_local_module(
            "page_gauge_heterogeneous_fa2_for_live_component_ab",
            HETEROGENEOUS_FA2_PATH,
        )
        legacy_metadata = decoder.old_wrapper
        v_fragment_indptr = legacy_metadata.indptr.clone()
        v_fragment_indices = legacy_metadata.indices.clone()
        v_fragment_last_len = legacy_metadata.last_len.clone()
        workspace = torch.empty(
            128 * 1024 * 1024, device="cuda", dtype=torch.uint8
        )
        v_fragment_wrapper = heterogeneous.make_v_fragment_page_gauge_wrapper(
            flashinfer,
            workspace,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=v_fragment_indptr,
            paged_kv_indices_buffer=v_fragment_indices,
            paged_kv_last_page_len_buffer=v_fragment_last_len,
        )
        total_old_pages = decoder.batch_size * decoder.old_pages
        heterogeneous.plan_decode(
            v_fragment_wrapper,
            v_fragment_indptr,
            v_fragment_indices[:total_old_pages],
            v_fragment_last_len,
            fixed_split_pages=decoder.candidate_split_pages,
        )
        v_fragment_output = torch.empty_like(decoder.attention_output[0])
        v_fragment_lse = torch.empty_like(decoder.old_lse[0])
        result["v_fragment_scale_ab"] = {
            "enabled": True,
            "module_uri": v_fragment_wrapper.page_gauge_module_uri,
            "module_source_hashes": v_fragment_wrapper.page_gauge_source_hashes,
            "semantic_change": (
                "K path unchanged; apply each page V scale to converted FP16 "
                "V fragments before MMA instead of to softmax probabilities"
            ),
            "required_correctness": (
                "legacy and V-fragment LSE bitwise; output error/cosine recorded"
            ),
        }
    for layer_index in selected_layers:
        real_query = real_queries[layer_index]
        random_query = _matched_random_query(
            real_query, seed + layer_index * 1009
        )
        events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

        def timed(label: str, operation) -> None:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            operation()
            end.record()
            events.setdefault(label, []).append((begin, end))

        if decoder.backend == "flashinfer_fp16":
            cache = decoder.cache
            wrapper = decoder.baseline_wrapper.wrapper

            def run_baseline(query: torch.Tensor) -> None:
                wrapper.run(
                    query,
                    (cache.key[layer_index], cache.value[layer_index]),
                    out=decoder.attention_output[layer_index],
                )

            for repeat in range(repeats):
                query_order = (
                    (("real", real_query), ("random", random_query))
                    if repeat % 2 == 0
                    else (("random", random_query), ("real", real_query))
                )
                for query_name, query in query_order:
                    timed(
                        f"baseline_wrapper_{query_name}_query",
                        lambda query=query: run_baseline(query),
                    )
        else:
            import flashinfer

            cache = decoder.cache
            old_wrapper = decoder.old_wrapper.wrapper
            exact_wrapper = decoder.exact_wrapper.wrapper

            def run_old(query: torch.Tensor) -> None:
                old_wrapper.run(
                    query,
                    (cache.key_codes[layer_index], cache.value_codes[layer_index]),
                    cache.key_scales[layer_index],
                    cache.value_scales[layer_index],
                    1.0 / (PG.DIM**0.5),
                    out=decoder.attention_output[layer_index],
                    lse=decoder.old_lse[layer_index],
                    return_lse=True,
                )

            def run_v_fragment(query: torch.Tensor) -> None:
                assert (
                    v_fragment_wrapper is not None
                    and v_fragment_output is not None
                    and v_fragment_lse is not None
                )
                v_fragment_wrapper.run(
                    query,
                    (cache.key_codes[layer_index], cache.value_codes[layer_index]),
                    cache.key_scales[layer_index],
                    cache.value_scales[layer_index],
                    1.0 / (PG.DIM**0.5),
                    out=v_fragment_output,
                    lse=v_fragment_lse,
                    return_lse=True,
                )

            def run_exact(query: torch.Tensor) -> None:
                exact_wrapper.run(
                    query,
                    (cache.exact_key[layer_index], cache.exact_value[layer_index]),
                    out=decoder.exact_output[layer_index],
                    lse=decoder.exact_lse[layer_index],
                    return_lse=True,
                )

            def run_merge() -> None:
                flashinfer.merge_state_in_place(
                    decoder.attention_output[layer_index],
                    decoder.old_lse[layer_index],
                    decoder.exact_output[layer_index],
                    decoder.exact_lse[layer_index],
                )

            def run_center_add() -> None:
                decoder.attention_output[layer_index].add_(
                    cache.output_center[layer_index]
                )

            # Keep JIT/module initialization and the first physical-page touch
            # outside all recorded samples for both implementations.
            run_old(real_query)
            run_v_fragment(real_query)
            torch.cuda.synchronize()
            for repeat in range(repeats):
                query_order = (
                    (("real", real_query), ("random", random_query))
                    if repeat % 2 == 0
                    else (("random", random_query), ("real", real_query))
                )
                for query_name, query in query_order:
                    wrapper_order = (
                        ("legacy", "v_fragment")
                        if repeat % 2 == 0
                        else ("v_fragment", "legacy")
                    )
                    for wrapper_name in wrapper_order:
                        if wrapper_name == "legacy":
                            timed(
                                f"old_wrapper_{query_name}_query",
                                lambda query=query: run_old(query),
                            )
                        else:
                            timed(
                                f"v_fragment_wrapper_{query_name}_query",
                                lambda query=query: run_v_fragment(query),
                            )
                    if query_name != "real":
                        continue
                    # Re-establish the legacy old state before decomposing its
                    # exact-tail merge, independent of the alternating A/B order.
                    run_old(query)
                    if decoder.tail_attention == "flashinfer_merge":
                        timed("exact_wrapper_real_query", lambda: run_exact(query))
                        timed("merge_state_real_query", run_merge)
                        if decoder.center_restore == "attention_add":
                            timed("center_add_real_query", run_center_add)
                    else:
                        timed(
                            "fused_exact_merge_center_real_query",
                            lambda query=query: decoder.append_extension.exact_tail_merge_center(
                                query,
                                cache.exact_key[layer_index],
                                cache.exact_value[layer_index],
                                decoder.exact_wrapper.indices[
                                    : decoder.batch_size * decoder.exact_pages
                                ],
                                decoder.exact_wrapper.last_len,
                                decoder.attention_output[layer_index],
                                decoder.old_lse[layer_index],
                                cache.value_center[layer_index],
                                1.0 / (PG.DIM**0.5),
                            ),
                        )
        torch.cuda.synchronize()
        component_summaries = {
            label: _event_sample_summary(
                [float(begin.elapsed_time(end)) for begin, end in pairs]
            )
            for label, pairs in events.items()
        }
        layer_result = {
            "real_query_rms": float(
                real_query.float().square().mean().sqrt().item()
            ),
            "random_query_rms": float(
                random_query.float().square().mean().sqrt().item()
            ),
            "components": component_summaries,
        }
        if decoder.backend == "page_gauge":
            assert v_fragment_output is not None and v_fragment_lse is not None
            run_old(real_query)
            legacy_output = decoder.attention_output[layer_index].detach().clone()
            legacy_lse = decoder.old_lse[layer_index].detach().clone()
            run_v_fragment(real_query)
            alternative_output = v_fragment_output.detach().clone()
            alternative_lse = v_fragment_lse.detach().clone()
            torch.cuda.synchronize()
            output_difference = (
                legacy_output.float() - alternative_output.float()
            )
            reference_norm = legacy_output.float().norm()
            layer_result["legacy_vs_v_fragment_correctness"] = {
                "lse_bitwise_identical": bool(
                    torch.equal(legacy_lse, alternative_lse)
                ),
                "lse_maximum_absolute_error": float(
                    (legacy_lse.float() - alternative_lse.float())
                    .abs()
                    .max()
                    .item()
                ),
                "output_bitwise_identical": bool(
                    torch.equal(legacy_output, alternative_output)
                ),
                "output_maximum_absolute_error": float(
                    output_difference.abs().max().item()
                ),
                "output_relative_l2_error": float(
                    (output_difference.norm() / reference_norm).item()
                ),
                "output_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        legacy_output.float().reshape(1, -1),
                        alternative_output.float().reshape(1, -1),
                    ).item()
                ),
            }
        result["layers"][str(layer_index)] = layer_result
    return result


def cuda_graph_node_summary(graph: torch.cuda.CUDAGraph) -> dict[str, Any]:
    """Return CUDA Runtime graph-node counts when cuda-python is available."""
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
        return {
            "available": True,
            "raw_graph_node_count": int(returned_count),
            "node_type_counts": type_counts,
        }
    except Exception as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _profile_graph_index_pattern(
    captured: CapturedTokenPage,
    graph_indices: list[int],
    initial_token: torch.Tensor,
    teacher_forced_tokens: torch.Tensor,
    input_mode: str,
) -> dict[str, Any]:
    """Measure GPU segments and host enqueue calls without per-step sync."""
    if not graph_indices or any(
        index < 0 or index >= len(captured.steps) for index in graph_indices
    ):
        raise ValueError("graph indices must select at least one captured step")
    captured.assert_replay_safe()
    validate_token_inputs(
        teacher_forced_tokens, TEACHER_FORCED, captured.batch_size
    )
    if tuple(initial_token.shape) != (captured.batch_size,):
        raise ValueError("pattern seed must have shape [B]")

    starts = [torch.cuda.Event(enable_timing=True) for _ in graph_indices]
    ends = [torch.cuda.Event(enable_timing=True) for _ in graph_indices]
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    host_calls: list[dict[str, float]] = []
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    host_enqueue_start = time.perf_counter()
    with torch.cuda.stream(captured.replay_stream):
        total_start.record()
        for iteration, graph_index in enumerate(graph_indices):
            starts[iteration].record()
            if iteration == 0:
                graph_input = initial_token
            elif input_mode == GENERATED_FEEDBACK:
                graph_input = captured.dynamic_token
            else:
                graph_input = teacher_forced_tokens[graph_index]

            before_copy = time.perf_counter_ns()
            captured.static_token.copy_(graph_input)
            after_copy = time.perf_counter_ns()
            before_replay = after_copy
            captured.steps[graph_index].graph.replay()
            after_replay = time.perf_counter_ns()
            torch.argmax(
                captured.steps[graph_index].logits,
                dim=-1,
                out=captured.dynamic_token,
            )
            after_argmax = time.perf_counter_ns()
            ends[iteration].record()
            host_calls.append(
                {
                    "device_to_device_copy_us": (after_copy - before_copy)
                    / 1e3,
                    "cuda_graph_replay_call_us": (
                        after_replay - before_replay
                    )
                    / 1e3,
                    "argmax_enqueue_us": (after_argmax - after_replay) / 1e3,
                }
            )
        total_end.record()
    host_enqueue_ms = (time.perf_counter() - host_enqueue_start) * 1e3
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    gpu_ms_by_iteration = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends)
    ]
    replay_call_us = [
        payload["cuda_graph_replay_call_us"] for payload in host_calls
    ]
    return {
        "graph_indices": graph_indices,
        "positions": [
            captured.steps[index].position for index in graph_indices
        ],
        "gpu_ms_by_iteration": gpu_ms_by_iteration,
        "gpu_total_event_ms": float(total_start.elapsed_time(total_end)),
        "gpu_sum_iteration_events_ms": sum(gpu_ms_by_iteration),
        "host_total_enqueue_ms": host_enqueue_ms,
        "wall_total_ms": wall_ms,
        "host_calls_by_iteration": host_calls,
        "host_cuda_graph_replay_call_us": {
            "minimum": min(replay_call_us),
            "median": statistics.median(replay_call_us),
            "maximum": max(replay_call_us),
            "sum": sum(replay_call_us),
        },
    }


def _prepare_offset_state(
    captured: CapturedTokenPage,
    target_offset: int,
    teacher_forced_tokens: torch.Tensor,
    input_mode: str,
) -> torch.Tensor:
    """Populate earlier slots and return the valid input for target_offset."""
    if target_offset == 0:
        return teacher_forced_tokens[0]
    token = teacher_forced_tokens[0]
    with torch.cuda.stream(captured.replay_stream):
        for offset in range(target_offset):
            source = (
                token
                if input_mode == GENERATED_FEEDBACK
                else teacher_forced_tokens[offset]
            )
            captured.static_token.copy_(source)
            captured.steps[offset].graph.replay()
            torch.argmax(
                captured.steps[offset].logits,
                dim=-1,
                out=captured.dynamic_token,
            )
            token = captured.dynamic_token
    torch.cuda.synchronize()
    if input_mode == GENERATED_FEEDBACK:
        return captured.dynamic_token.detach().clone()
    return teacher_forced_tokens[target_offset]


def profile_graph_replay_patterns(
    captured: dict[str, CapturedTokenPage],
    teacher_forced_tokens: torch.Tensor,
    input_mode: str,
    initial_caches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Contrast graph switching with repeated replay of one graph object."""
    result: dict[str, Any] = {}
    for name, page in captured.items():
        backend: dict[str, Any] = {
            "cuda_graph_nodes_by_offset": [
                cuda_graph_node_summary(step.graph) for step in page.steps
            ]
        }
        _restore_fixture_caches({name: page.decoder}, {name: initial_caches[name]})
        backend["cycling_offsets_0_to_15"] = _profile_graph_index_pattern(
            page,
            list(range(PG.PAGE)),
            teacher_forced_tokens[0],
            teacher_forced_tokens,
            input_mode,
        )
        for target_offset in (0, PG.PAGE // 2, PG.PAGE - 1):
            _restore_fixture_caches(
                {name: page.decoder}, {name: initial_caches[name]}
            )
            target_token = _prepare_offset_state(
                page,
                target_offset,
                teacher_forced_tokens,
                input_mode,
            )
            backend[f"same_offset_{target_offset}_repeated_16x"] = (
                _profile_graph_index_pattern(
                    page,
                    [target_offset] * PG.PAGE,
                    target_token,
                    teacher_forced_tokens,
                    input_mode,
                )
            )
        cycling = backend["cycling_offsets_0_to_15"]
        repeated_zero = backend["same_offset_0_repeated_16x"]
        backend["cycling_vs_same_offset_0"] = {
            "gpu_total_ratio_cycling_over_same": (
                cycling["gpu_total_event_ms"]
                / repeated_zero["gpu_total_event_ms"]
            ),
            "host_enqueue_ratio_cycling_over_same": (
                cycling["host_total_enqueue_ms"]
                / repeated_zero["host_total_enqueue_ms"]
            ),
            "wall_ratio_cycling_over_same": (
                cycling["wall_total_ms"] / repeated_zero["wall_total_ms"]
            ),
        }
        result[name] = backend
    _restore_fixture_caches(
        {name: page.decoder for name, page in captured.items()}, initial_caches
    )
    return {
        "input_mode": input_mode,
        "purpose": (
            "diagnose host cudaGraphLaunch blocking, per-offset latency, graph "
            "node-count asymmetry, and graph-object switching overhead"
        ),
        "events_include": (
            "one D2D graph-input copy, one whole-decoder graph replay, and one "
            "FP32 argmax per iteration; no host synchronization between steps"
        ),
        "backends": result,
    }


def snapshot_fixture_caches(
    decoders: dict[str, Any], start_position: int
) -> dict[str, dict[str, Any]]:
    """Clone only the page that a 16-token fixture will mutate."""
    expected = {"flashinfer_fp16", "page_gauge"}
    if set(decoders) != expected:
        raise ValueError(f"decoder keys must be exactly {sorted(expected)}")
    snapshots = {
        name: OUTER.snapshot_mutated_page(decoder, start_position)
        for name, decoder in decoders.items()
    }
    torch.cuda.synchronize()
    return snapshots


def _restore_fixture_caches(
    decoders: dict[str, Any], snapshots: dict[str, dict[str, Any]]
) -> None:
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, snapshots[name])
    torch.cuda.synchronize()


@torch.inference_mode()
def run_direct_generated_feedback_fixture(
    *,
    model: Any,
    decoders: dict[str, Any],
    initial_tokens: torch.Tensor,
    start_position: int,
    warmups: int,
    repeats: int,
    cache_scrub_mib: int,
    seed: int,
    min_logits_cosine: float,
    min_top1_agreement: float,
    direct_feedback_protocol: str = "deep_queued",
    initial_caches: dict[str, dict[str, Any]] | None = None,
    fixture_provenance: dict[str, Any] | None = None,
    cache_storage: dict[str, Any] | None = None,
    additional_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Measure generated feedback at two non-whole-graph boundaries."""
    expected_names = {"flashinfer_fp16", "page_gauge"}
    if set(decoders) != expected_names:
        raise ValueError(f"decoder keys must be exactly {sorted(expected_names)}")
    if warmups < 0 or repeats <= 0 or repeats % 2:
        raise ValueError("warmups must be non-negative; repeats positive and even")
    if cache_scrub_mib <= 0:
        raise ValueError("cache scrub size must be positive")
    if direct_feedback_protocol not in {"deep_queued", "token_synchronous"}:
        raise ValueError(
            "direct feedback protocol must be 'deep_queued' or "
            "'token_synchronous'"
        )
    synchronize_each_token = direct_feedback_protocol == "token_synchronous"
    if start_position % PG.PAGE:
        raise ValueError("direct feedback fixture must start on a page boundary")
    batch_size = decoders["flashinfer_fp16"].batch_size
    if any(decoder.batch_size != batch_size for decoder in decoders.values()):
        raise ValueError("both decoders must have the same batch size")
    initial_tokens = initial_tokens.to(device="cuda", dtype=torch.long).contiguous()
    if tuple(initial_tokens.shape) != (batch_size,):
        raise ValueError("initial tokens must have shape [B]")
    seeds = {name: initial_tokens for name in decoders}
    dynamic_tokens = {
        name: torch.empty_like(initial_tokens) for name in decoders
    }
    if initial_caches is None:
        initial_caches = snapshot_fixture_caches(decoders, start_position)
    elif set(initial_caches) != expected_names:
        raise ValueError("initial cache snapshots do not match decoder keys")

    memory_at_entry = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    source_paths = tuple(
        dict.fromkeys(
            (
                *OUTER.DEPENDENCY_PATHS,
                OUTER_GRAPH_PATH,
                Path(__file__).resolve(),
                *(Path(path).resolve() for path in additional_source_paths),
            )
        )
    )
    source_hashes = {
        _source_label(path): source_sha256(path)
        for path in source_paths
        if path.is_file()
    }

    # First establish the fully eager generated trajectory from the untouched
    # coherent page. No whole-token or attention CUDA graphs are installed.
    for decoder in decoders.values():
        decoder.attention_graphs = None
    gc.collect()
    torch.cuda.empty_cache()
    eager_outputs: dict[str, list[torch.Tensor]] = {}
    eager_tokens: dict[str, list[torch.Tensor]] = {}
    eager_caches: dict[str, dict[str, Any]] = {}
    eager_dispatch: dict[str, Any] = {}
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, initial_caches[name])
        decoder.reset_attention_dispatch_counts()
        outputs, generated = execute_direct_feedback(
            decoder,
            seeds[name],
            start_position,
            dynamic_tokens[name],
            collect=True,
            synchronize_each_token=synchronize_each_token,
        )
        torch.cuda.synchronize()
        eager_outputs[name] = outputs
        eager_tokens[name] = generated
        eager_caches[name] = OUTER.snapshot_mutated_page(
            decoder, start_position
        )
        eager_dispatch[name] = decoder.attention_dispatch_counts()

    # Install exactly the production attention-only graph boundary: one
    # captured attention path per layer, with projections/MLP/feedback eager.
    attention_capture_memory: dict[str, Any] = {}
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, initial_caches[name])
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        decoder.capture_attention_graphs(start_position + 1)
        attention_capture_memory[name] = {
            "allocated_before_bytes": before_allocated,
            "allocated_after_bytes": int(torch.cuda.memory_allocated()),
            "allocated_delta_bytes": int(torch.cuda.memory_allocated())
            - before_allocated,
            "reserved_before_bytes": before_reserved,
            "reserved_after_bytes": int(torch.cuda.memory_reserved()),
            "reserved_delta_bytes": int(torch.cuda.memory_reserved())
            - before_reserved,
        }

    graph_outputs: dict[str, list[torch.Tensor]] = {}
    graph_tokens: dict[str, list[torch.Tensor]] = {}
    graph_caches: dict[str, dict[str, Any]] = {}
    graph_dispatch: dict[str, Any] = {}
    expected_attention_calls = int(model.config.num_hidden_layers) * PG.PAGE
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, initial_caches[name])
        decoder.reset_attention_dispatch_counts()
        outputs, generated = execute_direct_feedback(
            decoder,
            seeds[name],
            start_position,
            dynamic_tokens[name],
            collect=True,
            synchronize_each_token=synchronize_each_token,
        )
        torch.cuda.synchronize()
        graph_outputs[name] = outputs
        graph_tokens[name] = generated
        graph_caches[name] = OUTER.snapshot_mutated_page(
            decoder, start_position
        )
        graph_dispatch[name] = decoder.attention_dispatch_counts()

    same_backend = {}
    for name in decoders:
        logits = PG.compare_logit_sequences(
            eager_outputs[name], graph_outputs[name]
        )
        tokens = compare_token_sequences(
            eager_tokens[name], graph_tokens[name]
        )
        cache = OUTER.compare_cache_snapshots(
            eager_caches[name], graph_caches[name]
        )
        dispatch_ok = (
            eager_dispatch[name]["eager_calls"] == expected_attention_calls
            and eager_dispatch[name]["graph_replays"] == 0
            and graph_dispatch[name]["graph_replays"]
            == expected_attention_calls
            and graph_dispatch[name]["eager_calls"] == 0
        )
        same_backend[name] = {
            "logits": logits,
            "generated_tokens": tokens,
            "cache": cache,
            "eager_attention_dispatch": eager_dispatch[name],
            "per_layer_graph_attention_dispatch": graph_dispatch[name],
            "dispatch_passed": dispatch_ok,
            "passed": (
                logits["bitwise_identical"]
                and tokens["bitwise_identical"]
                and cache["passed"]
                and dispatch_ok
            ),
        }

    cross_backend = {}
    for execution, outputs, generated in (
        ("fully_eager", eager_outputs, eager_tokens),
        ("per_layer_attention_graphs", graph_outputs, graph_tokens),
    ):
        cross_backend[execution] = {
            "logits": OUTER.compare_backends(
                outputs["flashinfer_fp16"], outputs["page_gauge"]
            ),
            "generated_tokens": compare_token_sequences(
                generated["flashinfer_fp16"], generated["page_gauge"]
            ),
        }
    cross_backend_passed = all(
        payload["logits"]["minimum_logits_cosine"] >= min_logits_cosine
        and payload["logits"]["top1_agreement_fraction"]
        >= min_top1_agreement
        and payload["generated_tokens"]["bitwise_identical"]
        for payload in cross_backend.values()
    )
    correctness_passed = (
        all(payload["passed"] for payload in same_backend.values())
        and cross_backend_passed
    )

    timing_modes: dict[str, Any] = {}
    timing_skipped_reason = None
    if correctness_passed:
        cache_scrub = torch.zeros(
            cache_scrub_mib * 1024 * 1024 // 4,
            device="cuda",
            dtype=torch.int32,
        )
        # Per-layer graphs are already installed from validation.
        timing_modes["per_layer_attention_graphs"] = (
            measure_direct_feedback_modes(
                decoders,
                seeds,
                dynamic_tokens,
                initial_caches,
                start_position,
                cache_scrub,
                warmups,
                repeats,
                seed + 5000,
                synchronize_each_token=synchronize_each_token,
            )
        )
        # Remove every graph owner and empty only unused allocator blocks; the
        # live model and full caches remain untouched.
        for decoder in decoders.values():
            decoder.attention_graphs = None
        gc.collect()
        torch.cuda.empty_cache()
        timing_modes["fully_eager"] = measure_direct_feedback_modes(
            decoders,
            seeds,
            dynamic_tokens,
            initial_caches,
            start_position,
            cache_scrub,
            warmups,
            repeats,
            seed + 6000,
            synchronize_each_token=synchronize_each_token,
        )
    else:
        timing_skipped_reason = (
            "same-backend eager/attention-graph equivalence, dispatch, or "
            "cross-backend generated-trajectory gate failed"
        )

    _restore_fixture_caches(decoders, initial_caches)
    for decoder in decoders.values():
        decoder.attention_graphs = None
    import flashinfer

    major, minor = torch.cuda.get_device_capability()
    return {
        "schema_version": 1,
        "experiment": "page_gauge_live_prefill_direct_generated_feedback",
        "claim_scope": (
            f"synchronized batch-{batch_size}, fixed-context, one-page greedy "
            f"feedback using the {direct_feedback_protocol} protocol with no "
            "whole-decoder graph; compares fully eager "
            "attention against production per-layer attention graphs"
        ),
        "batch_size": batch_size,
        "start_position": start_position,
        "decode_steps": PG.PAGE,
        "output_tokens_per_sequence": batch_size * PG.PAGE,
        "initial_token_ids": initial_tokens.detach().cpu().tolist(),
        "fixture_provenance": fixture_provenance or {},
        "direct_feedback_protocol": direct_feedback_protocol,
        "execution_boundaries": {
            "common": [
                "one D2D actual-successor seed copy per 16-token block",
                (
                    "16 complete decoder steps issued sequentially from Python"
                    if synchronize_each_token
                    else "16 complete decoder steps deep-queued from Python"
                ),
                "16 FP32 GPU argmax writes feeding the next embedding",
                (
                    "one synchronization after every token argmax, plus before "
                    "and after each measured block"
                    if synchronize_each_token
                    else "one synchronization before and after each measured block"
                ),
                (
                    "token boundaries include host stream synchronization"
                    if synchronize_each_token
                    else "no host synchronization at token boundaries"
                ),
            ],
            "fully_eager": (
                "all append, attention, projection, MLP, normalization, LM-head, "
                "and argmax operations enqueued individually"
            ),
            "per_layer_attention_graphs": (
                "one production attention-only graph replay per layer; append, "
                "projections, MLP, normalization, LM head, and argmax remain eager"
            ),
            "whole_token_or_whole_sequence_cuda_graphs": False,
            "pair_order": "AB, BA repeated; raw blocks retained",
            "cache_modes": ["cache_neutral_start", "cache_hot"],
        },
        "correctness": {
            "same_backend_fully_eager_vs_per_layer_graphs": same_backend,
            "cross_backend": cross_backend,
            "thresholds": {
                "minimum_logits_cosine": min_logits_cosine,
                "minimum_top1_agreement": min_top1_agreement,
                "generated_tokens_must_be_bitwise_identical": True,
                "same_backend_logits_tokens_and_cache_must_be_bitwise_identical": True,
                "attention_dispatch_must_be_exact": True,
            },
            "cross_backend_passed": cross_backend_passed,
            "passed": correctness_passed,
        },
        "timing_modes": timing_modes,
        "timing_skipped_reason": timing_skipped_reason,
        "cache_storage": cache_storage or {},
        "physical_page_layout": "request-major",
        "live_fixture_reuse": {
            "model_reused": True,
            "decoder_objects_reused": True,
            "cache_objects_reused": True,
            "cache_serialization": False,
            "cache_duplication": False,
            "only_mutated_page_snapshots_cloned": True,
        },
        "cuda_memory": {
            "at_fixture_entry": memory_at_entry,
            "attention_graph_capture_deltas": attention_capture_memory,
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
        "source_sha256_at_fixture_entry": source_hashes,
    }


@torch.inference_mode()
def run_static_teacher_forced_fixture(
    *,
    model: Any,
    decoders: dict[str, Any],
    teacher_forced_tokens: torch.Tensor,
    start_position: int,
    warmups: int,
    repeats: int,
    cache_scrub_mib: int,
    seed: int,
    min_logits_cosine: float,
    min_top1_agreement: float,
    initial_caches: dict[str, dict[str, Any]] | None = None,
    fixture_provenance: dict[str, Any] | None = None,
    cache_storage: dict[str, Any] | None = None,
    component_repeats: int = 20,
    run_matrix_timing: bool = True,
    release_baseline_for_candidate_residency: bool = False,
    relocate_candidate_cache_after_release: bool = False,
    additional_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Live coherent static-token control for feedback and output lifetime."""
    expected_names = {"flashinfer_fp16", "page_gauge"}
    if set(decoders) != expected_names:
        raise ValueError(f"decoder keys must be exactly {sorted(expected_names)}")
    if warmups < 0 or repeats <= 0 or repeats % 2:
        raise ValueError("warmups must be non-negative; repeats positive and even")
    if cache_scrub_mib <= 0 or component_repeats <= 0:
        raise ValueError("scrub size and component repeats must be positive")
    if release_baseline_for_candidate_residency and run_matrix_timing:
        raise ValueError(
            "terminal candidate-only residency control requires matrix timing disabled"
        )
    if (
        relocate_candidate_cache_after_release
        and not release_baseline_for_candidate_residency
    ):
        raise ValueError("candidate relocation requires terminal baseline release")
    if start_position % PG.PAGE:
        raise ValueError("static teacher fixture must start on a page boundary")
    batch_size = decoders["flashinfer_fp16"].batch_size
    teacher_forced_tokens = teacher_forced_tokens.to(
        device="cuda", dtype=torch.long
    ).contiguous()
    validate_token_inputs(teacher_forced_tokens, TEACHER_FORCED, batch_size)
    if any(decoder.batch_size != batch_size for decoder in decoders.values()):
        raise ValueError("both decoders must have the same batch size")
    if initial_caches is None:
        initial_caches = snapshot_fixture_caches(decoders, start_position)
    elif set(initial_caches) != expected_names:
        raise ValueError("initial cache snapshots do not match decoder keys")

    memory_at_entry = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    scale_distributions = live_page_gauge_scale_distributions(
        decoders["page_gauge"],
        start_position,
        include_code_distributions=(
            not release_baseline_for_candidate_residency
        ),
    )
    source_paths = tuple(
        dict.fromkeys(
            (
                *OUTER.DEPENDENCY_PATHS,
                OUTER_GRAPH_PATH,
                HETEROGENEOUS_FA2_PATH,
                Path(__file__).resolve(),
                *(Path(path).resolve() for path in additional_source_paths),
            )
        )
    )
    source_hashes = {
        _source_label(path): source_sha256(path)
        for path in source_paths
        if path.is_file()
    }

    eager_outputs: dict[str, list[torch.Tensor]] = {}
    eager_caches: dict[str, dict[str, Any]] = {}
    eager_dispatch: dict[str, Any] = {}
    for decoder in decoders.values():
        decoder.attention_graphs = None
    gc.collect()
    torch.cuda.empty_cache()
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, initial_caches[name])
        decoder.reset_attention_dispatch_counts()
        eager_outputs[name] = execute_static_teacher_forced_page(
            decoder,
            teacher_forced_tokens,
            start_position,
            retain_logits=True,
        )
        torch.cuda.synchronize()
        eager_caches[name] = OUTER.snapshot_mutated_page(
            decoder, start_position
        )
        eager_dispatch[name] = decoder.attention_dispatch_counts()

    attention_capture_memory: dict[str, Any] = {}
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, initial_caches[name])
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        decoder.capture_attention_graphs(start_position + 1)
        attention_capture_memory[name] = {
            "allocated_before_bytes": before_allocated,
            "allocated_after_bytes": int(torch.cuda.memory_allocated()),
            "allocated_delta_bytes": int(torch.cuda.memory_allocated())
            - before_allocated,
            "reserved_before_bytes": before_reserved,
            "reserved_after_bytes": int(torch.cuda.memory_reserved()),
            "reserved_delta_bytes": int(torch.cuda.memory_reserved())
            - before_reserved,
        }

    graph_outputs: dict[str, list[torch.Tensor]] = {}
    graph_caches: dict[str, dict[str, Any]] = {}
    graph_dispatch: dict[str, Any] = {}
    expected_attention_calls = int(model.config.num_hidden_layers) * PG.PAGE
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, initial_caches[name])
        decoder.reset_attention_dispatch_counts()
        graph_outputs[name] = execute_static_teacher_forced_page(
            decoder,
            teacher_forced_tokens,
            start_position,
            retain_logits=True,
        )
        torch.cuda.synchronize()
        graph_caches[name] = OUTER.snapshot_mutated_page(
            decoder, start_position
        )
        graph_dispatch[name] = decoder.attention_dispatch_counts()

    same_backend = {}
    for name in decoders:
        logits = PG.compare_logit_sequences(
            eager_outputs[name], graph_outputs[name]
        )
        cache = OUTER.compare_cache_snapshots(
            eager_caches[name], graph_caches[name]
        )
        dispatch_ok = (
            eager_dispatch[name]["eager_calls"] == expected_attention_calls
            and eager_dispatch[name]["graph_replays"] == 0
            and graph_dispatch[name]["graph_replays"]
            == expected_attention_calls
            and graph_dispatch[name]["eager_calls"] == 0
        )
        same_backend[name] = {
            "logits": logits,
            "cache": cache,
            "eager_attention_dispatch": eager_dispatch[name],
            "per_layer_graph_attention_dispatch": graph_dispatch[name],
            "dispatch_passed": dispatch_ok,
            "passed": logits["bitwise_identical"] and cache["passed"] and dispatch_ok,
        }
    cross_backend = {
        "fully_eager": OUTER.compare_backends(
            eager_outputs["flashinfer_fp16"], eager_outputs["page_gauge"]
        ),
        "per_layer_attention_graphs": OUTER.compare_backends(
            graph_outputs["flashinfer_fp16"], graph_outputs["page_gauge"]
        ),
    }
    cross_backend_passed = all(
        payload["minimum_logits_cosine"] >= min_logits_cosine
        and payload["top1_agreement_fraction"] >= min_top1_agreement
        for payload in cross_backend.values()
    )
    correctness_passed = (
        all(payload["passed"] for payload in same_backend.values())
        and cross_backend_passed
    )

    representative_profiles: dict[str, Any] = {
        "per_layer_attention_graphs": {},
        "fully_eager": {},
    }
    component_profiles: dict[str, Any] = {}
    candidate_residency_control: dict[str, Any] = {
        "enabled": release_baseline_for_candidate_residency,
        "terminal_fixture_mutation": release_baseline_for_candidate_residency,
        "co_resident_memory": None,
        "candidate_only_memory_before_profile": None,
        "candidate_only_memory_after_profile": None,
        "candidate_only_component_profile": None,
        "relocation_enabled": relocate_candidate_cache_after_release,
        "relocation": None,
        "relocated_component_profile": None,
    }
    timing_modes: dict[str, Any] = {}
    timing_skipped_reason = None
    if correctness_passed:
        cache_scrub = (
            torch.zeros(
                cache_scrub_mib * 1024 * 1024 // 4,
                device="cuda",
                dtype=torch.int32,
            )
            if run_matrix_timing
            else None
        )
        # Production per-layer attention graphs remain installed here.
        for name, decoder in decoders.items():
            profile = profile_representative_static_token(
                decoder,
                teacher_forced_tokens,
                start_position,
                initial_caches[name],
            )
            real_queries = profile.pop("selected_real_queries")
            representative_profiles["per_layer_attention_graphs"][name] = profile
            component_profiles[name] = profile_live_attention_components(
                decoder,
                real_queries,
                repeats=component_repeats,
                seed=seed + (0 if name == "flashinfer_fp16" else 100000),
            )
            OUTER.restore_mutated_page(decoder, initial_caches[name])
            torch.cuda.synchronize()
        if run_matrix_timing:
            assert cache_scrub is not None
            timing_modes["per_layer_attention_graphs"] = (
                measure_static_teacher_forced_modes(
                    decoders,
                    teacher_forced_tokens,
                    initial_caches,
                    start_position,
                    cache_scrub,
                    warmups,
                    repeats,
                    seed + 7000,
                )
            )

        for decoder in decoders.values():
            decoder.attention_graphs = None
        gc.collect()
        torch.cuda.empty_cache()
        for name, decoder in decoders.items():
            profile = profile_representative_static_token(
                decoder,
                teacher_forced_tokens,
                start_position,
                initial_caches[name],
            )
            profile.pop("selected_real_queries")
            representative_profiles["fully_eager"][name] = profile
            OUTER.restore_mutated_page(decoder, initial_caches[name])
            torch.cuda.synchronize()
        if run_matrix_timing:
            assert cache_scrub is not None
            timing_modes["fully_eager"] = measure_static_teacher_forced_modes(
                decoders,
                teacher_forced_tokens,
                initial_caches,
                start_position,
                cache_scrub,
                warmups,
                repeats,
                seed + 8000,
            )
        if release_baseline_for_candidate_residency:
            candidate_residency_control["co_resident_memory"] = gpu_memory_state()
            baseline_decoder = decoders["flashinfer_fp16"]
            baseline_decoder.attention_graphs = None
            baseline_key_bytes = (
                baseline_decoder.cache.key.numel()
                * baseline_decoder.cache.key.element_size()
            )
            baseline_value_bytes = (
                baseline_decoder.cache.value.numel()
                * baseline_decoder.cache.value.element_size()
            )
            # Terminal diagnostic mutation: update the shared cache object's
            # attributes so the outer baseline_cache reference releases the
            # same CUDA storages.  No correctness result is changed afterward.
            baseline_decoder.cache.key = torch.empty(
                0, device="cpu", dtype=torch.float16
            )
            baseline_decoder.cache.value = torch.empty(
                0, device="cpu", dtype=torch.float16
            )
            baseline_decoder.baseline_wrapper = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            candidate_residency_control.update(
                {
                    "released_baseline_key_bytes": int(baseline_key_bytes),
                    "released_baseline_value_bytes": int(baseline_value_bytes),
                    "candidate_only_memory_before_profile": gpu_memory_state(),
                }
            )
            candidate = decoders["page_gauge"]
            candidate_profile = profile_representative_static_token(
                candidate,
                teacher_forced_tokens,
                start_position,
                initial_caches["page_gauge"],
            )
            candidate_queries = candidate_profile.pop("selected_real_queries")
            co_resident_query_hashes = representative_profiles[
                "per_layer_attention_graphs"
            ]["page_gauge"]["selected_real_query_sha256"]
            candidate_only_query_hashes = candidate_profile[
                "selected_real_query_sha256"
            ]
            candidate_only_components = profile_live_attention_components(
                candidate,
                candidate_queries,
                repeats=component_repeats,
                seed=seed + 200000,
            )
            OUTER.restore_mutated_page(
                candidate, initial_caches["page_gauge"]
            )
            torch.cuda.synchronize()
            candidate_residency_control.update(
                {
                    "candidate_only_representative_profile": candidate_profile,
                    "candidate_only_component_profile": candidate_only_components,
                    "candidate_only_memory_after_profile": gpu_memory_state(),
                    "co_resident_query_sha256": co_resident_query_hashes,
                    "candidate_only_query_sha256": candidate_only_query_hashes,
                    "query_hashes_bitwise_identical": (
                        co_resident_query_hashes == candidate_only_query_hashes
                    ),
                    "interpretation": (
                        "compare live_attention_component_profiles.page_gauge "
                        "(FI+PG co-resident) against candidate_only_component_profile"
                    ),
                }
            )
            if relocate_candidate_cache_after_release:
                candidate.attention_graphs = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                relocation = relocate_page_gauge_cache_tensors(candidate)
                if not (
                    relocation["all_sample_hashes_bitwise_identical"]
                    and relocation["all_pointers_changed"]
                ):
                    raise RuntimeError(
                        "PageGauge cache relocation pointer/hash gate failed"
                    )
                relocated_profile = profile_representative_static_token(
                    candidate,
                    teacher_forced_tokens,
                    start_position,
                    initial_caches["page_gauge"],
                )
                relocated_queries = relocated_profile.pop(
                    "selected_real_queries"
                )
                relocated_query_hashes = relocated_profile[
                    "selected_real_query_sha256"
                ]
                relocated_components = profile_live_attention_components(
                    candidate,
                    relocated_queries,
                    repeats=component_repeats,
                    seed=seed + 300000,
                )
                OUTER.restore_mutated_page(
                    candidate, initial_caches["page_gauge"]
                )
                torch.cuda.synchronize()
                candidate_residency_control.update(
                    {
                        "relocation": relocation,
                        "relocated_representative_profile": relocated_profile,
                        "relocated_component_profile": relocated_components,
                        "relocated_memory_after_profile": gpu_memory_state(),
                        "relocated_query_sha256": relocated_query_hashes,
                        "relocated_query_hashes_bitwise_identical": (
                            co_resident_query_hashes
                            == candidate_only_query_hashes
                            == relocated_query_hashes
                        ),
                    }
                )
    else:
        timing_skipped_reason = (
            "same-backend eager/attention-graph equivalence, dispatch, or "
            "cross-backend static-token correctness gate failed"
        )

    if release_baseline_for_candidate_residency:
        OUTER.restore_mutated_page(
            decoders["page_gauge"], initial_caches["page_gauge"]
        )
        torch.cuda.synchronize()
    else:
        _restore_fixture_caches(decoders, initial_caches)
    for decoder in decoders.values():
        decoder.attention_graphs = None
    import flashinfer

    major, minor = torch.cuda.get_device_capability()
    return {
        "schema_version": 1,
        "experiment": "page_gauge_live_prefill_static_teacher_forced_control",
        "claim_scope": (
            f"diagnostic batch-{batch_size}, fixed-context, one-page coherent "
            "teacher-forced control; no generated causal dependency"
        ),
        "batch_size": batch_size,
        "start_position": start_position,
        "decode_steps": PG.PAGE,
        "output_tokens_per_sequence": batch_size * PG.PAGE,
        "teacher_forced_token_ids": teacher_forced_tokens.detach().cpu().tolist(),
        "fixture_provenance": fixture_provenance or {},
        "execution_boundaries": {
            "common": [
                "one predeclared CUDA int64 [16,B] coherent corpus tensor",
                "16 decoder.step calls with no argmax or feedback dependency",
                "one synchronization before and after each measured block",
            ],
            "logits_lifetimes": {
                "discard_logits": "explicitly release each logits tensor per step",
                "retain_logits": "retain all 16 FP32 logits through block completion",
            },
            "pair_order": "AB, BA repeated; raw warmup and measurement blocks retained",
            "cache_modes": ["cache_neutral_start", "cache_hot"],
        },
        "correctness": {
            "same_backend_fully_eager_vs_per_layer_graphs": same_backend,
            "cross_backend": cross_backend,
            "thresholds": {
                "minimum_logits_cosine": min_logits_cosine,
                "minimum_top1_agreement": min_top1_agreement,
                "same_backend_logits_and_cache_must_be_bitwise_identical": True,
                "attention_dispatch_must_be_exact": True,
            },
            "passed": correctness_passed,
        },
        "timing_modes": timing_modes,
        "matrix_timing_enabled": run_matrix_timing,
        "timing_skipped_reason": timing_skipped_reason,
        "representative_token_profiles": representative_profiles,
        "live_attention_component_profiles": component_profiles,
        "candidate_residency_control": candidate_residency_control,
        "page_gauge_live_scale_distributions": scale_distributions,
        "cache_storage": cache_storage or {},
        "physical_page_layout": "request-major",
        "live_fixture_reuse": {
            "model_reused": True,
            "decoder_objects_reused": True,
            "cache_objects_reused": True,
            "cache_serialization": False,
            "cache_duplication": False,
            "only_mutated_page_snapshots_cloned": True,
        },
        "cuda_memory": {
            "at_fixture_entry": memory_at_entry,
            "attention_graph_capture_deltas": attention_capture_memory,
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
        "source_sha256_at_fixture_entry": source_hashes,
    }


def _cross_backend_validation(
    validation_outputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for execution in ("eager", "graph"):
        output_key = f"{execution}_outputs"
        token_key = f"{execution}_tokens"
        result[execution] = {
            "logits": OUTER.compare_backends(
                validation_outputs["flashinfer_fp16"][output_key],
                validation_outputs["page_gauge"][output_key],
            ),
            "generated_tokens": compare_token_sequences(
                validation_outputs["flashinfer_fp16"][token_key],
                validation_outputs["page_gauge"][token_key],
            ),
        }
    return result


def _cross_logits_passed(
    cross_backend: dict[str, Any],
    min_logits_cosine: float,
    min_top1_agreement: float,
) -> bool:
    return all(
        payload["logits"]["minimum_logits_cosine"] >= min_logits_cosine
        and payload["logits"]["top1_agreement_fraction"]
        >= min_top1_agreement
        for payload in cross_backend.values()
    )


def _source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@torch.inference_mode()
def run_token_step_graph_fixture(
    *,
    model: Any,
    decoders: dict[str, Any],
    teacher_forced_tokens: torch.Tensor,
    start_position: int,
    capture_warmups: int,
    warmups: int,
    repeats: int,
    cache_scrub_mib: int,
    seed: int,
    min_logits_cosine: float,
    min_top1_agreement: float,
    initial_caches: dict[str, dict[str, Any]] | None = None,
    fixture_provenance: dict[str, Any] | None = None,
    cache_storage: dict[str, Any] | None = None,
    additional_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Capture and time token-step graphs on a caller-owned live fixture.

    The function never constructs or serializes a model or cache.  The caller
    passes the two decoders that share its live model and the page snapshots
    taken immediately after prefill.  Generated feedback is the preferred
    timing mode.  If and only if the backends' generated tokens diverge, the
    fixture falls back to the supplied teacher-forced inputs so both backends
    still execute the same coherent token stream.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if capture_warmups <= 0 or warmups < 0:
        raise ValueError("capture warmups must be positive and warmups non-negative")
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    if cache_scrub_mib <= 0:
        raise ValueError("cache scrub size must be positive")
    if not (0.0 <= min_top1_agreement <= 1.0):
        raise ValueError("top-1 threshold must lie in [0,1]")
    if start_position % PG.PAGE:
        raise ValueError("token-step fixture must begin on a page boundary")
    expected_names = {"flashinfer_fp16", "page_gauge"}
    if set(decoders) != expected_names:
        raise ValueError(f"decoder keys must be exactly {sorted(expected_names)}")
    batch_size = decoders["flashinfer_fp16"].batch_size
    if any(decoder.batch_size != batch_size for decoder in decoders.values()):
        raise ValueError("both decoders must have the same batch size")

    teacher_forced_tokens = teacher_forced_tokens.to(
        device="cuda", dtype=torch.long
    ).contiguous()
    validate_token_inputs(teacher_forced_tokens, TEACHER_FORCED, batch_size)
    generated_seed = teacher_forced_tokens[0]
    generated_inputs = {name: generated_seed for name in decoders}
    teacher_inputs = {name: teacher_forced_tokens for name in decoders}
    if initial_caches is None:
        initial_caches = snapshot_fixture_caches(decoders, start_position)
    elif set(initial_caches) != expected_names:
        raise ValueError("initial cache snapshots do not match decoder keys")

    source_paths = tuple(
        dict.fromkeys(
            (
                *OUTER.DEPENDENCY_PATHS,
                OUTER_GRAPH_PATH,
                Path(__file__).resolve(),
                *(Path(path).resolve() for path in additional_source_paths),
            )
        )
    )
    source_hashes = {
        _source_label(path): source_sha256(path)
        for path in source_paths
        if path.is_file()
    }
    memory_at_entry = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    # Existing per-layer attention graphs are outside this diagnostic's graph
    # boundary. Dropping only those graph owners preserves the live model and
    # both caches while allowing the whole-decoder graphs to own their pools.
    for decoder in decoders.values():
        decoder.attention_graphs = None
    gc.collect()
    torch.cuda.empty_cache()
    _restore_fixture_caches(decoders, initial_caches)

    captured: dict[str, CapturedTokenPage] = {}
    capture_memory: dict[str, Any] = {}
    for name, decoder in decoders.items():
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        print(
            f"Capturing {PG.PAGE} coherent per-token graphs: {name}",
            flush=True,
        )
        captured[name] = capture_token_page(
            decoder,
            generated_seed,
            start_position,
            initial_caches[name],
            capture_warmups,
        )
        capture_memory[name] = {
            "allocated_before_bytes": before_allocated,
            "allocated_after_bytes": int(torch.cuda.memory_allocated()),
            "allocated_delta_bytes": int(torch.cuda.memory_allocated())
            - before_allocated,
            "reserved_before_bytes": before_reserved,
            "reserved_after_bytes": int(torch.cuda.memory_reserved()),
            "reserved_delta_bytes": int(torch.cuda.memory_reserved())
            - before_reserved,
        }

    same_backend: dict[str, Any] = {}
    validation_outputs: dict[str, Any] = {}
    for input_mode, token_inputs in (
        (GENERATED_FEEDBACK, generated_inputs),
        (TEACHER_FORCED, teacher_inputs),
    ):
        same_backend[input_mode] = {}
        validation_outputs[input_mode] = {}
        for name in decoders:
            validation, outputs = validate_backend(
                captured[name],
                token_inputs[name],
                input_mode,
                initial_caches[name],
            )
            same_backend[input_mode][name] = validation
            validation_outputs[input_mode][name] = outputs

    cross_backend = {
        input_mode: _cross_backend_validation(validation_outputs[input_mode])
        for input_mode in (GENERATED_FEEDBACK, TEACHER_FORCED)
    }
    graph_equivalence_passed = all(
        validation["passed"]
        for mode_payload in same_backend.values()
        for validation in mode_payload.values()
    )
    generated_logits_passed = _cross_logits_passed(
        cross_backend[GENERATED_FEEDBACK],
        min_logits_cosine,
        min_top1_agreement,
    )
    generated_tokens_identical = all(
        payload["generated_tokens"]["bitwise_identical"]
        for payload in cross_backend[GENERATED_FEEDBACK].values()
    )
    teacher_forced_passed = _cross_logits_passed(
        cross_backend[TEACHER_FORCED],
        min_logits_cosine,
        min_top1_agreement,
    )

    selected_input_mode: str | None = None
    selection_reason: str
    if generated_tokens_identical and generated_logits_passed:
        selected_input_mode = GENERATED_FEEDBACK
        selection_reason = (
            "both backends generated exactly the same tokens and passed the "
            "cross-backend logits gate"
        )
    elif not generated_tokens_identical and teacher_forced_passed:
        selected_input_mode = TEACHER_FORCED
        selection_reason = (
            "generated feedback diverged, so matched actual-continuation "
            "teacher-forced inputs were selected; no generated-feedback "
            "throughput claim is made"
        )
    elif not generated_tokens_identical:
        selection_reason = (
            "generated feedback diverged and teacher-forced cross-backend "
            "correctness also failed"
        )
    else:
        selection_reason = "generated-feedback cross-backend logits gate failed"

    timing_modes: dict[str, Any] = {}
    timing_skipped_reason: str | None = None
    selected_cross_backend_passed = selected_input_mode is not None
    if not graph_equivalence_passed:
        timing_skipped_reason = "same-backend eager/graph/cache bitwise gate failed"
    elif selected_input_mode is None:
        timing_skipped_reason = selection_reason
    else:
        cache_scrub = torch.zeros(
            cache_scrub_mib * 1024 * 1024 // 4,
            device="cuda",
            dtype=torch.int32,
        )
        timing_inputs = (
            generated_inputs
            if selected_input_mode == GENERATED_FEEDBACK
            else teacher_inputs
        )
        timing_modes = measure_modes(
            captured,
            timing_inputs,
            selected_input_mode,
            initial_caches,
            cache_scrub,
            warmups,
            repeats,
            seed + 3000,
        )

    replay_profile_input_mode = selected_input_mode or GENERATED_FEEDBACK
    replay_pattern_diagnostics = profile_graph_replay_patterns(
        captured,
        teacher_forced_tokens,
        replay_profile_input_mode,
        initial_caches,
    )

    _restore_fixture_caches(decoders, initial_caches)
    import flashinfer

    major, minor = torch.cuda.get_device_capability()
    token_ids = teacher_forced_tokens.detach().cpu().tolist()
    result = {
        "schema_version": 2,
        "experiment": "page_gauge_live_prefill_per_token_decoder_graph",
        "claim_scope": (
            f"synchronized batch-{batch_size}, fixed-context, one-page "
            "whole-decoder graph timing on caller-owned live coherent caches; "
            "not a ragged, cross-page, scheduler, or prefill claim"
        ),
        "batch_size": batch_size,
        "start_position": start_position,
        "decode_steps": PG.PAGE,
        "output_tokens_per_sequence": batch_size * PG.PAGE,
        "fixture_provenance": fixture_provenance or {},
        "token_protocol": {
            "teacher_forced_layout": "step-major [16,B]",
            "teacher_forced_token_ids": token_ids,
            "teacher_forced_token_ids_sha256": hashlib.sha256(
                json.dumps(
                    token_ids,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "initial_token_semantics": (
                "actual successor of each model-prefilled prefix"
            ),
            "initial_token_ids": token_ids[0],
            "preferred_timing_mode": GENERATED_FEEDBACK,
            "selected_timing_mode": selected_input_mode,
            "selection_reason": selection_reason,
            "generated_feedback_requires_exact_cross_backend_tokens": True,
        },
        "graph_protocol": {
            "graphs_per_backend": PG.PAGE,
            "one_graph_per_page_offset": True,
            "captured_boundary_per_graph": (
                "embedding, every decoder layer, append/finalization, attention, "
                "projections, MLP, final norm, LM head, FP32 logits"
            ),
            "timed_per_page_operations": {
                "device_to_device_graph_input_copies": PG.PAGE,
                "whole_decoder_graph_replays": PG.PAGE,
                "fp32_greedy_argmax_operations": PG.PAGE,
                "host_synchronizations_between_tokens": 0,
            },
            "seed_copy_inside_timed_boundary": True,
            "all_feedback_or_teacher_token_copies_inside_timed_boundary": True,
            "nested_attention_graphs_disabled": True,
            "shared_private_graph_pool_per_backend": True,
            "required_replay_order": "page offsets 0 through 15",
            "plan_bucket_invariant_over_page": True,
            "plan_signature": {
                name: OUTER.jsonable_signature(page.invariant_plan_signature)
                for name, page in captured.items()
            },
            "per_offset_position_and_signature": {
                name: [
                    {
                        "offset": offset,
                        "position": step.position,
                        "plan_signature": OUTER.jsonable_signature(
                            step.plan_signature
                        ),
                    }
                    for offset, step in enumerate(page.steps)
                ]
                for name, page in captured.items()
            },
            "cache_state_restored_and_checked_for_every_request": True,
            "pair_order": "AB, BA repeated; raw blocks retained",
            "cache_modes": ["cache_neutral_start", "cache_hot"],
            "excluded": [
                "tokenizer and host-to-device tokenizer output",
                "non-greedy sampling and sampling policy",
                "request scheduling and ragged batches",
                "next-page plan selection or graph recapture",
                "prefill latency",
            ],
        },
        "correctness": {
            "same_backend_eager_vs_graph_and_cache": same_backend,
            "cross_backend": cross_backend,
            "thresholds": {
                "minimum_logits_cosine": min_logits_cosine,
                "minimum_top1_agreement": min_top1_agreement,
                "generated_tokens_must_be_bitwise_identical_for_dynamic_timing": True,
            },
            "graph_equivalence_passed": graph_equivalence_passed,
            "generated_feedback_logits_passed": generated_logits_passed,
            "generated_feedback_tokens_bitwise_identical": (
                generated_tokens_identical
            ),
            "teacher_forced_cross_backend_passed": teacher_forced_passed,
            "selected_cross_backend_passed": selected_cross_backend_passed,
            "passed": graph_equivalence_passed and selected_cross_backend_passed,
        },
        "timing_input_mode": selected_input_mode,
        "timing_modes": timing_modes,
        "timing_skipped_reason": timing_skipped_reason,
        "graph_replay_pattern_diagnostics": replay_pattern_diagnostics,
        "cache_storage": cache_storage or {},
        "physical_page_layout": "request-major",
        "live_fixture_reuse": {
            "model_reused": True,
            "decoder_objects_reused": True,
            "cache_objects_reused": True,
            "cache_serialization": False,
            "cache_duplication": False,
            "only_mutated_page_snapshots_cloned": True,
        },
        "cuda_memory": {
            "at_fixture_entry": memory_at_entry,
            "capture_deltas": capture_memory,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "final_allocated_bytes_while_graphs_live": int(
                torch.cuda.memory_allocated()
            ),
            "final_reserved_bytes_while_graphs_live": int(
                torch.cuda.memory_reserved()
            ),
        },
        "model": {
            "name_or_path": getattr(model.config, "_name_or_path", None),
            "revision": getattr(model.config, "_commit_hash", None),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
        "source_sha256_at_fixture_entry": source_hashes,
    }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.decode_steps != PG.PAGE:
        raise SystemExit("the token-step graph diagnostic requires exactly 16 steps")
    if (
        args.context <= args.exact_tail
        or args.context % PG.PAGE
        or args.exact_tail % PG.PAGE
    ):
        raise SystemExit(
            "context and exact tail must be page aligned; context must exceed tail"
        )
    if args.repeats <= 0 or args.repeats % 2:
        raise SystemExit("repeats must be a positive even number")
    if args.warmups < 0 or args.capture_warmups <= 0:
        raise SystemExit("warmups are invalid")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    if args.batch_size > 1 and args.center_restore == "projection_bias":
        raise SystemExit(
            "batched request-specific centers require attention_add"
        )
    if (
        args.tail_attention == "fused_kernel"
        and args.center_restore != "attention_add"
    ):
        raise SystemExit("the fused tail kernel requires attention_add")

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    source_paths = (
        *OUTER.DEPENDENCY_PATHS,
        OUTER_GRAPH_PATH,
        Path(__file__),
    )
    source_hashes = {
        str(path.relative_to(ROOT)): source_sha256(path)
        for path in source_paths
    }
    torch.cuda.reset_peak_memory_stats()
    print("Loading model, request-major caches, and kernels...", flush=True)
    model, baseline, candidate, cache_storage = OUTER.make_decoders(args)
    decoders = {
        "flashinfer_fp16": baseline,
        "page_gauge": candidate,
    }

    generator = torch.Generator().manual_seed(args.seed)
    initial_values = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.batch_size,),
        generator=generator,
        dtype=torch.long,
    )
    initial_tokens = {
        name: initial_values.to(device="cuda") for name in decoders
    }
    initial_caches = {
        name: OUTER.snapshot_mutated_page(decoder, args.context)
        for name, decoder in decoders.items()
    }
    torch.cuda.synchronize()

    captured: dict[str, CapturedTokenPage] = {}
    capture_memory = {}
    for name, decoder in decoders.items():
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        print(f"Capturing 16 per-token decoder graphs: {name}", flush=True)
        captured[name] = capture_token_page(
            decoder,
            initial_tokens[name],
            args.context,
            initial_caches[name],
            args.capture_warmups,
        )
        capture_memory[name] = {
            "allocated_before": before_allocated,
            "allocated_after": int(torch.cuda.memory_allocated()),
            "allocated_delta": int(torch.cuda.memory_allocated())
            - before_allocated,
            "reserved_before": before_reserved,
            "reserved_after": int(torch.cuda.memory_reserved()),
            "reserved_delta": int(torch.cuda.memory_reserved())
            - before_reserved,
        }

    same_backend = {}
    validation_outputs = {}
    for name in decoders:
        validation, outputs = validate_backend(
            captured[name],
            initial_tokens[name],
            GENERATED_FEEDBACK,
            initial_caches[name],
        )
        same_backend[name] = validation
        validation_outputs[name] = outputs

    cross_backend = {}
    for mode in ("eager", "graph"):
        output_key = f"{mode}_outputs"
        token_key = f"{mode}_tokens"
        cross_backend[mode] = {
            "logits": OUTER.compare_backends(
                validation_outputs["flashinfer_fp16"][output_key],
                validation_outputs["page_gauge"][output_key],
            ),
            "generated_tokens": compare_token_sequences(
                validation_outputs["flashinfer_fp16"][token_key],
                validation_outputs["page_gauge"][token_key],
            ),
        }
    graph_equivalence_passed = all(
        result["passed"] for result in same_backend.values()
    )
    cross_backend_passed = all(
        payload["logits"]["minimum_logits_cosine"]
        >= args.min_logits_cosine
        and payload["logits"]["top1_agreement_fraction"]
        >= args.min_top1_agreement
        and payload["generated_tokens"]["bitwise_identical"]
        for payload in cross_backend.values()
    )

    timing_modes: dict[str, Any] = {}
    timing_skipped_reason = None
    if graph_equivalence_passed and cross_backend_passed:
        cache_scrub = torch.zeros(
            args.cache_scrub_mib * 1024 * 1024 // 4,
            device="cuda",
            dtype=torch.int32,
        )
        timing_modes = measure_modes(
            captured,
            initial_tokens,
            GENERATED_FEEDBACK,
            initial_caches,
            cache_scrub,
            args.warmups,
            args.repeats,
            args.seed + 3000,
        )
    else:
        timing_skipped_reason = (
            "same-backend graph equivalence or cross-backend generated-token "
            "agreement failed"
        )

    import flashinfer

    result = {
        "schema_version": 1,
        "experiment": "page_gauge_per_token_whole_decoder_cuda_graph",
        "claim_scope": (
            f"production-like synchronized batch-{args.batch_size} per-token "
            "whole-decoder graph with timed greedy feedback over one fixed page; "
            "not a ragged or cross-page serving claim"
        ),
        "batch_size": args.batch_size,
        "decode_steps": PG.PAGE,
        "output_tokens_per_sequence": args.batch_size * PG.PAGE,
        "configuration": {
            **{
                key: value
                for key, value in vars(args).items()
                if key != "output"
            },
            "output": str(args.output),
            "initial_tokens": initial_values.tolist(),
        },
        "graph_protocol": {
            "graphs_per_backend": PG.PAGE,
            "one_graph_per_page_offset": True,
            "captured_boundary_per_graph": (
                "embedding, every decoder layer, append/finalization, attention, "
                "projections, MLP, final norm, LM head, FP32 logits"
            ),
            "timed_between_token_work": [
                "device-to-device copy into shared static token buffer",
                "FP32 greedy argmax into a preallocated dynamic token buffer",
            ],
            "timed_initial_work": (
                "device-to-device copy from a pre-created seed token buffer"
            ),
            "no_host_synchronization_between_tokens": True,
            "wall_time_includes_python_graph_selection_and_launch": True,
            "cuda_event_time_includes_all_stream_work_and_stream_idle_gaps": True,
            "nested_attention_graphs_disabled": True,
            "shared_private_graph_pool_per_backend": True,
            "required_replay_order": "page offsets 0 through 15",
            "plan_bucket_invariant_over_page": True,
            "plan_signature": {
                name: OUTER.jsonable_signature(
                    page.invariant_plan_signature
                )
                for name, page in captured.items()
            },
            "per_offset_position_and_signature": {
                name: [
                    {
                        "offset": offset,
                        "position": step.position,
                        "plan_signature": OUTER.jsonable_signature(
                            step.plan_signature
                        ),
                    }
                    for offset, step in enumerate(page.steps)
                ]
                for name, page in captured.items()
            },
            "cache_state_restored_and_checked_for_every_request": True,
            "pair_order": "AB, BA repeated; raw blocks retained",
            "excluded": [
                "tokenizer and host-to-device tokenizer output",
                "non-greedy sampling and sampling policy",
                "request scheduling and ragged batches",
                "next-page plan selection or graph recapture",
                "prefill",
            ],
        },
        "correctness": {
            "same_backend_eager_vs_graph": same_backend,
            "cross_backend": cross_backend,
            "thresholds": {
                "minimum_logits_cosine": args.min_logits_cosine,
                "minimum_top1_agreement": args.min_top1_agreement,
                "generated_tokens_must_match": True,
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
        raise SystemExit("per-token graph correctness validation failed")


if __name__ == "__main__":
    main()
