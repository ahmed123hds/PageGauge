#!/usr/bin/env python3
"""GPU-resident greedy 16-token whole-decoder CUDA-graph diagnostic.

One matched CUDA graph is captured per backend.  Each graph contains a
device-to-device copy from a mutable seed-token buffer followed by 16 complete
decoder steps.  The FP32 logits from every step are reduced by a GPU argmax,
and that generated token tensor directly feeds the next decoder step.  Thus a
timed block uses one host graph launch and has no host intervention between
tokens.

The measurement is a throughput-oriented, fixed-batch, fixed-context greedy
block-decode experiment.  It reports latency for all 16 output positions as a
single block; it is not a streaming-token-latency, ragged-batch, sampling, or
cross-page serving claim.  Positions are deliberately frozen to
20480..20495 so both backends use the independently selected 20K scheduler.
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
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTER_GRAPH_PATH = ROOT / "diagnostics/benchmark_full_sequence_graph.py"
START_POSITION = 20480
BACKENDS = ("flashinfer_fp16", "page_gauge")


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OUTER = load_local_module(
    "page_gauge_outer_graph_helpers_for_generated_sequence",
    OUTER_GRAPH_PATH,
)
PG = OUTER.PG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--context", type=int, default=START_POSITION)
    parser.add_argument("--decode-steps", type=int, default=PG.PAGE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=128)
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
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--capture-warmups", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_fixture_shape(
    decoders: dict[str, Any], seed_tokens: torch.Tensor, start_position: int
) -> int:
    if set(decoders) != set(BACKENDS):
        raise ValueError(f"decoder keys must be exactly {list(BACKENDS)}")
    if start_position != START_POSITION:
        raise ValueError(
            "generated block protocol freezes positions to "
            f"{START_POSITION}..{START_POSITION + PG.PAGE - 1}"
        )
    batch_size = int(decoders[BACKENDS[0]].batch_size)
    if batch_size <= 0 or any(
        int(decoder.batch_size) != batch_size for decoder in decoders.values()
    ):
        raise ValueError("both decoders must have the same positive batch size")
    if (
        seed_tokens.device.type != "cuda"
        or seed_tokens.dtype != torch.long
        or tuple(seed_tokens.shape) != (batch_size,)
    ):
        raise ValueError("seed tokens must be a CUDA int64 tensor with shape [B]")
    return batch_size


def clone_sequence(values: list[torch.Tensor]) -> list[torch.Tensor]:
    result = [value.detach().clone() for value in values]
    torch.cuda.synchronize()
    return result


@torch.inference_mode()
def execute_generated_block(
    decoder: Any,
    seed_source: torch.Tensor,
    start_position: int,
    initial_token: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Execute the exact operation sequence captured by the outer graph."""
    initial_token.copy_(seed_source)
    token = initial_token
    logits_sequence: list[torch.Tensor] = []
    generated_sequence: list[torch.Tensor] = []
    for offset in range(PG.PAGE):
        logits = decoder.step(token, start_position + offset)[0]
        generated = torch.argmax(logits, dim=-1)
        logits_sequence.append(logits)
        generated_sequence.append(generated)
        # No copy or CPU scalar extraction: the argmax output from this step is
        # the embedding input tensor for the following captured decoder step.
        token = generated
    return logits_sequence, generated_sequence


class CapturedGeneratedBlock:
    def __init__(
        self,
        *,
        decoder: Any,
        graph: torch.cuda.CUDAGraph,
        replay_stream: torch.cuda.Stream,
        seed_source: torch.Tensor,
        initial_token: torch.Tensor,
        logits: list[torch.Tensor],
        generated_tokens: list[torch.Tensor],
        start_position: int,
        plan_signature: Any,
    ) -> None:
        self.decoder = decoder
        self.graph = graph
        self.replay_stream = replay_stream
        self.seed_source = seed_source
        self.initial_token = initial_token
        self.logits = logits
        self.generated_tokens = generated_tokens
        self.start_position = start_position
        self.plan_signature = plan_signature

    @property
    def batch_size(self) -> int:
        return int(self.seed_source.numel())

    def assert_replay_safe(self) -> None:
        if self.decoder.attention_graphs is not None:
            raise RuntimeError("nested attention graphs must remain disabled")
        if len(self.logits) != PG.PAGE or len(self.generated_tokens) != PG.PAGE:
            raise RuntimeError("generated block must own exactly 16 output steps")
        if self.decoder.plan_signature() != self.plan_signature:
            raise RuntimeError("generated-block graph plan signature is stale")

    def stage_seed(self, seed_tokens: torch.Tensor) -> None:
        if (
            seed_tokens.device.type != "cuda"
            or seed_tokens.dtype != torch.long
            or tuple(seed_tokens.shape) != (self.batch_size,)
        ):
            raise ValueError("seed tokens must be CUDA int64 with shape [B]")
        self.seed_source.copy_(seed_tokens)

    def replay(self) -> None:
        self.assert_replay_safe()
        with torch.cuda.stream(self.replay_stream):
            self.graph.replay()

    def replay_and_synchronize(self) -> None:
        self.replay()
        torch.cuda.synchronize()

    def timed_replay(self) -> tuple[float, float]:
        self.assert_replay_safe()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        with torch.cuda.stream(self.replay_stream):
            begin.record()
            self.graph.replay()
            end.record()
        torch.cuda.synchronize()
        return (
            float(begin.elapsed_time(end)),
            (time.perf_counter() - wall_start) * 1e3,
        )


def capture_generated_block(
    decoder: Any,
    seed_tokens: torch.Tensor,
    start_position: int,
    initial_cache: dict[str, Any],
    capture_warmups: int,
) -> CapturedGeneratedBlock:
    """Capture one graph containing seed copy, 16 steps, and 16 argmaxes."""
    decoder.attention_graphs = None
    plan_signature = OUTER.assert_plan_bucket_is_invariant(
        decoder, start_position, PG.PAGE
    )

    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    warm_seed = seed_tokens.detach().clone()
    warm_initial = torch.empty_like(warm_seed)
    capture_stream.wait_stream(current_stream)
    warm_outputs = None
    with torch.cuda.stream(capture_stream):
        for _ in range(capture_warmups):
            warm_outputs = execute_generated_block(
                decoder, warm_seed, start_position, warm_initial
            )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    del warm_outputs

    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    seed_source = seed_tokens.detach().clone()
    initial_token = torch.empty_like(seed_source)
    graph = torch.cuda.CUDAGraph()
    graph_logits: list[torch.Tensor] = []
    graph_tokens: list[torch.Tensor] = []
    capture_stream.wait_stream(current_stream)
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_logits, graph_tokens = execute_generated_block(
            decoder, seed_source, start_position, initial_token
        )
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    return CapturedGeneratedBlock(
        decoder=decoder,
        graph=graph,
        replay_stream=capture_stream,
        seed_source=seed_source,
        initial_token=initial_token,
        logits=graph_logits,
        generated_tokens=graph_tokens,
        start_position=start_position,
        plan_signature=plan_signature,
    )


def compare_token_sequences(
    expected: list[torch.Tensor], observed: list[torch.Tensor]
) -> dict[str, Any]:
    if len(expected) != len(observed) or not expected:
        raise ValueError("token sequences must have equal non-zero length")
    if any(left.shape != right.shape for left, right in zip(expected, observed)):
        raise ValueError("token sequence tensor shapes do not match")
    per_step = [left.eq(right).tolist() for left, right in zip(expected, observed)]
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


def snapshot_fixture_caches(
    decoders: dict[str, Any], start_position: int
) -> dict[str, dict[str, Any]]:
    if set(decoders) != set(BACKENDS):
        raise ValueError(f"decoder keys must be exactly {list(BACKENDS)}")
    snapshots = {
        name: OUTER.snapshot_mutated_page(decoder, start_position)
        for name, decoder in decoders.items()
    }
    torch.cuda.synchronize()
    return snapshots


def restore_fixture_caches(
    decoders: dict[str, Any], snapshots: dict[str, dict[str, Any]]
) -> None:
    for name, decoder in decoders.items():
        OUTER.restore_mutated_page(decoder, snapshots[name])
    torch.cuda.synchronize()


def validate_backend(
    captured: CapturedGeneratedBlock,
    seed_tokens: torch.Tensor,
    initial_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decoder = captured.decoder
    eager_initial_token = torch.empty_like(seed_tokens)
    OUTER.restore_mutated_page(decoder, initial_cache)
    torch.cuda.synchronize()
    eager_logits_raw, eager_tokens_raw = execute_generated_block(
        decoder,
        seed_tokens,
        captured.start_position,
        eager_initial_token,
    )
    eager_logits = clone_sequence(eager_logits_raw)
    eager_tokens = clone_sequence(eager_tokens_raw)
    eager_cache = OUTER.snapshot_mutated_page(decoder, captured.start_position)
    torch.cuda.synchronize()

    OUTER.restore_mutated_page(decoder, initial_cache)
    captured.stage_seed(seed_tokens)
    torch.cuda.synchronize()
    captured.replay_and_synchronize()
    graph_logits = clone_sequence(captured.logits)
    graph_tokens = clone_sequence(captured.generated_tokens)
    graph_cache = OUTER.snapshot_mutated_page(decoder, captured.start_position)
    torch.cuda.synchronize()

    # A no-restore replay verifies the steady-state timing semantics.  Every
    # slot becomes visible only after the graph overwrites it causally.
    captured.replay_and_synchronize()
    repeated_logits = clone_sequence(captured.logits)
    repeated_tokens = clone_sequence(captured.generated_tokens)
    repeated_cache = OUTER.snapshot_mutated_page(decoder, captured.start_position)
    torch.cuda.synchronize()

    eager_vs_graph_logits = PG.compare_logit_sequences(eager_logits, graph_logits)
    repeated_graph_logits = PG.compare_logit_sequences(
        graph_logits, repeated_logits
    )
    eager_vs_graph_tokens = compare_token_sequences(eager_tokens, graph_tokens)
    repeated_graph_tokens = compare_token_sequences(graph_tokens, repeated_tokens)
    eager_vs_graph_cache = OUTER.compare_cache_snapshots(eager_cache, graph_cache)
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
            "eager_logits": eager_logits,
            "graph_logits": graph_logits,
            "eager_tokens": eager_tokens,
            "graph_tokens": graph_tokens,
        },
    )


def cross_backend_validation(
    validation_outputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for execution in ("eager", "graph"):
        result[execution] = {
            "logits": OUTER.compare_backends(
                validation_outputs[BACKENDS[0]][f"{execution}_logits"],
                validation_outputs[BACKENDS[1]][f"{execution}_logits"],
            ),
            "generated_tokens": compare_token_sequences(
                validation_outputs[BACKENDS[0]][f"{execution}_tokens"],
                validation_outputs[BACKENDS[1]][f"{execution}_tokens"],
            ),
        }
    return result


def add_block_latency(summary: dict[str, Any]) -> None:
    """Make whole-block latency explicit; do not imply streaming latency."""
    for backend in BACKENDS:
        gpu = summary[backend]["gpu_ms"]
        wall = summary[backend]["wall_ms"]
        summary[backend]["gpu_resident_16_token_block_latency_ms"] = {
            "gpu_p50": statistics.median(gpu),
            "gpu_mean": statistics.mean(gpu),
            "wall_p50": statistics.median(wall),
            "wall_mean": statistics.mean(wall),
            "wall_p95": PG.percentile(wall, 0.95),
        }
        amortized_keys = (
            "gpu_p50_ms_per_decode_step",
            "wall_p50_ms_per_decode_step",
            "wall_mean_ms_per_decode_step",
            "wall_p95_ms_per_decode_step",
            "gpu_p50_ms_per_token",
            "wall_p50_ms_per_token",
            "wall_mean_ms_per_token",
            "wall_p95_ms_per_token",
            "wall_decode_steps_per_second",
        )
        summary[backend]["derived_amortized_metrics_not_streaming_latency"] = {
            key: summary[backend].pop(key) for key in amortized_keys
        }
        summary[backend]["latency_semantics"] = (
            "one synchronized 16-position generated block; not time to first "
            "token or per-token streaming latency"
        )


def measure_modes(
    captured: dict[str, CapturedGeneratedBlock],
    initial_caches: dict[str, dict[str, Any]],
    cache_scrub: torch.Tensor,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode_index, mode in enumerate(("cache_neutral_start", "cache_hot")):
        restore_fixture_caches(
            {name: captured[name].decoder for name in BACKENDS}, initial_caches
        )
        for warmup in range(warmups):
            order = BACKENDS if warmup % 2 == 0 else tuple(reversed(BACKENDS))
            for name in order:
                if mode == "cache_neutral_start":
                    cache_scrub.add_(1)
                    torch.cuda.synchronize()
                else:
                    captured[name].replay_and_synchronize()
                captured[name].replay_and_synchronize()

        samples = {name: {"gpu": [], "wall": []} for name in BACKENDS}
        blocks: list[dict[str, Any]] = []
        measurement_start = time.perf_counter()
        for repeat in range(repeats):
            order = BACKENDS if repeat % 2 == 0 else tuple(reversed(BACKENDS))
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
                    captured[name].replay_and_synchronize()
                gpu_ms, wall_ms = captured[name].timed_replay()
                samples[name]["gpu"].append(gpu_ms)
                samples[name]["wall"].append(wall_ms)
                block["measurements"][name] = {
                    "gpu_ms_per_generated_16_token_block": gpu_ms,
                    "wall_ms_per_generated_16_token_block": wall_ms,
                }
            block["end_seconds"] = time.perf_counter() - measurement_start
            blocks.append(block)
        summary = OUTER.summarize_mode(
            samples,
            blocks,
            PG.PAGE,
            captured[BACKENDS[0]].batch_size,
            seed + mode_index * 100,
        )
        add_block_latency(summary)
        result[mode] = summary
    return result


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@torch.inference_mode()
def run_generated_sequence_graph_fixture(
    *,
    model: Any,
    decoders: dict[str, Any],
    initial_tokens: torch.Tensor,
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
    """Capture and measure a generated block on caller-owned live caches."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    initial_tokens = initial_tokens.to(device="cuda", dtype=torch.long).contiguous()
    batch_size = validate_fixture_shape(decoders, initial_tokens, start_position)
    if capture_warmups <= 0 or warmups < 0:
        raise ValueError("capture warmups must be positive and warmups non-negative")
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    if cache_scrub_mib <= 0:
        raise ValueError("cache scrub size must be positive")
    if not (0.0 <= min_top1_agreement <= 1.0):
        raise ValueError("top-1 threshold must lie in [0,1]")
    if initial_caches is None:
        initial_caches = snapshot_fixture_caches(decoders, start_position)
    elif set(initial_caches) != set(BACKENDS):
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
        source_label(path): source_sha256(path)
        for path in source_paths
        if path.is_file()
    }
    memory_at_entry = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    # Whole-decoder capture must own the attention kernels.  Remove only the
    # old nested graph owners; model weights and both full caches remain live.
    for decoder in decoders.values():
        decoder.attention_graphs = None
    gc.collect()
    torch.cuda.empty_cache()
    restore_fixture_caches(decoders, initial_caches)

    captured: dict[str, CapturedGeneratedBlock] = {}
    capture_memory: dict[str, Any] = {}
    for name in BACKENDS:
        decoder = decoders[name]
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        print(f"Capturing one generated 16-token graph: {name}", flush=True)
        captured[name] = capture_generated_block(
            decoder,
            initial_tokens,
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
    validation_outputs: dict[str, dict[str, Any]] = {}
    for name in BACKENDS:
        validation, outputs = validate_backend(
            captured[name], initial_tokens, initial_caches[name]
        )
        same_backend[name] = validation
        validation_outputs[name] = outputs

    cross_backend = cross_backend_validation(validation_outputs)
    graph_equivalence_passed = all(
        validation["passed"] for validation in same_backend.values()
    )
    cross_backend_logits_passed = all(
        payload["logits"]["minimum_logits_cosine"] >= min_logits_cosine
        and payload["logits"]["top1_agreement_fraction"]
        >= min_top1_agreement
        for payload in cross_backend.values()
    )
    generated_trajectory_exact = all(
        payload["generated_tokens"]["bitwise_identical"]
        for payload in cross_backend.values()
    )
    correctness_passed = (
        graph_equivalence_passed
        and cross_backend_logits_passed
        and generated_trajectory_exact
    )

    timing_modes: dict[str, Any] = {}
    timing_skipped_reason: str | None = None
    if correctness_passed:
        cache_scrub = torch.zeros(
            cache_scrub_mib * 1024 * 1024 // 4,
            device="cuda",
            dtype=torch.int32,
        )
        timing_modes = measure_modes(
            captured,
            initial_caches,
            cache_scrub,
            warmups,
            repeats,
            seed + 3000,
        )
    elif not graph_equivalence_passed:
        timing_skipped_reason = (
            "same-backend eager/graph logits, generated-token, or cache "
            "bitwise gate failed"
        )
    elif not generated_trajectory_exact:
        timing_skipped_reason = (
            "cross-backend generated trajectories differ; no generated-block "
            "throughput comparison is valid"
        )
    else:
        timing_skipped_reason = "cross-backend logits correctness gate failed"

    restore_fixture_caches(decoders, initial_caches)
    import flashinfer

    major, minor = torch.cuda.get_device_capability()
    initial_token_ids = initial_tokens.detach().cpu().tolist()
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_gpu_resident_greedy_16_token_block_graph",
        "claim_scope": (
            f"throughput-oriented synchronized batch-{batch_size}, fixed-position "
            "GPU-resident greedy 16-token block decode on live coherent caches; "
            "not streaming-token latency, ragged serving, sampling, or prefill"
        ),
        "batch_size": batch_size,
        "start_position": start_position,
        "end_position_inclusive": start_position + PG.PAGE - 1,
        "decode_steps": PG.PAGE,
        "output_tokens_per_block": batch_size * PG.PAGE,
        "latency_unit": "milliseconds per complete generated 16-token block",
        "fixture_provenance": fixture_provenance or {},
        "token_protocol": {
            "input_layout": "one CUDA int64 seed token per synchronized request",
            "initial_token_ids": initial_token_ids,
            "initial_token_ids_sha256": hashlib.sha256(
                json.dumps(
                    initial_token_ids, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "seed_source_buffer_mutable_between_replays": True,
            "seed_staging_into_source_buffer_outside_timed_boundary": True,
            "source_to_working_token_d2d_copy_inside_graph_and_timing": True,
            "greedy_feedback": (
                "each FP32 logits argmax stays on GPU and directly feeds the "
                "next captured decoder step"
            ),
            "cross_backend_generated_trajectory_must_be_exact": True,
        },
        "graph_protocol": {
            "graphs_per_backend": 1,
            "host_graph_launches_per_generated_block": 1,
            "positions": list(range(start_position, start_position + PG.PAGE)),
            "captured_boundary": (
                "initial D2D token copy; 16 embeddings; all model layers with "
                "append/finalization, attention, projections, and MLP; 16 final "
                "norm/LM-head/FP32 logits computations; 16 GPU argmax nodes"
            ),
            "captured_operations_per_block": {
                "initial_device_to_device_seed_copies": 1,
                "complete_decoder_steps": PG.PAGE,
                "fp32_greedy_argmax_nodes": PG.PAGE,
                "host_synchronizations_between_tokens": 0,
                "host_token_reads_between_tokens": 0,
            },
            "nested_attention_graphs_disabled": True,
            "plan_bucket_invariant_over_block": True,
            "plan_signatures": {
                name: OUTER.jsonable_signature(block.plan_signature)
                for name, block in captured.items()
            },
            "cache_state_restored_before_capture_validation_and_each_mode": True,
            "mutated_page_snapshot_covers_every_request": True,
            "pair_order": "AB, BA repeated; raw balanced blocks retained",
            "cache_modes": {
                "cache_neutral_start": (
                    "one synchronized cache scrub before each timed graph launch"
                ),
                "cache_hot": (
                    "one synchronized same-backend generated-block graph replay "
                    "immediately before each timed graph launch"
                ),
            },
            "excluded": [
                "tokenizer and host-to-device tokenizer output",
                "non-greedy sampling and sampling policy",
                "request scheduling and ragged batches",
                "next-page plan selection or graph recapture",
                "prefill latency",
                "time-to-first-token and per-token streaming latency",
            ],
        },
        "correctness": {
            "same_backend_eager_vs_graph_logits_tokens_and_cache": same_backend,
            "cross_backend": cross_backend,
            "thresholds": {
                "minimum_logits_cosine": min_logits_cosine,
                "minimum_top1_agreement": min_top1_agreement,
                "generated_trajectory_bitwise_identical": True,
            },
            "graph_equivalence_passed": graph_equivalence_passed,
            "cross_backend_logits_passed": cross_backend_logits_passed,
            "cross_backend_generated_trajectory_bitwise_identical": (
                generated_trajectory_exact
            ),
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


def validate_cli_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.context != START_POSITION or args.decode_steps != PG.PAGE:
        raise SystemExit(
            "generated block protocol requires context=20480 and decode-steps=16"
        )
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.context <= args.exact_tail or args.exact_tail % PG.PAGE:
        raise SystemExit("exact tail must be page aligned and below context")
    if args.repeats <= 0 or args.repeats % 2:
        raise SystemExit("repeats must be a positive even number")
    if args.capture_warmups <= 0 or args.warmups < 0:
        raise SystemExit("warmups are invalid")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    if args.batch_size > 1 and args.center_restore == "projection_bias":
        raise SystemExit("batched request centers require attention_add")
    if (
        args.tail_attention == "fused_kernel"
        and args.center_restore != "attention_add"
    ):
        raise SystemExit("fused tail requires attention_add")


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    torch.cuda.reset_peak_memory_stats()
    print("Loading model, caches, and current attention kernels...", flush=True)
    model, baseline, candidate, cache_storage = OUTER.make_decoders(args)
    decoders = {BACKENDS[0]: baseline, BACKENDS[1]: candidate}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial_tokens = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.batch_size,),
        generator=generator,
        dtype=torch.long,
    ).to(device="cuda")
    initial_caches = snapshot_fixture_caches(decoders, args.context)
    result = run_generated_sequence_graph_fixture(
        model=model,
        decoders=decoders,
        initial_tokens=initial_tokens,
        start_position=args.context,
        capture_warmups=args.capture_warmups,
        warmups=args.warmups,
        repeats=args.repeats,
        cache_scrub_mib=args.cache_scrub_mib,
        seed=args.seed,
        min_logits_cosine=args.min_logits_cosine,
        min_top1_agreement=args.min_top1_agreement,
        initial_caches=initial_caches,
        fixture_provenance={
            "kind": "deterministic_random_cache_and_seed_standalone_diagnostic",
            "seed": args.seed,
            "live_model_and_cache_objects": True,
        },
        cache_storage=cache_storage,
    )
    result["configuration"] = {
        **{key: value for key, value in vars(args).items() if key != "output"},
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    if not result["correctness"]["passed"]:
        raise SystemExit("generated-sequence graph correctness validation failed")


if __name__ == "__main__":
    main()
