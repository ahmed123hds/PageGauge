#!/usr/bin/env python3
"""Temporary component profiler for the PageGauge full-decoder integration.

This diagnostic imports the production runner without changing it.  It executes
the same decoder code and records representative-layer CUDA-event timings for
the eager and graph-replayed attention paths.  Optional NVTX and Kineto traces
provide kernel-level evidence; event results are diagnostic rather than paper
headline measurements because instrumentation perturbs an eager launch stream.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/benchmark_page_gauge_transformer.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "page_gauge_component_profile_runner", RUNNER_PATH
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
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--baseline-split-pages", type=int, default=64)
    parser.add_argument("--candidate-split-pages", type=int, default=0)
    parser.add_argument(
        "--center-restore",
        choices=("attention_add", "projection_bias"),
        default="attention_add",
        help="Match the production candidate's value-center restoration path.",
    )
    parser.add_argument(
        "--tail-attention",
        choices=("fused_kernel", "flashinfer_merge"),
        default="fused_kernel",
        help="Match the production candidate's exact-tail implementation.",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=(0, 15, 31),
        help="Representative zero-based layers to instrument.",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=(0, 15),
        help="Decode offsets to profile; 15 is the page-closing token.",
    )
    parser.add_argument(
        "--attention-modes",
        choices=("eager", "graph"),
        nargs="+",
        default=("eager", "graph"),
    )
    parser.add_argument(
        "--cache-mode", choices=("hot", "neutral_start"), default="hot"
    )
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit named NVTX ranges for Nsight Systems.",
    )
    parser.add_argument(
        "--torch-trace",
        type=Path,
        help="Optionally export a Kineto/CUPTI Chrome trace.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class EventCollector:
    """Preallocate events so event construction is outside the measured span."""

    def __init__(self, stage_names: list[str], *, nvtx: bool) -> None:
        self._events = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in stage_names
        }
        self._nvtx = nvtx

    @contextlib.contextmanager
    def range(self, name: str) -> Iterator[None]:
        start, end = self._events[name]
        start.record()
        if self._nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            with torch.autograd.profiler.record_function(name):
                yield
        finally:
            if self._nvtx:
                torch.cuda.nvtx.range_pop()
            end.record()

    def elapsed(self) -> dict[str, float]:
        return {
            name: float(start.elapsed_time(end))
            for name, (start, end) in self._events.items()
        }


COMMON_LAYER_STAGES = (
    "input_norm",
    "qkv_projection",
    "append",
    "output_projection",
    "attention_residual_add",
    "post_attention_norm",
    "mlp_gate_up",
    "mlp_silu_mul",
    "mlp_down_projection",
    "mlp_residual_add",
)


def stage_names(
    backend: str,
    attention_mode: str,
    finalizes_page: bool,
    center_restore: str,
    tail_attention: str,
) -> list[str]:
    names = ["plan_metadata", "embedding", *COMMON_LAYER_STAGES]
    names[names.index("append")] = (
        "append_page_finalize" if finalizes_page else "append_no_finalize"
    )
    if attention_mode == "graph":
        names.append("attention_graph_replay")
    elif backend == "flashinfer_fp16":
        names.append("fp16_attention_eager")
    else:
        names.append("old_int8_attention_eager")
        if tail_attention == "fused_kernel":
            names.append("exact_tail_merge_center_fused_eager")
        else:
            names.extend(
                (
                    "exact_tail_attention_eager",
                    "attention_state_merge_eager",
                )
            )
            if center_restore == "attention_add":
                names.append("value_center_restore_eager")
    names.extend(("final_norm", "lm_head", "logits_fp32_cast"))
    return names


def eager_attention_components(decoder, layer_index: int, collector: EventCollector):
    if decoder.backend == "flashinfer_fp16":
        with collector.range("fp16_attention_eager"):
            return decoder.baseline_wrapper.wrapper.run(
                decoder.rotated_query,
                (decoder.cache.key[layer_index], decoder.cache.value[layer_index]),
                out=decoder.attention_output[layer_index],
            )

    with collector.range("old_int8_attention_eager"):
        decoder.old_wrapper.wrapper.run(
            decoder.rotated_query,
            (
                decoder.cache.key_codes[layer_index],
                decoder.cache.value_codes[layer_index],
            ),
            decoder.cache.key_scales[layer_index],
            decoder.cache.value_scales[layer_index],
            1.0 / math.sqrt(PG.DIM),
            out=decoder.attention_output[layer_index],
            lse=decoder.old_lse[layer_index],
            return_lse=True,
        )
    if decoder.tail_attention == "fused_kernel":
        with collector.range("exact_tail_merge_center_fused_eager"):
            decoder.append_extension.exact_tail_merge_center(
                decoder.rotated_query,
                decoder.cache.exact_key[layer_index],
                decoder.cache.exact_value[layer_index],
                decoder.exact_wrapper.indices[: decoder.exact_pages],
                decoder.exact_wrapper.last_len,
                decoder.attention_output[layer_index],
                decoder.old_lse[layer_index],
                decoder.cache.value_center[layer_index],
                1.0 / math.sqrt(PG.DIM),
            )
        return decoder.attention_output[layer_index]
    with collector.range("exact_tail_attention_eager"):
        decoder.exact_wrapper.wrapper.run(
            decoder.rotated_query,
            (
                decoder.cache.exact_key[layer_index],
                decoder.cache.exact_value[layer_index],
            ),
            out=decoder.exact_output[layer_index],
            lse=decoder.exact_lse[layer_index],
            return_lse=True,
        )
    with collector.range("attention_state_merge_eager"):
        import flashinfer

        flashinfer.merge_state_in_place(
            decoder.attention_output[layer_index],
            decoder.old_lse[layer_index],
            decoder.exact_output[layer_index],
            decoder.exact_lse[layer_index],
        )
    if decoder.center_restore == "attention_add":
        with collector.range("value_center_restore_eager"):
            decoder.attention_output[layer_index].add_(
                decoder.cache.output_center[layer_index]
            )
    return decoder.attention_output[layer_index]


@torch.inference_mode()
def profiled_step(
    decoder,
    token: torch.Tensor,
    position: int,
    target_layer: int,
    attention_mode: str,
    collector: EventCollector,
) -> torch.Tensor:
    with collector.range("plan_metadata"):
        decoder.plan(position + 1)
    with collector.range("embedding"):
        hidden = decoder.model.model.embed_tokens(token.view(1, 1))[:, 0]

    for layer_index, layer in enumerate(decoder.model.model.layers):
        instrument = layer_index == target_layer
        residual = hidden
        if instrument:
            with collector.range("input_norm"):
                normalized = layer.input_layernorm(hidden)
        else:
            normalized = layer.input_layernorm(hidden)

        attention = layer.self_attn
        if instrument:
            with collector.range("qkv_projection"):
                qkv = F.linear(
                    normalized,
                    attention.qkv_weight,
                    getattr(attention, "qkv_bias", None),
                )
        else:
            qkv = F.linear(
                normalized,
                attention.qkv_weight,
                getattr(attention, "qkv_bias", None),
            )
        q_flat, k_flat, v_flat = torch.split(
            qkv,
            (
                attention._pkv_q_width,
                attention._pkv_kv_width,
                attention._pkv_kv_width,
            ),
            dim=-1,
        )
        query = q_flat.view(1, decoder.hq, PG.DIM)
        key = k_flat.view(1, decoder.hkv, PG.DIM)
        value = v_flat.view(1, decoder.hkv, PG.DIM)
        append_stage = (
            "append_page_finalize"
            if position % PG.PAGE == PG.PAGE - 1
            else "append_no_finalize"
        )
        if instrument:
            with collector.range(append_stage):
                decoder.append(layer_index, query, key, value, position)
        else:
            decoder.append(layer_index, query, key, value, position)

        if instrument:
            if attention_mode == "graph":
                with collector.range("attention_graph_replay"):
                    attended = decoder.attention(layer_index)
            else:
                attended = eager_attention_components(
                    decoder, layer_index, collector
                )
        else:
            attended = (
                decoder.attention(layer_index)
                if attention_mode == "graph"
                else decoder.eager_attention(layer_index)
            )

        if instrument:
            with collector.range("output_projection"):
                projected = F.linear(
                    attended.reshape(1, -1),
                    attention.o_proj.weight,
                    decoder.output_projection_biases[layer_index],
                )
            with collector.range("attention_residual_add"):
                hidden = residual + projected
            with collector.range("post_attention_norm"):
                mlp_input = layer.post_attention_layernorm(hidden)
        else:
            projected = F.linear(
                attended.reshape(1, -1),
                attention.o_proj.weight,
                decoder.output_projection_biases[layer_index],
            )
            hidden = residual + projected
            mlp_input = layer.post_attention_layernorm(hidden)

        mlp = layer.mlp
        if instrument:
            with collector.range("mlp_gate_up"):
                gate_up = F.linear(
                    mlp_input,
                    mlp.gate_up_weight,
                    getattr(mlp, "gate_up_bias", None),
                )
        else:
            gate_up = F.linear(
                mlp_input,
                mlp.gate_up_weight,
                getattr(mlp, "gate_up_bias", None),
            )
        gate, up = gate_up.split(mlp._pkv_intermediate, dim=-1)
        if instrument:
            with collector.range("mlp_silu_mul"):
                activated = F.silu(gate) * up
            with collector.range("mlp_down_projection"):
                mlp_output = mlp.down_proj(activated)
            with collector.range("mlp_residual_add"):
                hidden = hidden + mlp_output
        else:
            activated = F.silu(gate) * up
            hidden = hidden + mlp.down_proj(activated)

    with collector.range("final_norm"):
        normalized = decoder.model.model.norm(hidden)
    with collector.range("lm_head"):
        logits = decoder.model.lm_head(normalized)
    with collector.range("logits_fp32_cast"):
        return logits.float()


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for name, value in sample["stages_ms"].items():
            stages[name].append(value)
    return {
        "num_samples": len(samples),
        "whole_step_gpu_ms": {
            "median": statistics.median(s["whole_gpu_ms"] for s in samples),
            "p05": percentile([s["whole_gpu_ms"] for s in samples], 0.05),
            "p95": percentile([s["whole_gpu_ms"] for s in samples], 0.95),
        },
        "whole_step_wall_ms": {
            "median": statistics.median(s["wall_ms"] for s in samples),
            "p05": percentile([s["wall_ms"] for s in samples], 0.05),
            "p95": percentile([s["wall_ms"] for s in samples], 0.95),
        },
        "stage_ms": {
            name: {
                "median": statistics.median(values),
                "p05": percentile(values, 0.05),
                "p95": percentile(values, 0.95),
                "raw": values,
            }
            for name, values in sorted(stages.items())
        },
        "raw_samples": samples,
    }


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
    )
    common = (
        model,
        flashinfer,
        append_extension,
    )
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
    )
    return model, baseline, candidate


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.context % PG.PAGE or args.decode_steps < PG.PAGE:
        raise SystemExit("use a page-aligned context and at least 16 decode steps")
    if args.repeats <= 0 or args.warmups < 0:
        raise SystemExit("repeat counts are invalid")
    if any(layer < 0 for layer in args.layers):
        raise SystemExit("layer indices must be non-negative")
    if any(offset < 0 or offset >= args.decode_steps for offset in args.offsets):
        raise SystemExit("offsets must fall inside the decode window")

    print("Loading model, caches, and kernels...", flush=True)
    model, baseline, candidate = make_decoders(args)
    if any(layer >= baseline.layers for layer in args.layers):
        raise SystemExit(f"model contains {baseline.layers} layers")
    baseline.capture_attention_graphs(args.context + 1)
    candidate.capture_attention_graphs(args.context + 1)
    saved_graphs = {
        "flashinfer_fp16": baseline.attention_graphs,
        "page_gauge": candidate.attention_graphs,
    }
    generator = torch.Generator().manual_seed(args.seed)
    tokens = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.decode_steps,),
        generator=generator,
    ).tolist()
    token_tensor = torch.tensor(tokens, device="cuda", dtype=torch.long)

    # Populate every slot in the measured page before repeatedly profiling the
    # page-closing token. This also establishes graph/eager kernel warmup.
    for decoder in (baseline, candidate):
        PG.timed_sequence(decoder, tokens, args.context)

    scrub = torch.zeros(
        args.cache_scrub_mib * 1024 * 1024 // 4,
        device="cuda",
        dtype=torch.int32,
    )
    samples_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)

    profiler_context: Any = contextlib.nullcontext()
    profiler = None
    if args.torch_trace is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        profiler_context = profiler

    with profiler_context:
        for attention_mode in args.attention_modes:
            for decoder in (baseline, candidate):
                decoder.attention_graphs = (
                    saved_graphs[decoder.backend]
                    if attention_mode == "graph"
                    else None
                )
            for offset in args.offsets:
                position = args.context + offset
                token = token_tensor[offset : offset + 1]
                for target_layer in args.layers:
                    for decoder in (baseline, candidate):
                        for _ in range(args.warmups):
                            decoder.step(token, position)
                        for repeat in range(args.repeats):
                            if args.cache_mode == "hot":
                                decoder.step(token, position)
                            else:
                                scrub.add_(1)
                                torch.cuda.synchronize()
                            names = stage_names(
                                decoder.backend,
                                attention_mode,
                                position % PG.PAGE == PG.PAGE - 1,
                                decoder.center_restore,
                                decoder.tail_attention,
                            )
                            collector = EventCollector(names, nvtx=args.nvtx)
                            whole_start = torch.cuda.Event(enable_timing=True)
                            whole_end = torch.cuda.Event(enable_timing=True)
                            label = (
                                f"{decoder.backend}/{attention_mode}/offset_{offset}/"
                                f"layer_{target_layer}/repeat_{repeat}"
                            )
                            torch.cuda.synchronize()
                            wall_start = time.perf_counter()
                            whole_start.record()
                            if args.nvtx:
                                torch.cuda.nvtx.range_push(label)
                            profiled_step(
                                decoder,
                                token,
                                position,
                                target_layer,
                                attention_mode,
                                collector,
                            )
                            if args.nvtx:
                                torch.cuda.nvtx.range_pop()
                            whole_end.record()
                            torch.cuda.synchronize()
                            wall_ms = (time.perf_counter() - wall_start) * 1e3
                            stage_values = collector.elapsed()
                            whole_gpu_ms = float(whole_start.elapsed_time(whole_end))
                            key = (
                                f"{decoder.backend}/{attention_mode}/offset_{offset}/"
                                f"layer_{target_layer}"
                            )
                            samples_by_key[key].append(
                                {
                                    "repeat": repeat,
                                    "whole_gpu_ms": whole_gpu_ms,
                                    "wall_ms": wall_ms,
                                    "instrumented_stage_sum_ms": sum(
                                        stage_values.values()
                                    ),
                                    "unattributed_stream_span_ms": whole_gpu_ms
                                    - sum(stage_values.values()),
                                    "wall_minus_gpu_ms": wall_ms - whole_gpu_ms,
                                    "stages_ms": stage_values,
                                }
                            )
                            if profiler is not None:
                                profiler.step()

    if profiler is not None and args.torch_trace is not None:
        args.torch_trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.torch_trace))

    major, minor = torch.cuda.get_device_capability()
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_transformer_component_diagnostic",
        "warning": (
            "Representative-layer instrumentation perturbs an eager launch stream; "
            "use Nsight/CUPTI kernel timelines for pure device durations and use "
            "the production runner for headline throughput."
        ),
        "configuration": {
            **{
                key: value
                for key, value in vars(args).items()
                if key not in {"output", "torch_trace"}
            },
            "output": str(args.output),
            "torch_trace": str(args.torch_trace) if args.torch_trace else None,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": platform.python_version(),
            "flashinfer": __import__("flashinfer").__version__,
            "kineto_available": torch.profiler.kineto_available(),
            "supported_profiler_activities": sorted(
                str(activity) for activity in torch.profiler.supported_activities()
            ),
        },
        "source_sha256": {
            str(RUNNER_PATH.relative_to(ROOT)): hashlib.sha256(
                RUNNER_PATH.read_bytes()
            ).hexdigest(),
            str(Path(__file__).relative_to(ROOT)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "profiles": {
            key: summarize_samples(samples)
            for key, samples in sorted(samples_by_key.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    if args.torch_trace is not None:
        print(f"wrote {args.torch_trace}", flush=True)


if __name__ == "__main__":
    main()
