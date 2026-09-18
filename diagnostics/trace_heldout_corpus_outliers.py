#!/usr/bin/env python3
"""Trace strict PageGauge outliers on the real held-out corpus trajectory.

This diagnostic reproduces the correctness worker's B1 co-resident execution:
HF SDPA builds a prefix, the FP16 paged cache is copied from that prefix,
PageGauge is derived from the copied cache, FlashInfer FP16 runs first, and
PageGauge runs second on the same corpus teacher tokens.  Only explicitly
failed rows retain per-layer tensors.  Production kernels, quantization,
thresholds, and publication artifacts are not modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
HELDOUT_PATH = ROOT / "diagnostics/benchmark_model_prefill_correctness.py"
TRACE_PATH = ROOT / "diagnostics/trace_quality_outlier_layers.py"


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELDOUT = load_local_module("page_gauge_heldout_for_corpus_trace", HELDOUT_PATH)
TRACE = load_local_module("page_gauge_layer_trace_for_corpus_trace", TRACE_PATH)
PG = HELDOUT.PG
BACKEND = TRACE.BACKEND


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-top-k", type=int, default=16)
    parser.add_argument(
        "--projection-mode",
        choices=("packed", "hf_separate"),
        default="packed",
        help="Diagnostic control for common Q/K/V and gate/up GEMM granularity.",
    )
    parser.add_argument("--wikitext-zip", type=Path, default=ROOT / "data/wikitext-2-raw-v1.zip")
    return parser.parse_args()


def source_targets(source: dict[str, Any]) -> list[dict[str, Any]]:
    pg_threshold = float(
        source["correctness"]["thresholds"]["page_gauge_minimum_logits_cosine"]
    )
    fi_threshold = float(
        source["correctness"]["thresholds"]["flashinfer_minimum_hf_logits_cosine"]
    )
    pg_fi = source["correctness"]["page_gauge_vs_flashinfer_fp16"]
    fi_hf = source["correctness"]["flashinfer_fp16_vs_hf_sdpa_fp16"]
    pg_hf = source["correctness"]["page_gauge_vs_hf_sdpa_fp16"]
    targets = []
    for step in range(int(source["decode_steps"])):
        for request in range(int(source["batch_size"])):
            pg_cosine = float(pg_fi["logits_cosine_by_step_request"][step][request])
            fi_cosine = float(fi_hf["logits_cosine_by_step_request"][step][request])
            if pg_cosine < pg_threshold or fi_cosine < fi_threshold:
                targets.append(
                    {
                        "step": step,
                        "request": request,
                        "absolute_position": int(source["context"]) + step,
                        "source_page_gauge_vs_flashinfer_cosine": pg_cosine,
                        "source_flashinfer_vs_hf_cosine": fi_cosine,
                        "source_page_gauge_vs_hf_cosine": float(
                            pg_hf["logits_cosine_by_step_request"][step][request]
                        ),
                    }
                )
    if not targets:
        raise RuntimeError("source contains no row below either strict cosine gate")
    return targets


def validate_source(source: dict[str, Any]) -> None:
    required = {
        "batch_size": 1,
        "context": 20_480,
        "decode_steps": 1_536,
        "exact_tail_tokens": 768,
        "baseline_split_pages": 256,
        "candidate_split_pages": 256,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
    }
    mismatches = {
        key: {"expected": expected, "observed": source.get(key)}
        for key, expected in required.items()
        if source.get(key) != expected
    }
    if source.get("exact_prefix_pages") not in (3, 4):
        mismatches["exact_prefix_pages"] = {
            "expected_one_of": [3, 4],
            "observed": source.get("exact_prefix_pages"),
        }
    token_source = source.get("token_source", {})
    if token_source.get("archive_member") != "wikitext-2-raw/wiki.test.raw":
        mismatches["archive_member"] = {
            "expected": "wikitext-2-raw/wiki.test.raw",
            "observed": token_source.get("archive_member"),
        }
    starts = token_source.get("corpus_window_start_offsets")
    allowed_starts = {0, 23_600, 47_200, 70_800, 94_400, 118_000}
    if not isinstance(starts, list) or len(starts) != 1 or starts[0] not in allowed_starts:
        mismatches["corpus_window_start_offsets"] = {
            "expected_one_of": sorted(allowed_starts),
            "observed": starts,
        }
    if mismatches:
        raise RuntimeError(f"source is not the frozen exposed B1 window: {mismatches}")


def copy_prefix_to_baseline(
    dynamic_cache: Any,
    baseline: Any,
    *,
    layers: int,
    hkv: int,
    context: int,
) -> None:
    initial_pages = context // PG.PAGE
    for layer in range(layers):
        key, value = HELDOUT.cache_layer_tensors(dynamic_cache, layer)
        expected = (1, hkv, context, PG.DIM)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise RuntimeError(
                f"HF prefix layer {layer} has K={tuple(key.shape)}, "
                f"V={tuple(value.shape)}; expected={expected}"
            )
        baseline.key[layer, :initial_pages].view(context, hkv, PG.DIM).copy_(
            key[0].transpose(0, 1)
        )
        baseline.value[layer, :initial_pages].view(context, hkv, PG.DIM).copy_(
            value[0].transpose(0, 1)
        )


@torch.inference_mode()
def build_exact_heldout_fixture(
    *,
    model: Any,
    tokens: torch.Tensor,
    baseline: Any,
    layers: int,
    hkv: int,
    pages: int,
    context: int,
    decode_steps: int,
    exact_pages: int,
    exact_prefix_pages: int,
    chunk_tokens: int,
    targets: list[dict[str, Any]],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    page_top_k: int,
) -> tuple[Any, torch.Tensor, dict[tuple[int, int], dict[str, Any]]]:
    from transformers.cache_utils import DynamicCache

    dynamic_cache = DynamicCache()
    for begin in range(0, context, chunk_tokens):
        end = min(begin + chunk_tokens, context)
        input_ids = tokens[0, begin:end].to("cuda")[None]
        positions = torch.arange(begin, end, device="cuda", dtype=torch.long)
        outputs = model.model(
            input_ids=input_ids,
            position_ids=positions[None],
            cache_position=positions,
            past_key_values=dynamic_cache,
            use_cache=True,
            return_dict=True,
        )
        dynamic_cache = outputs.past_key_values
        del input_ids, positions, outputs
    if int(dynamic_cache.get_seq_length()) != context:
        raise RuntimeError("HF prefix length mismatch")

    copy_prefix_to_baseline(
        dynamic_cache,
        baseline,
        layers=layers,
        hkv=hkv,
        context=context,
    )
    gauge = HELDOUT.build_gauge_cache_from_baseline(
        baseline,
        layers,
        pages,
        context // PG.PAGE,
        exact_pages,
        1,
        hkv,
        exact_prefix_pages,
    )

    logits = []
    with TRACE.HFTargetInterceptor(
        model=model,
        cache=gauge,
        targets=targets,
        context=context,
        exact_tail_tokens=exact_pages * PG.PAGE,
        exact_sink_pages=exact_prefix_pages,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        page_top_k=page_top_k,
        pre_tail_band_ablation_enabled=False,
    ) as interceptor:
        for step in range(decode_steps):
            position = context + step
            input_ids = tokens[0, position].reshape(1, 1).to("cuda")
            cache_position = torch.tensor([position], device="cuda", dtype=torch.long)
            outputs = model.model(
                input_ids=input_ids,
                position_ids=cache_position[None],
                cache_position=cache_position,
                past_key_values=dynamic_cache,
                use_cache=True,
                return_dict=True,
            )
            dynamic_cache = outputs.past_key_values
            logits.append(model.lm_head(outputs.last_hidden_state[:, -1]).float().cpu())
            del input_ids, cache_position, outputs
    torch.cuda.synchronize()
    hf_logits = torch.stack(logits)
    traces = interceptor.traces
    del dynamic_cache, logits
    gc.collect()
    torch.cuda.empty_cache()
    return gauge, hf_logits, traces


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    # Match the publication worker exactly: logits are reduced in FP64 on CPU.
    reference_f = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_f = candidate.detach().to(device="cpu", dtype=torch.float64)
    cosine = F.cosine_similarity(reference_f, candidate_f, dim=-1)
    top1 = reference_f.argmax(dim=-1).eq(candidate_f.argmax(dim=-1))
    absolute = (reference_f - candidate_f).abs().amax(dim=-1)
    relative = torch.linalg.vector_norm(reference_f - candidate_f, dim=-1) / torch.linalg.vector_norm(
        reference_f, dim=-1
    ).clamp_min(1.0e-12)
    worst = int(cosine.reshape(-1).argmin())
    batch = int(cosine.shape[1])
    return {
        "checked_request_steps": int(cosine.numel()),
        "minimum_logits_cosine": float(cosine.min()),
        "worst_logits_cosine_location": {
            "step": worst // batch,
            "request": worst % batch,
            "value": float(cosine.reshape(-1)[worst]),
        },
        "top1_agreement_fraction": float(top1.float().mean()),
        "maximum_logits_absolute_error": float(absolute.max()),
        "maximum_relative_logits_l2": float(relative.max()),
        "logits_cosine_by_step_request": cosine.tolist(),
    }


def install_hf_separate_projection_control(decoder: Any) -> None:
    """Use HF-equivalent separate GEMM shapes without changing any weights."""

    def execute_layer(
        self: Any,
        layer_index: int,
        hidden: torch.Tensor,
        position: int,
        destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer = self.model.model.layers[layer_index]
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        q_width = int(attention._pkv_q_width)
        kv_width = int(attention._pkv_kv_width)
        qkv_bias = getattr(attention, "qkv_bias", None)
        q_flat = F.linear(
            normalized,
            attention.qkv_weight[:q_width],
            None if qkv_bias is None else qkv_bias[:q_width],
        )
        k_flat = F.linear(
            normalized,
            attention.qkv_weight[q_width : q_width + kv_width],
            None if qkv_bias is None else qkv_bias[q_width : q_width + kv_width],
        )
        v_flat = F.linear(
            normalized,
            attention.qkv_weight[q_width + kv_width :],
            None if qkv_bias is None else qkv_bias[q_width + kv_width :],
        )
        query = q_flat.reshape(self.batch_size, self.hq, PG.DIM)
        key = k_flat.reshape(self.batch_size, self.hkv, PG.DIM)
        value = v_flat.reshape(self.batch_size, self.hkv, PG.DIM)
        self.append(layer_index, query, key, value, position)
        attended = self.attention(layer_index)
        projected = F.linear(
            attended.reshape(self.batch_size, -1),
            attention.o_proj.weight,
            self.output_projection_biases[layer_index],
        )
        hidden = residual + projected
        mlp_input = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        width = int(mlp._pkv_intermediate)
        gate_up_bias = getattr(mlp, "gate_up_bias", None)
        gate = F.linear(
            mlp_input,
            mlp.gate_up_weight[:width],
            None if gate_up_bias is None else gate_up_bias[:width],
        )
        up = F.linear(
            mlp_input,
            mlp.gate_up_weight[width:],
            None if gate_up_bias is None else gate_up_bias[width:],
        )
        hidden = hidden + mlp.down_proj(F.silu(gate) * up)
        if destination is not None:
            destination.copy_(hidden)
            return destination
        return hidden

    decoder._execute_decoder_layer = MethodType(execute_layer, decoder)


@torch.inference_mode()
def collect_decoder_tensor(
    decoder: Any, teacher_inputs: torch.Tensor, context: int
) -> torch.Tensor:
    rows = []
    for step in range(int(teacher_inputs.shape[0])):
        rows.append(decoder.step(teacher_inputs[step], context + step)[0].cpu())
    return torch.stack(rows)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    validate_source(source)
    targets = source_targets(source)
    if args.page_top_k <= 0:
        raise SystemExit("page-top-k must be positive")

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    seed = 20_260_861
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        source["model"],
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    layers, _hq, hkv, hidden = PG.BASE_E2E.check_model(model)
    token_args = SimpleNamespace(
        model=source["model"],
        context=int(source["context"]),
        decode_steps=int(source["decode_steps"]),
        batch_size=1,
        token_source="wikitext2",
        seed=seed,
        wikitext_zip=args.wikitext_zip,
        wikitext_member=source["token_source"]["archive_member"],
        token_offset=int(source["token_source"]["corpus_window_start_offsets"][0]),
        token_stride=23_600,
    )
    tokens, token_provenance = HELDOUT.build_token_matrix(
        token_args, int(model.config.vocab_size)
    )
    if token_provenance["token_ids_sha256"] != source["token_source"]["token_ids_sha256"]:
        raise RuntimeError("rebuilt held-out token tensor does not match source")
    model = model.cuda()

    context = int(source["context"])
    decode_steps = int(source["decode_steps"])
    max_context = context + decode_steps
    pages = math.ceil(max_context / PG.PAGE)
    exact_pages = int(source["exact_tail_tokens"]) // PG.PAGE
    exact_prefix_pages = int(source["exact_prefix_pages"])
    baseline_cache = HELDOUT.allocate_baseline_cache(layers, pages, 1, hkv)
    with torch.inference_mode():
        positions = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        rope_probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(rope_probe, positions)
        if rope_cos.dim() == 3:
            rope_cos, rope_sin = rope_cos[0], rope_sin[0]
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()
    del positions, rope_probe

    print("Replaying exact held-out HF corpus trajectory and tracing failed rows...", flush=True)
    gauge_cache, hf_logits, hf_traces = build_exact_heldout_fixture(
        model=model,
        tokens=tokens,
        baseline=baseline_cache,
        layers=layers,
        hkv=hkv,
        pages=pages,
        context=context,
        decode_steps=decode_steps,
        exact_pages=exact_pages,
        exact_prefix_pages=exact_prefix_pages,
        chunk_tokens=int(source["prefill_chunk_tokens"]),
        targets=targets,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        page_top_k=args.page_top_k,
    )
    if set(hf_traces) != {(row["step"], row["request"]) for row in targets}:
        raise RuntimeError("HF trace did not cover every failed source row")

    prefix_snapshot = HELDOUT.exact_prefix_snapshot(
        gauge_cache,
        exact_tail_pages=exact_pages,
        exact_prefix_pages=exact_prefix_pages,
        batch_size=1,
    )
    expected_prefix_sha = source["exact_prefix_canary"]["population"]["expected_combined_sha256"]
    if prefix_snapshot["combined_sha256"] != expected_prefix_sha:
        raise RuntimeError("derived exact-prefix bytes do not reproduce the held-out source")

    PG.BASE_E2E.pack_model_projections(model)
    gc.collect()
    torch.cuda.empty_cache()
    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    common = (model, flashinfer, append_extension)
    baseline = PG.TransformerDecoder(
        *common,
        "flashinfer_fp16",
        baseline_cache,
        max_context,
        int(source["exact_tail_tokens"]),
        int(source["baseline_split_pages"]),
        int(source["candidate_split_pages"]),
        rope_cos,
        rope_sin,
        "attention_add",
        tail_attention=str(source["tail_attention"]),
        batch_size=1,
        old_value_scale_placement=str(source["old_value_scale_placement"]),
        exact_sink_pages=0,
    )
    candidate = PG.TransformerDecoder(
        *common,
        "page_gauge",
        gauge_cache,
        max_context,
        int(source["exact_tail_tokens"]),
        int(source["baseline_split_pages"]),
        int(source["candidate_split_pages"]),
        rope_cos,
        rope_sin,
        "attention_add",
        tail_attention=str(source["tail_attention"]),
        batch_size=1,
        old_value_scale_placement=str(source["old_value_scale_placement"]),
        exact_sink_pages=exact_prefix_pages,
    )
    teacher_inputs = tokens[:, context : context + decode_steps].T.contiguous().cuda()
    print("Running co-resident FlashInfer reference then traced PageGauge...", flush=True)
    if args.projection_mode == "hf_separate":
        install_hf_separate_projection_control(baseline)
        install_hf_separate_projection_control(candidate)
        baseline_logits = collect_decoder_tensor(baseline, teacher_inputs, context)
        candidate_logits = collect_decoder_tensor(candidate, teacher_inputs, context)
        observed = {
            "page_gauge_vs_flashinfer_fp16": comparison(
                baseline_logits, candidate_logits
            ),
            "flashinfer_fp16_vs_hf_sdpa_fp16": comparison(
                hf_logits, baseline_logits
            ),
            "page_gauge_vs_hf_sdpa_fp16": comparison(hf_logits, candidate_logits),
        }
        return {
            "schema_version": 1,
            "experiment": "page_gauge_hf_separate_projection_control",
            "protocol_passed": True,
            "claim_scope": "diagnostic-only common-projection numerical control",
            "production_math_modified": False,
            "production_files_modified": False,
            "projection_mode": "three Q/K/V plus two gate/up GEMMs using exact packed-weight slices",
            "source_result": str(args.source_result.resolve()),
            "source_result_sha256": sha256_file(args.source_result),
            "observed": observed,
            "exact_prefix_combined_sha256": prefix_snapshot["combined_sha256"],
            "token_source": token_provenance,
            "source_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (
                    Path(__file__),
                    HELDOUT_PATH,
                    TRACE_PATH,
                    ROOT / "scripts/benchmark_page_gauge_transformer.py",
                    ROOT / "scripts/benchmark_page_gauge_overheads.py",
                    ROOT / "tests/page_gauge_append_extension.cu",
                )
            },
        }
    baseline.plan(context)
    baseline_logits, _baseline_tokens, baseline_traces = TRACE.trace_candidate_teacher(
        baseline,
        teacher_inputs=teacher_inputs,
        context=context,
        targets=targets,
    )
    candidate.plan(context)
    candidate_logits, candidate_tokens, candidate_traces = TRACE.trace_candidate_teacher(
        candidate,
        teacher_inputs=teacher_inputs,
        context=context,
        targets=targets,
    )
    torch.cuda.synchronize()
    baseline_merged = TRACE.merge_target_traces(
        targets=targets,
        hf_traces=hf_traces,
        candidate_traces=baseline_traces,
        hf_logits=hf_logits,
        candidate_logits=baseline_logits,
        hidden_cosine_trigger=0.999,
        hidden_relative_l2_trigger=0.01,
    )
    candidate_merged = TRACE.merge_target_traces(
        targets=targets,
        hf_traces=hf_traces,
        candidate_traces=candidate_traces,
        hf_logits=hf_logits,
        candidate_logits=candidate_logits,
        hidden_cosine_trigger=0.999,
        hidden_relative_l2_trigger=0.01,
    )

    observed = {
        "page_gauge_vs_flashinfer_fp16": comparison(baseline_logits, candidate_logits),
        "flashinfer_fp16_vs_hf_sdpa_fp16": comparison(hf_logits, baseline_logits),
        "page_gauge_vs_hf_sdpa_fp16": comparison(hf_logits, candidate_logits),
    }
    source_names = tuple(observed)
    scalar_deltas = {}
    reproduction_passed = True
    for name in source_names:
        expected = source["correctness"][name]
        scalar_deltas[name] = {
            field: abs(float(observed[name][field]) - float(expected[field]))
            for field in (
                "minimum_logits_cosine",
                "top1_agreement_fraction",
                "maximum_logits_absolute_error",
                "maximum_relative_logits_l2",
            )
        }
        reproduction_passed = reproduction_passed and all(
            delta <= 1.0e-6 for delta in scalar_deltas[name].values()
        )
        reproduction_passed = reproduction_passed and (
            observed[name]["worst_logits_cosine_location"]["step"]
            == expected["worst_logits_cosine_location"]["step"]
        )
    if not reproduction_passed:
        raise RuntimeError(f"held-out scalar reproduction failed: {scalar_deltas}")

    return {
        "schema_version": 1,
        "experiment": "page_gauge_heldout_corpus_outlier_layer_trace",
        "protocol_passed": True,
        "claim_scope": "diagnostic-only replay of already-exposed held-out rows",
        "production_math_modified": False,
        "source_result": str(args.source_result.resolve()),
        "source_result_sha256": sha256_file(args.source_result),
        "source_reproduction": {
            "passed": True,
            "tolerance": 1.0e-6,
            "scalar_absolute_deltas": scalar_deltas,
            "observed": observed,
            "exact_prefix_combined_sha256": prefix_snapshot["combined_sha256"],
            "teacher_trajectory": "real WikiText corpus continuation",
        },
        "flashinfer_targets": baseline_merged,
        "page_gauge_targets": candidate_merged,
        "token_source": token_provenance,
        "candidate_argmax_tokens_sha256": hashlib.sha256(
            candidate_tokens.contiguous().numpy().tobytes()
        ).hexdigest(),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "flashinfer": str(flashinfer.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                HELDOUT_PATH,
                TRACE_PATH,
                ROOT / "scripts/benchmark_page_gauge_transformer.py",
                ROOT / "scripts/benchmark_page_gauge_overheads.py",
                ROOT / "tests/page_gauge_append_extension.cu",
            )
        },
    }


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for target in result.get("flashinfer_targets", []):
            first = target["first_diagnostic_divergence"]
            print(
                f"FI step={target['step']} request={target['request']} "
                f"cos={target['logits']['cosine']:.9f} "
                f"first={first['layer']}:{first['stage']}",
                flush=True,
            )
        for target in result.get("page_gauge_targets", []):
            first = target["first_diagnostic_divergence"]
            print(
                f"PG step={target['step']} request={target['request']} "
                f"cos={target['logits']['cosine']:.9f} "
                f"first={first['layer']}:{first['stage']}",
                flush=True,
            )
        if "observed" in result and not result.get("flashinfer_targets"):
            print(
                json.dumps(
                    {
                        name: {
                            "minimum_logits_cosine": values[
                                "minimum_logits_cosine"
                            ],
                            "top1_agreement_fraction": values[
                                "top1_agreement_fraction"
                            ],
                        }
                        for name, values in result["observed"].items()
                    },
                    indent=2,
                ),
                flush=True,
            )
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment": "page_gauge_heldout_corpus_outlier_layer_trace",
                    "protocol_passed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "source_result": str(args.source_result.resolve()),
                    "source_result_sha256": (
                        sha256_file(args.source_result)
                        if args.source_result.is_file()
                        else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
