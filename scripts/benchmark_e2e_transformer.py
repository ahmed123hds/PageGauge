#!/usr/bin/env python3
"""Paired full-transformer decode with FlashInfer or PersistentKV attention.

This benchmark executes every decoder layer of a pretrained Llama/Mistral-family
causal LM.  Embedding, RMSNorm, RoPE, Q/K/V/O projections, gated MLPs, final
normalization, and LM head are identical between runs; only native-paged
attention changes.  The long prefix KV state is deterministically synthesized
in already-rotated form.  This measures steady-state full-model decode and is
not a prompt-prefill benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PKV_EXTENSION = ROOT / "tests" / "persistentkv_torch_extension.cu"
PAGE_SIZE = 16
HEAD_DIM = 128


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    at = (len(ordered) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - at) + ordered[hi] * (at - lo)


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def bootstrap_ci(values: list[float], seed: int, samples: int = 10_000) -> list[float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(
        len(values), (samples, len(values)), generator=generator
    )
    boot = torch.exp(torch.log(tensor[indices]).mean(dim=1))
    return [float(torch.quantile(boot, 0.025)), float(torch.quantile(boot, 0.975))]


def import_flashinfer():
    try:
        import flashinfer
    except ModuleNotFoundError as error:
        raise RuntimeError("Install flashinfer-python before running this benchmark") from error
    return flashinfer


def import_persistentkv():
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(ROOT / "build" / "e2e_torch_extensions")
    )
    python_bin = str(Path(sys.prefix) / "bin")
    if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join(
            [python_bin, os.environ.get("PATH", "")]
        )
    major, minor = torch.cuda.get_device_capability()
    sm = f"{major}{minor}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    from torch.utils.cpp_extension import load

    return load(
        name="persistentkv_e2e_ext",
        sources=[str(PKV_EXTENSION)],
        extra_include_paths=[str(ROOT)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            f"-gencode=arch=compute_{sm},code=sm_{sm}",
        ],
        verbose=False,
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def rms_epsilon(module) -> float:
    for name in ("variance_epsilon", "eps"):
        if hasattr(module, name):
            return float(getattr(module, name))
    raise RuntimeError(f"cannot determine RMSNorm epsilon for {type(module)}")


def check_model(model) -> tuple[int, int, int, int]:
    config = model.config
    layers = int(config.num_hidden_layers)
    hq = int(config.num_attention_heads)
    hkv = int(config.num_key_value_heads)
    hidden = int(config.hidden_size)
    configured_head_dim = getattr(config, "head_dim", None)
    head_dim = int(hidden // hq if configured_head_dim is None else configured_head_dim)
    if getattr(config, "model_type", "") not in {
        "mistral",
        "llama",
        "ministral3",
    }:
        raise RuntimeError(
            f"this integration supports Mistral/Llama-family modules, got "
            f"model_type={getattr(config, 'model_type', None)!r}"
        )
    if head_dim != HEAD_DIM:
        raise RuntimeError(f"PersistentKV requires head_dim=128, model has {head_dim}")
    if hq % hkv or hq // hkv not in (1, 2, 4, 8):
        raise RuntimeError(f"unsupported GQA mapping Hq={hq}, Hkv={hkv}")
    return layers, hq, hkv, hidden


def pack_model_projections(model) -> None:
    """Replace separate eager projections with production-style packed GEMMs.

    This transformation is common to both attention backends and preserves all
    weights. It reduces Q/K/V from three GEMM launches to one and gate/up from
    two GEMM launches to one in every decoder layer.
    """
    _, hq, hkv, hidden = check_model(model)
    q_width = hq * HEAD_DIM
    kv_width = hkv * HEAD_DIM
    for layer in model.model.layers:
        attention = layer.self_attn
        qkv_weight = torch.cat(
            (
                attention.q_proj.weight.detach(),
                attention.k_proj.weight.detach(),
                attention.v_proj.weight.detach(),
            ),
            dim=0,
        )
        attention.register_parameter(
            "qkv_weight",
            torch.nn.Parameter(qkv_weight, requires_grad=False),
        )
        projection_biases = [
            projection.bias
            for projection in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
            )
        ]
        if any(bias is not None for bias in projection_biases):
            widths = (q_width, kv_width, kv_width)
            qkv_bias = torch.cat(
                tuple(
                    bias.detach()
                    if bias is not None
                    else torch.zeros(
                        width,
                        device=qkv_weight.device,
                        dtype=qkv_weight.dtype,
                    )
                    for bias, width in zip(projection_biases, widths)
                )
            )
            attention.register_parameter(
                "qkv_bias",
                torch.nn.Parameter(qkv_bias, requires_grad=False),
            )
        attention.q_proj = None
        attention.k_proj = None
        attention.v_proj = None
        attention._pkv_q_width = q_width
        attention._pkv_kv_width = kv_width

        mlp = layer.mlp
        gate_up_weight = torch.cat(
            (mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()), dim=0
        )
        mlp.register_parameter(
            "gate_up_weight",
            torch.nn.Parameter(gate_up_weight, requires_grad=False),
        )
        mlp_biases = (mlp.gate_proj.bias, mlp.up_proj.bias)
        if any(bias is not None for bias in mlp_biases):
            gate_up_bias = torch.cat(
                tuple(
                    bias.detach()
                    if bias is not None
                    else torch.zeros(
                        gate_up_weight.shape[0] // 2,
                        device=gate_up_weight.device,
                        dtype=gate_up_weight.dtype,
                    )
                    for bias in mlp_biases
                )
            )
            mlp.register_parameter(
                "gate_up_bias",
                torch.nn.Parameter(gate_up_bias, requires_grad=False),
            )
        mlp.gate_proj = None
        mlp.up_proj = None
        mlp._pkv_intermediate = gate_up_weight.shape[0] // 2


class PagedCache:
    def __init__(
        self,
        layers: int,
        pages: int,
        hkv: int,
        seed: int,
        *,
        pkv_layout: bool,
    ) -> None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        shape = (
            (layers, pages, hkv, PAGE_SIZE, HEAD_DIM)
            if pkv_layout
            else (layers, pages, PAGE_SIZE, hkv, HEAD_DIM)
        )
        # A modest scale resembles post-projection activations and avoids
        # pathological saturated logits from a synthetic long prefix.
        self.k = torch.empty(shape, device="cuda", dtype=torch.float16)
        self.v = torch.empty_like(self.k)
        self.k.normal_(mean=0.0, std=0.02, generator=generator)
        self.v.normal_(mean=0.0, std=0.02, generator=generator)
        self.pkv_layout = pkv_layout

    def append(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        page, offset = divmod(position, PAGE_SIZE)
        if self.pkv_layout:
            self.k[layer, page, :, offset, :].copy_(k[0])
            self.v[layer, page, :, offset, :].copy_(v[0])
        else:
            self.k[layer, page, offset, :, :].copy_(k[0])
            self.v[layer, page, offset, :, :].copy_(v[0])


class TextOnlyCausalLM(torch.nn.Module):
    """Retain only a multimodal checkpoint's pretrained text decoder."""

    def __init__(self, text_model, lm_head, text_config) -> None:
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.config = text_config


class FullDecoder:
    def __init__(
        self,
        model,
        backend: str,
        cache: PagedCache,
        flashinfer,
        persistentkv,
        max_context: int,
        splits: int,
        flashinfer_backend: str,
        pkv_route: str,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> None:
        self.model = model
        self.backend = backend
        self.cache = cache
        self.layers, self.hq, self.hkv, self.hidden = check_model(model)
        self.intermediate = int(model.config.intermediate_size)
        self.splits = splits
        self.pkv_route = pkv_route
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.pages = math.ceil(max_context / PAGE_SIZE)
        self.block_table = torch.arange(
            self.pages, device="cuda", dtype=torch.int32
        )[None, :]
        self.indptr = torch.tensor([0, self.pages], device="cuda", dtype=torch.int32)
        self.last_len = torch.tensor([PAGE_SIZE], device="cuda", dtype=torch.int32)
        self.seq_len = torch.tensor([PAGE_SIZE], device="cuda", dtype=torch.int32)
        self.current_pages = 1
        self.flashinfer_plan_signature: tuple[int, int] | None = None
        self.flashinfer_plan_calls = 0
        self.workspace = torch.empty(
            self.hq * splits * (HEAD_DIM + 2),
            device="cuda",
            dtype=torch.float32,
        )
        self.output = torch.empty(
            1, self.hq, HEAD_DIM, device="cuda", dtype=torch.float16
        )
        self.rotated_q = torch.empty_like(self.output)
        self.normed = torch.empty(
            1, self.hidden, device="cuda", dtype=torch.float16
        )
        self.mlp_input = torch.empty_like(self.normed)
        self.hidden_after_attention = torch.empty_like(self.normed)
        self.hidden_after_mlp = torch.empty_like(self.normed)
        self.silu_product = torch.empty(
            1, self.intermediate, device="cuda", dtype=torch.float16
        )
        total_tiles = math.ceil(max_context / 32)
        work_items = []
        for split in range(splits):
            tile_begin = (total_tiles * split) // splits
            tile_end = (total_tiles * (split + 1)) // splits
            for kvh in range(self.hkv):
                work_items.append([0, kvh, split, tile_begin, tile_end])
        self.work_items = torch.tensor(
            work_items, device="cuda", dtype=torch.int32
        )
        self.split_offsets = torch.tensor(
            [0, splits], device="cuda", dtype=torch.int32
        )
        self.merge_counters = torch.zeros(
            1, self.hkv, device="cuda", dtype=torch.int32
        )
        self.q_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.out_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.wrapper = None
        self.persistentkv = persistentkv
        self.flashinfer_backend = flashinfer_backend
        if backend == "flashinfer":
            fi_workspace = torch.empty(
                128 * 1024 * 1024, device="cuda", dtype=torch.uint8
            )
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                fi_workspace,
                "HND" if cache.pkv_layout else "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                backend=flashinfer_backend,
                paged_kv_indptr_buffer=self.indptr,
                paged_kv_indices_buffer=torch.empty(
                    self.pages, device="cuda", dtype=torch.int32
                ),
                paged_kv_last_page_len_buffer=self.last_len,
            )

    def plan(self, context: int) -> None:
        pages = math.ceil(context / PAGE_SIZE)
        self.current_pages = pages
        self.seq_len[0] = context
        if self.wrapper is not None:
            self.last_len[0] = (context - 1) % PAGE_SIZE + 1
            # Match the serving-trace control: retain graph-compatible page
            # buffers and replan only when page or 32-token KV-tile counts
            # change. The current last-page length is updated every token.
            signature = (pages, math.ceil(context / 32))
            if signature != self.flashinfer_plan_signature:
                self.indptr[1] = pages
                self.wrapper.plan(
                    self.indptr,
                    self.block_table[0, :pages],
                    self.last_len,
                    self.hq,
                    self.hkv,
                    HEAD_DIM,
                    PAGE_SIZE,
                    pos_encoding_mode="NONE",
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16,
                    seq_lens=self.seq_len,
                )
                self.flashinfer_plan_signature = signature
                self.flashinfer_plan_calls += 1

    def attention(self, layer: int, q: torch.Tensor, context: int) -> torch.Tensor:
        pages = self.current_pages
        if self.backend == "flashinfer":
            assert self.wrapper is not None
            if self.flashinfer_backend == "cute-dsl":
                # CuTe DSL may use atomic split reduction, whose documented
                # output contract requires a zero-initialized destination.
                self.output.zero_()
            return self.wrapper.run(
                q,
                (self.cache.k[layer], self.cache.v[layer]),
                out=self.output,
            )
        if self.pkv_route == "workqueue-fused":
            self.persistentkv.paged_attention_workqueue(
                q,
                self.cache.k[layer],
                self.cache.v[layer],
                self.block_table[:, :pages],
                self.seq_len,
                self.work_items,
                self.split_offsets,
                self.merge_counters,
                self.q_indices,
                self.out_indices,
                self.output,
                self.workspace,
                pages * PAGE_SIZE,
                PAGE_SIZE,
                self.splits,
                True,
            )
        elif self.pkv_route == "rectangular-fused":
            self.persistentkv.paged_attention_fused(
                q,
                self.cache.k[layer],
                self.cache.v[layer],
                self.block_table[:, :pages],
                self.split_offsets,
                self.merge_counters,
                self.output,
                self.workspace,
                context,
                PAGE_SIZE,
                self.splits,
            )
        else:
            self.persistentkv.paged_attention(
                q,
                self.cache.k[layer],
                self.cache.v[layer],
                self.block_table[:, :pages],
                self.output,
                self.workspace,
                context,
                PAGE_SIZE,
                self.splits,
            )
        return self.output

    @torch.inference_mode()
    def step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        context = position + 1
        self.plan(context)
        hidden = self.model.model.embed_tokens(token.view(1, 1))[:, 0, :]
        first_layer = self.model.model.layers[0]
        self.persistentkv.rms_norm(
            hidden,
            first_layer.input_layernorm.weight,
            self.normed,
            rms_epsilon(first_layer.input_layernorm),
        )
        normed = self.normed
        cos = self.rope_cos[position]
        sin = self.rope_sin[position]
        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden
            attention = layer.self_attn
            if hasattr(attention, "qkv_weight"):
                qkv = F.linear(
                    normed,
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
            else:
                q_flat = attention.q_proj(normed)
                k_flat = attention.k_proj(normed)
                v_flat = attention.v_proj(normed)
            q = q_flat.view(1, self.hq, HEAD_DIM)
            k = k_flat.view(1, self.hkv, HEAD_DIM)
            v = v_flat.view(1, self.hkv, HEAD_DIM)
            self.persistentkv.rope_append_paged_kv(
                q,
                k,
                v,
                cos,
                sin,
                self.rotated_q,
                self.cache.k[layer_index],
                self.cache.v[layer_index],
                position,
                PAGE_SIZE,
                self.cache.pkv_layout,
            )
            if getattr(self.model.config, "model_type", "") == "ministral3":
                rope_parameters = self.model.config.rope_parameters
                beta = float(rope_parameters["llama_4_scaling_beta"])
                original = int(
                    rope_parameters["original_max_position_embeddings"]
                )
                query_scale = 1.0 + beta * math.log(
                    1.0 + math.floor(position / original)
                )
                if query_scale != 1.0:
                    self.rotated_q.mul_(query_scale)
            attended = self.attention(
                layer_index, self.rotated_q, context
            )
            attention_branch = attention.o_proj(
                attended.reshape(1, self.hq * HEAD_DIM)
            )
            self.persistentkv.add_rms_norm(
                residual,
                attention_branch,
                layer.post_attention_layernorm.weight,
                self.hidden_after_attention,
                self.mlp_input,
                rms_epsilon(layer.post_attention_layernorm),
            )
            hidden = self.hidden_after_attention
            mlp = layer.mlp
            if hasattr(mlp, "gate_up_weight"):
                gate_up = F.linear(
                    self.mlp_input,
                    mlp.gate_up_weight,
                    getattr(mlp, "gate_up_bias", None),
                )
                gate, up = gate_up.split(mlp._pkv_intermediate, dim=-1)
            else:
                gate = mlp.gate_proj(self.mlp_input)
                up = mlp.up_proj(self.mlp_input)
            if getattr(self.model.config, "hidden_act", "silu") in {
                "silu", "swish"
            }:
                self.persistentkv.silu_mul(gate, up, self.silu_product)
                activated = self.silu_product
            else:
                activated = mlp.act_fn(gate) * up
            mlp_output = mlp.down_proj(activated)

            if layer_index + 1 < self.layers:
                next_norm = self.model.model.layers[
                    layer_index + 1
                ].input_layernorm
            else:
                next_norm = self.model.model.norm
            self.persistentkv.add_rms_norm(
                hidden,
                mlp_output,
                next_norm.weight,
                self.hidden_after_mlp,
                self.normed,
                rms_epsilon(next_norm),
            )
            hidden = self.hidden_after_mlp
            normed = self.normed
        return self.model.lm_head(normed).float()


def timed_sequence(
    decoder: FullDecoder, tokens: list[int], start_position: int
) -> tuple[list[torch.Tensor], float, float]:
    token_tensor = torch.tensor(tokens, device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    logits = [
        decoder.step(token_tensor[index : index + 1], start_position + index)
        for index in range(len(tokens))
    ]
    end.record()
    torch.cuda.synchronize()
    return logits, begin.elapsed_time(end), (time.perf_counter() - wall_start) * 1e3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--context", type=int, default=32752)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--splits",
        type=int,
        default=0,
        help=(
            "sequence splits; 0 selects one wave using runtime CUDA "
            "occupancy, SM count, and the model KV-head count"
        ),
    )
    parser.add_argument(
        "--flashinfer-backend",
        choices=("auto", "fa2", "fa3", "cute-dsl"),
        default="auto",
        help="explicit FlashInfer backend; use cute-dsl when validating SM120",
    )
    parser.add_argument(
        "--pkv-route",
        choices=(
            "rectangular",
            "rectangular-fused",
            "workqueue-fused",
        ),
        default="rectangular-fused",
    )
    parser.add_argument(
        "--unfused-projections",
        action="store_true",
        help="retain separate Q/K/V and gate/up GEMMs (diagnostic only)",
    )
    parser.add_argument(
        "--shared-hnd-cache",
        action="store_true",
        help=(
            "give FlashInfer and PersistentKV the exact same HND KV tensor; "
            "this is the memory-efficient protocol for 32K pretrained-model runs"
        ),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-logits-cosine", type=float, default=0.999)
    parser.add_argument("--min-top1-agreement", type=float, default=0.95)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[20260708, 20260709, 20260710, 20260711, 20260712],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="do not download a missing Hugging Face model",
    )
    return parser.parse_args()


def summarize_times(gpu_ms: list[float], wall_ms: list[float], steps: int) -> dict:
    return {
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "gpu_mean_ms": statistics.mean(gpu_ms),
        "wall_mean_ms": statistics.mean(wall_ms),
        "wall_p95_ms": percentile(wall_ms, 0.95),
        "wall_tokens_per_s": steps / (statistics.mean(wall_ms) / 1e3),
        "wall_ms_per_token": statistics.mean(wall_ms) / steps,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.context < PAGE_SIZE or args.decode_steps < 1 or args.splits < 0:
        raise SystemExit("context must be >=16 and decode-steps must be positive")

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    print("Loading full model...", flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM

    if args.model in {
        "synthetic-mistral-smoke",
        "synthetic-llama-smoke",
        "synthetic-ministral3-smoke",
        "synthetic-mistral-32layer",
    }:
        # Test-only shape: it preserves the production attention dimensions
        # required by the CUDA kernel while minimizing layers, vocabulary, and
        # MLP width. It is never suitable for a paper result.
        if args.model == "synthetic-mistral-32layer":
            config_kwargs = dict(
                vocab_size=4096,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=32,
                num_attention_heads=8,
                num_key_value_heads=2,
                max_position_embeddings=args.context + args.decode_steps,
            )
        else:
            config_kwargs = dict(
                vocab_size=256,
                hidden_size=4096,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=args.context + args.decode_steps,
            )
        if args.model == "synthetic-ministral3-smoke":
            from transformers import Ministral3Config, Ministral3ForCausalLM

            config_kwargs.update(
                hidden_size=1024,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=128,
            )
            model = Ministral3ForCausalLM(
                Ministral3Config(**config_kwargs)
            )
        elif args.model == "synthetic-llama-smoke":
            from transformers import LlamaConfig, LlamaForCausalLM

            model = LlamaForCausalLM(LlamaConfig(**config_kwargs))
        else:
            from transformers import MistralConfig, MistralForCausalLM

            model = MistralForCausalLM(MistralConfig(**config_kwargs))
        model = model.half().eval()
    else:
        pretrained_config = AutoConfig.from_pretrained(
            args.model, local_files_only=args.local_files_only
        )
        if getattr(pretrained_config, "model_type", "") == "mistral3":
            from transformers import Mistral3ForConditionalGeneration

            multimodal_model = (
                Mistral3ForConditionalGeneration.from_pretrained(
                    args.model,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    local_files_only=args.local_files_only,
                ).eval()
            )
            model = TextOnlyCausalLM(
                multimodal_model.model.language_model,
                multimodal_model.lm_head,
                pretrained_config.text_config,
            ).eval()
            del multimodal_model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=args.local_files_only,
            ).eval()
    layers, hq, hkv, hidden = check_model(model)
    if not args.unfused_projections:
        print("Packing QKV and gate/up projections...", flush=True)
        pack_model_projections(model)
    model = model.cuda()
    print("Loading FlashInfer and PersistentKV kernels...", flush=True)
    flashinfer = import_flashinfer()
    persistentkv = import_persistentkv()
    max_context = args.context + args.decode_steps
    total_tiles = math.ceil(max_context / 32)
    active_blocks_per_sm = int(
        persistentkv.paged_partial_active_blocks_per_sm(hq, hkv)
    )
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if args.splits:
        selected_splits = args.splits
        split_selection = "explicit"
    else:
        selected_splits = max(
            1,
            min(total_tiles, (active_blocks_per_sm * sm_count) // hkv),
        )
        split_selection = "runtime_occupancy_one_wave"
    if selected_splits > total_tiles:
        raise SystemExit(
            f"splits={selected_splits} exceeds available KV tiles={total_tiles}"
        )
    if (
        args.pkv_route == "rectangular-fused"
        and selected_splits == 1
    ):
        raise SystemExit("selected merge route requires at least two splits")
    print(
        f"PersistentKV splits={selected_splits} source={split_selection} "
        f"active_blocks_per_sm={active_blocks_per_sm} sm_count={sm_count}",
        flush=True,
    )
    with torch.inference_mode():
        position_ids = torch.arange(
            max_context, device="cuda", dtype=torch.long
        )[None, :]
        rope_probe = torch.empty(
            1, 1, hidden, device="cuda", dtype=torch.float16
        )
        rope_cos, rope_sin = model.model.rotary_emb(rope_probe, position_ids)
        if rope_cos.dim() == 3:
            rope_cos = rope_cos[0]
            rope_sin = rope_sin[0]
        if tuple(rope_cos.shape) != (max_context, HEAD_DIM):
            raise RuntimeError(
                f"unexpected RoPE table shape {tuple(rope_cos.shape)}"
            )
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()
    pages = math.ceil(max_context / PAGE_SIZE)
    vocab = int(model.config.vocab_size)
    model_runtime_allocated_bytes = torch.cuda.memory_allocated()
    model_runtime_reserved_bytes = torch.cuda.memory_reserved()
    seed_results = []

    for seed in args.seeds:
        print(f"seed={seed}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        tokens = [
            int(value)
            for value in torch.randint(
                0, vocab, (args.decode_steps,), generator=torch.Generator().manual_seed(seed)
            )
        ]
        pkv_cache = PagedCache(layers, pages, hkv, seed + 1000, pkv_layout=True)
        if args.shared_hnd_cache:
            # Both backends consume one physical HND tensor. This removes a
            # duplicate multi-GB KV allocation without changing any values.
            flash_cache = pkv_cache
        else:
            # Derive FlashInfer's NHD cache from the exact same logical values.
            flash_cache = PagedCache.__new__(PagedCache)
            flash_cache.k = pkv_cache.k.permute(0, 1, 3, 2, 4).contiguous()
            flash_cache.v = pkv_cache.v.permute(0, 1, 3, 2, 4).contiguous()
            flash_cache.pkv_layout = False
        flash_decoder = FullDecoder(
            model, "flashinfer", flash_cache, flashinfer, persistentkv,
            max_context, selected_splits, args.flashinfer_backend, args.pkv_route,
            rope_cos, rope_sin,
        )
        pkv_decoder = FullDecoder(
            model, "persistentkv", pkv_cache, flashinfer, persistentkv,
            max_context, selected_splits, args.flashinfer_backend, args.pkv_route,
            rope_cos, rope_sin,
        )
        start = args.context - 1
        # Compile/lazily initialize both paths and validate every full-model
        # decode step before timing.
        references, _, _ = timed_sequence(flash_decoder, tokens, start)
        candidates, _, _ = timed_sequence(pkv_decoder, tokens, start)
        differences = [
            (reference - candidate).abs()
            for reference, candidate in zip(references, candidates)
        ]
        cosines = [
            torch.nn.functional.cosine_similarity(reference, candidate).item()
            for reference, candidate in zip(references, candidates)
        ]
        top1 = [
            reference.argmax().item() == candidate.argmax().item()
            for reference, candidate in zip(references, candidates)
        ]
        correctness_pass = (
            min(cosines) >= args.min_logits_cosine
            and sum(top1) / len(top1) >= args.min_top1_agreement
        )

        for _ in range(args.warmups):
            timed_sequence(flash_decoder, tokens, start)
            timed_sequence(pkv_decoder, tokens, start)

        fi_gpu, fi_wall, pkv_gpu, pkv_wall = [], [], [], []
        for repeat in range(args.repeats):
            # Reverse order on alternating repeats to balance drift.
            order: list[tuple[str, FullDecoder]]
            if repeat % 2:
                order = [("pkv", pkv_decoder), ("fi", flash_decoder)]
            else:
                order = [("fi", flash_decoder), ("pkv", pkv_decoder)]
            for name, decoder in order:
                _, gpu, wall = timed_sequence(decoder, tokens, start)
                if name == "fi":
                    fi_gpu.append(gpu)
                    fi_wall.append(wall)
                else:
                    pkv_gpu.append(gpu)
                    pkv_wall.append(wall)
        fi = summarize_times(fi_gpu, fi_wall, args.decode_steps)
        pkv = summarize_times(pkv_gpu, pkv_wall, args.decode_steps)
        ratio = fi["wall_mean_ms"] / pkv["wall_mean_ms"]
        shared_cache_bytes = (
            pkv_cache.k.numel() * pkv_cache.k.element_size()
            + pkv_cache.v.numel() * pkv_cache.v.element_size()
        )
        seed_results.append(
            {
                "seed": seed,
                "flashinfer": fi,
                "persistentkv": pkv,
                "persistentkv_over_flashinfer_wall_tokens_s": ratio,
                "flashinfer_cached_plan_calls": (
                    flash_decoder.flashinfer_plan_calls
                ),
                "shared_kv_cache_bytes": shared_cache_bytes,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "correctness": {
                    "checked_decode_steps": args.decode_steps,
                    "logits_max_abs": max(float(diff.max()) for diff in differences),
                    "logits_mean_abs_max": max(
                        float(diff.mean()) for diff in differences
                    ),
                    "logits_cosine_min": min(float(value) for value in cosines),
                    "top1_agreement_fraction": sum(top1) / len(top1),
                    "passed": correctness_pass,
                },
            }
        )
        print(
            f"  ratio={ratio:.4f}x cosine_min={min(cosines):.8f} "
            f"top1={sum(top1)}/{len(top1)}", flush=True
        )
        # ``order`` and the loop variable ``decoder`` retain the decoder (and
        # therefore its multi-GB KV cache) after the timing loop. Release every
        # tensor-owning reference before constructing the next seed's cache.
        del order, decoder, references, candidates, differences
        del flash_decoder, pkv_decoder, flash_cache, pkv_cache
        torch.cuda.empty_cache()

    ratios = [
        item["persistentkv_over_flashinfer_wall_tokens_s"] for item in seed_results
    ]
    extension_hash = hashlib.sha256(PKV_EXTENSION.read_bytes()).hexdigest()
    kernel_hash = hashlib.sha256(
        (ROOT / "kernels" / "persistentkv_attention.cuh").read_bytes()
    ).hexdigest()
    result = {
        "schema_version": 1,
        "experiment": "persistentkv_full_transformer_decode",
        "claim_scope": (
            (
                "Functional/performance validation of the complete multi-layer "
                "runtime with randomly initialized weights; not a paper result."
            )
            if args.model.startswith("synthetic-")
            else (
                "Full pretrained Mistral/Llama-family decoder steady-state decode "
                "with a deterministically synthesized long-prefix paged KV cache; "
                "excludes prefill."
            )
        ),
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": HEAD_DIM,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "page_size": PAGE_SIZE,
        "kv_cache_protocol": (
            "one shared physical HND tensor"
            if args.shared_hnd_cache
            else "logically identical backend-preferred HND and NHD tensors"
        ),
        "splits": selected_splits,
        "requested_splits": args.splits,
        "split_selection": split_selection,
        "paged_partial_active_blocks_per_sm": active_blocks_per_sm,
        "flashinfer_backend": args.flashinfer_backend,
        "flashinfer_plan_protocol": (
            "graph-compatible persistent buffers; replan only on page-count "
            "or 32-token KV-tile-count changes; update last-page length each token"
        ),
        "persistentkv_route": args.pkv_route,
        "packed_projections": not args.unfused_projections,
        "fused_rope_kv_append": True,
        "fused_rmsnorm_swiglu": True,
        "model_runtime_cuda_allocated_bytes_before_kv_cache": (
            model_runtime_allocated_bytes
        ),
        "model_runtime_cuda_reserved_bytes_before_kv_cache": (
            model_runtime_reserved_bytes
        ),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seeds": args.seeds,
        "seed_results": seed_results,
        "aggregate": {
            "persistentkv_over_flashinfer_wall_tokens_s_geomean": geometric_mean(ratios),
            "paired_seed_bootstrap_95_ci": bootstrap_ci(ratios, args.seeds[0] + 5090),
            "top1_agreement_fraction": statistics.mean(
                item["correctness"]["top1_agreement_fraction"]
                for item in seed_results
            ),
            "minimum_logits_cosine": min(
                item["correctness"]["logits_cosine_min"] for item in seed_results
            ),
            "maximum_logits_max_abs": max(
                item["correctness"]["logits_max_abs"] for item in seed_results
            ),
            "correctness_passed": all(
                item["correctness"]["passed"] for item in seed_results
            ),
            "correctness_thresholds": {
                "minimum_logits_cosine": args.min_logits_cosine,
                "minimum_top1_agreement": args.min_top1_agreement,
            },
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"{major}.{minor}",
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "pkv_extension_sha256": extension_hash,
            "pkv_kernel_header_sha256": kernel_hash,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")
    if not result["aggregate"]["correctness_passed"]:
        raise SystemExit("full-model correctness thresholds failed; see output JSON")


if __name__ == "__main__":
    main()
