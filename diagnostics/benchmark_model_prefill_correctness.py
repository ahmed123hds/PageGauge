#!/usr/bin/env python3
"""Validate PageGauge on KV states produced by an HF SDPA FP16 prefill.

The default path is a correctness-only diagnostic.  It creates each request's
coherent prefix with the pretrained Hugging Face SDPA decoder cast to FP16,
copies the post-RoPE K/V states into the production request-major FP16 paged
layout, and then derives the PageGauge INT8-plus-exact-tail cache from those
same tensors.  Only after the HF reference logits have been recorded are the
common packed QKV and gate/up projections installed and the production
PageGauge decoder run for one synchronized, teacher-forced decode page.

With ``--run-token-step-graphs``, the same live model, decoder objects, and
caches are additionally passed in-process to the whole-decoder per-token graph
diagnostic.  Prefill is still excluded from all latency and throughput claims.

With ``--run-generated-sequence-graph``, the same fixture is instead passed to
a single GPU-resident 16-token greedy graph per backend.  Its latency is for a
complete generated block, not for streaming one token to a client.

The request-at-a-time prefill is deliberate: it keeps the temporary HF dynamic
cache small enough to coexist with the batch-4 FP16 paged destination on a
32-GiB device.  It does not enter a latency or throughput claim.
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
import re
import statistics
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOKEN_STEP_GRAPH_PATH = ROOT / "diagnostics/benchmark_token_step_graphs.py"
GENERATED_SEQUENCE_GRAPH_PATH = (
    ROOT / "diagnostics/benchmark_generated_sequence_graph.py"
)
WIKITEXT2_RAW_V1_SHA256 = (
    "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
)
FUTURE_PAGE_CODE_CANARY = -128

# Frozen, untouched full-distribution confirmation of the TRAIN-selected
# S3/T768 mechanism.  These constants are deliberately kept in the raw worker
# as well as the CPU-only reducer: a publication shard must fail before the
# TEST member is opened if its command line does not describe one of the five
# preregistered groups.
HELDOUT_PROTOCOL_NAME = "wikitext2_test_s3_t768_d1536_v1"
HELDOUT_MODEL = "mistralai/Mistral-7B-v0.3"
HELDOUT_MODEL_REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
HELDOUT_MODEL_CONFIG_SHA256 = (
    "f223f73de240195fe40495201ecbe85c9dac820842e4fe7e50c38d576b8d22ca"
)
HELDOUT_MODEL_PARAMETER_COUNT = 7_248_023_552
HELDOUT_TOKENIZER_MANIFEST_SHA256 = (
    "c97b022b02f1b0c13a96ed355471032bdbeefe28eaad8db51b488207e3cac67a"
)
HELDOUT_MEMBER = "wikitext-2-raw/wiki.test.raw"
HELDOUT_TEST_TOKEN_COUNT = 328_879
HELDOUT_CONTEXT = 20_480
HELDOUT_DECODE_STEPS = 1_536
HELDOUT_EXACT_TAIL = 768
HELDOUT_EXACT_PREFIX_PAGES = 3
HELDOUT_SPLIT_PAGES = 256
HELDOUT_PREFILL_CHUNK_TOKENS = 1_024
HELDOUT_WINDOW_STRIDE = 23_600
HELDOUT_WINDOW_TOKENS = HELDOUT_CONTEXT + HELDOUT_DECODE_STEPS
HELDOUT_STARTS = tuple(index * HELDOUT_WINDOW_STRIDE for index in range(14))
HELDOUT_GROUPS = (
    HELDOUT_STARTS[0:3],
    HELDOUT_STARTS[3:6],
    HELDOUT_STARTS[6:9],
    HELDOUT_STARTS[9:12],
    HELDOUT_STARTS[12:14],
)
HELDOUT_MIN_LOGITS_COSINE = 0.995
HELDOUT_MIN_TOP1_AGREEMENT = 0.99
HELDOUT_MIN_BASELINE_HF_COSINE = 0.999
HELDOUT_SEED = 20_260_861

# The raw quality path keeps the FP16 model, FP16 paged cache, and PageGauge
# cache resident together.  This projection reproduces the tensor formula,
# and a fixed 2-GiB reserve covers decoder workspaces and allocator overhead.
HELDOUT_MODEL_FP16_RESIDENT_BYTES = 14_496_047_616
HELDOUT_MEMORY_RESERVE_BYTES = 2 * 1024**3
HELDOUT_MAX_CORESIDENT_BATCH = 3

MODEL_PREFILL_SOURCE_PATHS = (
    Path(__file__),
    ROOT / "diagnostics/aggregate_heldout_quality.py",
    TOKEN_STEP_GRAPH_PATH,
    GENERATED_SEQUENCE_GRAPH_PATH,
    ROOT / "scripts/benchmark_page_gauge_transformer.py",
    ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py",
    ROOT / "scripts/benchmark_page_gauge_overheads.py",
    ROOT / "scripts/benchmark_e2e_transformer.py",
    ROOT / "scripts/page_gauge_runtime.py",
    ROOT / "tests/page_gauge_append_extension.cu",
    ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch",
)


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PG = load_local_module(
    "page_gauge_transformer_for_model_prefill_correctness",
    ROOT / "scripts/benchmark_page_gauge_transformer.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=20480)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument(
        "--exact-prefix-pages",
        type=int,
        default=0,
        help=(
            "fixed exact FP16 prefix length S in pages; stored before the "
            "modulo exact-tail ring"
        ),
    )
    parser.add_argument(
        "--exact-static-suffix-pages",
        type=int,
        default=0,
        help=(
            "fixed exact FP16 pages at the end of the original prefill; "
            "stored beside the prefix before the modulo tail ring"
        ),
    )
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=128)
    parser.add_argument(
        "--tail-attention",
        choices=("flashinfer_merge", "fused_kernel"),
        default="flashinfer_merge",
        help="PageGauge exact-tail attention/online-state merge implementation",
    )
    parser.add_argument(
        "--old-value-scale-placement",
        choices=("probability", "value_fragment"),
        default="probability",
        help="placement of the old-cache INT8 V scale in PageGauge attention",
    )
    parser.add_argument(
        "--heldout-policy",
        action="store_true",
        help=(
            "fail closed on the frozen untouched WikiText-2 TEST S3/T768/D1536 "
            "full-distribution protocol"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260861)
    parser.add_argument(
        "--token-source",
        choices=("wikitext2", "random"),
        default="wikitext2",
        help=(
            "wikitext2 reads and locally tokenizes the selected raw member; "
            "random is a deterministic synthetic-token negative control"
        ),
    )
    parser.add_argument(
        "--wikitext-zip",
        type=Path,
        default=ROOT / "data/wikitext-2-raw-v1.zip",
        help="local WikiText-2 raw ZIP (read in-memory; never extracted)",
    )
    parser.add_argument(
        "--wikitext-member",
        default="wikitext-2-raw/wiki.train.raw",
        help="UTF-8 train/valid/test member within --wikitext-zip",
    )
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="first corpus token used by wikitext2",
    )
    parser.add_argument(
        "--token-stride",
        type=int,
        default=0,
        help=(
            "distance between wikitext2 request windows; zero means one disjoint "
            "context-plus-decode-minus-BOS sequence"
        ),
    )
    parser.add_argument(
        "--disable-attention-cuda-graphs",
        action="store_true",
        help="run only eager attention; production graphs are checked by default",
    )
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument(
        "--min-baseline-hf-cosine",
        type=float,
        default=0.999,
        help="conversion/decoder fidelity gate against the HF SDPA FP16 reference",
    )
    parser.add_argument(
        "--run-token-step-graphs",
        action="store_true",
        help=(
            "reuse this live coherent model/cache fixture for matched per-token "
            "whole-decoder CUDA-graph correctness and timing"
        ),
    )
    parser.add_argument("--token-graph-capture-warmups", type=int, default=3)
    parser.add_argument("--token-graph-warmups", type=int, default=10)
    parser.add_argument("--token-graph-repeats", type=int, default=30)
    parser.add_argument("--token-graph-cache-scrub-mib", type=int, default=256)
    parser.add_argument(
        "--run-direct-feedback-paths",
        action="store_true",
        help=(
            "reuse the coherent fixture for deep-queued generated feedback "
            "with fully eager attention and production per-layer attention graphs"
        ),
    )
    parser.add_argument(
        "--direct-feedback-protocol",
        choices=("deep-queued", "token-synchronous"),
        default="deep-queued",
        help="generated-feedback synchronization protocol for the direct path",
    )
    parser.add_argument("--direct-feedback-warmups", type=int, default=10)
    parser.add_argument("--direct-feedback-repeats", type=int, default=30)
    parser.add_argument("--direct-feedback-cache-scrub-mib", type=int, default=256)
    parser.add_argument(
        "--run-static-teacher-control",
        action="store_true",
        help=(
            "reuse the coherent live fixture for predeclared corpus-token "
            "retain/discard-logits timing and component localization"
        ),
    )
    parser.add_argument("--static-teacher-warmups", type=int, default=3)
    parser.add_argument("--static-teacher-repeats", type=int, default=4)
    parser.add_argument("--static-teacher-cache-scrub-mib", type=int, default=256)
    parser.add_argument("--static-teacher-component-repeats", type=int, default=20)
    parser.add_argument(
        "--static-teacher-skip-matrix-timing",
        action="store_true",
        help="run correctness/component localization without retain/discard timing",
    )
    parser.add_argument(
        "--static-teacher-terminal-pg-only-residency",
        action="store_true",
        help=(
            "after co-resident correctness/profile, release the FP16 baseline "
            "cache and repeat only PageGauge components; terminal diagnostic"
        ),
    )
    parser.add_argument(
        "--static-teacher-terminal-relocate-pg-cache",
        action="store_true",
        help=(
            "after terminal baseline release/control, clone/swap every "
            "PageGauge cache tensor and repeat component profiling"
        ),
    )
    parser.add_argument(
        "--run-generated-sequence-graph",
        action="store_true",
        help=(
            "reuse the live coherent fixture for one matched GPU-resident "
            "greedy 16-token whole-decoder graph per backend"
        ),
    )
    parser.add_argument("--generated-graph-capture-warmups", type=int, default=3)
    parser.add_argument("--generated-graph-warmups", type=int, default=10)
    parser.add_argument("--generated-graph-repeats", type=int, default=30)
    parser.add_argument("--generated-graph-cache-scrub-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def model_prefill_source_manifest() -> dict[str, str]:
    """Hash the complete worker closure in deterministic relative-path order."""

    missing = [path for path in MODEL_PREFILL_SOURCE_PATHS if not path.is_file()]
    if missing:
        raise ValueError(f"model-prefill source closure is incomplete: {missing}")
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in MODEL_PREFILL_SOURCE_PATHS
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_heldout_command(args: argparse.Namespace) -> tuple[int, ...] | None:
    """Fail before dataset access unless ``args`` is a frozen TEST shard."""

    if not args.heldout_policy:
        return None
    fixed = {
        "model": (args.model, HELDOUT_MODEL),
        "context": (args.context, HELDOUT_CONTEXT),
        "decode steps": (args.decode_steps, HELDOUT_DECODE_STEPS),
        "exact tail": (args.exact_tail, HELDOUT_EXACT_TAIL),
        "exact prefix pages": (
            args.exact_prefix_pages,
            HELDOUT_EXACT_PREFIX_PAGES,
        ),
        "exact static suffix pages": (
            getattr(args, "exact_static_suffix_pages", 0),
            0,
        ),
        "prefill chunk": (
            args.prefill_chunk_tokens,
            HELDOUT_PREFILL_CHUNK_TOKENS,
        ),
        "baseline split": (args.baseline_split_pages, HELDOUT_SPLIT_PAGES),
        "candidate split": (args.candidate_split_pages, HELDOUT_SPLIT_PAGES),
        "tail attention": (args.tail_attention, "flashinfer_merge"),
        "old V-scale placement": (
            args.old_value_scale_placement,
            "probability",
        ),
        "seed": (args.seed, HELDOUT_SEED),
        "token source": (args.token_source, "wikitext2"),
        "WikiText member": (args.wikitext_member, HELDOUT_MEMBER),
        "token stride": (args.token_stride, HELDOUT_WINDOW_STRIDE),
        "minimum logits cosine": (
            args.min_logits_cosine,
            HELDOUT_MIN_LOGITS_COSINE,
        ),
        "minimum top-1 agreement": (
            args.min_top1_agreement,
            HELDOUT_MIN_TOP1_AGREEMENT,
        ),
        "minimum baseline/HF cosine": (
            args.min_baseline_hf_cosine,
            HELDOUT_MIN_BASELINE_HF_COSINE,
        ),
    }
    mismatches = [
        f"{name}: got {observed!r}, expected {expected!r}"
        for name, (observed, expected) in fixed.items()
        if observed != expected
    ]
    if mismatches:
        raise ValueError(
            "held-out command differs from the frozen protocol: "
            + "; ".join(mismatches)
        )
    if not args.disable_attention_cuda_graphs:
        raise ValueError(
            "held-out D1536 collection requires eager attention "
            "(--disable-attention-cuda-graphs)"
        )
    optional_modes = {
        "token-step graphs": args.run_token_step_graphs,
        "direct feedback": args.run_direct_feedback_paths,
        "static teacher control": args.run_static_teacher_control,
        "generated-sequence graph": args.run_generated_sequence_graph,
    }
    enabled = [name for name, value in optional_modes.items() if value]
    if enabled:
        raise ValueError(
            "held-out correctness collection forbids optional timing modes: "
            + ", ".join(enabled)
        )
    matching = [
        group
        for group in HELDOUT_GROUPS
        if args.batch_size == len(group) and args.token_offset == group[0]
    ]
    if len(matching) != 1:
        raise ValueError(
            "held-out shard must be one frozen (offset,batch) group; got "
            f"({args.token_offset},{args.batch_size})"
        )
    return tuple(matching[0])


def heldout_co_resident_memory_gate(
    *,
    batch_size: int,
    context: int,
    decode_steps: int,
    exact_tail: int,
    exact_prefix_pages: int,
    exact_static_suffix_pages: int = 0,
    free_bytes: int,
) -> dict[str, Any]:
    """Project the simultaneously resident model and two cache formats."""

    if min(batch_size, context, decode_steps, exact_tail, free_bytes) <= 0:
        raise ValueError("co-resident memory inputs must be positive")
    if context % PG.PAGE or exact_tail % PG.PAGE:
        raise ValueError("co-resident cache projection requires page alignment")
    exact_prefix_pages = PG.validate_exact_prefix_pages(exact_prefix_pages)
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_prefix_pages,
        initial_context_pages=context // PG.PAGE,
    )
    layers = 32
    hq = 32
    hkv = 8
    dim = PG.DIM
    pages = math.ceil((context + decode_steps) / PG.PAGE)
    tail_pages = exact_tail // PG.PAGE
    fp16 = torch.tensor([], dtype=torch.float16).element_size()
    int8 = torch.tensor([], dtype=torch.int8).element_size()
    baseline_bytes = 2 * layers * batch_size * pages * PG.PAGE * hkv * dim * fp16
    codes_bytes = 2 * layers * batch_size * pages * PG.PAGE * hkv * dim * int8
    scales_bytes = 2 * layers * batch_size * pages * hkv * fp16
    centers_bytes = layers * batch_size * (2 * hkv * dim + hq * dim) * fp16
    exact_bytes = (
        2
        * layers
        * batch_size
        * (tail_pages + exact_prefix_pages + exact_static_suffix_pages)
        * PG.PAGE
        * hkv
        * dim
        * fp16
    )
    page_gauge_bytes = codes_bytes + scales_bytes + centers_bytes + exact_bytes
    projected = HELDOUT_MODEL_FP16_RESIDENT_BYTES + baseline_bytes + page_gauge_bytes
    required = projected + HELDOUT_MEMORY_RESERVE_BYTES
    passed = bool(batch_size <= HELDOUT_MAX_CORESIDENT_BATCH and required <= free_bytes)
    return {
        "passed": passed,
        "policy": "model + matched FP16 cache + PageGauge cache + fixed reserve",
        "max_batch_size": HELDOUT_MAX_CORESIDENT_BATCH,
        "observed_cuda_free_bytes_before_model_load": int(free_bytes),
        "model_fp16_resident_bytes": HELDOUT_MODEL_FP16_RESIDENT_BYTES,
        "flashinfer_fp16_cache_bytes": baseline_bytes,
        "page_gauge_cache_bytes": page_gauge_bytes,
        "page_gauge_components": {
            "int8_codes_bytes": codes_bytes,
            "fp16_scales_bytes": scales_bytes,
            "fp16_centers_and_output_center_bytes": centers_bytes,
            "fp16_exact_prefix_and_tail_bytes": exact_bytes,
        },
        "projected_co_resident_bytes": projected,
        "fixed_workspace_allocator_reserve_bytes": HELDOUT_MEMORY_RESERVE_BYTES,
        "required_free_bytes": required,
        "headroom_after_required_bytes": int(free_bytes - required),
        "logical_pages_per_request": pages,
        "exact_tail_pages": tail_pages,
        "exact_prefix_pages": exact_prefix_pages,
        "exact_static_suffix_pages": exact_static_suffix_pages,
    }


def validate_heldout_model_identity(
    *,
    model_revision: Any,
    model_config_sha256: str,
    parameter_count: int,
    expected_starts: tuple[int, ...] | None,
) -> None:
    """Fail on a wrong local checkpoint before opening the TEST member."""

    if expected_starts is None:
        return
    model_values = {
        "revision": (model_revision, HELDOUT_MODEL_REVISION),
        "config SHA256": (model_config_sha256, HELDOUT_MODEL_CONFIG_SHA256),
        "parameter count": (parameter_count, HELDOUT_MODEL_PARAMETER_COUNT),
    }
    mismatches = [
        f"{name}: got {observed!r}, expected {expected!r}"
        for name, (observed, expected) in model_values.items()
        if observed != expected
    ]
    if mismatches:
        raise ValueError(
            "held-out model identity differs from the frozen protocol: "
            + "; ".join(mismatches)
        )


def validate_heldout_model_and_tokens(
    *,
    model_revision: Any,
    model_config_sha256: str,
    parameter_count: int,
    token_provenance: dict[str, Any],
    expected_starts: tuple[int, ...] | None,
) -> None:
    """Validate post-load identities and corpus coordinates for heldout mode."""

    if expected_starts is None:
        return
    validate_heldout_model_identity(
        model_revision=model_revision,
        model_config_sha256=model_config_sha256,
        parameter_count=parameter_count,
        expected_starts=expected_starts,
    )
    mismatches: list[str] = []
    tokenizer = token_provenance.get("tokenizer", {})
    source_values = {
        "archive SHA256": (
            token_provenance.get("archive_sha256"),
            WIKITEXT2_RAW_V1_SHA256,
        ),
        "archive member": (
            token_provenance.get("archive_member"),
            HELDOUT_MEMBER,
        ),
        "dataset split": (token_provenance.get("split"), "test"),
        "TEST token count": (
            token_provenance.get("available_corpus_token_count"),
            HELDOUT_TEST_TOKEN_COUNT,
        ),
        "window starts": (
            tuple(token_provenance.get("corpus_window_start_offsets", ())),
            expected_starts,
        ),
        "window ends": (
            tuple(token_provenance.get("corpus_window_end_offsets_exclusive", ())),
            tuple(start + HELDOUT_WINDOW_TOKENS for start in expected_starts),
        ),
        "tokenizer manifest SHA256": (
            tokenizer.get("manifest_sha256"),
            HELDOUT_TOKENIZER_MANIFEST_SHA256,
        ),
        "tokenizer revision": (
            tokenizer.get("resolved_snapshot_revision"),
            HELDOUT_MODEL_REVISION,
        ),
    }
    mismatches.extend(
        f"{name}: got {observed!r}, expected {expected!r}"
        for name, (observed, expected) in source_values.items()
        if observed != expected
    )
    if mismatches:
        raise ValueError(
            "held-out model/source identity differs from the frozen protocol: "
            + "; ".join(mismatches)
        )


def wikitext_split_from_member(member: str) -> str:
    """Derive the canonical split from a frozen WikiText raw member name."""
    name = PurePosixPath(member).name
    match = re.fullmatch(r"wiki\.(train|valid|test)\.raw", name)
    if match is None:
        raise ValueError(
            "WikiText member must end in wiki.train.raw, wiki.valid.raw, or "
            f"wiki.test.raw; got {member!r}"
        )
    return {"train": "train", "valid": "validation", "test": "test"}[match.group(1)]


def tokenizer_artifact_provenance(
    tokenizer: Any, model_name_or_path: str
) -> dict[str, Any]:
    """Resolve and hash every local file that defines token-ID semantics."""
    requested_revision = tokenizer.init_kwargs.get("_commit_hash")
    model_path = Path(model_name_or_path).expanduser()
    if model_path.is_dir():
        snapshot_path = model_path.resolve()
    else:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=model_name_or_path,
                revision=requested_revision,
                local_files_only=True,
            )
        ).resolve()

    resolved_revision = requested_revision
    if (
        resolved_revision is None
        and snapshot_path.parent.name == "snapshots"
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", snapshot_path.name)
    ):
        resolved_revision = snapshot_path.name.lower()

    artifact_names = {
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    artifact_names.update(
        str(name)
        for name in getattr(tokenizer, "vocab_files_names", {}).values()
        if name
    )
    artifacts = []
    for name in sorted(artifact_names):
        path = snapshot_path / name
        if path.is_file():
            artifacts.append(
                {
                    "relative_path": name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    if not artifacts:
        raise ValueError(
            f"no tokenizer artifacts were found in local snapshot {snapshot_path}"
        )

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    backend_state_sha256 = None
    if backend_tokenizer is not None:
        backend_state_sha256 = sha256_bytes(backend_tokenizer.to_str().encode("utf-8"))
    manifest_payload = {
        "resolved_revision": resolved_revision,
        "artifacts": artifacts,
        "backend_tokenizer_state_sha256": backend_state_sha256,
    }
    return {
        "requested_name_or_path": model_name_or_path,
        "resolved_snapshot_revision": resolved_revision,
        "class": type(tokenizer).__name__,
        "artifacts": artifacts,
        "backend_tokenizer_state_sha256": backend_state_sha256,
        "manifest_sha256": canonical_json_sha256(manifest_payload),
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0,1]")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return float(ordered[lower])
    weight = coordinate - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def finite_exp(value: float) -> float:
    try:
        result = math.exp(value)
    except OverflowError as error:
        raise ValueError(f"exponential overflow for {value}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite exponential for {value}")
    return result


def request_window_metadata(
    token_provenance: dict[str, Any],
    context: int,
    decode_steps: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    starts = token_provenance.get("corpus_window_start_offsets")
    ends = token_provenance.get("corpus_window_end_offsets_exclusive")
    split = token_provenance.get("split")
    member_sha256 = token_provenance.get("archive_member_sha256")
    records = []
    for request in range(batch_size):
        if starts is None:
            corpus_begin = None
            corpus_end = None
            label_begin = None
            label_end = None
            unit_id = f"random-seed-{token_provenance.get('seed')}-request-{request}"
        else:
            # Model position one is the first corpus token because position zero
            # is the prepended BOS.  Decode output step zero predicts position
            # context+1, hence corpus offset start+context.
            corpus_begin = int(starts[request])
            corpus_end = int(ends[request])
            label_begin = corpus_begin + context
            label_end = label_begin + decode_steps
            unit_id = (
                f"wikitext2-{split}-{member_sha256[:16]}-{label_begin}-{label_end}"
            )
        records.append(
            {
                "request": request,
                "cluster_unit_id": unit_id,
                "dataset_split": split,
                "archive_member": token_provenance.get("archive_member"),
                "archive_member_sha256": member_sha256,
                "corpus_window_start_offset": corpus_begin,
                "corpus_window_end_offset_exclusive": corpus_end,
                "corpus_label_start_offset": label_begin,
                "corpus_label_end_offset_exclusive": label_end,
                "model_predicted_position_start": context + 1,
                "model_predicted_position_end_exclusive": (context + decode_steps + 1),
            }
        )
    return records


def build_token_matrix(
    args: argparse.Namespace, vocab_size: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return request-major IDs with one true label beyond every decode input.

    WikiText is read directly from its ZIP without creating an untracked
    extracted corpus.  Each request is ``BOS + a disjoint corpus window``;
    therefore the first decode token is the real continuation of its prefix,
    and position ``context + decode_steps`` labels the final decode logit.
    """
    needed = args.context + args.decode_steps + 1
    if args.token_source == "random":
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        tokens = torch.randint(
            0,
            vocab_size,
            (args.batch_size, needed),
            generator=generator,
            dtype=torch.long,
        )
        provenance: dict[str, Any] = {
            "kind": "random",
            "role": "synthetic-token negative control; not the publication default",
            "generator": "torch.Generator(device='cpu')",
            "seed": args.seed,
            "range": [0, vocab_size],
            "includes_special_ids": True,
        }
    else:
        from transformers import AutoTokenizer

        if not args.wikitext_zip.is_file():
            raise ValueError(f"WikiText archive not found: {args.wikitext_zip}")
        archive_sha256 = sha256_file(args.wikitext_zip)
        if archive_sha256 != WIKITEXT2_RAW_V1_SHA256:
            raise ValueError(
                "WikiText archive SHA256 does not match the frozen raw-v1 "
                f"artifact: {archive_sha256}"
            )
        with zipfile.ZipFile(args.wikitext_zip, "r") as archive:
            try:
                member_info = archive.getinfo(args.wikitext_member)
            except KeyError as error:
                raise ValueError(
                    f"WikiText archive has no member {args.wikitext_member!r}"
                ) from error
            member_bytes = archive.read(member_info)
        text = member_bytes.decode("utf-8")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokenizer_manifest = tokenizer_artifact_provenance(tokenizer, args.model)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("the local model tokenizer has no BOS token")
        corpus_tokens_per_request = needed - 1
        stride = args.token_stride or corpus_tokens_per_request
        if stride < corpus_tokens_per_request:
            raise ValueError(
                "WikiText request windows must be disjoint: token stride must "
                f"be at least {corpus_tokens_per_request}"
            )
        starts = [
            args.token_offset + request * stride for request in range(args.batch_size)
        ]
        if starts[-1] + corpus_tokens_per_request > len(encoded):
            raise ValueError(
                "WikiText does not contain enough locally tokenized IDs for "
                f"B={args.batch_size}, length={corpus_tokens_per_request}, "
                f"offset={args.token_offset}, "
                f"stride={stride}; available={len(encoded)}"
            )
        tokens = torch.tensor(
            [
                [bos_token_id] + encoded[start : start + corpus_tokens_per_request]
                for start in starts
            ],
            dtype=torch.long,
        )
        provenance = {
            "kind": "wikitext2",
            "dataset": "WikiText-2 raw",
            "split": wikitext_split_from_member(args.wikitext_member),
            "archive_path": str(args.wikitext_zip.resolve()),
            "archive_sha256": archive_sha256,
            "archive_sha256_verified": True,
            "archive_member": args.wikitext_member,
            "archive_member_sha256": sha256_bytes(member_bytes),
            "archive_member_crc32": f"{member_info.CRC:08x}",
            "archive_member_uncompressed_bytes": member_info.file_size,
            "archive_member_compressed_bytes": member_info.compress_size,
            "tokenizer": tokenizer_manifest,
            "add_special_tokens": False,
            "bos_prepended_per_request": True,
            "bos_token_id": bos_token_id,
            "corpus_window_start_offsets": starts,
            "corpus_window_end_offsets_exclusive": [
                start + corpus_tokens_per_request for start in starts
            ],
            "corpus_window_stride": stride,
            "corpus_windows_disjoint": all(
                right >= left + corpus_tokens_per_request
                for left, right in zip(starts, starts[1:])
            ),
            "corpus_tokens_per_request": corpus_tokens_per_request,
            "available_corpus_token_count": len(encoded),
            "continuation_semantics": (
                "prefix is positions [0,context); decode position context and "
                "all later decode IDs are actual WikiText tokens; each decode "
                "logit at input position p is labeled by position p+1"
            ),
        }
    if tokens.shape != (args.batch_size, needed):
        raise RuntimeError("token provider returned the wrong shape")
    if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
        raise ValueError(
            f"token IDs must lie in [0,{vocab_size}); got "
            f"[{int(tokens.min())},{int(tokens.max())}]"
        )
    provenance.update(
        {
            "layout": "request-major [batch,context_plus_decode_plus_label]",
            "shape": list(tokens.shape),
            "token_ids_sha256": canonical_json_sha256(tokens.tolist()),
            "prefix_first_16_ids_by_request": tokens[:, :16].tolist(),
            "prefix_last_16_ids_by_request": tokens[
                :, args.context - 16 : args.context
            ].tolist(),
            "teacher_forced_decode_ids_step_major": tokens[
                :, args.context : args.context + args.decode_steps
            ].T.tolist(),
            "decode_label_ids_step_major": tokens[
                :, args.context + 1 : args.context + args.decode_steps + 1
            ].T.tolist(),
            "label_alignment": (
                "decode input at model position context+s predicts the true "
                "corpus token at model position context+s+1"
            ),
        }
    )
    return tokens, provenance


def allocate_baseline_cache(
    layers: int,
    pages: int,
    batch_size: int,
    hkv: int,
) -> PG.BaselineCache:
    shape = (layers, batch_size * pages, PG.PAGE, hkv, PG.DIM)
    key = torch.empty(shape, device="cuda", dtype=torch.float16)
    value = torch.empty_like(key)
    return PG.BaselineCache(key, value)


def cache_layer_tensors(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read both current and legacy Hugging Face Cache interfaces."""
    try:
        key, value = cache[layer]
    except (KeyError, TypeError, IndexError):
        cache_layers = getattr(cache, "layers", None)
        if cache_layers is None or layer >= len(cache_layers):
            raise RuntimeError(f"HF cache does not contain layer {layer}")
        key = cache_layers[layer].keys
        value = cache_layers[layer].values
    if key is None or value is None:
        raise RuntimeError(f"HF cache layer {layer} is uninitialized")
    return key, value


@torch.inference_mode()
def prefill_model_cache(
    model,
    tokens: torch.Tensor,
    baseline: PG.BaselineCache,
    pages_per_request: int,
    context: int,
    decode_steps: int,
    chunk_tokens: int,
) -> tuple[
    list[torch.Tensor],
    torch.Tensor,
    list[dict[str, Any]],
    str,
]:
    """Prefill B requests serially and return step-major HF logits on CPU."""
    from transformers.cache_utils import DynamicCache

    layers, _, hkv, _ = PG.BASE_E2E.check_model(model)
    batch_size = int(tokens.shape[0])
    hf_logits_by_request: list[list[torch.Tensor]] = []
    hf_prefix_next_logits_by_request: list[torch.Tensor] = []
    prefill_records: list[dict[str, Any]] = []
    sampled_fingerprint = hashlib.sha256()
    for request in range(batch_size):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        started = time.perf_counter()
        cache = DynamicCache()
        prefix_next_logits = None
        for begin in range(0, context, chunk_tokens):
            end = min(begin + chunk_tokens, context)
            input_ids = tokens[request, begin:end].to(
                device="cuda", non_blocking=False
            )[None]
            cache_position = torch.arange(begin, end, device="cuda", dtype=torch.long)
            outputs = model.model(
                input_ids=input_ids,
                position_ids=cache_position[None],
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            if end == context:
                # The final prefill hidden state is HF's oracle for the first
                # real continuation token, before any custom cache is used.
                prefix_next_logits = (
                    model.lm_head(outputs.last_hidden_state[:, -1]).float().cpu()
                )
            del input_ids, cache_position, outputs
        if prefix_next_logits is None:
            raise RuntimeError("prefill did not produce a transition logit")
        if int(cache.get_seq_length()) != context:
            raise RuntimeError(
                f"request {request} HF cache length={cache.get_seq_length()}, "
                f"expected {context}"
            )
        physical_begin = request * pages_per_request
        physical_end = physical_begin + context // PG.PAGE
        sampled_copy_bitwise = True
        for layer in range(layers):
            key, value = cache_layer_tensors(cache, layer)
            expected = (1, hkv, context, PG.DIM)
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise RuntimeError(
                    f"HF layer {layer} cache has K={tuple(key.shape)}, "
                    f"V={tuple(value.shape)}; expected {expected}. Context must "
                    "not exceed the checkpoint's sliding-window cache capacity."
                )
            destination_key = baseline.key[layer, physical_begin:physical_end].view(
                context, hkv, PG.DIM
            )
            destination_value = baseline.value[layer, physical_begin:physical_end].view(
                context, hkv, PG.DIM
            )
            source_key = key[0].transpose(0, 1)
            source_value = value[0].transpose(0, 1)
            destination_key.copy_(source_key)
            destination_value.copy_(source_value)
            for index in (0, context - 1):
                sampled_copy_bitwise = sampled_copy_bitwise and torch.equal(
                    destination_key[index], source_key[index]
                )
                sampled_copy_bitwise = sampled_copy_bitwise and torch.equal(
                    destination_value[index], source_value[index]
                )
            if layer in (0, layers - 1):
                sample = (
                    torch.cat(
                        (
                            source_key[0].reshape(-1)[:64],
                            source_key[-1].reshape(-1)[:64],
                            source_value[0].reshape(-1)[:64],
                            source_value[-1].reshape(-1)[:64],
                        )
                    )
                    .contiguous()
                    .cpu()
                )
                sampled_fingerprint.update(sample.numpy().tobytes())

        request_logits = []
        for step in range(decode_steps):
            position = context + step
            input_ids = tokens[request, position].reshape(1, 1).to("cuda")
            cache_position = torch.tensor([position], device="cuda", dtype=torch.long)
            outputs = model.model(
                input_ids=input_ids,
                position_ids=cache_position[None],
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits = model.lm_head(outputs.last_hidden_state[:, -1]).float()
            request_logits.append(logits.cpu())
            del input_ids, cache_position, logits, outputs
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1e3
        peak = int(torch.cuda.max_memory_allocated())
        prefill_records.append(
            {
                "request": request,
                "prefix_tokens": context,
                "reference_decode_tokens": decode_steps,
                "chunk_tokens": chunk_tokens,
                "chunks": math.ceil(context / chunk_tokens),
                "sampled_boundary_copy_bitwise_identical": sampled_copy_bitwise,
                "wall_ms_including_reference_decode": elapsed_ms,
                "allocated_before_bytes": allocated_before,
                "peak_allocated_bytes": peak,
                "incremental_peak_allocated_bytes": max(0, peak - allocated_before),
            }
        )
        hf_logits_by_request.append(request_logits)
        hf_prefix_next_logits_by_request.append(prefix_next_logits)
        del cache, request_logits, prefix_next_logits
        gc.collect()
        torch.cuda.empty_cache()
    hf_logits = [
        torch.cat(
            [hf_logits_by_request[request][step] for request in range(batch_size)],
            dim=0,
        )
        for step in range(decode_steps)
    ]
    return (
        hf_logits,
        torch.cat(hf_prefix_next_logits_by_request, dim=0),
        prefill_records,
        sampled_fingerprint.hexdigest(),
    )


@torch.inference_mode()
def exact_prefix_snapshot(
    cache: Any,
    *,
    exact_tail_pages: int,
    exact_prefix_pages: int,
    batch_size: int,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> dict[str, Any]:
    """Hash all immutable exact prefix/suffix slots in logical request order."""

    exact_prefix_pages = PG.validate_exact_prefix_pages(exact_prefix_pages)
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_prefix_pages,
        initial_context_pages=initial_context_pages,
    )
    if batch_size <= 0 or exact_tail_pages <= 0:
        raise ValueError("exact-prefix snapshot dimensions must be positive")
    storage_pages = (
        exact_tail_pages + exact_prefix_pages + exact_static_suffix_pages
    )
    if int(cache.exact_key.shape[1]) != batch_size * storage_pages:
        raise ValueError("exact-prefix snapshot found the wrong exact layout")
    fixed_logical_pages = tuple(range(exact_prefix_pages)) + tuple(
        range(
            int(initial_context_pages) - exact_static_suffix_pages,
            int(initial_context_pages),
        )
        if exact_static_suffix_pages
        else ()
    )
    mapping = [
        [
            PG.exact_physical_page_index(
                request,
                logical_page,
                exact_tail_pages,
                exact_prefix_pages,
                exact_static_suffix_pages,
                initial_context_pages,
            )
            for logical_page in fixed_logical_pages
        ]
        for request in range(batch_size)
    ]
    flat = [physical for request_pages in mapping for physical in request_pages]
    if flat:
        index = torch.tensor(flat, device=cache.exact_key.device, dtype=torch.long)
        key = cache.exact_key.index_select(1, index).detach().cpu().contiguous()
        value = cache.exact_value.index_select(1, index).detach().cpu().contiguous()
        key_sha256 = sha256_bytes(key.numpy().tobytes(order="C"))
        value_sha256 = sha256_bytes(value.numpy().tobytes(order="C"))
        byte_count = (
            key.numel() * key.element_size() + value.numel() * value.element_size()
        )
    else:
        key_sha256 = sha256_bytes(b"")
        value_sha256 = sha256_bytes(b"")
        byte_count = 0
    digest_payload = {
        "mapping": mapping,
        "key_sha256": key_sha256,
        "value_sha256": value_sha256,
        "exact_prefix_pages": exact_prefix_pages,
        "exact_static_suffix_pages": exact_static_suffix_pages,
        "initial_context_pages": initial_context_pages,
        "fixed_logical_pages": list(fixed_logical_pages),
        "exact_tail_pages": exact_tail_pages,
        "batch_size": batch_size,
    }
    return {
        **digest_payload,
        "combined_sha256": canonical_json_sha256(digest_payload),
        "bytes_hashed": int(byte_count),
    }


@torch.inference_mode()
def audit_exact_prefix_population(
    baseline: Any,
    cache: Any,
    *,
    pages_per_request: int,
    exact_tail_pages: int,
    exact_prefix_pages: int,
    batch_size: int,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> dict[str, Any]:
    """Prove that each fixed slot is the centered FP16 baseline prefix."""

    snapshot = exact_prefix_snapshot(
        cache,
        exact_tail_pages=exact_tail_pages,
        exact_prefix_pages=exact_prefix_pages,
        batch_size=batch_size,
        exact_static_suffix_pages=exact_static_suffix_pages,
        initial_context_pages=initial_context_pages,
    )
    fixed_logical_pages = tuple(snapshot["fixed_logical_pages"])
    if not fixed_logical_pages:
        return {"enabled": False, "passed": True, **snapshot}
    expected_key = []
    expected_value = []
    for request in range(batch_size):
        physical = torch.tensor(
            [
                request * pages_per_request + logical
                for logical in fixed_logical_pages
            ],
            device=baseline.key.device,
            dtype=torch.long,
        )
        key_center = cache.key_center[:, request][:, None, None]
        value_center = cache.value_center[:, request][:, None, None]
        expected_key.append(
            (
                baseline.key.index_select(1, physical).float()
                - key_center.float()
            ).half()
        )
        expected_value.append(
            (
                baseline.value.index_select(1, physical).float()
                - value_center.float()
            ).half()
        )
    expected_key_tensor = torch.cat(expected_key, dim=1)
    expected_value_tensor = torch.cat(expected_value, dim=1)
    flat = [physical for pages in snapshot["mapping"] for physical in pages]
    index = torch.tensor(flat, device=cache.exact_key.device, dtype=torch.long)
    observed_key = cache.exact_key.index_select(1, index)
    observed_value = cache.exact_value.index_select(1, index)
    key_identical = bool(torch.equal(expected_key_tensor, observed_key))
    value_identical = bool(torch.equal(expected_value_tensor, observed_value))
    expected_key_cpu = expected_key_tensor.detach().cpu().contiguous()
    expected_value_cpu = expected_value_tensor.detach().cpu().contiguous()
    expected_digest = canonical_json_sha256(
        {
            "mapping": snapshot["mapping"],
            "key_sha256": sha256_bytes(expected_key_cpu.numpy().tobytes(order="C")),
            "value_sha256": sha256_bytes(expected_value_cpu.numpy().tobytes(order="C")),
            "exact_prefix_pages": exact_prefix_pages,
            "exact_static_suffix_pages": exact_static_suffix_pages,
            "initial_context_pages": initial_context_pages,
            "fixed_logical_pages": list(fixed_logical_pages),
            "exact_tail_pages": exact_tail_pages,
            "batch_size": batch_size,
        }
    )
    passed = bool(
        key_identical
        and value_identical
        and expected_digest == snapshot["combined_sha256"]
    )
    result = {
        "enabled": True,
        "passed": passed,
        "source": "centered FP16 pages copied from the matched baseline cache",
        "key_bitwise_identical": key_identical,
        "value_bitwise_identical": value_identical,
        "expected_combined_sha256": expected_digest,
        "observed": snapshot,
    }
    if not passed:
        raise RuntimeError(f"exact-prefix population canary failed: {result}")
    return result


def audit_exact_prefix_immutability(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Compare fixed-prefix digests before and after the complete decode."""

    comparable = (
        "mapping",
        "key_sha256",
        "value_sha256",
        "combined_sha256",
        "bytes_hashed",
    )
    passed = all(before.get(name) == after.get(name) for name in comparable)
    result = {
        "passed": passed,
        "before": before,
        "after": after,
        "fixed_slots_unchanged": passed,
    }
    if not passed:
        raise RuntimeError(f"exact-prefix immutability canary failed: {result}")
    return result


@torch.inference_mode()
def audit_exact_prefix_page_tables(
    decoder: Any,
    *,
    exact_tail_pages: int,
    exact_prefix_pages: int,
    batch_size: int,
    exact_static_suffix_pages: int = 0,
    initial_context_pages: int | None = None,
) -> dict[str, Any]:
    """Prove that both production logical and active tables retain S slots."""

    exact_prefix_pages = PG.validate_exact_prefix_pages(exact_prefix_pages)
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_prefix_pages,
        initial_context_pages=initial_context_pages,
    )
    fixed_logical_pages = tuple(range(exact_prefix_pages)) + tuple(
        range(
            int(initial_context_pages) - exact_static_suffix_pages,
            int(initial_context_pages),
        )
        if exact_static_suffix_pages
        else ()
    )
    expected = [
        [
            PG.exact_physical_page_index(
                request,
                logical,
                exact_tail_pages,
                exact_prefix_pages,
                exact_static_suffix_pages,
                initial_context_pages,
            )
            for logical in fixed_logical_pages
        ]
        for request in range(batch_size)
    ]
    if not fixed_logical_pages:
        return {
            "enabled": False,
            "passed": True,
            "expected_prefix_physical_pages_by_request": expected,
        }
    logical_table = getattr(decoder, "exact_ring_pages", None)
    active_table = getattr(decoder, "exact_plan_pages", None)
    if logical_table is None or active_table is None:
        raise RuntimeError("S>0 decoder did not allocate both exact page tables")
    logical_index = torch.tensor(
        fixed_logical_pages, device=logical_table.device, dtype=torch.long
    )
    logical_observed = (
        logical_table.index_select(1, logical_index).detach().cpu().tolist()
    )
    fixed_pages = len(fixed_logical_pages)
    active_observed = active_table[:, :fixed_pages].detach().cpu().tolist()
    passed = logical_observed == expected and active_observed == expected
    result = {
        "enabled": True,
        "passed": passed,
        "source_api": (
            "request_major_exact_page_table + write_exact_prefix_tail_page_table"
        ),
        "expected_prefix_physical_pages_by_request": expected,
        "logical_to_exact_table_prefix": logical_observed,
        "active_exact_plan_table_prefix": active_observed,
        "exact_page_table_updates": int(decoder.exact_page_table_updates),
        "final_layout_signature": list(decoder.exact_layout_signature),
    }
    if not passed:
        raise RuntimeError(f"exact-prefix page-table canary failed: {result}")
    return result


@torch.inference_mode()
def build_gauge_cache_from_baseline(
    baseline: PG.BaselineCache,
    layers: int,
    pages: int,
    initial_pages: int,
    exact_pages: int,
    batch_size: int,
    hkv: int,
    exact_prefix_pages: int = 0,
    exact_static_suffix_pages: int = 0,
) -> PG.GaugeCache:
    """Apply the production PageGauge representation to model-derived pages."""
    exact_prefix_pages = PG.validate_exact_prefix_attention_path(
        exact_prefix_pages, "flashinfer_merge"
    )
    exact_static_suffix_pages = PG.validate_exact_static_suffix_pages(
        exact_static_suffix_pages,
        exact_prefix_pages=exact_prefix_pages,
        initial_context_pages=initial_pages,
    )
    if not 0 < exact_pages < initial_pages <= pages:
        raise ValueError("cache page counts must satisfy 0 < exact < initial <= total")
    if exact_prefix_pages + exact_static_suffix_pages >= initial_pages:
        raise ValueError("fixed exact pages must leave quantized old pages")
    shape = (layers, batch_size * pages, PG.PAGE, hkv, PG.DIM)
    key_codes = torch.empty(shape, device="cuda", dtype=torch.int8)
    value_codes = torch.empty_like(key_codes)
    key_scales = torch.empty(
        layers, batch_size * pages, hkv, device="cuda", dtype=torch.float16
    )
    value_scales = torch.empty_like(key_scales)
    key_center = torch.empty(
        layers, batch_size, hkv, PG.DIM, device="cuda", dtype=torch.float16
    )
    value_center = torch.empty_like(key_center)
    exact_shape = (
        layers,
        batch_size
        * (exact_prefix_pages + exact_static_suffix_pages + exact_pages),
        PG.PAGE,
        hkv,
        PG.DIM,
    )
    exact_key = torch.empty(exact_shape, device="cuda", dtype=torch.float16)
    exact_value = torch.empty_like(exact_key)

    for layer in range(layers):
        for request in range(batch_size):
            page_begin = request * pages
            prefix_end = page_begin + initial_pages
            prefix_key = baseline.key[layer, page_begin:prefix_end]
            prefix_value = baseline.value[layer, page_begin:prefix_end]
            key_center[layer, request].copy_(prefix_key.float().mean(dim=(0, 1)).half())
            value_center[layer, request].copy_(
                prefix_value.float().mean(dim=(0, 1)).half()
            )
            PG.OVERHEAD.quantize_completed_kv_pages(
                prefix_key,
                prefix_value,
                key_center[layer, request],
                value_center[layer, request],
                key_codes[layer, page_begin:prefix_end],
                value_codes[layer, page_begin:prefix_end],
                key_scales[layer, page_begin:prefix_end],
                value_scales[layer, page_begin:prefix_end],
            )
            exact_logical_pages = tuple(range(exact_prefix_pages)) + tuple(
                range(
                    initial_pages - exact_static_suffix_pages,
                    initial_pages,
                )
                if exact_static_suffix_pages
                else ()
            ) + tuple(range(initial_pages - exact_pages, initial_pages))
            exact_logical_pages = tuple(dict.fromkeys(exact_logical_pages))
            for logical_page in exact_logical_pages:
                source_page = page_begin + logical_page
                exact_page = PG.exact_physical_page_index(
                    request,
                    logical_page,
                    exact_pages,
                    exact_prefix_pages,
                    exact_static_suffix_pages,
                    initial_pages,
                )
                exact_key[layer, exact_page].copy_(
                    (
                        baseline.key[layer, source_page].float()
                        - key_center[layer, request][None].float()
                    ).half()
                )
                exact_value[layer, exact_page].copy_(
                    (
                        baseline.value[layer, source_page].float()
                        - value_center[layer, request][None].float()
                    ).half()
                )
    torch.cuda.synchronize()
    return PG.GaugeCache(
        exact_key,
        exact_value,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        key_center,
        value_center,
        exact_tail_pages=exact_pages,
        exact_sink_pages=exact_prefix_pages,
        exact_static_suffix_pages=exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )


@torch.inference_mode()
def initialize_future_page_canary(
    cache: Any,
    *,
    pages_per_request: int,
    initial_pages_per_request: int,
    batch_size: int,
) -> dict[str, Any]:
    """Poison not-yet-written compressed pages with impossible sentinels.

    The symmetric PageGauge encoder emits only [-127,127], so -128 is outside
    its representation.  Runtime finalization must also replace every NaN
    scale before a completed page can enter the old-cache attention segment.
    """
    if not 0 < initial_pages_per_request < pages_per_request:
        raise ValueError("future-page canary requires non-empty future capacity")
    if batch_size <= 0:
        raise ValueError("future-page canary batch size must be positive")
    expected_physical_pages = batch_size * pages_per_request
    tensors = (
        cache.key_codes,
        cache.value_codes,
        cache.key_scales,
        cache.value_scales,
    )
    if any(int(tensor.shape[1]) != expected_physical_pages for tensor in tensors):
        raise ValueError("future-page canary cache layout is not request-major")
    for request in range(batch_size):
        begin = request * pages_per_request + initial_pages_per_request
        end = (request + 1) * pages_per_request
        cache.key_codes[:, begin:end].fill_(FUTURE_PAGE_CODE_CANARY)
        cache.value_codes[:, begin:end].fill_(FUTURE_PAGE_CODE_CANARY)
        cache.key_scales[:, begin:end].fill_(float("nan"))
        cache.value_scales[:, begin:end].fill_(float("nan"))
    return {
        "initialized": True,
        "code_canary": FUTURE_PAGE_CODE_CANARY,
        "scale_canary": "NaN",
        "encoder_valid_code_range": [-127, 127],
        "future_pages_per_request": pages_per_request - initial_pages_per_request,
        "future_physical_pages": batch_size
        * (pages_per_request - initial_pages_per_request),
    }


def audit_runtime_page_finalization(
    cache: Any,
    *,
    pages_per_request: int,
    initial_pages_per_request: int,
    batch_size: int,
    layers: int,
    context: int,
    decode_steps: int,
    exact_tail: int,
    exact_prefix_pages: int = 0,
    exact_static_suffix_pages: int = 0,
) -> dict[str, Any]:
    """Assert and describe completed-page overwrite/finalization coverage."""
    if context % PG.PAGE or exact_tail % PG.PAGE:
        raise ValueError("runtime-finalization audit requires page alignment")
    if decode_steps <= 0:
        raise ValueError("runtime-finalization audit requires decode steps")
    completed_pages = decode_steps // PG.PAGE
    partial_tokens = decode_steps % PG.PAGE
    future_pages = pages_per_request - initial_pages_per_request
    expected_future_pages = math.ceil(decode_steps / PG.PAGE)
    if future_pages != expected_future_pages:
        raise ValueError(
            "future cache capacity does not match the decode window: "
            f"{future_pages} != {expected_future_pages}"
        )

    completed_indices = [
        request * pages_per_request + initial_pages_per_request + page
        for request in range(batch_size)
        for page in range(completed_pages)
    ]
    if completed_indices:
        index = torch.tensor(
            completed_indices,
            device=cache.key_codes.device,
            dtype=torch.long,
        )
        completed_key_codes = cache.key_codes.index_select(1, index)
        completed_value_codes = cache.value_codes.index_select(1, index)
        completed_key_scales = cache.key_scales.index_select(1, index)
        completed_value_scales = cache.value_scales.index_select(1, index)
        code_overwritten = bool(
            completed_key_codes.ne(FUTURE_PAGE_CODE_CANARY).all().item()
            and completed_value_codes.ne(FUTURE_PAGE_CODE_CANARY).all().item()
        )
        scales_valid = bool(
            torch.isfinite(completed_key_scales).all().item()
            and torch.isfinite(completed_value_scales).all().item()
            and completed_key_scales.gt(0).all().item()
            and completed_value_scales.gt(0).all().item()
        )
    else:
        code_overwritten = True
        scales_valid = True

    partial_page_retained_canary = True
    if partial_tokens:
        partial_indices = torch.tensor(
            [
                request * pages_per_request
                + initial_pages_per_request
                + completed_pages
                for request in range(batch_size)
            ],
            device=cache.key_codes.device,
            dtype=torch.long,
        )
        partial_page_retained_canary = bool(
            cache.key_codes.index_select(1, partial_indices)
            .eq(FUTURE_PAGE_CODE_CANARY)
            .all()
            .item()
            and cache.value_codes.index_select(1, partial_indices)
            .eq(FUTURE_PAGE_CODE_CANARY)
            .all()
            .item()
            and torch.isnan(cache.key_scales.index_select(1, partial_indices))
            .all()
            .item()
            and torch.isnan(cache.value_scales.index_select(1, partial_indices))
            .all()
            .item()
        )

    exact_prefix_pages = PG.validate_exact_prefix_pages(exact_prefix_pages)
    endpoint_tokens = context + decode_steps
    endpoint_pages = math.ceil(endpoint_tokens / PG.PAGE)
    endpoint_last_page_len = endpoint_tokens % PG.PAGE or PG.PAGE
    endpoint_partition = PG.page_gauge_logical_partition(
        endpoint_pages,
        exact_tail // PG.PAGE,
        exact_prefix_pages,
        endpoint_last_page_len,
        exact_static_suffix_pages,
        initial_pages_per_request,
    )
    generated_completed_begin = initial_pages_per_request
    generated_completed_end = initial_pages_per_request + completed_pages
    endpoint_old_begin = exact_prefix_pages
    endpoint_old_end = int(endpoint_partition["tail_logical_begin"])
    generated_int8_begin = max(generated_completed_begin, endpoint_old_begin)
    generated_int8_end = min(generated_completed_end, endpoint_old_end)
    runtime_pages_in_old = max(0, generated_int8_end - generated_int8_begin)
    passed = bool(code_overwritten and scales_valid and partial_page_retained_canary)
    result = {
        "passed": passed,
        "canary_initialized": True,
        "code_canary": FUTURE_PAGE_CODE_CANARY,
        "scale_canary": "NaN",
        "completed_pages_per_request": completed_pages,
        "partial_page_tokens": partial_tokens,
        "future_capacity_pages_per_request": future_pages,
        "all_completed_code_slots_overwritten": code_overwritten,
        "all_completed_scales_finite_positive": scales_valid,
        "partial_unfinalized_page_retained_canary": (partial_page_retained_canary),
        "runtime_finalized_logical_pages_total": (batch_size * completed_pages),
        "runtime_finalized_layer_pages_total": (layers * batch_size * completed_pages),
        "first_decode_step_consuming_runtime_finalized_old_page": (
            exact_tail if decode_steps > exact_tail else None
        ),
        "decode_steps_consuming_runtime_finalized_old_pages": max(
            0, decode_steps - exact_tail
        ),
        "request_token_outputs_consuming_runtime_finalized_old_pages": (
            batch_size * max(0, decode_steps - exact_tail)
        ),
        "runtime_finalized_pages_in_old_segment_at_endpoint": (runtime_pages_in_old),
        "runtime_generated_int8_logical_page_range": [
            generated_int8_begin,
            generated_int8_end,
        ],
        "runtime_generated_int8_logical_page_count": runtime_pages_in_old,
        "generated_completed_logical_page_range": [
            generated_completed_begin,
            generated_completed_end,
        ],
        "endpoint_prefix_logical_page_range": [0, exact_prefix_pages],
        "endpoint_static_suffix_logical_page_range": (
            [
                initial_pages_per_request - exact_static_suffix_pages,
                initial_pages_per_request,
            ]
            if exact_static_suffix_pages
            else None
        ),
        "endpoint_old_logical_page_range": [
            endpoint_old_begin,
            endpoint_old_end,
        ],
        "endpoint_tail_logical_page_range": [
            endpoint_old_end,
            endpoint_pages,
        ],
        "endpoint_prefix_pages_per_request": exact_prefix_pages,
        "endpoint_static_suffix_pages_per_request": exact_static_suffix_pages,
        "endpoint_old_pages_per_request": int(endpoint_partition["old_page_count"]),
        "endpoint_tail_pages_per_request": exact_tail // PG.PAGE,
        "endpoint_partition_coverage_disjoint": bool(
            endpoint_partition["coverage_disjoint"]
        ),
        "recurrence": (
            "at endpoint, generated completed logical pages intersect the "
            "production prefix/old/tail partition; the intersection with old "
            "is stored as generated INT8 pages"
        ),
        "coverage_note": (
            "completed future pages are proven overwritten by an impossible "
            "INT8-code/NaN-scale canary; after exact-tail displacement, later "
            "teacher-forced logits consume those pages through old-cache attention"
        ),
    }
    if not passed:
        raise RuntimeError(f"runtime page-finalization canary failed: {result}")
    return result


@torch.inference_mode()
def collect_decoder_logits_on_cpu(
    decoder: Any, tokens: Any, start_position: int
) -> list[torch.Tensor]:
    """Run correctness decode while immediately releasing GPU logits.

    This helper intentionally makes no latency claim.  Moving each full-vocab
    result to CPU prevents a D512 validation from retaining two roughly
    256-MiB GPU logit streams beside the model and both long-context caches.
    """
    token_tensor = PG.token_tensor_from_sequence(tokens, decoder.batch_size)
    outputs: list[torch.Tensor] = []
    for index in range(int(token_tensor.shape[0])):
        logits = decoder.step(token_tensor[index], start_position + index)[0]
        outputs.append(logits.detach().to(device="cpu", non_blocking=False))
        del logits
    return outputs


def aggregate_distribution_records(
    records: list[dict[str, Any]],
    request_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce scalar token records into exact request/bootstrap summaries."""
    if not records:
        raise ValueError("cannot aggregate an empty metric stream")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        reference_nll = [row["reference_nll_nats"] for row in rows]
        candidate_nll = [row["candidate_nll_nats"] for row in rows]
        nll_delta = [row["nll_delta_nats"] for row in rows]
        forward_kl = [row["forward_kl_nats"] for row in rows]
        js = [row["jensen_shannon_nats"] for row in rows]
        tv = [row["total_variation"] for row in rows]
        reference_rank = [float(row["reference_true_token_rank"]) for row in rows]
        candidate_rank = [float(row["candidate_true_token_rank"]) for row in rows]
        top1 = [bool(row["top1_agreement"]) for row in rows]
        reference_top1_true = [
            bool(row["reference_true_token_rank"] == 1) for row in rows
        ]
        candidate_top1_true = [
            bool(row["candidate_true_token_rank"] == 1) for row in rows
        ]
        reference_nll_sum = float(sum(reference_nll))
        candidate_nll_sum = float(sum(candidate_nll))
        mean_reference_nll = reference_nll_sum / count
        mean_candidate_nll = candidate_nll_sum / count
        return {
            "token_count": count,
            "raw_sufficient_statistics": {
                "reference_nll_sum_nats": reference_nll_sum,
                "candidate_nll_sum_nats": candidate_nll_sum,
                "nll_delta_sum_nats": float(sum(nll_delta)),
                "forward_kl_sum_nats": float(sum(forward_kl)),
                "jensen_shannon_sum_nats": float(sum(js)),
                "total_variation_sum": float(sum(tv)),
                "top1_agreement_count": int(sum(top1)),
                "reference_true_token_top1_count": int(sum(reference_top1_true)),
                "candidate_true_token_top1_count": int(sum(candidate_top1_true)),
            },
            "reference_mean_nll_nats": mean_reference_nll,
            "candidate_mean_nll_nats": mean_candidate_nll,
            "mean_nll_delta_nats": mean_candidate_nll - mean_reference_nll,
            "reference_perplexity": finite_exp(mean_reference_nll),
            "candidate_perplexity": finite_exp(mean_candidate_nll),
            "candidate_to_reference_perplexity_ratio": finite_exp(
                mean_candidate_nll - mean_reference_nll
            ),
            "candidate_minus_reference_perplexity": (
                finite_exp(mean_candidate_nll) - finite_exp(mean_reference_nll)
            ),
            "nll_delta_nats": scalar_summary(nll_delta),
            "forward_kl_nats": scalar_summary(forward_kl),
            "jensen_shannon_nats": scalar_summary(js),
            "total_variation": scalar_summary(tv),
            "reference_true_token_rank": scalar_summary(reference_rank),
            "candidate_true_token_rank": scalar_summary(candidate_rank),
            "top1_agreement_fraction": float(sum(top1) / count),
            "reference_true_token_top1_accuracy": float(
                sum(reference_top1_true) / count
            ),
            "candidate_true_token_top1_accuracy": float(
                sum(candidate_top1_true) / count
            ),
        }

    request_aggregates = []
    for window in request_windows:
        request = int(window["request"])
        rows = [row for row in records if int(row["request"]) == request]
        if not rows:
            raise ValueError(f"metric stream has no rows for request {request}")
        request_aggregates.append({**window, **aggregate(rows)})
    return {
        "overall": aggregate(records),
        "cluster_bootstrap_unit": "request/corpus window",
        "cluster_bootstrap_units": request_aggregates,
    }


def compare_logits(
    reference: list[torch.Tensor],
    observed: list[torch.Tensor],
    true_token_ids: torch.Tensor | None = None,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    predicted_position_start: int | None = None,
    request_windows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare logits one step at a time and retain scalar metrics only.

    Probability vectors are materialized only for the current decode step and
    immediately discarded.  The returned schema contains token scalars and
    request-level sufficient statistics, never logits/probabilities over the
    vocabulary, so its size is O(batch * decode_steps), not O(vocabulary).
    """
    if not reference or len(reference) != len(observed):
        raise ValueError("logit sequences must have the same non-zero length")
    if true_token_ids is not None:
        true_token_ids = true_token_ids.detach().long().cpu()
        if true_token_ids.dim() != 2:
            raise ValueError("true token IDs must have shape [steps,batch]")
        if int(true_token_ids.shape[0]) != len(reference):
            raise ValueError("true token step count does not match logits")
    cosine_by_step_request: list[list[float]] = []
    top1_by_step_request: list[list[bool]] = []
    maximum_absolute_by_step_request: list[list[float]] = []
    relative_l2_by_step_request: list[list[float]] = []
    distribution_records: list[dict[str, Any]] = []
    batch_size = None
    for step, (expected, actual) in enumerate(zip(reference, observed)):
        expected_cpu = expected.detach().to(device="cpu", dtype=torch.float64)
        actual_cpu = actual.detach().to(device="cpu", dtype=torch.float64)
        if expected_cpu.shape != actual_cpu.shape:
            raise ValueError(
                f"logit shape mismatch: {expected_cpu.shape} != {actual_cpu.shape}"
            )
        if expected_cpu.dim() != 2:
            raise ValueError("each logit tensor must have shape [batch,vocabulary]")
        if (
            not torch.isfinite(expected_cpu).all()
            or not torch.isfinite(actual_cpu).all()
        ):
            raise ValueError(f"non-finite logits at decode step {step}")
        if batch_size is None:
            batch_size = int(expected_cpu.shape[0])
        elif int(expected_cpu.shape[0]) != batch_size:
            raise ValueError("batch size changed within the logit stream")
        delta = expected_cpu - actual_cpu
        cosine_by_step_request.append(
            [
                float(value)
                for value in F.cosine_similarity(
                    expected_cpu, actual_cpu, dim=-1
                ).tolist()
            ]
        )

        if true_token_ids is not None:
            if int(true_token_ids.shape[1]) != batch_size:
                raise ValueError("true token batch size does not match logits")
            labels = true_token_ids[step]
            vocabulary = int(expected_cpu.shape[1])
            if bool(((labels < 0) | (labels >= vocabulary)).any()):
                raise ValueError(f"true token ID outside vocabulary at step {step}")

            reference_log_probability = F.log_softmax(expected_cpu, dim=-1)
            candidate_log_probability = F.log_softmax(actual_cpu, dim=-1)
            reference_probability = reference_log_probability.exp()
            candidate_probability = candidate_log_probability.exp()
            mixture_log_probability = torch.logaddexp(
                reference_log_probability, candidate_log_probability
            ) - math.log(2.0)
            reference_nll = -reference_log_probability.gather(1, labels[:, None])[:, 0]
            candidate_nll = -candidate_log_probability.gather(1, labels[:, None])[:, 0]
            forward_kl = (
                (
                    reference_probability
                    * (reference_log_probability - candidate_log_probability)
                )
                .sum(dim=-1)
                .clamp_min(0.0)
            )
            js = 0.5 * (
                (
                    reference_probability
                    * (reference_log_probability - mixture_log_probability)
                ).sum(dim=-1)
                + (
                    candidate_probability
                    * (candidate_log_probability - mixture_log_probability)
                ).sum(dim=-1)
            )
            js = js.clamp_min(0.0)
            tv = 0.5 * (reference_probability - candidate_probability).abs().sum(dim=-1)
            identical_rows = expected_cpu.eq(actual_cpu).all(dim=-1)
            forward_kl = torch.where(
                identical_rows, torch.zeros_like(forward_kl), forward_kl
            )
            js = torch.where(identical_rows, torch.zeros_like(js), js)
            tv = torch.where(identical_rows, torch.zeros_like(tv), tv)
            reference_true_logits = expected_cpu.gather(1, labels[:, None])
            candidate_true_logits = actual_cpu.gather(1, labels[:, None])
            reference_rank = (expected_cpu > reference_true_logits).sum(dim=-1) + 1
            candidate_rank = (actual_cpu > candidate_true_logits).sum(dim=-1) + 1
            reference_top1 = expected_cpu.argmax(dim=-1)
            candidate_top1 = actual_cpu.argmax(dim=-1)
            for request in range(batch_size):
                nll_delta = float(candidate_nll[request] - reference_nll[request])
                distribution_records.append(
                    {
                        "step": step,
                        "request": request,
                        "input_position": (
                            None
                            if predicted_position_start is None
                            else predicted_position_start - 1 + step
                        ),
                        "predicted_position": (
                            None
                            if predicted_position_start is None
                            else predicted_position_start + step
                        ),
                        "true_token_id": int(labels[request]),
                        "reference_nll_nats": float(reference_nll[request]),
                        "candidate_nll_nats": float(candidate_nll[request]),
                        "nll_delta_nats": nll_delta,
                        "candidate_to_reference_token_perplexity_ratio": (
                            finite_exp(nll_delta)
                        ),
                        "forward_kl_nats": float(forward_kl[request]),
                        "jensen_shannon_nats": float(js[request]),
                        "total_variation": float(tv[request]),
                        "reference_true_token_rank": int(reference_rank[request]),
                        "candidate_true_token_rank": int(candidate_rank[request]),
                        "reference_top1_token_id": int(reference_top1[request]),
                        "candidate_top1_token_id": int(candidate_top1[request]),
                        "top1_agreement": bool(
                            reference_top1[request] == candidate_top1[request]
                        ),
                    }
                )
            del (
                reference_log_probability,
                candidate_log_probability,
                reference_probability,
                candidate_probability,
                mixture_log_probability,
            )
        top1_by_step_request.append(
            [
                bool(value)
                for value in expected_cpu.argmax(dim=-1)
                .eq(actual_cpu.argmax(dim=-1))
                .tolist()
            ]
        )
        maximum_absolute_by_step_request.append(
            [float(value) for value in delta.abs().amax(dim=-1).tolist()]
        )
        relative_l2_by_step_request.append(
            [
                float(value)
                for value in (
                    delta.norm(dim=-1) / expected_cpu.norm(dim=-1).clamp_min(1e-12)
                ).tolist()
            ]
        )
    flat_cosine = [value for row in cosine_by_step_request for value in row]
    flat_top1 = [value for row in top1_by_step_request for value in row]
    flat_maximum = [value for row in maximum_absolute_by_step_request for value in row]
    flat_relative = [value for row in relative_l2_by_step_request for value in row]
    worst_flat_index = min(range(len(flat_cosine)), key=flat_cosine.__getitem__)
    assert batch_size is not None
    mismatches = [
        {"step": step, "request": request}
        for step, row in enumerate(top1_by_step_request)
        for request, agrees in enumerate(row)
        if not agrees
    ]
    result = {
        "reference": reference_name,
        "candidate": candidate_name,
        "checked_decode_steps": len(reference),
        "checked_request_steps": len(flat_cosine),
        "minimum_logits_cosine": min(flat_cosine),
        "logits_cosine_by_step_request": cosine_by_step_request,
        "worst_logits_cosine_location": {
            "step": worst_flat_index // batch_size,
            "request": worst_flat_index % batch_size,
            "value": flat_cosine[worst_flat_index],
        },
        "top1_agreement_fraction": sum(flat_top1) / len(flat_top1),
        "top1_agreement_by_step_request": top1_by_step_request,
        "top1_mismatch_locations": mismatches,
        "maximum_logits_absolute_error": max(flat_maximum),
        "maximum_logits_absolute_error_by_step_request": (
            maximum_absolute_by_step_request
        ),
        "maximum_relative_logits_l2": max(flat_relative),
        "relative_logits_l2_by_step_request": relative_l2_by_step_request,
    }
    if true_token_ids is not None:
        if request_windows is None:
            request_windows = [
                {
                    "request": request,
                    "cluster_unit_id": f"request-{request}",
                    "dataset_split": None,
                    "corpus_label_start_offset": None,
                    "corpus_label_end_offset_exclusive": None,
                    "model_predicted_position_start": predicted_position_start,
                    "model_predicted_position_end_exclusive": (
                        None
                        if predicted_position_start is None
                        else predicted_position_start + len(reference)
                    ),
                }
                for request in range(batch_size)
            ]
        if len(request_windows) != batch_size or {
            int(window["request"]) for window in request_windows
        } != set(range(batch_size)):
            raise ValueError(
                "request-window metadata must contain each batch request exactly once"
            )
        result["distribution_quality"] = {
            "units": {
                "nll": "natural-log nats/token",
                "perplexity": "exp(mean NLL)",
                "forward_kl": "KL(reference || candidate), natural-log nats",
                "jensen_shannon": "natural-log nats",
                "total_variation": "half L1 distance in [0,1]",
                "true_token_rank": (
                    "one-based strict rank: 1 + count(logit > true-token logit)"
                ),
            },
            "label_count": len(distribution_records),
            "serialized_full_vocabulary_logits_or_probabilities": False,
            "per_token_metrics_step_major": distribution_records,
            **aggregate_distribution_records(distribution_records, request_windows),
            "cluster_bootstrap_note": (
                "resample cluster_bootstrap_units with replacement and combine "
                "their raw_sufficient_statistics; do not resample correlated "
                "tokens as independent observations"
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    try:
        heldout_expected_starts = validate_heldout_command(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    source_sha256_at_start = model_prefill_source_manifest()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0 or args.decode_steps <= 0:
        raise SystemExit("batch size and decode steps must be positive")
    if args.prefill_chunk_tokens <= 0:
        raise SystemExit("prefill chunk size must be positive")
    if args.token_offset < 0 or args.token_stride < 0:
        raise SystemExit("token offset and stride cannot be negative")
    if (
        args.context <= args.exact_tail
        or args.context % PG.PAGE
        or args.exact_tail % PG.PAGE
    ):
        raise SystemExit(
            "context and exact tail must be page aligned; context must be larger"
        )
    try:
        PG.validate_exact_prefix_attention_path(
            args.exact_prefix_pages, args.tail_attention
        )
        PG.validate_exact_static_suffix_pages(
            args.exact_static_suffix_pages,
            exact_prefix_pages=args.exact_prefix_pages,
            initial_context_pages=args.context // PG.PAGE,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.exact_static_suffix_pages and args.tail_attention != "flashinfer_merge":
        raise SystemExit("exact static suffix pages require flashinfer_merge")
    if (
        args.exact_prefix_pages + args.exact_static_suffix_pages
        >= args.context // PG.PAGE
    ):
        raise SystemExit("fixed exact pages must leave quantized old pages")
    if args.decode_steps != PG.PAGE and not args.disable_attention_cuda_graphs:
        raise SystemExit(
            "production attention graph validation requires exactly one 16-token page"
        )
    if not (0.0 <= args.min_top1_agreement <= 1.0):
        raise SystemExit("top-1 threshold must lie in [0,1]")
    if args.run_token_step_graphs:
        if args.decode_steps != PG.PAGE:
            raise SystemExit(
                "token-step graph timing requires exactly one 16-token page"
            )
        if args.token_graph_capture_warmups <= 0:
            raise SystemExit("token graph capture warmups must be positive")
        if args.token_graph_warmups < 0:
            raise SystemExit("token graph warmups cannot be negative")
        if args.token_graph_repeats <= 0 or args.token_graph_repeats % 2:
            raise SystemExit("token graph repeats must be a positive even number")
        if args.token_graph_cache_scrub_mib <= 0:
            raise SystemExit("token graph cache scrub size must be positive")
    if args.run_direct_feedback_paths:
        if args.decode_steps != PG.PAGE:
            raise SystemExit(
                "direct generated-feedback timing requires exactly 16 steps"
            )
        if args.direct_feedback_warmups < 0:
            raise SystemExit("direct feedback warmups cannot be negative")
        if args.direct_feedback_repeats <= 0 or args.direct_feedback_repeats % 2:
            raise SystemExit("direct feedback repeats must be a positive even number")
        if args.direct_feedback_cache_scrub_mib <= 0:
            raise SystemExit("direct feedback cache scrub size must be positive")
    if args.run_static_teacher_control:
        if args.decode_steps != PG.PAGE:
            raise SystemExit(
                "static teacher control requires exactly one 16-token page"
            )
        if args.static_teacher_warmups < 0:
            raise SystemExit("static teacher warmups cannot be negative")
        if args.static_teacher_repeats <= 0 or args.static_teacher_repeats % 2:
            raise SystemExit("static teacher repeats must be a positive even number")
        if (
            args.static_teacher_cache_scrub_mib <= 0
            or args.static_teacher_component_repeats <= 0
        ):
            raise SystemExit(
                "static teacher scrub size and component repeats must be positive"
            )
        if (
            args.static_teacher_terminal_pg_only_residency
            and not args.static_teacher_skip_matrix_timing
        ):
            raise SystemExit(
                "terminal PG-only residency requires "
                "--static-teacher-skip-matrix-timing"
            )
        if args.static_teacher_terminal_pg_only_residency and (
            args.run_direct_feedback_paths
            or args.run_token_step_graphs
            or args.run_generated_sequence_graph
        ):
            raise SystemExit(
                "terminal PG-only residency cannot precede another nested diagnostic"
            )
        if (
            args.static_teacher_terminal_relocate_pg_cache
            and not args.static_teacher_terminal_pg_only_residency
        ):
            raise SystemExit(
                "terminal PG relocation requires terminal PG-only residency"
            )
    elif (
        args.static_teacher_skip_matrix_timing
        or args.static_teacher_terminal_pg_only_residency
        or args.static_teacher_terminal_relocate_pg_cache
    ):
        raise SystemExit(
            "static teacher control modifiers require --run-static-teacher-control"
        )
    if args.run_generated_sequence_graph:
        if args.context != 20480 or args.decode_steps != PG.PAGE:
            raise SystemExit(
                "generated-sequence graph timing requires context=20480 and "
                "decode-steps=16"
            )
        if args.generated_graph_capture_warmups <= 0:
            raise SystemExit("generated graph capture warmups must be positive")
        if args.generated_graph_warmups < 0:
            raise SystemExit("generated graph warmups cannot be negative")
        if args.generated_graph_repeats <= 0 or args.generated_graph_repeats % 2:
            raise SystemExit("generated graph repeats must be a positive even number")
        if args.generated_graph_cache_scrub_mib <= 0:
            raise SystemExit("generated graph cache scrub size must be positive")

    major, minor = torch.cuda.get_device_capability()
    heldout_memory_gate: dict[str, Any] = {
        "enabled": False,
        "passed": True,
    }
    if heldout_expected_starts is not None:
        cuda_free_bytes, _ = torch.cuda.mem_get_info()
        heldout_memory_gate = {
            "enabled": True,
            **heldout_co_resident_memory_gate(
                batch_size=args.batch_size,
                context=args.context,
                decode_steps=args.decode_steps,
                exact_tail=args.exact_tail,
                exact_prefix_pages=args.exact_prefix_pages,
                exact_static_suffix_pages=args.exact_static_suffix_pages,
                free_bytes=int(cuda_free_bytes),
            ),
        }
        if not heldout_memory_gate["passed"]:
            raise SystemExit(
                f"held-out co-resident memory gate failed: {heldout_memory_gate}"
            )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading the local checkpoint as the HF SDPA FP16 reference...", flush=True)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    layers, hq, hkv, hidden = PG.BASE_E2E.check_model(model)
    if (hq, hkv) != (32, 8):
        raise RuntimeError(
            f"diagnostic currently requires Mistral Hq/Hkv=32/8; got {hq}/{hkv}"
        )
    config_payload = model.config.to_dict()
    model_config_sha256 = canonical_json_sha256(config_payload)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    vocab_size = int(model.config.vocab_size)
    validate_heldout_model_identity(
        model_revision=getattr(model.config, "_commit_hash", None),
        model_config_sha256=model_config_sha256,
        parameter_count=parameter_count,
        expected_starts=heldout_expected_starts,
    )
    tokens, token_provenance = build_token_matrix(args, vocab_size)
    validate_heldout_model_and_tokens(
        model_revision=getattr(model.config, "_commit_hash", None),
        model_config_sha256=model_config_sha256,
        parameter_count=parameter_count,
        token_provenance=token_provenance,
        expected_starts=heldout_expected_starts,
    )
    decode_labels = tokens[
        :, args.context + 1 : args.context + args.decode_steps + 1
    ].T.contiguous()
    quality_windows = request_window_metadata(
        token_provenance,
        args.context,
        args.decode_steps,
        args.batch_size,
    )
    model = model.cuda()

    max_context = args.context + args.decode_steps
    pages = math.ceil(max_context / PG.PAGE)
    initial_pages = args.context // PG.PAGE
    exact_pages = args.exact_tail // PG.PAGE
    baseline_cache = allocate_baseline_cache(layers, pages, args.batch_size, hkv)
    print(
        f"Generating {args.batch_size} coherent B1 prefixes with chunked HF SDPA...",
        flush=True,
    )
    (
        hf_logits,
        hf_prefix_next_logits,
        prefill_records,
        kv_sample_sha256,
    ) = prefill_model_cache(
        model,
        tokens,
        baseline_cache,
        pages,
        args.context,
        args.decode_steps,
        args.prefill_chunk_tokens,
    )
    if not all(
        record["sampled_boundary_copy_bitwise_identical"] for record in prefill_records
    ):
        raise RuntimeError("HF-to-paged cache boundary copy was not bitwise exact")

    print(
        "Packing common projections after the HF reference is complete...", flush=True
    )
    PG.BASE_E2E.pack_model_projections(model)
    print("Quantizing the same model-derived prefix into PageGauge...", flush=True)
    gauge_cache = build_gauge_cache_from_baseline(
        baseline_cache,
        layers,
        pages,
        initial_pages,
        exact_pages,
        args.batch_size,
        hkv,
        args.exact_prefix_pages,
        args.exact_static_suffix_pages,
    )
    exact_prefix_population = audit_exact_prefix_population(
        baseline_cache,
        gauge_cache,
        pages_per_request=pages,
        exact_tail_pages=exact_pages,
        exact_prefix_pages=args.exact_prefix_pages,
        batch_size=args.batch_size,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )
    exact_prefix_before_decode = exact_prefix_snapshot(
        gauge_cache,
        exact_tail_pages=exact_pages,
        exact_prefix_pages=args.exact_prefix_pages,
        batch_size=args.batch_size,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )
    future_page_canary = initialize_future_page_canary(
        gauge_cache,
        pages_per_request=pages,
        initial_pages_per_request=initial_pages,
        batch_size=args.batch_size,
    )
    storage = PG.cache_storage(baseline_cache, gauge_cache)

    import flashinfer

    append_extension = PG.RUNTIME.load_append_extension()
    with torch.inference_mode():
        position_ids = torch.arange(max_context, device="cuda", dtype=torch.long)[None]
        rope_probe = torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16)
        rope_cos, rope_sin = model.model.rotary_emb(rope_probe, position_ids)
        if rope_cos.dim() == 3:
            rope_cos, rope_sin = rope_cos[0], rope_sin[0]
        rope_cos = rope_cos.to(dtype=torch.float16).contiguous()
        rope_sin = rope_sin.to(dtype=torch.float16).contiguous()

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
        "attention_add",
        tail_attention=args.tail_attention,
        batch_size=args.batch_size,
        old_value_scale_placement=args.old_value_scale_placement,
        exact_sink_pages=0,
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
        "attention_add",
        tail_attention=args.tail_attention,
        batch_size=args.batch_size,
        old_value_scale_placement=args.old_value_scale_placement,
        exact_sink_pages=args.exact_prefix_pages,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )
    decoders = {
        "flashinfer_fp16": baseline,
        "page_gauge": candidate,
    }
    token_step_graph_module = None
    generated_sequence_graph_module = None
    if (
        args.run_token_step_graphs
        or args.run_direct_feedback_paths
        or args.run_static_teacher_control
    ):
        token_step_graph_module = load_local_module(
            "page_gauge_token_step_graphs_for_model_prefill",
            TOKEN_STEP_GRAPH_PATH,
        )
    if args.run_generated_sequence_graph:
        generated_sequence_graph_module = load_local_module(
            "page_gauge_generated_sequence_graph_for_model_prefill",
            GENERATED_SEQUENCE_GRAPH_PATH,
        )
    graph_initial_caches = None
    if (
        args.run_token_step_graphs
        or args.run_direct_feedback_paths
        or args.run_static_teacher_control
        or args.run_generated_sequence_graph
    ):
        # Capture the untouched, model-derived destination page before any of
        # the correctness paths append into it. Only this one page is cloned;
        # the model and both full caches remain live and are never serialized.
        snapshot_module = (
            token_step_graph_module
            if token_step_graph_module is not None
            else generated_sequence_graph_module
        )
        assert snapshot_module is not None
        graph_initial_caches = snapshot_module.snapshot_fixture_caches(
            decoders, args.context
        )
    decode_tokens = tokens[
        :, args.context : args.context + args.decode_steps
    ].T.tolist()

    print("Running eager model-derived cache correctness...", flush=True)
    eager_baseline = collect_decoder_logits_on_cpu(
        baseline, decode_tokens, args.context
    )
    eager_candidate = collect_decoder_logits_on_cpu(
        candidate, decode_tokens, args.context
    )
    graph_check: dict[str, Any] = {
        "enabled": not args.disable_attention_cuda_graphs,
        "scope": "matched complete per-layer attention paths",
        "passed": True,
    }
    if args.disable_attention_cuda_graphs:
        baseline_logits = eager_baseline
        candidate_logits = eager_candidate
    else:
        print(
            "Capturing and replaying matched production attention graphs...", flush=True
        )
        baseline.capture_attention_graphs(args.context + 1)
        candidate.capture_attention_graphs(args.context + 1)
        baseline.reset_attention_dispatch_counts()
        candidate.reset_attention_dispatch_counts()
        baseline_logits = collect_decoder_logits_on_cpu(
            baseline, decode_tokens, args.context
        )
        candidate_logits = collect_decoder_logits_on_cpu(
            candidate, decode_tokens, args.context
        )
        expected_replays = layers * args.decode_steps
        dispatch = {
            "flashinfer_fp16": baseline.attention_dispatch_counts(),
            "page_gauge": candidate.attention_dispatch_counts(),
        }
        graph_check.update(
            {
                "eager_vs_graph": {
                    "flashinfer_fp16": PG.compare_logit_sequences(
                        eager_baseline, baseline_logits
                    ),
                    "page_gauge": PG.compare_logit_sequences(
                        eager_candidate, candidate_logits
                    ),
                },
                "dispatch": dispatch,
                "expected_graph_replays_per_backend": expected_replays,
            }
        )
        graph_check["passed"] = all(
            comparison["bitwise_identical"]
            for comparison in graph_check["eager_vs_graph"].values()
        ) and all(
            counts["graph_replays"] == expected_replays and counts["eager_calls"] == 0
            for counts in dispatch.values()
        )

    exact_prefix_after_decode = exact_prefix_snapshot(
        gauge_cache,
        exact_tail_pages=exact_pages,
        exact_prefix_pages=args.exact_prefix_pages,
        batch_size=args.batch_size,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )
    exact_prefix_immutability = audit_exact_prefix_immutability(
        exact_prefix_before_decode, exact_prefix_after_decode
    )
    exact_prefix_page_tables = audit_exact_prefix_page_tables(
        candidate,
        exact_tail_pages=exact_pages,
        exact_prefix_pages=args.exact_prefix_pages,
        batch_size=args.batch_size,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
        initial_context_pages=initial_pages,
    )
    exact_prefix_canary = {
        "passed": bool(
            exact_prefix_population["passed"]
            and exact_prefix_immutability["passed"]
            and exact_prefix_page_tables["passed"]
        ),
        "exact_prefix_pages": args.exact_prefix_pages,
        "exact_static_suffix_pages": args.exact_static_suffix_pages,
        "fixed_exact_pages": (
            args.exact_prefix_pages + args.exact_static_suffix_pages
        ),
        "exact_tail_pages": exact_pages,
        "storage_pages_per_request": (
            args.exact_prefix_pages
            + args.exact_static_suffix_pages
            + exact_pages
        ),
        "population": exact_prefix_population,
        "immutability": exact_prefix_immutability,
        "page_tables": exact_prefix_page_tables,
    }

    runtime_page_finalization = audit_runtime_page_finalization(
        gauge_cache,
        pages_per_request=pages,
        initial_pages_per_request=initial_pages,
        batch_size=args.batch_size,
        layers=layers,
        context=args.context,
        decode_steps=args.decode_steps,
        exact_tail=args.exact_tail,
        exact_prefix_pages=args.exact_prefix_pages,
        exact_static_suffix_pages=args.exact_static_suffix_pages,
    )

    page_gauge_vs_baseline = compare_logits(
        baseline_logits,
        candidate_logits,
        decode_labels,
        reference_name="FlashInfer FP16",
        candidate_name="PageGauge INT8 plus exact FP16 tail",
        predicted_position_start=args.context + 1,
        request_windows=quality_windows,
    )
    baseline_vs_hf = compare_logits(
        hf_logits,
        baseline_logits,
        decode_labels,
        reference_name="HF SDPA FP16",
        candidate_name="FlashInfer FP16",
        predicted_position_start=args.context + 1,
        request_windows=quality_windows,
    )
    page_gauge_vs_hf = compare_logits(
        hf_logits,
        candidate_logits,
        decode_labels,
        reference_name="HF SDPA FP16",
        candidate_name="PageGauge INT8 plus exact FP16 tail",
        predicted_position_start=args.context + 1,
        request_windows=quality_windows,
    )
    page_gauge_threshold_passed = (
        page_gauge_vs_baseline["minimum_logits_cosine"] >= args.min_logits_cosine
        and page_gauge_vs_baseline["top1_agreement_fraction"] >= args.min_top1_agreement
    )
    baseline_hf_threshold_passed = (
        baseline_vs_hf["minimum_logits_cosine"] >= args.min_baseline_hf_cosine
    )
    correctness_passed = (
        page_gauge_threshold_passed
        and baseline_hf_threshold_passed
        and graph_check["passed"]
        and runtime_page_finalization["passed"]
        and exact_prefix_canary["passed"]
        and heldout_memory_gate["passed"]
    )
    prefix_actual_ids = tokens[:, args.context]
    prefix_oracle_top1 = hf_prefix_next_logits.argmax(dim=-1)
    prefix_log_probability = F.log_softmax(hf_prefix_next_logits.double(), dim=-1)
    prefix_reference_nll = -prefix_log_probability.gather(
        1, prefix_actual_ids[:, None]
    )[:, 0]
    prefix_true_logits = hf_prefix_next_logits.double().gather(
        1, prefix_actual_ids[:, None]
    )
    prefix_true_rank = (hf_prefix_next_logits.double() > prefix_true_logits).sum(
        dim=-1
    ) + 1
    prefix_oracle = {
        "definition": (
            "HF SDPA FP16 final-prefill logits predict the actual WikiText "
            "token at position context"
        ),
        "predicted_position": args.context,
        "actual_token_ids": prefix_actual_ids.tolist(),
        "hf_top1_token_ids": prefix_oracle_top1.tolist(),
        "hf_top1_agrees_with_actual": prefix_oracle_top1.eq(prefix_actual_ids).tolist(),
        "hf_actual_token_logits": hf_prefix_next_logits.gather(
            1, prefix_actual_ids[:, None]
        )[:, 0].tolist(),
        "hf_actual_token_nll_nats": prefix_reference_nll.tolist(),
        "hf_actual_token_perplexity_contribution": [
            finite_exp(float(value)) for value in prefix_reference_nll
        ],
        "hf_actual_token_rank": prefix_true_rank.tolist(),
    }
    decode_transition_oracle = {
        "definition": (
            "the first custom decode step consumes the actual continuation at "
            "position context; its logits are compared with an independent HF "
            "step over the same copied-prefix state"
        ),
        "input_position": args.context,
        "input_token_ids": prefix_actual_ids.tolist(),
        "predicted_position": args.context + 1,
        "actual_next_token_ids": (
            tokens[:, args.context + 1].tolist() if args.decode_steps > 1 else None
        ),
        "hf_top1_token_ids": hf_logits[0].argmax(dim=-1).tolist(),
        "flashinfer_fp16_top1_token_ids": baseline_logits[0]
        .argmax(dim=-1)
        .cpu()
        .tolist(),
        "page_gauge_top1_token_ids": candidate_logits[0].argmax(dim=-1).cpu().tolist(),
        "flashinfer_fp16_vs_hf": {
            "logits_cosine_by_request": baseline_vs_hf["logits_cosine_by_step_request"][
                0
            ],
            "top1_agreement_by_request": baseline_vs_hf[
                "top1_agreement_by_step_request"
            ][0],
            "maximum_absolute_error_by_request": baseline_vs_hf[
                "maximum_logits_absolute_error_by_step_request"
            ][0],
        },
        "page_gauge_vs_flashinfer_fp16": {
            "logits_cosine_by_request": page_gauge_vs_baseline[
                "logits_cosine_by_step_request"
            ][0],
            "top1_agreement_by_request": page_gauge_vs_baseline[
                "top1_agreement_by_step_request"
            ][0],
            "maximum_absolute_error_by_request": page_gauge_vs_baseline[
                "maximum_logits_absolute_error_by_step_request"
            ][0],
        },
    }

    base_correctness_passed = correctness_passed
    static_teacher_diagnostic: dict[str, Any] = {
        "enabled": False,
        "completed": False,
        "passed": None,
        "result": None,
        "error": None,
    }
    if args.run_static_teacher_control:
        assert token_step_graph_module is not None
        assert graph_initial_caches is not None
        print(
            "Measuring coherent static teacher-forced retain/discard control...",
            flush=True,
        )
        static_teacher_diagnostic["enabled"] = True
        static_fixture_provenance = {
            "source_experiment": "page_gauge_model_prefill_correctness",
            "cache_origin": (
                "HF SDPA FP16 post-RoPE K/V copied into the FP16 "
                "request-major cache, then PageGauge derived in-process from "
                "those same tensors"
            ),
            "same_live_model_and_decoder_objects": True,
            "same_live_cache_objects": True,
            "cache_serialization_or_reload": False,
            "prefill_execution": (
                "requests serialized at batch one with a growing DynamicCache"
            ),
            "sampled_boundary_kv_sha256": kv_sample_sha256,
            "token_source_kind": token_provenance["kind"],
            "token_ids_sha256": token_provenance["token_ids_sha256"],
            "archive_sha256": token_provenance.get("archive_sha256"),
            "archive_member_sha256": token_provenance.get("archive_member_sha256"),
            "continuation_semantics": token_provenance.get("continuation_semantics"),
            "teacher_forced_semantics": (
                "the 16 actual corpus successor tokens beginning at context, "
                "predeclared once as a CUDA [16,B] tensor"
            ),
        }
        try:
            static_result = token_step_graph_module.run_static_teacher_forced_fixture(
                model=model,
                decoders=decoders,
                teacher_forced_tokens=tokens[
                    :, args.context : args.context + PG.PAGE
                ].T,
                start_position=args.context,
                warmups=args.static_teacher_warmups,
                repeats=args.static_teacher_repeats,
                cache_scrub_mib=args.static_teacher_cache_scrub_mib,
                seed=args.seed,
                min_logits_cosine=args.min_logits_cosine,
                min_top1_agreement=args.min_top1_agreement,
                initial_caches=graph_initial_caches,
                fixture_provenance=static_fixture_provenance,
                cache_storage=storage,
                component_repeats=args.static_teacher_component_repeats,
                run_matrix_timing=(not args.static_teacher_skip_matrix_timing),
                release_baseline_for_candidate_residency=(
                    args.static_teacher_terminal_pg_only_residency
                ),
                relocate_candidate_cache_after_release=(
                    args.static_teacher_terminal_relocate_pg_cache
                ),
                additional_source_paths=(Path(__file__),),
            )
            static_passed = bool(static_result["correctness"]["passed"])
            static_teacher_diagnostic.update(
                {
                    "completed": True,
                    "passed": static_passed,
                    "result": static_result,
                }
            )
        except Exception as error:
            static_passed = False
            static_teacher_diagnostic["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        correctness_passed = correctness_passed and static_passed

    direct_feedback_diagnostic: dict[str, Any] = {
        "enabled": False,
        "completed": False,
        "passed": None,
        "result": None,
        "error": None,
    }
    if args.run_direct_feedback_paths:
        assert token_step_graph_module is not None
        assert graph_initial_caches is not None
        print(
            "Measuring coherent generated-feedback path "
            f"({args.direct_feedback_protocol})...",
            flush=True,
        )
        direct_feedback_diagnostic["enabled"] = True
        direct_fixture_provenance = {
            "source_experiment": "page_gauge_model_prefill_correctness",
            "cache_origin": (
                "HF SDPA FP16 post-RoPE K/V copied into the FP16 "
                "request-major cache, then PageGauge derived in-process from "
                "those same tensors"
            ),
            "same_live_model_and_decoder_objects": True,
            "same_live_cache_objects": True,
            "cache_serialization_or_reload": False,
            "prefill_execution": (
                "requests serialized at batch one with a growing DynamicCache"
            ),
            "sampled_boundary_kv_sha256": kv_sample_sha256,
            "token_source_kind": token_provenance["kind"],
            "token_ids_sha256": token_provenance["token_ids_sha256"],
            "archive_sha256": token_provenance.get("archive_sha256"),
            "archive_member_sha256": token_provenance.get("archive_member_sha256"),
            "continuation_semantics": token_provenance.get("continuation_semantics"),
            "initial_seed_semantics": (
                "actual successor token at the first decode position for each "
                "coherent model-prefilled request"
            ),
        }
        try:
            direct_result = (
                token_step_graph_module.run_direct_generated_feedback_fixture(
                    model=model,
                    decoders=decoders,
                    initial_tokens=tokens[:, args.context],
                    start_position=args.context,
                    warmups=args.direct_feedback_warmups,
                    repeats=args.direct_feedback_repeats,
                    cache_scrub_mib=args.direct_feedback_cache_scrub_mib,
                    seed=args.seed,
                    min_logits_cosine=args.min_logits_cosine,
                    min_top1_agreement=args.min_top1_agreement,
                    initial_caches=graph_initial_caches,
                    fixture_provenance=direct_fixture_provenance,
                    cache_storage=storage,
                    additional_source_paths=(Path(__file__),),
                    direct_feedback_protocol=(
                        args.direct_feedback_protocol.replace("-", "_")
                    ),
                )
            )
            direct_passed = bool(direct_result["correctness"]["passed"])
            direct_feedback_diagnostic.update(
                {
                    "completed": True,
                    "passed": direct_passed,
                    "result": direct_result,
                }
            )
        except Exception as error:
            direct_passed = False
            direct_feedback_diagnostic["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        correctness_passed = correctness_passed and direct_passed

    token_step_graph_diagnostic: dict[str, Any] = {
        "enabled": False,
        "completed": False,
        "passed": None,
        "result": None,
        "error": None,
    }
    if args.run_token_step_graphs:
        assert token_step_graph_module is not None
        assert graph_initial_caches is not None
        print(
            "Reusing the live coherent caches for per-token decoder graphs...",
            flush=True,
        )
        token_step_graph_diagnostic["enabled"] = True
        fixture_provenance = {
            "source_experiment": "page_gauge_model_prefill_correctness",
            "cache_origin": (
                "HF SDPA FP16 post-RoPE K/V copied into the FP16 "
                "request-major cache, then PageGauge derived in-process from "
                "those same tensors"
            ),
            "same_live_model_and_decoder_objects": True,
            "same_live_cache_objects": True,
            "cache_serialization_or_reload": False,
            "prefill_execution": (
                "requests serialized at batch one with a growing DynamicCache"
            ),
            "sampled_boundary_kv_sha256": kv_sample_sha256,
            "token_source_kind": token_provenance["kind"],
            "token_ids_sha256": token_provenance["token_ids_sha256"],
            "archive_sha256": token_provenance.get("archive_sha256"),
            "archive_member_sha256": token_provenance.get("archive_member_sha256"),
            "continuation_semantics": token_provenance.get("continuation_semantics"),
        }
        try:
            token_step_result = token_step_graph_module.run_token_step_graph_fixture(
                model=model,
                decoders=decoders,
                teacher_forced_tokens=tokens[
                    :, args.context : args.context + PG.PAGE
                ].T.contiguous(),
                start_position=args.context,
                capture_warmups=args.token_graph_capture_warmups,
                warmups=args.token_graph_warmups,
                repeats=args.token_graph_repeats,
                cache_scrub_mib=args.token_graph_cache_scrub_mib,
                seed=args.seed,
                min_logits_cosine=args.min_logits_cosine,
                min_top1_agreement=args.min_top1_agreement,
                initial_caches=graph_initial_caches,
                fixture_provenance=fixture_provenance,
                cache_storage=storage,
                additional_source_paths=(Path(__file__),),
            )
            token_step_passed = bool(token_step_result["correctness"]["passed"])
            token_step_graph_diagnostic.update(
                {
                    "completed": True,
                    "passed": token_step_passed,
                    "result": token_step_result,
                }
            )
        except Exception as error:
            token_step_passed = False
            token_step_graph_diagnostic["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        correctness_passed = correctness_passed and token_step_passed

    generated_sequence_graph_diagnostic: dict[str, Any] = {
        "enabled": False,
        "completed": False,
        "passed": None,
        "result": None,
        "error": None,
    }
    if args.run_generated_sequence_graph:
        assert generated_sequence_graph_module is not None
        assert graph_initial_caches is not None
        print(
            "Reusing the live coherent caches for one generated block graph...",
            flush=True,
        )
        generated_sequence_graph_diagnostic["enabled"] = True
        generated_fixture_provenance = {
            "source_experiment": "page_gauge_model_prefill_correctness",
            "cache_origin": (
                "HF SDPA FP16 post-RoPE K/V copied into the FP16 "
                "request-major cache, then PageGauge derived in-process from "
                "those same tensors"
            ),
            "same_live_model_and_decoder_objects": True,
            "same_live_cache_objects": True,
            "cache_serialization_or_reload": False,
            "prefill_execution": (
                "requests serialized at batch one with a growing DynamicCache"
            ),
            "sampled_boundary_kv_sha256": kv_sample_sha256,
            "token_source_kind": token_provenance["kind"],
            "token_ids_sha256": token_provenance["token_ids_sha256"],
            "archive_sha256": token_provenance.get("archive_sha256"),
            "archive_member_sha256": token_provenance.get("archive_member_sha256"),
            "continuation_semantics": token_provenance.get("continuation_semantics"),
            "initial_seed_semantics": (
                "actual successor token at position 20480 for each coherent "
                "model-prefilled request"
            ),
        }
        try:
            generated_result = (
                generated_sequence_graph_module.run_generated_sequence_graph_fixture(
                    model=model,
                    decoders=decoders,
                    initial_tokens=tokens[:, args.context],
                    start_position=args.context,
                    capture_warmups=args.generated_graph_capture_warmups,
                    warmups=args.generated_graph_warmups,
                    repeats=args.generated_graph_repeats,
                    cache_scrub_mib=args.generated_graph_cache_scrub_mib,
                    seed=args.seed,
                    min_logits_cosine=args.min_logits_cosine,
                    min_top1_agreement=args.min_top1_agreement,
                    initial_caches=graph_initial_caches,
                    fixture_provenance=generated_fixture_provenance,
                    cache_storage=storage,
                    additional_source_paths=(Path(__file__),),
                )
            )
            generated_passed = bool(generated_result["correctness"]["passed"])
            generated_sequence_graph_diagnostic.update(
                {
                    "completed": True,
                    "passed": generated_passed,
                    "result": generated_result,
                }
            )
        except Exception as error:
            generated_passed = False
            generated_sequence_graph_diagnostic["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        correctness_passed = correctness_passed and generated_passed

    source_sha256_at_end = model_prefill_source_manifest()
    if source_sha256_at_end != source_sha256_at_start:
        raise RuntimeError("model-prefill source closure changed during execution")
    source_integrity_gate = {
        "passed": True,
        "hashed_before_model_or_dataset_access": True,
        "unchanged_through_result_finalization": True,
        "paths": list(source_sha256_at_start),
    }
    heldout_protocol: dict[str, Any] = {
        "enabled": heldout_expected_starts is not None,
        "name": HELDOUT_PROTOCOL_NAME if heldout_expected_starts is not None else None,
        "passed": bool(correctness_passed),
    }
    if heldout_expected_starts is not None:
        heldout_protocol.update(
            {
                "preregistered_before_test_access": True,
                "selected_on_split": "train",
                "confirmation_split": "test",
                "all_window_start_offsets": list(HELDOUT_STARTS),
                "all_window_end_offsets_exclusive": [
                    start + HELDOUT_WINDOW_TOKENS for start in HELDOUT_STARTS
                ],
                "this_shard_start_offsets": list(heldout_expected_starts),
                "this_shard_index": HELDOUT_GROUPS.index(heldout_expected_starts),
                "shard_batch_sizes": [len(group) for group in HELDOUT_GROUPS],
                "window_stride": HELDOUT_WINDOW_STRIDE,
                "window_tokens": HELDOUT_WINDOW_TOKENS,
                "last_window_end_offset_exclusive": (
                    HELDOUT_STARTS[-1] + HELDOUT_WINDOW_TOKENS
                ),
                "test_tokens_remaining_after_last_window": (
                    HELDOUT_TEST_TOKEN_COUNT
                    - HELDOUT_STARTS[-1]
                    - HELDOUT_WINDOW_TOKENS
                ),
                "source_content_lock": {
                    "archive_sha256": WIKITEXT2_RAW_V1_SHA256,
                    "archive_member": HELDOUT_MEMBER,
                    "available_corpus_token_count": HELDOUT_TEST_TOKEN_COUNT,
                    "tokenizer_manifest_sha256": (HELDOUT_TOKENIZER_MANIFEST_SHA256),
                    "archive_hash_cryptographically_locks_member_bytes": True,
                },
                "model_lock": {
                    "model": HELDOUT_MODEL,
                    "revision": HELDOUT_MODEL_REVISION,
                    "config_sha256": HELDOUT_MODEL_CONFIG_SHA256,
                    "parameter_count": HELDOUT_MODEL_PARAMETER_COUNT,
                },
                "configuration_lock": {
                    "context": HELDOUT_CONTEXT,
                    "decode_steps": HELDOUT_DECODE_STEPS,
                    "exact_tail_tokens": HELDOUT_EXACT_TAIL,
                    "exact_prefix_pages": HELDOUT_EXACT_PREFIX_PAGES,
                    "baseline_split_pages": HELDOUT_SPLIT_PAGES,
                    "candidate_split_pages": HELDOUT_SPLIT_PAGES,
                    "tail_attention": "flashinfer_merge",
                    "old_value_scale_placement": "probability",
                    "prefill_chunk_tokens": HELDOUT_PREFILL_CHUNK_TOKENS,
                    "seed": HELDOUT_SEED,
                },
                "worker_threshold_lock": {
                    "minimum_logits_cosine": HELDOUT_MIN_LOGITS_COSINE,
                    "minimum_top1_agreement": HELDOUT_MIN_TOP1_AGREEMENT,
                    "minimum_flashinfer_hf_cosine": (HELDOUT_MIN_BASELINE_HF_COSINE),
                },
                "co_resident_memory_gate": heldout_memory_gate,
                "source_closure_paths": [
                    str(path.relative_to(ROOT)) for path in MODEL_PREFILL_SOURCE_PATHS
                ],
            }
        )
    result = {
        "schema_version": 8,
        "experiment": "page_gauge_model_prefill_correctness",
        "claim_scope": (
            "model-derived KV correctness with serialized batch-one prefill; "
            + (
                "the optional nested result times fixed-batch, fixed-context, "
                "one-page whole-decoder graphs only, while prefill itself makes "
                "no latency or throughput claim"
                if (
                    args.run_token_step_graphs
                    or args.run_direct_feedback_paths
                    or args.run_static_teacher_control
                    or args.run_generated_sequence_graph
                )
                else "no latency or throughput claim is made"
            )
        ),
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_config_sha256": model_config_sha256,
        "parameters": parameter_count,
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": PG.DIM,
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail_tokens": args.exact_tail,
        "exact_prefix_pages": args.exact_prefix_pages,
        "exact_sink_pages": args.exact_prefix_pages,
        "exact_static_suffix_pages": args.exact_static_suffix_pages,
        "exact_static_suffix_logical_range": (
            [
                initial_pages - args.exact_static_suffix_pages,
                initial_pages,
            ]
            if args.exact_static_suffix_pages
            else None
        ),
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "center_restore": "attention_add",
        "tail_attention": args.tail_attention,
        "old_value_scale_placement": args.old_value_scale_placement,
        "heldout_protocol": heldout_protocol,
        "source_integrity_gate": source_integrity_gate,
        "token_source": token_provenance,
        "prefill": {
            "reference_label": "HF SDPA FP16",
            "implementation": (
                "Hugging Face Mistral-family decoder loaded with dtype=torch.float16"
            ),
            "checkpoint_config_torch_dtype": str(config_payload.get("torch_dtype")),
            "reference_weight_dtype": "torch.float16",
            "reported_logits_dtype": (
                "torch.float32 after the model lm_head output is cast for comparison"
            ),
            "attention_implementation": model.config._attn_implementation,
            "execution": (
                "requests serialized at batch one; chunks are causal with a "
                "growing DynamicCache"
            ),
            "cache_semantics": (
                "HF post-RoPE keys and values copied before reference decode"
            ),
            "hf_cache_layout": "[1,Hkv,tokens,head_dim]",
            "paged_destination_layout": "request-major [L,B*P,16,Hkv,D]",
            "chunk_tokens": args.prefill_chunk_tokens,
            "request_records": prefill_records,
            "sampled_boundary_kv_sha256": kv_sample_sha256,
        },
        "cache_storage": {
            **storage,
            "logical_prefix_fp16_kv_bytes": (
                layers
                * args.batch_size
                * args.context
                * hkv
                * PG.DIM
                * torch.tensor([], dtype=torch.float16).element_size()
                * 2
            ),
            "derivation": (
                "both backends derive from the same copied HF prefix; PageGauge "
                "uses per-request, per-layer, per-head channel centers, one FP16 "
                "scale per page/head, fixed centered FP16 prefix slots, and a "
                "centered FP16 ring for the exact tail"
            ),
        },
        "logit_collection": {
            "immediate_cpu_offload": True,
            "timing_claim": False,
            "reason": (
                "correctness-only collection releases each full-vocabulary GPU "
                "logit tensor before the next decode step"
            ),
        },
        "runtime_page_finalization": {
            **future_page_canary,
            **runtime_page_finalization,
        },
        "exact_prefix_canary": exact_prefix_canary,
        "correctness": {
            "teacher_forced": True,
            "logit_layout": "per-step/per-request over the full vocabulary",
            "hf_prefix_next_token_oracle": prefix_oracle,
            "hf_to_custom_decode_transition_oracle": decode_transition_oracle,
            "page_gauge_vs_flashinfer_fp16": page_gauge_vs_baseline,
            "flashinfer_fp16_vs_hf_sdpa_fp16": baseline_vs_hf,
            "page_gauge_vs_hf_sdpa_fp16": page_gauge_vs_hf,
            "quality_metric_protocol": {
                "decode_inputs": ("positions [context, context+decode_steps)"),
                "true_labels": ("positions [context+1, context+decode_steps+1)"),
                "required_token_shape": ("[batch, context+decode_steps+1]"),
                "distribution_reduction": (
                    "one decode step at a time; only scalar per-token metrics "
                    "and per-request sufficient statistics are serialized"
                ),
                "cluster_bootstrap_unit": "one disjoint corpus request/window",
                "window_metadata": quality_windows,
            },
            "graph_replay": graph_check,
            "runtime_page_finalization": runtime_page_finalization,
            "exact_prefix_canary": exact_prefix_canary,
            "thresholds": {
                "page_gauge_minimum_logits_cosine": args.min_logits_cosine,
                "page_gauge_minimum_top1_agreement": args.min_top1_agreement,
                "flashinfer_minimum_hf_logits_cosine": (args.min_baseline_hf_cosine),
            },
            "page_gauge_threshold_passed": page_gauge_threshold_passed,
            "baseline_hf_conversion_threshold_passed": (baseline_hf_threshold_passed),
            "base_model_prefill_correctness_passed": base_correctness_passed,
            "static_teacher_diagnostic_required": (args.run_static_teacher_control),
            "static_teacher_diagnostic_passed": (static_teacher_diagnostic["passed"]),
            "direct_feedback_diagnostic_required": (args.run_direct_feedback_paths),
            "direct_feedback_diagnostic_passed": (direct_feedback_diagnostic["passed"]),
            "token_step_graph_diagnostic_required": (args.run_token_step_graphs),
            "token_step_graph_diagnostic_passed": (
                token_step_graph_diagnostic["passed"]
            ),
            "generated_sequence_graph_diagnostic_required": (
                args.run_generated_sequence_graph
            ),
            "generated_sequence_graph_diagnostic_passed": (
                generated_sequence_graph_diagnostic["passed"]
            ),
            "passed": correctness_passed,
        },
        "static_teacher_diagnostic": static_teacher_diagnostic,
        "direct_feedback_diagnostic": direct_feedback_diagnostic,
        "token_step_graph_diagnostic": token_step_graph_diagnostic,
        "generated_sequence_graph_diagnostic": (generated_sequence_graph_diagnostic),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "local_files_only": True,
        },
        "invocation": {
            "argv": sys.argv,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "source_sha256": source_sha256_at_start,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "page_gauge_vs_flashinfer_fp16": {
                    "minimum_logits_cosine": page_gauge_vs_baseline[
                        "minimum_logits_cosine"
                    ],
                    "top1_agreement_fraction": page_gauge_vs_baseline[
                        "top1_agreement_fraction"
                    ],
                },
                "flashinfer_fp16_vs_hf_sdpa_fp16": {
                    "minimum_logits_cosine": baseline_vs_hf["minimum_logits_cosine"],
                    "top1_agreement_fraction": baseline_vs_hf[
                        "top1_agreement_fraction"
                    ],
                },
                "static_teacher_control": {
                    "enabled": static_teacher_diagnostic["enabled"],
                    "completed": static_teacher_diagnostic["completed"],
                    "passed": static_teacher_diagnostic["passed"],
                    "timing_completed": bool(
                        static_teacher_diagnostic["result"]
                        and static_teacher_diagnostic["result"]["timing_modes"]
                    ),
                },
                "direct_feedback": {
                    "enabled": direct_feedback_diagnostic["enabled"],
                    "completed": direct_feedback_diagnostic["completed"],
                    "passed": direct_feedback_diagnostic["passed"],
                    "timing_completed": bool(
                        direct_feedback_diagnostic["result"]
                        and direct_feedback_diagnostic["result"]["timing_modes"]
                    ),
                },
                "token_step_graphs": {
                    "enabled": token_step_graph_diagnostic["enabled"],
                    "completed": token_step_graph_diagnostic["completed"],
                    "passed": token_step_graph_diagnostic["passed"],
                    "timing_input_mode": (
                        token_step_graph_diagnostic["result"]["timing_input_mode"]
                        if token_step_graph_diagnostic["result"] is not None
                        else None
                    ),
                },
                "generated_sequence_graph": {
                    "enabled": generated_sequence_graph_diagnostic["enabled"],
                    "completed": generated_sequence_graph_diagnostic["completed"],
                    "passed": generated_sequence_graph_diagnostic["passed"],
                    "timing_completed": bool(
                        generated_sequence_graph_diagnostic["result"]
                        and generated_sequence_graph_diagnostic["result"][
                            "timing_modes"
                        ]
                    ),
                },
                "passed": correctness_passed,
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    if not correctness_passed:
        raise SystemExit("model-prefill correctness thresholds failed")


if __name__ == "__main__":
    main()
