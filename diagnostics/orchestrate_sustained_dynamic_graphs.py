#!/usr/bin/env python3
"""Run sustained PageGauge/FI workers in fresh-process Williams ABBA order."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import orchestrate_backend_exclusive as BASE
from analyze_sustained_dynamic_graphs import (
    ANALYSIS_EXPERIMENT,
    DYNAMIC_GRAPH_SCOPE,
    MANIFEST_EXPERIMENT,
    ProtocolError,
    REQUIRED_WORKER_SOURCES,
    analyze_run_directory,
    expected_orchestration_environment,
    expected_worker_configuration,
    sha256_file,
    validate_worker_result,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 3
DEFAULT_WORKER = ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py"
DEFAULT_ANALYZER = ROOT / "diagnostics/analyze_sustained_dynamic_graphs.py"
DEFAULT_WIKITEXT = ROOT / "data/wikitext-2-raw-v1.zip"
MANIFEST_NAME = "orchestration_manifest.json"
POLICY_PRESET = "exact_prefix_s3_d1024_t768"
POLICY_EXACT_PREFIX_PAGES = 3
POLICY_DECODE_STEPS = 1024
POLICY_EXACT_TAIL_TOKENS = 768
POLICY_GENERATED_INT8_PAGES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-preset",
        choices=(POLICY_PRESET,),
        default=POLICY_PRESET,
        help=(
            "Frozen publication policy: Mistral B4/C20480/D1024, exact prefix "
            "S3, exact tail T768, segmented FlashInfer merge, probability-side "
            "old-value scaling, and frozen-HF teacher forcing."
        ),
    )
    parser.add_argument("--profile", choices=("pilot", "publication"), default="pilot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--pairs-per-seed", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=5090)

    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=20480)
    parser.add_argument("--decode-steps", type=int, default=POLICY_DECODE_STEPS)
    parser.add_argument("--exact-tail", type=int, default=POLICY_EXACT_TAIL_TOKENS)
    parser.add_argument(
        "--exact-sink-pages",
        type=int,
        default=POLICY_EXACT_PREFIX_PAGES,
        help=(
            "Legacy worker spelling for the contiguous exact-prefix length S. "
            "The frozen policy requires S=3 and records both prefix/sink aliases."
        ),
    )
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=256)
    parser.add_argument(
        "--tail-attention",
        choices=("flashinfer_merge", "fused_kernel", "heterogeneous_fa2"),
        default="flashinfer_merge",
    )
    parser.add_argument(
        "--old-value-scale-placement",
        choices=("probability", "value_fragment"),
        default="probability",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("greedy_feedback", "frozen_hf_teacher_forced"),
        default="frozen_hf_teacher_forced",
    )
    parser.add_argument("--capture-warmups", type=int, default=1)
    parser.add_argument("--maximum-graph-banks", type=int, default=16)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument("--min-logits-cosine", type=float, default=0.995)
    parser.add_argument("--min-top1-agreement", type=float, default=0.99)
    parser.add_argument("--quality-diagnostics-top-k", type=int, default=0)
    parser.add_argument("--quality-diagnostics-top-vocab", type=int, default=8)

    parser.add_argument(
        "--token-source", choices=("wikitext2", "random"), default="wikitext2"
    )
    parser.add_argument("--wikitext-zip", type=Path, default=DEFAULT_WIKITEXT)
    parser.add_argument("--wikitext-member", default="wikitext-2-raw/wiki.train.raw")
    parser.add_argument("--token-offset", type=int, default=100000)
    parser.add_argument("--token-stride", type=int, default=21984)
    parser.add_argument("--seed-token-offset-stride", type=int)

    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", type=Path)
    parser.add_argument("--idle-timeout-s", type=float, default=300.0)
    parser.add_argument("--idle-poll-s", type=float, default=1.0)
    parser.add_argument("--idle-consecutive-polls", type=int, default=3)
    parser.add_argument("--idle-max-utilization-percent", type=float, default=5.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=float, default=256.0)
    parser.add_argument("--max-initial-gpu-used-mib", type=float, default=4096.0)
    parser.add_argument("--telemetry-interval-ms", type=int, default=500)
    parser.add_argument("--worker-timeout-s", type=float, default=14400.0)
    parser.add_argument("--allow-missing-nvidia-smi", action="store_true")
    parser.add_argument(
        "--worker-env", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolved_profile(args: argparse.Namespace) -> dict[str, Any]:
    fixture_length = args.context + args.decode_steps
    request_stride = args.token_stride or fixture_length
    disjoint_seed_stride = (
        (args.batch_size - 1) * request_stride + fixture_length
    )
    if args.profile == "pilot":
        defaults = {
            "seeds": [20260861, 20260862],
            "pairs_per_seed": 2,
            "warmups": 1,
            "repeats": 2,
            "bootstrap_samples": 2_000,
            "seed_token_offset_stride": disjoint_seed_stride,
        }
    else:
        defaults = {
            "seeds": [20260861, 20260862, 20260863, 20260864],
            "pairs_per_seed": 4,
            "warmups": 3,
            "repeats": 10,
            "bootstrap_samples": 20_000,
            "seed_token_offset_stride": disjoint_seed_stride,
        }
    return {
        "profile": args.profile,
        "seeds": list(args.seeds if args.seeds is not None else defaults["seeds"]),
        "pairs_per_seed": args.pairs_per_seed
        if args.pairs_per_seed is not None
        else defaults["pairs_per_seed"],
        "warmups": args.warmups if args.warmups is not None else defaults["warmups"],
        "repeats": args.repeats if args.repeats is not None else defaults["repeats"],
        "bootstrap_samples": args.bootstrap_samples
        if args.bootstrap_samples is not None
        else defaults["bootstrap_samples"],
        "seed_token_offset_stride": args.seed_token_offset_stride
        if args.seed_token_offset_stride is not None
        else defaults["seed_token_offset_stride"],
    }


def validate_args(args: argparse.Namespace, profile: Mapping[str, Any]) -> None:
    if not profile["seeds"] or len(set(profile["seeds"])) != len(profile["seeds"]):
        raise ProtocolError("seeds must be non-empty and unique")
    if profile["pairs_per_seed"] <= 0 or profile["pairs_per_seed"] % 2:
        raise ProtocolError("pairs-per-seed must be a positive even number")
    if profile["warmups"] < 0 or profile["repeats"] <= 0:
        raise ProtocolError("warmups/repeats are invalid")
    if profile["bootstrap_samples"] <= 0:
        raise ProtocolError("bootstrap sample count must be positive")
    if len(profile["seeds"]) < 2:
        raise ProtocolError(
            "Williams pilot/publication requires at least two distinct seed cohorts"
        )
    if args.batch_size <= 0 or args.context <= args.exact_tail:
        raise ProtocolError("batch/context/exact-tail configuration is invalid")
    if args.context % 16 or args.decode_steps < 512 or args.decode_steps % 16:
        raise ProtocolError("sustained context/decode must be page aligned and D>=512")
    if args.exact_tail <= 0 or args.exact_tail % 16:
        raise ProtocolError("exact tail must contain whole pages")
    if args.exact_sink_pages < 0:
        raise ProtocolError("exact-prefix page count cannot be negative")
    if args.context // 16 <= args.exact_tail // 16 + args.exact_sink_pages:
        raise ProtocolError("context must retain at least one quantized old page")
    if args.baseline_split_pages < 0 or args.candidate_split_pages < 0:
        raise ProtocolError("fixed split sizes cannot be negative")
    if args.capture_warmups <= 0 or args.maximum_graph_banks <= 0:
        raise ProtocolError("capture controls must be positive")
    if args.cache_scrub_mib <= 0 or args.prefill_chunk_tokens <= 0:
        raise ProtocolError("cache scrub and prefill chunk must be positive")
    if args.quality_diagnostics_top_k < 0 or args.quality_diagnostics_top_vocab <= 0:
        raise ProtocolError("quality-diagnostic widths are invalid")
    if profile["seed_token_offset_stride"] < 0 or args.token_offset < 0:
        raise ProtocolError("token offsets cannot be negative")
    fixture_length = args.context + args.decode_steps
    request_stride = args.token_stride or fixture_length
    minimum_seed_stride = (args.batch_size - 1) * request_stride + fixture_length
    if (
        args.token_source == "wikitext2"
        and len(profile["seeds"]) > 1
        and profile["seed_token_offset_stride"] < minimum_seed_stride
    ):
        raise ProtocolError(
            "WikiText seed fixtures must be disjoint: seed-token-offset-stride "
            f"must be at least the B-request cohort span={minimum_seed_stride}"
        )
    if args.profile == "publication":
        if args.token_source != "wikitext2":
            raise ProtocolError(
                "publication profile requires the fixed WikiText-2 corpus artifact; "
                "random tokens are a non-publication negative control"
            )
        publication_minima = {
            "seeds": (len(profile["seeds"]), 4),
            "pairs-per-seed": (profile["pairs_per_seed"], 4),
            "warmups": (profile["warmups"], 3),
            "repeats": (profile["repeats"], 10),
            "bootstrap-samples": (profile["bootstrap_samples"], 20_000),
        }
        for label, (observed, minimum) in publication_minima.items():
            if observed < minimum:
                raise ProtocolError(
                    f"publication {label} must be at least {minimum}, got {observed}"
                )
    if args.allow_missing_nvidia_smi:
        raise ProtocolError(
            "Williams pilot/publication cannot skip nvidia-smi idle/telemetry gates"
        )
    if args.rerun_failed and not args.resume:
        raise ProtocolError("--rerun-failed requires --resume")
    if args.worker.resolve() != DEFAULT_WORKER.resolve():
        raise ProtocolError(
            "sustained-v3 source attestation requires the bundled canonical worker"
        )
    if (
        args.idle_timeout_s <= 0
        or args.idle_poll_s <= 0
        or args.worker_timeout_s <= 0
        or args.idle_consecutive_polls <= 0
        or args.telemetry_interval_ms <= 0
    ):
        raise ProtocolError("idle, telemetry, and worker timeout controls must be positive")
    frozen_policy = {
        "policy_preset": args.policy_preset,
        "model": args.model,
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail": args.exact_tail,
        "exact_sink_pages": args.exact_sink_pages,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "tail_attention": args.tail_attention,
        "old_value_scale_placement": args.old_value_scale_placement,
        "trajectory_mode": args.trajectory_mode,
        "token_source": args.token_source,
        "wikitext_member": args.wikitext_member,
        "token_offset": args.token_offset,
        "token_stride": args.token_stride,
        "min_logits_cosine": args.min_logits_cosine,
        "min_top1_agreement": args.min_top1_agreement,
        "quality_diagnostics_top_k": args.quality_diagnostics_top_k,
    }
    expected_frozen_policy = {
        "policy_preset": POLICY_PRESET,
        "model": "mistralai/Mistral-7B-v0.3",
        "batch_size": 4,
        "context": 20480,
        "decode_steps": POLICY_DECODE_STEPS,
        "exact_tail": POLICY_EXACT_TAIL_TOKENS,
        "exact_sink_pages": POLICY_EXACT_PREFIX_PAGES,
        "baseline_split_pages": 256,
        "candidate_split_pages": 256,
        "tail_attention": "flashinfer_merge",
        "old_value_scale_placement": "probability",
        "trajectory_mode": "frozen_hf_teacher_forced",
        "token_source": "wikitext2",
        "wikitext_member": "wikitext-2-raw/wiki.train.raw",
        "token_offset": 100000,
        "token_stride": 21984,
        "min_logits_cosine": 0.995,
        "min_top1_agreement": 0.99,
        "quality_diagnostics_top_k": 0,
    }
    if frozen_policy != expected_frozen_policy:
        mismatches = [
            f"{name}={frozen_policy[name]!r} (expected {expected!r})"
            for name, expected in expected_frozen_policy.items()
            if frozen_policy[name] != expected
        ]
        raise ProtocolError(
            f"{POLICY_PRESET} is immutable: " + "; ".join(mismatches)
        )
    generated_int8_pages = args.decode_steps // 16 - args.exact_tail // 16
    if generated_int8_pages != POLICY_GENERATED_INT8_PAGES:
        raise ProtocolError(
            "frozen recurrence policy must consume exactly 16 generated INT8 pages"
        )


def worker_command(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    block: Mapping[str, Any],
    output_path: Path,
) -> list[str]:
    exact_prefix_pages = (
        args.exact_sink_pages if block["backend"] == "page_gauge" else 0
    )
    return [
        args.python,
        str(args.worker.resolve()),
        "--backend",
        str(block["backend"]),
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
        "--context",
        str(args.context),
        "--decode-steps",
        str(args.decode_steps),
        "--exact-tail",
        str(args.exact_tail),
        "--exact-sink-pages",
        str(exact_prefix_pages),
        "--prefill-chunk-tokens",
        str(args.prefill_chunk_tokens),
        "--baseline-split-pages",
        str(args.baseline_split_pages),
        "--candidate-split-pages",
        str(args.candidate_split_pages),
        "--tail-attention",
        args.tail_attention,
        "--old-value-scale-placement",
        args.old_value_scale_placement,
        "--trajectory-mode",
        args.trajectory_mode,
        "--seed",
        str(block["seed"]),
        "--token-source",
        args.token_source,
        "--wikitext-zip",
        str(args.wikitext_zip.resolve()),
        "--wikitext-member",
        args.wikitext_member,
        "--token-offset",
        str(block["token_offset"]),
        "--token-stride",
        str(args.token_stride),
        "--capture-warmups",
        str(args.capture_warmups),
        "--maximum-graph-banks",
        str(args.maximum_graph_banks),
        "--warmups",
        str(profile["warmups"]),
        "--repeats",
        str(profile["repeats"]),
        "--cache-scrub-mib",
        str(args.cache_scrub_mib),
        "--min-logits-cosine",
        str(args.min_logits_cosine),
        "--min-top1-agreement",
        str(args.min_top1_agreement),
        "--quality-diagnostics-top-k",
        str(args.quality_diagnostics_top_k),
        "--quality-diagnostics-top-vocab",
        str(args.quality_diagnostics_top_vocab),
        "--output",
        str(output_path.resolve()),
    ]


def build_config(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    worker_environment: Mapping[str, str],
) -> dict[str, Any]:
    sources = (
        args.worker.resolve(),
        DEFAULT_ANALYZER.resolve(),
        Path(__file__).resolve(),
        (ROOT / "diagnostics/orchestrate_backend_exclusive.py").resolve(),
        (ROOT / "diagnostics/analyze_backend_exclusive.py").resolve(),
        (ROOT / "docs/exact_prefix_s3_williams.md").resolve(),
    )
    worker_source_closure = {
        name: sha256_file((ROOT / name).resolve())
        for name in REQUIRED_WORKER_SOURCES
    }
    if args.token_source == "wikitext2":
        wikitext_zip = args.wikitext_zip.resolve()
        if not wikitext_zip.is_file():
            raise ProtocolError(f"WikiText archive is missing: {wikitext_zip}")
        wikitext_archive_sha256: str | None = sha256_file(wikitext_zip)
    else:
        wikitext_archive_sha256 = None
    return {
        "policy_preset": args.policy_preset,
        "profile": dict(profile),
        "backend_symbols": dict(BASE.SYMBOL_TO_BACKEND),
        "williams_sequences": list(BASE.WILLIAMS_SEQUENCES),
        "fresh_process_per_backend_block": True,
        "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
        "model": args.model,
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail": args.exact_tail,
        "exact_sink_pages": args.exact_sink_pages,
        "exact_prefix_pages": args.exact_sink_pages,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "tail_attention": args.tail_attention,
        "old_value_scale_placement": args.old_value_scale_placement,
        "trajectory_mode": args.trajectory_mode,
        "capture_warmups": args.capture_warmups,
        "maximum_graph_banks": args.maximum_graph_banks,
        "token_source": args.token_source,
        "wikitext_zip": str(args.wikitext_zip.resolve()),
        "wikitext_member": args.wikitext_member,
        "wikitext_archive_sha256": wikitext_archive_sha256,
        "base_token_offset": args.token_offset,
        "token_stride": args.token_stride,
        "cache_scrub_mib": args.cache_scrub_mib,
        "min_logits_cosine": args.min_logits_cosine,
        "min_top1_agreement": args.min_top1_agreement,
        "quality_diagnostics_top_k": args.quality_diagnostics_top_k,
        "quality_diagnostics_top_vocab": args.quality_diagnostics_top_vocab,
        "backend_conditioned_exact_prefix_policy": {
            "flashinfer_fp16": 0,
            "page_gauge": args.exact_sink_pages,
            "candidate_exact_prefix_pages": args.exact_sink_pages,
            "candidate_exact_sink_pages_legacy_alias": args.exact_sink_pages,
            "reason": (
                "the FP16 baseline has no segmented exact-prefix representation; "
                "the PageGauge worker pairing key must attest S3"
            ),
        },
        "recurrence_attestation": {
            "page_tokens": 16,
            "generated_pages": args.decode_steps // 16,
            "exact_tail_pages": args.exact_tail // 16,
            "runtime_generated_pages_consumed_as_int8": (
                args.decode_steps // 16 - args.exact_tail // 16
            ),
            "required_runtime_generated_pages_consumed_as_int8": (
                POLICY_GENERATED_INT8_PAGES
            ),
            "passed": (
                args.decode_steps // 16 - args.exact_tail // 16
                == POLICY_GENERATED_INT8_PAGES
            ),
        },
        "python_executable": args.python,
        "worker": str(args.worker.resolve()),
        "worker_environment_overrides": dict(sorted(worker_environment.items())),
        "gpu_index": args.gpu_index,
        "idle_gate": {
            "timeout_s": args.idle_timeout_s,
            "poll_s": args.idle_poll_s,
            "consecutive_polls": args.idle_consecutive_polls,
            "max_utilization_percent": args.idle_max_utilization_percent,
            "memory_tolerance_mib": args.idle_memory_tolerance_mib,
            "max_initial_used_mib": args.max_initial_gpu_used_mib,
        },
        "telemetry_interval_ms": args.telemetry_interval_ms,
        "worker_timeout_s": args.worker_timeout_s,
        "bootstrap_seed": args.bootstrap_seed,
        "scheduler_capacity_policy": (
            "no orchestrator clamp; each worker derives every active wrapper's "
            "capacity from the actual GPU and exhaustive plan preflights all positions"
        ),
        "worker_source_closure_sha256": worker_source_closure,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sources
        },
    }


def initial_manifest(
    config: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]], dry_run: bool
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": MANIFEST_EXPERIMENT,
        "cuda_graph_scope": DYNAMIC_GRAPH_SCOPE,
        "claim_scope": (
            "fresh-process one-backend-at-a-time B4/C20480/D1024 serving; "
            "PageGauge exact-prefix S3/exact-tail T768; frozen-HF WikiText-2 "
            "TRAIN trajectories; Williams ABBA/BAAB order control"
        ),
        "status": "dry_run" if dry_run else "initialized",
        "created_utc": utc_now(),
        "config": dict(config),
        "config_sha256": BASE.canonical_json_sha256(config),
        "schedule": list(schedule),
        "executions": {},
        "analysis_experiment": ANALYSIS_EXPERIMENT,
        "analysis_expected_path": "publication_analysis.json",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": str(Path.cwd().resolve()),
        },
        "invocation": {"argv": list(sys.argv), "orchestrator_pid": os.getpid()},
    }


def prepare_manifest(
    output_dir: Path,
    config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    resume: bool,
    dry_run: bool,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    if resume:
        if not path.is_file():
            raise ProtocolError("resume requested but manifest is missing")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ProtocolError("existing manifest is not an object")
        observed_config = manifest.get("config")
        expected_config_sha256 = BASE.canonical_json_sha256(config)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("experiment") != MANIFEST_EXPERIMENT
            or not isinstance(observed_config, Mapping)
            or dict(observed_config) != dict(config)
            or BASE.canonical_json_sha256(observed_config)
            != manifest.get("config_sha256")
            or manifest.get("config_sha256") != expected_config_sha256
        ):
            raise ProtocolError("resume configuration differs from manifest")
        if manifest.get("schedule") != list(schedule):
            raise ProtocolError("resume schedule differs from manifest")
        return path, manifest
    if any(output_dir.iterdir()):
        raise ProtocolError("output directory is non-empty; use --resume or a new path")
    manifest = initial_manifest(config, schedule, dry_run)
    BASE.atomic_write_json(path, manifest)
    return path, manifest


def _next_attempt(block_dir: Path) -> int:
    attempts = []
    if block_dir.exists():
        for child in block_dir.iterdir():
            if child.is_dir() and child.name.startswith("attempt_"):
                try:
                    attempts.append(int(child.name.removeprefix("attempt_")))
                except ValueError:
                    pass
    return max(attempts, default=0) + 1


def validate_resume_worker_result_hash(
    output_dir: Path,
    block: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    relative = execution.get("worker_result_path")
    digest = execution.get("worker_result_sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ProtocolError(
            f"resume hash provenance is missing for {block['block_id']}"
        )
    result_path = (output_dir / relative).resolve()
    try:
        result_path.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ProtocolError(
            f"resume result path escapes output directory for {block['block_id']}"
        ) from error
    if not result_path.is_file() or sha256_file(result_path) != digest:
        raise ProtocolError(f"resume hash check failed for {block['block_id']}")


def run_process(
    command: Sequence[str],
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
    sampler: BASE.TelemetrySampler | None,
) -> tuple[int, int, float, str | None, list[dict[str, Any]]]:
    started = time.monotonic()
    timeout_error = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            text=True,
            shell=False,
        )
        pid = process.pid
        if sampler is not None:
            sampler.start()
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timeout_error = f"worker exceeded timeout of {timeout_s} seconds"
            process.terminate()
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=15)
        finally:
            telemetry = sampler.stop() if sampler is not None else []
    return return_code, pid, time.monotonic() - started, timeout_error, telemetry


def execute_block(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    config_sha256: str,
    manifest_config: Mapping[str, Any],
    orchestration_session_id: str,
    block: Mapping[str, Any],
    output_dir: Path,
    worker_environment: Mapping[str, str],
    smi: BASE.NvidiaSmi | None,
    baseline_used_mib: float | None,
) -> dict[str, Any]:
    block_dir = output_dir / "blocks" / str(block["block_id"])
    attempt_dir = block_dir / f"attempt_{_next_attempt(block_dir):03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = attempt_dir / "worker_result.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    telemetry_path = attempt_dir / "nvidia_smi_telemetry.jsonl"
    command = worker_command(args, profile, block, result_path)
    started_utc = utc_now()

    if smi is not None:
        if baseline_used_mib is None:
            raise ProtocolError("missing GPU idle baseline")
        idle_before = BASE.wait_for_gpu_idle(
            smi,
            baseline_used_mib=baseline_used_mib,
            memory_tolerance_mib=args.idle_memory_tolerance_mib,
            timeout_s=args.idle_timeout_s,
            poll_s=args.idle_poll_s,
            consecutive_required=args.idle_consecutive_polls,
            max_utilization=args.idle_max_utilization_percent,
        )
        sampler = BASE.TelemetrySampler(smi, args.telemetry_interval_ms / 1000.0)
    else:
        idle_before = {"skipped": True, "reason": "nvidia-smi unavailable"}
        sampler = None

    environment = os.environ.copy()
    environment.update(worker_environment)
    orchestration = expected_orchestration_environment(
        config_sha256, block, orchestration_session_id
    )
    environment.update(orchestration)
    return_code, pid, wall_s, timeout_error, telemetry = run_process(
        command,
        environment,
        stdout_path,
        stderr_path,
        args.worker_timeout_s,
        sampler,
    )
    BASE.write_jsonl(telemetry_path, telemetry)

    idle_after_error = None
    if smi is not None:
        try:
            idle_after = BASE.wait_for_gpu_idle(
                smi,
                baseline_used_mib=float(baseline_used_mib),
                memory_tolerance_mib=args.idle_memory_tolerance_mib,
                timeout_s=args.idle_timeout_s,
                poll_s=args.idle_poll_s,
                consecutive_required=args.idle_consecutive_polls,
                max_utilization=args.idle_max_utilization_percent,
            )
        except ProtocolError as error:
            idle_after = {"failed": True}
            idle_after_error = str(error)
    else:
        idle_after = {"skipped": True, "reason": "nvidia-smi unavailable"}

    relative = lambda path: str(path.relative_to(output_dir))
    execution: dict[str, Any] = {
        "status": "failed",
        "fresh_process_launched": True,
        "orchestration_session_id": orchestration_session_id,
        "worker_pid": pid,
        "backend": block["backend"],
        "seed": block["seed"],
        "pair_id": block["pair_id"],
        "pair_order": block["pair_order"],
        "slot": block["slot"],
        "command": command,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "cwd": str(ROOT),
        "return_code": return_code,
        "process_wall_seconds_including_setup": wall_s,
        "timeout_error": timeout_error,
        "idle_before": idle_before,
        "idle_after": idle_after,
        "idle_after_error": idle_after_error,
        "stdout_path": relative(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": relative(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "telemetry_path": relative(telemetry_path),
        "telemetry_sha256": sha256_file(telemetry_path),
        "telemetry_samples": len(telemetry),
    }
    errors = []
    if timeout_error:
        errors.append(timeout_error)
    if return_code != 0:
        errors.append(f"worker exited with return code {return_code}")
    if idle_after_error:
        errors.append(idle_after_error)
    if smi is not None and not any("query_error" not in item for item in telemetry):
        errors.append("no successful in-process nvidia-smi telemetry sample")
    if not result_path.is_file():
        errors.append("worker did not create result JSON")
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_worker_result(
                result,
                str(block["backend"]),
                int(block["seed"]),
                orchestration,
                expected_worker_configuration(manifest_config, block),
                manifest_config["worker_source_closure_sha256"],
            )
        except (json.JSONDecodeError, ProtocolError) as error:
            errors.append(f"worker result failed validation: {error}")
        execution["worker_result_path"] = relative(result_path)
        execution["worker_result_sha256"] = sha256_file(result_path)
    execution["errors"] = errors
    if not errors:
        execution["status"] = "completed"
    return execution


def main() -> None:
    args = parse_args()
    profile = resolved_profile(args)
    try:
        validate_args(args, profile)
        worker_environment = BASE.parse_worker_env(args.worker_env)
        if not args.worker.resolve().is_file():
            raise ProtocolError(f"sustained worker is missing: {args.worker}")
        schedule = BASE.build_schedule(
            profile["seeds"],
            int(profile["pairs_per_seed"]),
            args.token_offset,
            int(profile["seed_token_offset_stride"]),
        )
        config = build_config(args, profile, worker_environment)
        output_dir = args.output_dir.resolve()
        manifest_path, manifest = prepare_manifest(
            output_dir,
            config,
            schedule,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            manifest["dry_run_commands"] = [
                {
                    "block_id": block["block_id"],
                    "command": worker_command(
                        args,
                        profile,
                        block,
                        output_dir
                        / "blocks"
                        / block["block_id"]
                        / "attempt_001"
                        / "worker_result.json",
                    ),
                }
                for block in schedule
            ]
            manifest["status"] = "dry_run"
            BASE.atomic_write_json(manifest_path, manifest)
            print(f"Wrote sustained Williams dry-run: {manifest_path}")
            return

        smi_path = BASE.resolve_nvidia_smi(args.nvidia_smi)
        if smi_path is None and not args.allow_missing_nvidia_smi:
            raise ProtocolError("nvidia-smi not found; run fails closed")
        smi = BASE.NvidiaSmi(smi_path, args.gpu_index) if smi_path else None
        if smi is not None:
            idle_baseline = BASE.establish_idle_baseline(
                smi,
                timeout_s=args.idle_timeout_s,
                poll_s=args.idle_poll_s,
                consecutive_required=args.idle_consecutive_polls,
                max_utilization=args.idle_max_utilization_percent,
                max_initial_used_mib=args.max_initial_gpu_used_mib,
            )
            baseline_used_mib: float | None = float(
                idle_baseline["baseline_used_mib"]
            )
        else:
            idle_baseline = {"skipped": True, "reason": "nvidia-smi unavailable"}
            baseline_used_mib = None
        orchestration_session_id = (
            f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-"
            f"pid{os.getpid()}"
        )
        manifest["status"] = "running"
        manifest["run_started_utc"] = utc_now()
        manifest["nvidia_smi_path"] = str(smi_path) if smi_path else None
        manifest["initial_idle_baseline"] = idle_baseline
        manifest.setdefault("orchestration_sessions", []).append(
            {
                "session_id": orchestration_session_id,
                "started_utc": utc_now(),
                "resume": bool(args.resume),
                "rerun_failed": bool(args.rerun_failed),
            }
        )
        BASE.atomic_write_json(manifest_path, manifest)

        executions = manifest.setdefault("executions", {})
        quartets: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
        for block in schedule:
            key = (int(block["seed_index"]), int(block["quartet_index"]))
            quartets.setdefault(key, []).append(block)
        quartets_to_run: set[tuple[int, int]] = set()
        for key, quartet in quartets.items():
            if len(quartet) != 4:
                raise ProtocolError(f"Williams quartet {key} does not contain four blocks")
            prior = [executions.get(str(block["block_id"])) for block in quartet]
            for block, record in zip(quartet, prior):
                if (
                    isinstance(record, Mapping)
                    and record.get("status") == "completed"
                ):
                    validate_resume_worker_result_hash(output_dir, block, record)
            completed = [
                isinstance(record, Mapping) and record.get("status") == "completed"
                for record in prior
            ]
            if all(completed):
                sessions = {
                    str(record.get("orchestration_session_id"))
                    for record in prior
                    if isinstance(record, Mapping)
                }
                if len(sessions) != 1 or "None" in sessions:
                    raise ProtocolError(
                        f"completed quartet {key} spans orchestration sessions"
                    )
                continue
            if any(record is not None for record in prior) and not args.rerun_failed:
                raise ProtocolError(
                    f"quartet {key} is incomplete; resume with --rerun-failed "
                    "to rerun the entire ABBA/BAAB quartet"
                )
            quartets_to_run.add(key)

        for block in schedule:
            block_id = str(block["block_id"])
            quartet_key = (
                int(block["seed_index"]),
                int(block["quartet_index"]),
            )
            if quartet_key not in quartets_to_run:
                continue
            execution = execute_block(
                args,
                profile,
                str(manifest["config_sha256"]),
                config,
                orchestration_session_id,
                block,
                output_dir,
                worker_environment,
                smi,
                baseline_used_mib,
            )
            executions[block_id] = execution
            BASE.atomic_write_json(manifest_path, manifest)
            if execution["status"] != "completed":
                manifest["status"] = "failed"
                manifest["failed_block_id"] = block_id
                manifest["run_finished_utc"] = utc_now()
                BASE.atomic_write_json(manifest_path, manifest)
                raise ProtocolError(f"block {block_id} failed: {execution['errors']}")

        manifest["status"] = "completed"
        manifest["run_finished_utc"] = utc_now()
        BASE.atomic_write_json(manifest_path, manifest)
        analysis = analyze_run_directory(
            output_dir,
            manifest_path,
            bootstrap_samples=int(profile["bootstrap_samples"]),
            bootstrap_seed=args.bootstrap_seed,
        )
    except (ProtocolError, json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"sustained orchestration failed closed: {error}") from error

    primary = analysis["aggregates"]["cache_neutral"]["wall_ms"]
    interval = primary["hierarchical_seed_pair_bootstrap"]["speedup_95_ci"]
    print(
        f"Completed {len(schedule)} fresh-process blocks; sustained wall speedup "
        f"{primary['speedup_geomean']:.4f}x (95% CI "
        f"[{interval[0]:.4f}, {interval[1]:.4f}])."
    )


if __name__ == "__main__":
    main()
