#!/usr/bin/env python3
"""Run the frozen cross-GPU PageGauge timing and overhead protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts/benchmark_flashinfer_page_affine_int8.py"
OVERHEAD = ROOT / "scripts/benchmark_page_gauge_overheads.py"
SUMMARIZER = ROOT / "scripts/summarize_page_gauge_results.py"
PREPARE = ROOT / "scripts/prepare_flashinfer_page_gauge.py"
FA4_PROBE = ROOT / "scripts/probe_fa4_paged_decode.py"
TRANSFORMER = ROOT / "scripts/benchmark_page_gauge_transformer.py"
BASE_E2E = ROOT / "scripts/benchmark_e2e_transformer.py"
RUNTIME = ROOT / "scripts/page_gauge_runtime.py"
APPEND_EXTENSION = ROOT / "tests/page_gauge_append_extension.cu"
PATCH = ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch"
TESTS = (
    ROOT / "tests/test_gauge_int8_model.py",
    ROOT / "tests/test_flashinfer_page_affine_int8.py",
    ROOT / "tests/test_page_gauge_overheads.py",
    ROOT / "tests/test_page_gauge_append_extension.py",
    ROOT / "tests/test_page_gauge_transformer.py",
    ROOT / "tests/test_page_gauge_runner.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-splits", type=int, nargs="+", default=(0, 64, 128, 256, 512)
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=(20260840, 20260841, 20260842, 20260843, 20260844),
    )
    parser.add_argument("--pilot-seed", type=int, default=20260814)
    parser.add_argument("--pilot-repeats", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--cache-scrub-mib", type=int, default=256)
    parser.add_argument(
        "--fa4-policy",
        choices=("required", "record-only"),
        default="required",
        help=(
            "required creates the complete archive but exits nonzero unless the "
            "matching FA4 paged probe is supported and numerically correct"
        ),
    )
    parser.add_argument(
        "--transformer-policy",
        choices=("required", "record-only"),
        default="required",
    )
    parser.add_argument(
        "--transformer-model", default="mistralai/Mistral-7B-v0.3"
    )
    parser.add_argument("--transformer-context", type=int, default=16384)
    parser.add_argument("--transformer-decode-steps", type=int, default=16)
    parser.add_argument("--transformer-warmups", type=int, default=10)
    parser.add_argument("--transformer-repeats", type=int, default=30)
    parser.add_argument(
        "--transformer-seeds",
        type=int,
        nargs="+",
        default=(20260850, 20260851, 20260852),
    )
    parser.add_argument("--transformer-local-files-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def run_logged(
    command: list[str],
    log: Path,
    environment: dict[str, str],
    allow_failure: bool = False,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log.open("w") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode and not allow_failure:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise SystemExit(f"command failed ({process.returncode}); {log}\n{tail}")
    return process.returncode


def load_timing(path: Path, mode: str = "cache_neutral") -> tuple[float, float]:
    payload = json.loads(path.read_text())
    mode_payload = payload["timing_modes"][mode]
    timings = mode_payload["timings"]
    candidate = [name for name in timings if name != "flashinfer_fp16"]
    if len(candidate) != 1:
        raise ValueError(f"invalid candidate timing in {path}")
    return (
        timings["flashinfer_fp16"]["p50_ms"],
        timings[candidate[0]]["p50_ms"],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in vars(args).items()
    }


def benchmark_command(
    output: Path,
    seed: int,
    repeats: int,
    warmup: int,
    baseline_split: int,
    candidate_split: int,
    exact_sink: int = 0,
    lengths: str | None = None,
    cache_scrub_mib: int = 256,
) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK),
        "--representation",
        "page_gauge",
        "--exact-tail",
        "256",
        "--exact-sink",
        str(exact_sink),
        "--baseline-fixed-split-pages",
        str(baseline_split),
        "--candidate-fixed-split-pages",
        str(candidate_split),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--cache-scrub-mib",
        str(cache_scrub_mib),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    if lengths is not None:
        command.extend(("--lengths", lengths))
    return command


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if len(args.validation_seeds) < 3:
        raise SystemExit("at least three validation seeds are required")
    if any(split < 0 for split in args.pilot_splits):
        raise SystemExit("pilot splits must be non-negative")
    if args.cache_scrub_mib <= 0:
        raise SystemExit("cache scrub size must be positive")
    if args.transformer_context <= 256 or args.transformer_context % 16:
        raise SystemExit("transformer context must be page aligned and exceed 256")
    if args.transformer_decode_steps <= 0 or args.transformer_repeats <= 0:
        raise SystemExit("transformer decode steps and repeats must be positive")
    if args.transformer_decode_steps != 16:
        raise SystemExit(
            "the protocol-v2 graph-captured publication trace requires exactly "
            "16 decode steps; use the standalone transformer benchmark for other "
            "lengths"
        )
    if args.transformer_repeats % 2:
        raise SystemExit("transformer repeats must be even for balanced paired order")
    if len(args.transformer_seeds) < 3:
        raise SystemExit("at least three transformer seeds are required")
    output = args.output_dir.resolve()
    if (output / "FINAL_SUMMARY.json").exists():
        raise SystemExit(f"refusing to mix a completed run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(serializable_arguments(args), indent=2) + "\n"
    )
    source_paths = (
        Path(__file__).resolve(),
        BENCHMARK,
        OVERHEAD,
        SUMMARIZER,
        PREPARE,
        FA4_PROBE,
        TRANSFORMER,
        BASE_E2E,
        RUNTIME,
        APPEND_EXTENSION,
        PATCH,
    )
    (output / "RUNNER_SOURCE_SHA256.json").write_text(
        json.dumps(
            {
                str(path.relative_to(ROOT)): sha256(path)
                for path in source_paths
            },
            indent=2,
        )
        + "\n"
    )
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.setdefault("TORCH_EXTENSIONS_DIR", str(output / "torch_extensions"))
    environment.setdefault("MAX_JOBS", "4")

    import flashinfer
    import triton

    manifest: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "pilot_seed": args.pilot_seed,
        "validation_seeds": args.validation_seeds,
        "pilot_splits": args.pilot_splits,
    }
    (output / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run_logged(
        ["nvidia-smi", "-q"], output / "nvidia_smi.log", environment
    )
    run_logged(
        [sys.executable, str(PREPARE)], output / "prepare_patch.log", environment
    )
    if not args.skip_tests:
        run_logged(
            [sys.executable, "-m", "pytest", "-q", *map(str, TESTS)],
            output / "tests.log",
            environment,
        )

    pilot_dir = output / "pilot"
    pilot_results: list[tuple[int, Path, float, float]] = []
    for split in args.pilot_splits:
        result = pilot_dir / f"split_{split}.json"
        run_logged(
            benchmark_command(
                result,
                args.pilot_seed,
                args.pilot_repeats,
                min(args.warmup, 20),
                split,
                split,
                cache_scrub_mib=args.cache_scrub_mib,
            ),
            result.with_suffix(".log"),
            environment,
        )
        baseline_ms, candidate_ms = load_timing(result)
        pilot_results.append((split, result, baseline_ms, candidate_ms))
    baseline_split = min(pilot_results, key=lambda record: record[2])[0]
    candidate_split = min(pilot_results, key=lambda record: record[3])[0]
    selection = {
        "source": "independent pilot seed with explicit cache scrub",
        "selection_timing_mode": "cache_neutral",
        "pilot_seed": args.pilot_seed,
        "baseline_fixed_split_pages": baseline_split,
        "candidate_fixed_split_pages": candidate_split,
        "rows": [
            {
                "split_pages": split,
                "result": str(path.resolve()),
                "baseline_p50_ms": baseline_ms,
                "candidate_p50_ms": candidate_ms,
            }
            for split, path, baseline_ms, candidate_ms in pilot_results
        ],
    }
    (output / "FROZEN_SELECTION.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )

    validation_dir = output / "validation"
    validation_results: list[Path] = []
    for seed in args.validation_seeds:
        result = validation_dir / f"seed_{seed}.json"
        run_logged(
            benchmark_command(
                result,
                seed,
                args.repeats,
                args.warmup,
                baseline_split,
                candidate_split,
                cache_scrub_mib=args.cache_scrub_mib,
            ),
            result.with_suffix(".log"),
            environment,
        )
        validation_results.append(result)

    overhead_result = output / "page_finalization_overhead.json"
    run_logged(
        [
            sys.executable,
            str(OVERHEAD),
            "--batch",
            "8",
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--cache-scrub-mib",
            str(args.cache_scrub_mib),
            "--seed",
            str(args.validation_seeds[0]),
            "--output",
            str(overhead_result),
        ],
        overhead_result.with_suffix(".log"),
        environment,
    )
    summary_result = output / "FINAL_SUMMARY.json"
    run_logged(
        [
            sys.executable,
            str(SUMMARIZER),
            "--benchmarks",
            *map(str, validation_results),
            "--overheads",
            str(overhead_result),
            "--output",
            str(summary_result),
        ],
        output / "summary.log",
        environment,
    )

    # A matching B1/49K suite permits a direct comparison with FA4's paged API
    # on architectures where flash-attn-4 supports the physical GPU.
    b1_dir = output / "b1_49k"
    b1_pilot_results: list[tuple[int, Path, float, float]] = []
    for split in args.pilot_splits:
        result = b1_dir / "pilot" / f"split_{split}.json"
        run_logged(
            benchmark_command(
                result,
                args.pilot_seed,
                args.pilot_repeats,
                min(args.warmup, 20),
                split,
                split,
                lengths="49152",
                cache_scrub_mib=args.cache_scrub_mib,
            ),
            result.with_suffix(".log"),
            environment,
        )
        baseline_ms, candidate_ms = load_timing(result)
        b1_pilot_results.append((split, result, baseline_ms, candidate_ms))
    b1_baseline_split = min(b1_pilot_results, key=lambda record: record[2])[0]
    b1_candidate_split = min(b1_pilot_results, key=lambda record: record[3])[0]
    b1_selection = {
        "source": "independent pilot seed with explicit cache scrub",
        "selection_timing_mode": "cache_neutral",
        "pilot_seed": args.pilot_seed,
        "baseline_fixed_split_pages": b1_baseline_split,
        "candidate_fixed_split_pages": b1_candidate_split,
        "rows": [
            {
                "split_pages": split,
                "result": str(path.resolve()),
                "baseline_p50_ms": baseline_ms,
                "candidate_p50_ms": candidate_ms,
            }
            for split, path, baseline_ms, candidate_ms in b1_pilot_results
        ],
    }
    (b1_dir / "FROZEN_SELECTION.json").write_text(
        json.dumps(b1_selection, indent=2) + "\n"
    )
    b1_validation_results: list[Path] = []
    for seed in args.validation_seeds:
        result = b1_dir / "validation" / f"seed_{seed}.json"
        run_logged(
            benchmark_command(
                result,
                seed,
                args.repeats,
                args.warmup,
                b1_baseline_split,
                b1_candidate_split,
                lengths="49152",
                cache_scrub_mib=args.cache_scrub_mib,
            ),
            result.with_suffix(".log"),
            environment,
        )
        b1_validation_results.append(result)
    b1_overhead_result = b1_dir / "page_finalization_overhead.json"
    run_logged(
        [
            sys.executable,
            str(OVERHEAD),
            "--batch",
            "1",
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--cache-scrub-mib",
            str(args.cache_scrub_mib),
            "--seed",
            str(args.validation_seeds[0]),
            "--output",
            str(b1_overhead_result),
        ],
        b1_overhead_result.with_suffix(".log"),
        environment,
    )
    b1_summary = b1_dir / "B1_SUMMARY.json"
    run_logged(
        [
            sys.executable,
            str(SUMMARIZER),
            "--benchmarks",
            *map(str, b1_validation_results),
            "--overheads",
            str(b1_overhead_result),
            "--output",
            str(b1_summary),
        ],
        b1_dir / "summary.log",
        environment,
    )

    fa4_result = b1_dir / "flashattention4_paged_probe.json"
    fa4_exit_code = run_logged(
        [
            sys.executable,
            str(FA4_PROBE),
            "--output",
            str(fa4_result),
            "--batch-size",
            "1",
            "--kv-length",
            "49152",
            "--page-size",
            "16",
            "--q-heads",
            "32",
            "--kv-heads",
            "8",
            "--head-dim",
            "128",
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--cache-scrub-mib",
            str(args.cache_scrub_mib),
            "--seed",
            str(args.validation_seeds[0]),
        ],
        b1_dir / "flashattention4_paged_probe.log",
        environment,
        allow_failure=True,
    )
    (b1_dir / "flashattention4_paged_probe_exit_code.txt").write_text(
        f"{fa4_exit_code}\n"
    )
    fa4_payload = json.loads(fa4_result.read_text())
    fa4_ok = bool(
        fa4_payload.get("status") == "supported"
        and fa4_payload.get("supported")
        and fa4_payload.get("correctness", {}).get("passed")
    )
    b1_payload = json.loads(b1_summary.read_text())
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "page_gauge_vs_flashattention4_paged_decode",
        "fa4_policy": args.fa4_policy,
        "status": "complete" if fa4_ok else "failed",
        "fa4_probe_status": fa4_payload.get("status"),
        "fa4_probe": str(fa4_result.resolve()),
        "page_gauge_summary": str(b1_summary.resolve()),
    }
    if fa4_ok:
        comparison["timing_modes"] = {}
        for mode in ("cache_neutral", "cache_hot"):
            page_gauge_ms = statistics.median(
                record["timing_modes"][mode]["candidate_inclusive_p50_ms"]
                for record in b1_payload["runs"]
            )
            flashinfer_ms = statistics.median(
                record["timing_modes"][mode]["baseline_p50_ms"]
                for record in b1_payload["runs"]
            )
            fa4_ms = fa4_payload["timing_modes"][mode]["median"]
            comparison["timing_modes"][mode] = {
                "flashinfer_fp16_p50_ms": flashinfer_ms,
                "page_gauge_inclusive_p50_ms": page_gauge_ms,
                "flashattention4_p50_ms": fa4_ms,
                "page_gauge_speedup_over_flashattention4": fa4_ms
                / page_gauge_ms,
                "flashinfer_speedup_over_flashattention4": fa4_ms
                / flashinfer_ms,
            }
    else:
        comparison["failure"] = {
            "exit_code": fa4_exit_code,
            "error_type": fa4_payload.get("error_type"),
            "error": fa4_payload.get("error"),
            "required_action": (
                "install a working flash-attn-4 build with the paged varlen API "
                "for this GPU, then rerun; no modern-baseline claim is valid"
            ),
        }
    fa4_comparison = b1_dir / "FA4_COMPARISON.json"
    fa4_comparison.write_text(json.dumps(comparison, indent=2) + "\n")

    # Select schedulers at the actual full-transformer context instead of
    # reusing the 49K choice, then measure every decoder layer and wall tokens/s.
    transformer_dir = output / "transformer"
    transformer_pilot_results: list[tuple[int, Path, float, float]] = []
    for split in args.pilot_splits:
        result = transformer_dir / "pilot" / f"split_{split}.json"
        run_logged(
            benchmark_command(
                result,
                args.pilot_seed + 1,
                min(args.pilot_repeats, 40),
                min(args.warmup, 20),
                split,
                split,
                lengths=str(args.transformer_context),
                cache_scrub_mib=args.cache_scrub_mib,
            ),
            result.with_suffix(".log"),
            environment,
        )
        baseline_ms, candidate_ms = load_timing(result)
        transformer_pilot_results.append(
            (split, result, baseline_ms, candidate_ms)
        )
    transformer_baseline_split = min(
        transformer_pilot_results, key=lambda record: record[2]
    )[0]
    transformer_candidate_split = min(
        transformer_pilot_results, key=lambda record: record[3]
    )[0]
    transformer_selection = {
        "source": "independent cache-neutral pilot at full-transformer context",
        "pilot_seed": args.pilot_seed + 1,
        "context": args.transformer_context,
        "baseline_fixed_split_pages": transformer_baseline_split,
        "candidate_fixed_split_pages": transformer_candidate_split,
        "rows": [
            {
                "split_pages": split,
                "result": str(path.resolve()),
                "baseline_cache_neutral_p50_ms": baseline_ms,
                "candidate_cache_neutral_p50_ms": candidate_ms,
            }
            for split, path, baseline_ms, candidate_ms in transformer_pilot_results
        ],
    }
    transformer_dir.mkdir(parents=True, exist_ok=True)
    (transformer_dir / "FROZEN_SELECTION.json").write_text(
        json.dumps(transformer_selection, indent=2) + "\n"
    )
    transformer_result = transformer_dir / "FULL_MODEL.json"
    transformer_command = [
        sys.executable,
        str(TRANSFORMER),
        "--model",
        args.transformer_model,
        "--context",
        str(args.transformer_context),
        "--decode-steps",
        str(args.transformer_decode_steps),
        "--baseline-split-pages",
        str(transformer_baseline_split),
        "--candidate-split-pages",
        str(transformer_candidate_split),
        "--warmups",
        str(args.transformer_warmups),
        "--repeats",
        str(args.transformer_repeats),
        "--layer-profile-repeats",
        str(args.transformer_repeats),
        "--cache-scrub-mib",
        str(args.cache_scrub_mib),
        "--seeds",
        *map(str, args.transformer_seeds),
        "--output",
        str(transformer_result),
    ]
    if args.transformer_local_files_only:
        transformer_command.append("--local-files-only")
    transformer_exit_code = run_logged(
        transformer_command,
        transformer_dir / "full_model.log",
        environment,
        allow_failure=True,
    )
    (transformer_dir / "full_model_exit_code.txt").write_text(
        f"{transformer_exit_code}\n"
    )
    transformer_payload = (
        json.loads(transformer_result.read_text())
        if transformer_result.is_file()
        else None
    )
    transformer_ok = bool(
        transformer_exit_code == 0
        and transformer_payload is not None
        and transformer_payload.get("aggregate", {}).get("correctness_passed")
    )
    systems_status = {
        "schema_version": 1,
        "fa4_policy": args.fa4_policy,
        "fa4_complete": fa4_ok,
        "transformer_policy": args.transformer_policy,
        "transformer_complete": transformer_ok,
        "transformer_exit_code": transformer_exit_code,
        "publication_complete": fa4_ok and transformer_ok,
    }
    (output / "SYSTEMS_VALIDATION_STATUS.json").write_text(
        json.dumps(systems_status, indent=2) + "\n"
    )

    sink_result = output / "decode_sink16_ablation.json"
    run_logged(
        benchmark_command(
            sink_result,
            args.validation_seeds[0] + 100,
            args.repeats,
            args.warmup,
            baseline_split,
            candidate_split,
            exact_sink=16,
            cache_scrub_mib=args.cache_scrub_mib,
        ),
        sink_result.with_suffix(".log"),
        environment,
    )

    checksums = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and "torch_extensions" not in path.parts
    }
    (output / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
    archive = Path(shutil.make_archive(str(output), "zip", output.parent, output.name))
    archive_digest = sha256(archive)
    print("mixed_B8_summary")
    print(json.dumps(json.loads(summary_result.read_text())["aggregate"], indent=2))
    print("B1_49K_summary")
    print(json.dumps(json.loads(b1_summary.read_text())["aggregate"], indent=2))
    if fa4_result.is_file():
        print(
            "FA4_paged_probe",
            fa4_payload.get("status"),
            fa4_payload.get("latency_ms", {}).get("median"),
        )
    print(f"archive={archive}")
    print(f"sha256={archive_digest}")
    required_failures = []
    if args.fa4_policy == "required" and not fa4_ok:
        required_failures.append("FA4 paged comparison")
    if args.transformer_policy == "required" and not transformer_ok:
        required_failures.append("full-transformer validation")
    if required_failures:
        raise SystemExit(
            "required systems checks failed: "
            + ", ".join(required_failures)
            + "; the result archive was preserved with explicit diagnostics"
        )


if __name__ == "__main__":
    main()
