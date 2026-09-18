#!/usr/bin/env python3
"""Run publication timing with one GPU backend per fresh worker process.

This driver owns treatment order, process isolation, idle gating, system-level
telemetry, immutable provenance, and the paired analysis.  The worker owns only
one backend allocation and emits raw CUDA-event and wall-clock latency samples.

No shell is used to launch workers.  A completed worker process must disappear
and GPU memory/utilization must return to the recorded idle envelope before the
next Williams-design block begins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from analyze_backend_exclusive import (
    ProtocolError,
    analyze_run_directory,
    expected_orchestration_environment,
    sha256_file,
    validate_worker_result,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
BASELINE = "flashinfer_fp16"
CANDIDATE = "page_gauge"
SYMBOL_TO_BACKEND = {"A": BASELINE, "B": CANDIDATE}
WILLIAMS_SEQUENCES = ("ABBA", "BAAB")
DEFAULT_WORKER = ROOT / "diagnostics/benchmark_backend_exclusive.py"
DEFAULT_WIKITEXT = ROOT / "data/wikitext-2-raw-v1.zip"
MANIFEST_NAME = "orchestration_manifest.json"

# These values are part of the paired-block identity written to the manifest.
# Letting a trailing worker-extra argument override one would make the command
# disagree with the recorded protocol configuration (argparse uses the last
# duplicate option). Keep escape-hatch arguments for genuinely additional
# worker controls only.
CONTROLLED_WORKER_OPTIONS = frozenset(
    {
        "--backend",
        "--model",
        "--batch-size",
        "--context",
        "--decode-steps",
        "--exact-tail",
        "--prefill-chunk-tokens",
        "--baseline-split-pages",
        "--candidate-split-pages",
        "--tail-attention",
        "--cuda-graph-scope",
        "--seed",
        "--token-source",
        "--wikitext-zip",
        "--wikitext-member",
        "--token-offset",
        "--token-stride",
        "--warmups",
        "--repeats",
        "--cache-scrub-mib",
        "--output",
    }
)

IDLE_QUERY_FIELDS = (
    "index",
    "uuid",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
)
RICH_QUERY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.reserved",
    "memory.used",
    "memory.free",
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("pilot", "publication"),
        default="pilot",
        help="pilot defaults to 1 seed x 2 pairs, w3/r4; publication scales up",
    )
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
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--exact-tail", type=int, default=256)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--baseline-split-pages", type=int, default=256)
    parser.add_argument("--candidate-split-pages", type=int, default=128)
    parser.add_argument(
        "--tail-attention",
        choices=("flashinfer_merge", "fused_kernel", "heterogeneous_fa2"),
        default="flashinfer_merge",
    )
    parser.add_argument(
        "--cuda-graph-scope",
        choices=("attention", "decoder_layer"),
        default="attention",
        help="matched worker graph boundary; decoder_layer is opt-in",
    )
    parser.add_argument(
        "--token-source", choices=("wikitext2", "random"), default="wikitext2"
    )
    parser.add_argument("--wikitext-zip", type=Path, default=DEFAULT_WIKITEXT)
    parser.add_argument(
        "--wikitext-member", default="wikitext-2-raw/wiki.train.raw"
    )
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--token-stride", type=int, default=0)
    parser.add_argument(
        "--seed-token-offset-stride",
        type=int,
        help=(
            "corpus offset increment between seeds; publication default uses "
            "disjoint B*(context+decode) windows"
        ),
    )
    parser.add_argument("--cache-scrub-mib", type=int, default=256)

    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", type=Path)
    parser.add_argument("--idle-timeout-s", type=float, default=300.0)
    parser.add_argument("--idle-poll-s", type=float, default=1.0)
    parser.add_argument("--idle-consecutive-polls", type=int, default=3)
    parser.add_argument("--idle-max-utilization-percent", type=float, default=5.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=float, default=256.0)
    parser.add_argument("--max-initial-gpu-used-mib", type=float, default=4096.0)
    parser.add_argument("--telemetry-interval-ms", type=int, default=500)
    parser.add_argument("--worker-timeout-s", type=float, default=7200.0)
    parser.add_argument(
        "--allow-missing-nvidia-smi",
        action="store_true",
        help="diagnostic pilot escape hatch; forbidden for publication profile",
    )

    parser.add_argument(
        "--worker-extra-arg",
        action="append",
        default=[],
        help="append one literal worker argv token; repeat for multiple tokens",
    )
    parser.add_argument(
        "--worker-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="explicit environment override recorded in the manifest",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the deterministic schedule and commands without querying or using CUDA",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolved_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile == "pilot":
        defaults = {
            "seeds": [20260861],
            "pairs_per_seed": 2,
            "warmups": 3,
            "repeats": 4,
            "bootstrap_samples": 2_000,
            "seed_token_offset_stride": 0,
        }
    else:
        defaults = {
            "seeds": [20260861, 20260862, 20260863, 20260864],
            "pairs_per_seed": 4,
            "warmups": 10,
            "repeats": 30,
            "bootstrap_samples": 20_000,
            "seed_token_offset_stride": (
                args.batch_size * (args.context + args.decode_steps)
            ),
        }
    return {
        "profile": args.profile,
        "seeds": list(args.seeds if args.seeds is not None else defaults["seeds"]),
        "pairs_per_seed": (
            args.pairs_per_seed
            if args.pairs_per_seed is not None
            else defaults["pairs_per_seed"]
        ),
        "warmups": args.warmups if args.warmups is not None else defaults["warmups"],
        "repeats": args.repeats if args.repeats is not None else defaults["repeats"],
        "bootstrap_samples": (
            args.bootstrap_samples
            if args.bootstrap_samples is not None
            else defaults["bootstrap_samples"]
        ),
        "seed_token_offset_stride": (
            args.seed_token_offset_stride
            if args.seed_token_offset_stride is not None
            else defaults["seed_token_offset_stride"]
        ),
    }


def validate_args(args: argparse.Namespace, profile: Mapping[str, Any]) -> None:
    if not profile["seeds"] or len(set(profile["seeds"])) != len(profile["seeds"]):
        raise ProtocolError("seeds must be non-empty and unique")
    if int(profile["pairs_per_seed"]) <= 0 or int(profile["pairs_per_seed"]) % 2:
        raise ProtocolError("pairs per seed must be a positive even number")
    if int(profile["warmups"]) < 0:
        raise ProtocolError("warmups cannot be negative")
    if int(profile["repeats"]) <= 0 or int(profile["repeats"]) % 2:
        raise ProtocolError("repeats must be a positive even number")
    if int(profile["bootstrap_samples"]) <= 0:
        raise ProtocolError("bootstrap samples must be positive")
    if int(profile["seed_token_offset_stride"]) < 0 or args.token_offset < 0:
        raise ProtocolError("token offsets cannot be negative")
    positive_ints = {
        "batch size": args.batch_size,
        "context": args.context,
        "decode steps": args.decode_steps,
        "exact tail": args.exact_tail,
        "prefill chunk": args.prefill_chunk_tokens,
        "cache scrub MiB": args.cache_scrub_mib,
        "idle consecutive polls": args.idle_consecutive_polls,
        "telemetry interval ms": args.telemetry_interval_ms,
    }
    for label, value in positive_ints.items():
        if value <= 0:
            raise ProtocolError(f"{label} must be positive")
    if args.context <= args.exact_tail:
        raise ProtocolError("context must exceed exact-tail length")
    if args.idle_timeout_s <= 0 or args.idle_poll_s <= 0 or args.worker_timeout_s <= 0:
        raise ProtocolError("idle/worker timeouts and idle poll interval must be positive")
    if not 0 <= args.idle_max_utilization_percent <= 100:
        raise ProtocolError("idle utilization threshold must be in [0,100]")
    if args.idle_memory_tolerance_mib < 0 or args.max_initial_gpu_used_mib <= 0:
        raise ProtocolError("memory gate values are invalid")
    if args.profile == "publication" and args.allow_missing_nvidia_smi:
        raise ProtocolError("publication profile cannot allow missing nvidia-smi")
    if args.rerun_failed and not args.resume:
        raise ProtocolError("--rerun-failed requires --resume")
    for token in args.worker_extra_arg:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if any(controlled.startswith(option) for controlled in CONTROLLED_WORKER_OPTIONS):
            raise ProtocolError(
                f"worker-extra-arg cannot override controlled option {option!r}"
            )


def parse_worker_env(items: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ProtocolError(f"worker environment entry lacks '=': {item!r}")
        name, value = item.split("=", 1)
        if not name or "\x00" in name or "\x00" in value:
            raise ProtocolError(f"invalid worker environment entry: {item!r}")
        result[name] = value
    return result


def build_schedule(
    seeds: Sequence[int],
    pairs_per_seed: int,
    base_token_offset: int,
    seed_token_offset_stride: int,
) -> list[dict[str, Any]]:
    """Return adjacent pairs in balanced Williams ABBA/BAAB quartets."""

    if pairs_per_seed <= 0 or pairs_per_seed % 2:
        raise ValueError("pairs_per_seed must be positive and even")
    schedule: list[dict[str, Any]] = []
    chronological_index = 0
    for seed_index, seed in enumerate(seeds):
        token_offset = base_token_offset + seed_index * seed_token_offset_stride
        for pair_index in range(pairs_per_seed):
            quartet_index = pair_index // 2
            sequence_index = (seed_index + quartet_index) % len(WILLIAMS_SEQUENCES)
            sequence = WILLIAMS_SEQUENCES[sequence_index]
            within_quartet = pair_index % 2
            symbols = sequence[2 * within_quartet : 2 * within_quartet + 2]
            pair_id = f"seed{seed_index:03d}_pair{pair_index:03d}"
            for slot, symbol in enumerate(symbols):
                backend = SYMBOL_TO_BACKEND[symbol]
                block_id = (
                    f"block{chronological_index:04d}_seed{seed_index:03d}_"
                    f"pair{pair_index:03d}_slot{slot}_{symbol}"
                )
                schedule.append(
                    {
                        "block_id": block_id,
                        "chronological_index": chronological_index,
                        "seed": int(seed),
                        "seed_index": seed_index,
                        "token_offset": token_offset,
                        "pair_id": pair_id,
                        "pair_index": pair_index,
                        "quartet_index": quartet_index,
                        "williams_sequence": sequence,
                        "pair_order": symbols,
                        "slot": slot,
                        "symbol": symbol,
                        "backend": backend,
                    }
                )
                chronological_index += 1
    return schedule


def resolve_nvidia_smi(explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    discovered = shutil.which("nvidia-smi")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        [
            Path("/usr/lib/wsl/lib/nvidia-smi"),
            Path("C:/Windows/System32/nvidia-smi.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        converted = float(stripped)
    except ValueError:
        return stripped
    return int(converted) if converted.is_integer() else converted


class NvidiaSmi:
    def __init__(self, executable: Path, gpu_index: int) -> None:
        self.executable = executable
        self.gpu_index = gpu_index

    def query(self, fields: Sequence[str]) -> dict[str, Any]:
        command = [
            str(self.executable),
            f"--id={self.gpu_index}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if completed.returncode != 0:
            raise ProtocolError(
                "nvidia-smi query failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
        if len(rows) != 1 or len(rows[0]) != len(fields):
            raise ProtocolError(
                f"nvidia-smi returned {len(rows)} rows for GPU {self.gpu_index}: "
                f"{completed.stdout!r}"
            )
        snapshot = {
            field.replace(".", "_"): _parse_scalar(value)
            for field, value in zip(fields, rows[0], strict=True)
        }
        snapshot["orchestrator_utc"] = utc_now()
        snapshot["orchestrator_monotonic_s"] = time.monotonic()
        return snapshot

    def idle_snapshot(self) -> dict[str, Any]:
        return self.query(IDLE_QUERY_FIELDS)

    def telemetry_snapshot(self) -> dict[str, Any]:
        try:
            return self.query(RICH_QUERY_FIELDS)
        except ProtocolError as rich_error:
            fallback = self.query(IDLE_QUERY_FIELDS)
            fallback["rich_query_error"] = str(rich_error)
            return fallback


def _numeric(snapshot: Mapping[str, Any], key: str) -> float:
    value = snapshot.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"nvidia-smi snapshot lacks numeric {key}: {snapshot}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ProtocolError(f"nvidia-smi {key} is not finite")
    return converted


def establish_idle_baseline(
    smi: NvidiaSmi,
    *,
    timeout_s: float,
    poll_s: float,
    consecutive_required: int,
    max_utilization: float,
    max_initial_used_mib: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    consecutive: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        snapshot = smi.idle_snapshot()
        history.append(snapshot)
        idle = (
            _numeric(snapshot, "utilization_gpu") <= max_utilization
            and _numeric(snapshot, "memory_used") <= max_initial_used_mib
        )
        if idle:
            consecutive.append(snapshot)
            if len(consecutive) >= consecutive_required:
                return {
                    "baseline_used_mib": min(
                        _numeric(item, "memory_used") for item in consecutive
                    ),
                    "accepted_snapshots": consecutive,
                    "poll_count": len(history),
                    "criteria": {
                        "max_utilization_percent": max_utilization,
                        "max_initial_used_mib": max_initial_used_mib,
                        "consecutive_polls": consecutive_required,
                    },
                }
        else:
            consecutive = []
        time.sleep(poll_s)
    last = history[-1] if history else None
    raise ProtocolError(
        f"GPU did not reach initial idle gate within {timeout_s}s; last={last}"
    )


def wait_for_gpu_idle(
    smi: NvidiaSmi,
    *,
    baseline_used_mib: float,
    memory_tolerance_mib: float,
    timeout_s: float,
    poll_s: float,
    consecutive_required: int,
    max_utilization: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    consecutive: list[dict[str, Any]] = []
    history_count = 0
    memory_limit = baseline_used_mib + memory_tolerance_mib
    while time.monotonic() < deadline:
        snapshot = smi.idle_snapshot()
        history_count += 1
        idle = (
            _numeric(snapshot, "utilization_gpu") <= max_utilization
            and _numeric(snapshot, "memory_used") <= memory_limit
        )
        if idle:
            consecutive.append(snapshot)
            if len(consecutive) >= consecutive_required:
                return {
                    "accepted_snapshots": consecutive,
                    "poll_count": history_count,
                    "criteria": {
                        "baseline_used_mib": baseline_used_mib,
                        "memory_tolerance_mib": memory_tolerance_mib,
                        "memory_limit_mib": memory_limit,
                        "max_utilization_percent": max_utilization,
                        "consecutive_polls": consecutive_required,
                    },
                }
        else:
            consecutive = []
        time.sleep(poll_s)
    raise ProtocolError(
        f"GPU did not return to idle envelope within {timeout_s}s; "
        f"memory limit={memory_limit} MiB"
    )


class TelemetrySampler:
    def __init__(self, smi: NvidiaSmi, interval_s: float) -> None:
        self.smi = smi
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.records: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.records.append(self.smi.telemetry_snapshot())
            except Exception as error:  # telemetry must preserve transient failures
                self.records.append(
                    {
                        "orchestrator_utc": utc_now(),
                        "orchestrator_monotonic_s": time.monotonic(),
                        "query_error": f"{type(error).__name__}: {error}",
                    }
                )
            self.stop_event.wait(self.interval_s)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_s * 3.0))
        return self.records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def worker_command(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    block: Mapping[str, Any],
    output_path: Path,
) -> list[str]:
    command = [
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
        "--prefill-chunk-tokens",
        str(args.prefill_chunk_tokens),
        "--baseline-split-pages",
        str(args.baseline_split_pages),
        "--candidate-split-pages",
        str(args.candidate_split_pages),
        "--tail-attention",
        args.tail_attention,
        "--cuda-graph-scope",
        args.cuda_graph_scope,
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
        "--warmups",
        str(profile["warmups"]),
        "--repeats",
        str(profile["repeats"]),
        "--cache-scrub-mib",
        str(args.cache_scrub_mib),
        "--output",
        str(output_path.resolve()),
    ]
    command.extend(args.worker_extra_arg)
    return command


def build_config(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    worker_env: Mapping[str, str],
) -> dict[str, Any]:
    worker = args.worker.resolve()
    analyzer = ROOT / "diagnostics/analyze_backend_exclusive.py"
    orchestrator = Path(__file__).resolve()
    source_hashes = {
        str(worker.relative_to(ROOT)): sha256_file(worker),
        str(analyzer.relative_to(ROOT)): sha256_file(analyzer),
        str(orchestrator.relative_to(ROOT)): sha256_file(orchestrator),
    }
    return {
        "profile": dict(profile),
        "backend_symbols": dict(SYMBOL_TO_BACKEND),
        "williams_sequences": list(WILLIAMS_SEQUENCES),
        "fresh_process_per_backend_block": True,
        "model": args.model,
        "batch_size": args.batch_size,
        "context": args.context,
        "decode_steps": args.decode_steps,
        "exact_tail": args.exact_tail,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "baseline_split_pages": args.baseline_split_pages,
        "candidate_split_pages": args.candidate_split_pages,
        "tail_attention": args.tail_attention,
        "cuda_graph_scope": args.cuda_graph_scope,
        "token_source": args.token_source,
        "wikitext_zip": str(args.wikitext_zip.resolve()),
        "wikitext_member": args.wikitext_member,
        "base_token_offset": args.token_offset,
        "token_stride": args.token_stride,
        "cache_scrub_mib": args.cache_scrub_mib,
        "python_executable": args.python,
        "worker": str(worker),
        "worker_extra_args": list(args.worker_extra_arg),
        "worker_environment_overrides": dict(sorted(worker_env.items())),
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
        "source_sha256": source_hashes,
    }


def initial_manifest(
    config: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]], dry_run: bool
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "page_gauge_backend_exclusive_publication_orchestration",
        "claim_scope": (
            "fresh-process, one-backend-at-a-time fixed-batch decode timing with "
            "Williams-order paired blocks"
        ),
        "status": "dry_run" if dry_run else "initialized",
        "created_utc": utc_now(),
        "config": dict(config),
        "config_sha256": canonical_json_sha256(config),
        "schedule": list(schedule),
        "executions": {},
        "analysis_expected_path": "publication_analysis.json",
        "raw_samples_expected_path": "raw_latency_samples.csv",
        "paired_records_expected_path": "paired_log_speedups.csv",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": str(Path.cwd().resolve()),
        },
        "invocation": {
            "argv": list(sys.argv),
            "orchestrator_pid": os.getpid(),
        },
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
    manifest_path = output_dir / MANIFEST_NAME
    if resume:
        if not manifest_path.is_file():
            raise ProtocolError("--resume requested but orchestration manifest is missing")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ProtocolError("existing orchestration manifest is not an object")
        expected = canonical_json_sha256(config)
        if value.get("config_sha256") != expected:
            raise ProtocolError(
                "resume configuration differs from the existing manifest; "
                "use a new output directory"
            )
        if value.get("schedule") != list(schedule):
            raise ProtocolError("resume schedule differs from the existing manifest")
        manifest = value
    else:
        existing = [path for path in output_dir.iterdir()]
        if existing:
            raise ProtocolError(
                f"output directory is not empty: {output_dir}; use --resume or a new path"
            )
        manifest = initial_manifest(config, schedule, dry_run)
        atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest


def _next_attempt(block_dir: Path) -> int:
    attempts: list[int] = []
    if block_dir.exists():
        for child in block_dir.iterdir():
            if child.is_dir() and child.name.startswith("attempt_"):
                try:
                    attempts.append(int(child.name.removeprefix("attempt_")))
                except ValueError:
                    continue
    return max(attempts, default=0) + 1


def _run_worker_process(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
    sampler: TelemetrySampler | None,
) -> tuple[int, float, str | None, list[dict[str, Any]]]:
    start = time.monotonic()
    timed_out: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            text=True,
            shell=False,
        )
        if sampler is not None:
            sampler.start()
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = f"worker exceeded timeout of {timeout_s} seconds"
            process.terminate()
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=15)
        finally:
            telemetry = sampler.stop() if sampler is not None else []
    return return_code, time.monotonic() - start, timed_out, telemetry


def execute_block(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    config_sha256: str,
    block: Mapping[str, Any],
    output_dir: Path,
    worker_env_overrides: Mapping[str, str],
    smi: NvidiaSmi | None,
    idle_baseline_used_mib: float | None,
) -> dict[str, Any]:
    block_dir = output_dir / "blocks" / str(block["block_id"])
    attempt_number = _next_attempt(block_dir)
    attempt_dir = block_dir / f"attempt_{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = attempt_dir / "worker_result.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    telemetry_path = attempt_dir / "nvidia_smi_telemetry.jsonl"
    command = worker_command(args, profile, block, result_path)

    if smi is not None:
        if idle_baseline_used_mib is None:
            raise ProtocolError("internal error: missing idle baseline")
        idle_before = wait_for_gpu_idle(
            smi,
            baseline_used_mib=idle_baseline_used_mib,
            memory_tolerance_mib=args.idle_memory_tolerance_mib,
            timeout_s=args.idle_timeout_s,
            poll_s=args.idle_poll_s,
            consecutive_required=args.idle_consecutive_polls,
            max_utilization=args.idle_max_utilization_percent,
        )
        sampler = TelemetrySampler(smi, args.telemetry_interval_ms / 1000.0)
    else:
        idle_before = {"skipped": True, "reason": "nvidia-smi unavailable"}
        sampler = None

    environment = os.environ.copy()
    environment.update(worker_env_overrides)
    orchestration_environment = expected_orchestration_environment(
        config_sha256, block
    )
    environment.update(orchestration_environment)
    started_utc = utc_now()
    return_code, process_wall_s, timeout_error, telemetry = _run_worker_process(
        command,
        ROOT,
        environment,
        stdout_path,
        stderr_path,
        args.worker_timeout_s,
        sampler,
    )
    write_jsonl(telemetry_path, telemetry)

    idle_after: Mapping[str, Any]
    idle_after_error: str | None = None
    if smi is not None:
        try:
            idle_after = wait_for_gpu_idle(
                smi,
                baseline_used_mib=float(idle_baseline_used_mib),
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
        "attempt": attempt_number,
        "backend": block["backend"],
        "seed": block["seed"],
        "pair_id": block["pair_id"],
        "pair_order": block["pair_order"],
        "slot": block["slot"],
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "command": command,
        "cwd": str(ROOT),
        "worker_environment_overrides": dict(sorted(worker_env_overrides.items())),
        "return_code": return_code,
        "process_wall_seconds_including_setup": process_wall_s,
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
        "telemetry_successful_samples": sum(
            1 for item in telemetry if "query_error" not in item
        ),
    }
    errors: list[str] = []
    if timeout_error is not None:
        errors.append(timeout_error)
    if return_code != 0:
        errors.append(f"worker exited with return code {return_code}")
    if idle_after_error is not None:
        errors.append(idle_after_error)
    if smi is not None and not any("query_error" not in item for item in telemetry):
        errors.append("no successful in-process nvidia-smi telemetry samples")
    if not result_path.is_file():
        errors.append("worker did not create its result JSON")
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, Mapping):
                raise ProtocolError("worker result root is not an object")
            validate_worker_result(
                result,
                str(block["backend"]),
                int(block["seed"]),
                orchestration_environment,
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
        worker_env = parse_worker_env(args.worker_env)
        if not args.worker.resolve().is_file():
            raise ProtocolError(f"backend-exclusive worker is missing: {args.worker}")
        schedule = build_schedule(
            profile["seeds"],
            int(profile["pairs_per_seed"]),
            args.token_offset,
            int(profile["seed_token_offset_stride"]),
        )
        config = build_config(args, profile, worker_env)
        output_dir = args.output_dir.resolve()
        manifest_path, manifest = prepare_manifest(
            output_dir,
            config,
            schedule,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            commands = []
            for block in schedule:
                preview_output = (
                    output_dir
                    / "blocks"
                    / str(block["block_id"])
                    / "attempt_001"
                    / "worker_result.json"
                )
                commands.append(
                    {
                        "block_id": block["block_id"],
                        "command": worker_command(args, profile, block, preview_output),
                    }
                )
            manifest["dry_run_commands"] = commands
            manifest["status"] = "dry_run"
            atomic_write_json(manifest_path, manifest)
            print(f"Wrote dry-run Williams schedule: {manifest_path}")
            return

        smi_path = resolve_nvidia_smi(args.nvidia_smi)
        if smi_path is None and not args.allow_missing_nvidia_smi:
            raise ProtocolError("nvidia-smi not found; publication run fails closed")
        smi = NvidiaSmi(smi_path, args.gpu_index) if smi_path is not None else None
        if smi is not None:
            idle_baseline = establish_idle_baseline(
                smi,
                timeout_s=args.idle_timeout_s,
                poll_s=args.idle_poll_s,
                consecutive_required=args.idle_consecutive_polls,
                max_utilization=args.idle_max_utilization_percent,
                max_initial_used_mib=args.max_initial_gpu_used_mib,
            )
            idle_baseline_used_mib: float | None = float(
                idle_baseline["baseline_used_mib"]
            )
        else:
            idle_baseline = {"skipped": True, "reason": "nvidia-smi unavailable"}
            idle_baseline_used_mib = None

        manifest["status"] = "running"
        manifest["run_started_utc"] = utc_now()
        manifest["nvidia_smi_path"] = str(smi_path) if smi_path else None
        manifest["initial_idle_baseline"] = idle_baseline
        atomic_write_json(manifest_path, manifest)

        executions = manifest.setdefault("executions", {})
        for block in schedule:
            block_id = str(block["block_id"])
            existing = executions.get(block_id)
            if isinstance(existing, Mapping) and existing.get("status") == "completed":
                result_relative = existing.get("worker_result_path")
                if not isinstance(result_relative, str):
                    raise ProtocolError(f"completed block {block_id} lacks result path")
                result_path = output_dir / result_relative
                if not result_path.is_file() or sha256_file(result_path) != existing.get(
                    "worker_result_sha256"
                ):
                    raise ProtocolError(f"completed block {block_id} failed resume hash check")
                continue
            if isinstance(existing, Mapping) and not args.rerun_failed:
                raise ProtocolError(
                    f"block {block_id} previously failed; use --resume --rerun-failed"
                )
            executions[block_id] = {
                "status": "starting",
                "backend": block["backend"],
                "pair_id": block["pair_id"],
                "started_utc": utc_now(),
            }
            atomic_write_json(manifest_path, manifest)
            try:
                execution = execute_block(
                    args,
                    profile,
                    str(manifest["config_sha256"]),
                    block,
                    output_dir,
                    worker_env,
                    smi,
                    idle_baseline_used_mib,
                )
            except Exception as error:
                execution = {
                    "status": "failed",
                    "backend": block["backend"],
                    "pair_id": block["pair_id"],
                    "pair_order": block["pair_order"],
                    "slot": block["slot"],
                    "finished_utc": utc_now(),
                    "errors": [f"{type(error).__name__}: {error}"],
                }
            executions[block_id] = execution
            atomic_write_json(manifest_path, manifest)
            if execution["status"] != "completed":
                manifest["status"] = "failed"
                manifest["failed_block_id"] = block_id
                manifest["run_finished_utc"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                raise ProtocolError(
                    f"block {block_id} failed closed: {execution['errors']}"
                )

        manifest["status"] = "completed"
        manifest["run_finished_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        analysis = analyze_run_directory(
            output_dir,
            manifest_path,
            bootstrap_samples=int(profile["bootstrap_samples"]),
            bootstrap_seed=args.bootstrap_seed,
        )
    except (ProtocolError, json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"backend-exclusive orchestration failed closed: {error}") from error

    primary = analysis["aggregates"]["cache_neutral"]["wall_ms"]
    ci = primary["hierarchical_seed_pair_bootstrap"]["speedup_95_ci"]
    print(
        f"Completed {len(schedule)} fresh-process blocks. "
        f"Cache-neutral wall speedup {primary['speedup_geomean']:.4f}x "
        f"(hierarchical 95% CI [{ci[0]:.4f}, {ci[1]:.4f}])."
    )


if __name__ == "__main__":
    main()
