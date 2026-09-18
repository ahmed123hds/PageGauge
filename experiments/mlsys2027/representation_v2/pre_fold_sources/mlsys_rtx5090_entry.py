#!/usr/bin/env python3
"""Prepare or launch the first RTX 5090 MLSys experiment, one step at a time."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "diagnostics/mlsys_rtx5090_step01.py"
HEADER = ROOT / "build/gauge_affine_flashinfer/include/flashinfer/attention/prefill.cuh"
EXPECTED_HEADER_SHA256 = "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"
SOURCE_FILES = (
    "diagnostics/mlsys_rtx5090_entry.py",
    "diagnostics/mlsys_rtx5090_step01.py",
    "experiments/mlsys2027/run_rtx5090.sh",
    "scripts/benchmark_page_gauge_transformer.py",
    "scripts/benchmark_flashinfer_page_affine_int8.py",
    "scripts/page_gauge_heterogeneous_fa2.py",
    "scripts/benchmark_page_gauge_overheads.py",
    "scripts/page_gauge_runtime.py",
    "scripts/benchmark_e2e_transformer.py",
    "patches/flashinfer-0.6.17-page-gauge-int8.patch",
    "build/gauge_affine_flashinfer/include/flashinfer/attention/prefill.cuh",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_environment(extra_sources=()) -> dict:
    """Inspect metadata/syntax only; importing this module cannot initialize CUDA."""
    failures = []
    versions = {}
    for name in ("torch", "flashinfer-python", "numpy", "transformers", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing package: {name}")
    if versions.get("flashinfer-python") != "0.6.17":
        failures.append("This source requires flashinfer-python==0.6.17.")
    closure = {}
    for relative in dict.fromkeys((*SOURCE_FILES, *extra_sources)):
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"Missing source: {relative}")
            continue
        closure[relative] = sha256_file(path)
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as error:
                failures.append(f"Syntax error: {error}")
    if closure.get(str(HEADER.relative_to(ROOT)).replace("\\", "/")) != EXPECTED_HEADER_SHA256:
        failures.append("The production PageGauge header is missing or has changed; no rebuild was attempted.")
    cuda_home = os.environ.get("CUDA_HOME", "")
    if not cuda_home or not (Path(cuda_home) / "bin/nvcc").is_file():
        failures.append("CUDA_HOME/bin/nvcc is missing.")
    ninja_path = shutil.which("ninja")
    if ninja_path is None:
        failures.append("ninja is not on PATH; use the launcher with the PageGauge Python environment's bin directory.")
    return {
        "passed": not failures,
        "cpu_only": True,
        "gpu_queried": False,
        "python": sys.executable,
        "versions": versions,
        "build_tools": {"ninja": ninja_path},
        "source_sha256": closure,
        "failures": failures,
    }


def parse_device_snapshot(device_csv: str, process_csv: str) -> dict:
    rows = list(csv.reader(line for line in device_csv.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("Cannot establish device identity and idle state.")
    name, gpu_uuid, total, used, utilization = [item.strip() for item in rows[0]]
    try:
        total, used, utilization = float(total), float(used), float(utilization)
    except ValueError as error:
        raise RuntimeError("GPU memory/utilization telemetry is unavailable; refusing to launch.") from error
    if not all(math.isfinite(value) for value in (total, used, utilization)):
        raise RuntimeError("GPU telemetry is nonfinite.")
    if "RTX 5090" not in name or not gpu_uuid.startswith("GPU-") or total < 30000:
        raise RuntimeError(f"Expected a full RTX 5090, got {name} ({total} MiB).")
    if not 0 <= utilization <= 100 or not 0 <= used <= total:
        raise RuntimeError("GPU telemetry is out of range.")
    processes = []
    for row in csv.reader(line for line in process_csv.splitlines() if line.strip()):
        if len(row) != 2 or not row[1].strip().isdigit():
            raise RuntimeError("Compute-process enumeration is unavailable; refusing to launch.")
        if row[0].strip() == gpu_uuid:
            processes.append(int(row[1].strip()))
    if processes or used > 2048 or utilization > 5:
        raise RuntimeError(
            f"GPU BUSY: {len(processes)} compute processes, {used:.0f} MiB used, "
            f"{utilization:.0f}% utilization. No worker started; existing jobs were not changed."
        )
    return {"name": name, "uuid": gpu_uuid, "memory_total_mib": total,
            "memory_used_mib": used, "utilization_percent": utilization,
            "compute_process_count": len(processes)}


def idle_preflight(gpu_index: int) -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("nvidia-smi is unavailable; refusing to launch.")
    snapshots = []
    for index in range(3):
        device = subprocess.run(
            [executable, f"--id={gpu_index}", "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        processes = subprocess.run(
            [executable, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        snapshots.append(parse_device_snapshot(device.stdout, processes.stdout))
        if len({item["uuid"] for item in snapshots}) != 1:
            raise RuntimeError("Device identity changed during preflight.")
        if index != 2:
            time.sleep(1)
    return snapshots


def launch_step01(environment_check: dict, gpu_index: int) -> int:
    # No imports of torch/FlashInfer until the separate worker starts.
    snapshots = idle_preflight(gpu_index)
    import fcntl

    lock_path = Path("/tmp") / f"pagegauge-mlsys-{snapshots[-1]['uuid']}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another PageGauge MLSys command already holds this device.") from error
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        output_dir = ROOT / "results/mlsys2027_rtx5090/01_correctness" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        # Importing the worker here is CPU-only. GPU imports are inside its run function.
        import mlsys_rtx5090_step01 as worker

        manifest = {
            "schema_version": 1,
            "experiment": worker.EXPERIMENT,
            "status": "frozen_before_synthetic_execution",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": worker.CONFIG,
            "source_sha256": environment_check["source_sha256"],
            "environment": environment_check["versions"],
            "idle_preflight": snapshots,
            "scope": "Synthetic production-attention correctness; no end-to-end speed or task-quality claim.",
        }
        manifest["manifest_sha256"] = canonical_hash(manifest)
        manifest_path = output_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = snapshots[-1]["uuid"]
        command = [sys.executable, "-u", str(WORKER), "--manifest", str(manifest_path), "--output", str(output_dir / "result.json")]
        print(f"Step 01 output: {output_dir}", flush=True)
        print("Starting production-kernel correctness. First-use compilation may take several minutes.", flush=True)
        with (output_dir / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=worker_env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        result_path = output_dir / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            if not isinstance(result, dict):
                result = {}
        except (OSError, ValueError):
            result = {}
        unchanged = all((ROOT / name).is_file() and sha256_file(ROOT / name) == digest
                        for name, digest in manifest["source_sha256"].items())
        valid = (return_code == 0 and result.get("passed") is True
                 and result.get("manifest_sha256") == manifest["manifest_sha256"]
                 and result.get("experiment") == manifest["experiment"] and unchanged)
        completion = {
            "passed": valid, "worker_return_code": return_code,
            "source_unchanged": unchanged, "manifest_sha256": manifest["manifest_sha256"],
            "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
            "log_sha256": sha256_file(output_dir / "worker.log"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(output_dir / "completion.json", completion)
        print(f"STEP 01 {'PASS' if valid else 'FAIL'}: {result_path}", flush=True)
        return 0 if valid else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check", "01", "check02", "02"), nargs="?", default="check")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    performance = None
    if args.step in ("check02", "02"):
        import mlsys_rtx5090_step02 as performance
    checked = check_environment(performance.EXTRA_SOURCES if performance else ())
    if args.step in ("check", "check02") or not checked["passed"]:
        print(json.dumps({key: value for key, value in checked.items() if key != "source_sha256"}, indent=2))
    if not checked["passed"]:
        raise SystemExit(2)
    if performance:
        try:
            if args.step == "check02":
                print(json.dumps({"step02_prerequisites_passed": True, **performance.prerequisites()}, indent=2))
            else:
                raise SystemExit(performance.launch(checked, args.gpu_index))
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2) from error
    if args.step == "01":
        try:
            raise SystemExit(launch_step01(checked, args.gpu_index))
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2) from error


if __name__ == "__main__":
    main()
