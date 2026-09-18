#!/usr/bin/env python3
"""Fail closed unless this is a full NVIDIA A100 suitable for the B4 run."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

MIN_MEMORY_BYTES = 38 * 1024**3
EXPECTED_MULTIPROCESSORS = 108


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; select a Colab GPU runtime")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = list(torch.cuda.get_device_capability(index))
    name = torch.cuda.get_device_name(index)
    failures: list[str] = []
    if "A100" not in name:
        failures.append(f"device name is not A100: {name!r}")
    if capability != [8, 0]:
        failures.append(f"compute capability is {capability}, expected [8, 0]")
    if int(properties.multi_processor_count) != EXPECTED_MULTIPROCESSORS:
        failures.append(
            "expected a full 108-SM A100; another count is unsupported or a "
            f"MIG slice ({properties.multi_processor_count} SMs)"
        )
    if int(properties.total_memory) < MIN_MEMORY_BYTES:
        failures.append(
            "less than 38 GiB device memory; full B4 confirmation would not "
            f"fit safely ({properties.total_memory / 1024**3:.3f} GiB)"
        )
    nvcc = command_output(["nvcc", "--version"])
    match = re.search(r"release\s+(\d+)\.(\d+)", nvcc)
    if match is None:
        failures.append("could not parse nvcc version")
        nvcc_version = None
    else:
        nvcc_version = [int(match.group(1)), int(match.group(2))]
        if tuple(nvcc_version) < (12, 6):
            failures.append(
                f"CUDA toolkit {nvcc_version[0]}.{nvcc_version[1]} is older than 12.6"
            )
    try:
        import flashinfer

        flashinfer_version = getattr(flashinfer, "__version__", "unknown")
    except Exception as error:  # setup may intentionally call this before install
        flashinfer_version = f"unavailable: {type(error).__name__}: {error}"
    payload = {
        "schema_version": 1,
        "experiment": "page_gauge_a100_colab_preflight",
        "passed": not failures,
        "failures": failures,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": name,
            "compute_capability": capability,
            "multiprocessor_count": int(properties.multi_processor_count),
            "total_memory_bytes": int(properties.total_memory),
            "total_memory_gib": float(properties.total_memory / 1024**3),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer_version,
            "nvcc_version": nvcc_version,
            "nvcc_output": nvcc,
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
        "compile_contract": {
            "torch_cuda_arch_list": "8.0+PTX",
            "flashinfer_cuda_arch_list": "8.0",
            "clean_extension_cache_required": True,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
