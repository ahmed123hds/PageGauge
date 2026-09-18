#!/usr/bin/env python3
"""Wait for an idle single-GPU Colab runtime before a fresh worker block."""

from __future__ import annotations

import argparse
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--max-utilization", type=float, default=5.0)
    parser.add_argument("--max-used-mib", type=float, default=1024.0)
    return parser.parse_args()


def sample() -> tuple[float, float]:
    output = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(output) != 1:
        raise RuntimeError(f"expected one GPU, got {len(output)}")
    utilization, used = (float(value.strip()) for value in output[0].split(","))
    return utilization, used


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    good = 0
    while time.monotonic() < deadline:
        utilization, used = sample()
        if utilization <= args.max_utilization and used <= args.max_used_mib:
            good += 1
            if good >= args.consecutive:
                print(
                    f"GPU idle gate PASS: utilization={utilization:.1f}%, "
                    f"used={used:.1f} MiB"
                )
                return
        else:
            good = 0
        time.sleep(args.poll)
    raise SystemExit("GPU did not reach the required idle envelope")


if __name__ == "__main__":
    main()
