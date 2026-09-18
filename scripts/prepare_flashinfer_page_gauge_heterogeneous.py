#!/usr/bin/env python3
"""Prepare the additive heterogeneous PageGauge FlashInfer include tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREPARE = ROOT / "scripts/prepare_flashinfer_page_gauge.py"
LEGACY_INCLUDE = ROOT / "build/gauge_affine_flashinfer/include"
TARGET = ROOT / "build/gauge_heterogeneous_flashinfer/include"
PATCH = ROOT / "patches/flashinfer-0.6.17-page-gauge-heterogeneous.patch"
EXPECTED_VERSION = "0.6.17"
EXPECTED_LEGACY_SHA256 = "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"
EXPECTED_PATCHED_SHA256 = "2a3f3018576d04697477e3030507379ccc2d126036255ed81a238b63e9572e56"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-version",
        action="store_true",
        help="Attempt preparation even when FlashInfer's version differs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import flashinfer

    version = getattr(flashinfer, "__version__", "unknown")
    if version != EXPECTED_VERSION and not args.force_version:
        raise SystemExit(
            f"PageGauge patch targets FlashInfer {EXPECTED_VERSION}; found {version}."
        )
    legacy_header = LEGACY_INCLUDE / "flashinfer/attention/prefill.cuh"
    if not legacy_header.is_file() or sha256(legacy_header) != EXPECTED_LEGACY_SHA256:
        command = [sys.executable, str(LEGACY_PREPARE)]
        if args.force_version:
            command.append("--force-version")
        subprocess.run(command, check=True)
    legacy_digest = sha256(legacy_header)
    if legacy_digest != EXPECTED_LEGACY_SHA256:
        raise SystemExit(
            "Legacy PageGauge include checksum mismatch after preparation: "
            f"{legacy_digest}"
        )
    if not PATCH.is_file():
        raise SystemExit(f"Heterogeneous patch not found: {PATCH}")

    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(LEGACY_INCLUDE, TARGET)
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(PATCH)],
        cwd=TARGET,
        check=True,
    )
    patched_header = TARGET / "flashinfer/attention/prefill.cuh"
    patched_digest = sha256(patched_header)
    if patched_digest != EXPECTED_PATCHED_SHA256:
        raise SystemExit(f"Patched header checksum mismatch: {patched_digest}")
    print(f"Prepared {patched_header}")
    print(f"legacy_sha256={legacy_digest}")
    print(f"sha256={patched_digest}")


if __name__ == "__main__":
    main()
