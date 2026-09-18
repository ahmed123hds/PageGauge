#!/usr/bin/env python3
"""Create a reproducible patched FlashInfer 0.6.17 include tree under build/."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/flashinfer-0.6.17-page-gauge-int8.patch"
TARGET = ROOT / "build/gauge_affine_flashinfer/include"
EXPECTED_VERSION = "0.6.17"
EXPECTED_SOURCE_SHA256 = "2e5927bdc0d36ddb393cb4fab68c2e958d65d5b4b0085c969f7cfa777ecdfb5b"
EXPECTED_PATCHED_SHA256 = "db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54"


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
        help="Attempt the patch even when FlashInfer's version string differs.",
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
    source = Path(flashinfer.__file__).resolve().parent / "data/include"
    source_header = source / "flashinfer/attention/prefill.cuh"
    if not source_header.is_file():
        raise SystemExit(f"FlashInfer header not found: {source_header}")
    source_digest = sha256(source_header)
    if source_digest != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "FlashInfer prefill.cuh does not match the tested 0.6.17 source: "
            f"{source_digest}"
        )
    if not PATCH.is_file():
        raise SystemExit(f"Patch not found: {PATCH}")

    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(source, TARGET)
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(PATCH)],
        cwd=TARGET,
        check=True,
    )
    patched_header = TARGET / "flashinfer/attention/prefill.cuh"
    patched_digest = sha256(patched_header)
    if patched_digest != EXPECTED_PATCHED_SHA256:
        raise SystemExit(
            f"Patched header checksum mismatch: {patched_digest}"
        )
    print(f"Prepared {patched_header}")
    print(f"sha256={patched_digest}")


if __name__ == "__main__":
    main()
