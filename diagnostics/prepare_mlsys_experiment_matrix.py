#!/usr/bin/env python3
"""Validate and seal the CPU-only PageGauge MLSys experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlsys_experiment_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs/mlsys_2027_experiment_matrix.json"
DEFAULT_OUTPUT = ROOT / "results/mlsys_2027_design/experiment_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-runnable",
        action="store_true",
        help="Fail if any model, hardware, method, or suite runner is unresolved.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Validate and print the summary without writing a manifest.",
    )
    return parser.parse_args()


def prepare(spec_path: Path) -> dict:
    spec_path = spec_path.resolve()
    spec = protocol.load_json(spec_path)
    source_paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        spec_path,
    )
    return protocol.build_manifest(
        spec, spec_path=spec_path, source_paths=source_paths
    )


def main() -> None:
    args = parse_args()
    manifest = prepare(args.spec)
    if args.require_runnable and not manifest["runnable"]:
        unresolved = ", ".join(manifest["unresolved_requirements"])
        raise SystemExit(f"experiment matrix is not runnable: {unresolved}")
    report = protocol.summary(manifest)
    if not args.summary_only:
        protocol.atomic_write_json(args.output.resolve(), manifest)
        report["output"] = str(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

