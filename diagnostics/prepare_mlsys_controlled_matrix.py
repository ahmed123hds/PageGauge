#!/usr/bin/env python3
"""Seal the CPU-only MLSys 2027 controlled-decoder design manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlsys_controlled_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "experiments/mlsys2027/decoder_matrix_v1.json"
DEFAULT_OUTPUT = ROOT / "results/mlsys_2027_design/controlled_decoder_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--require-runnable", action="store_true")
    return parser.parse_args()


def prepare(matrix_path: Path) -> dict:
    matrix_path = matrix_path.resolve()
    matrix = protocol.load_json(matrix_path)
    return protocol.build_manifest(
        matrix,
        matrix_path=matrix_path,
        source_paths=(
            Path(__file__).resolve(),
            Path(protocol.__file__).resolve(),
            ROOT / "diagnostics/reduce_mlsys_controlled.py",
            matrix_path,
        ),
    )


def main() -> None:
    args = parse_args()
    manifest = prepare(args.matrix)
    protocol.verify_manifest(manifest)
    if args.require_runnable and not manifest["runnable"]:
        raise SystemExit(
            "controlled experiment is not runnable: "
            + ", ".join(manifest["unresolved_requirements"])
        )
    if not args.summary_only:
        protocol.atomic_write_json(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "runnable": manifest["runnable"],
                "cell_count": manifest["cell_count"],
                "physical_pair_count": manifest["physical_pair_count"],
                "fresh_process_block_count": manifest["fresh_process_block_count"],
                "unresolved_requirements": manifest["unresolved_requirements"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": None if args.summary_only else str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
