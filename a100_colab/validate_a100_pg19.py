#!/usr/bin/env python3
"""Bind the frozen six-book PG-19 aggregate to one full A100 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OBJECTS = ("10146", "10321", "10356", "10762", "15562", "22424")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_identity(payload: dict[str, Any], label: str) -> dict[str, Any]:
    environment = payload.get("environment", {})
    require("A100" in str(environment.get("gpu")), f"{label}: not A100")
    require(
        environment.get("compute_capability") == [8, 0],
        f"{label}: not compute capability 8.0",
    )
    keys = (
        "gpu",
        "compute_capability",
        "torch",
        "torch_cuda",
        "flashinfer",
        "transformers",
        "python",
    )
    return {key: environment.get(key) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=6, type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    require(preflight.get("passed") is True, "A100 preflight failed")
    device = preflight.get("device", {})
    require(device.get("compute_capability") == [8, 0], "preflight capability")
    require(device.get("multiprocessor_count") == 108, "preflight full A100")
    require(aggregate.get("passed") is True, "PG-19 aggregate failed")

    records: list[dict[str, Any]] = []
    reference_environment: dict[str, Any] | None = None
    observed_objects: list[str] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        object_name = str(payload.get("token_source", {}).get("object_name", ""))
        observed_objects.append(Path(object_name).stem)
        identity = environment_identity(payload, str(path))
        if reference_environment is None:
            reference_environment = identity
        require(identity == reference_environment, f"{path}: environment drift")
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "object_name": object_name,
            }
        )
    require(tuple(observed_objects) == OBJECTS, "PG-19 object order/identity")
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_a100_pg19_external_confirmation",
        "passed": True,
        "a100_preflight": {
            "path": str(args.preflight),
            "sha256": sha256_file(args.preflight),
        },
        "aggregate": {
            "path": str(args.aggregate),
            "sha256": sha256_file(args.aggregate),
        },
        "inputs": records,
        "shared_environment": reference_environment,
        "performance_samples_contributed": False,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
