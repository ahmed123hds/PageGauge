#!/usr/bin/env python3
"""CPU-only protocol for the MLSys 2027 controlled decoder experiment.

The module defines immutable cell identities and fresh-process schedules.  It
contains no worker launcher or GPU-library import.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "pagegauge_mlsys2027_controlled_decoder"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
# These are implementation blockers, not user-editable matrix readiness flags.
# Remove them only when an actual controlled runner and execution source lock
# are integrated and validated.  This module currently seals a design only.
DESIGN_ONLY_REQUIREMENTS = frozenset(
    {
        "runner:controlled_decoder:integration",
        "source_lock:controlled_worker",
        "source_lock:kernels_and_dependencies",
    }
)
EXPECTED_BASELINES = {
    "flashinfer_fp16",
    "page_gauge_explicit_reconstruction",
    "page_gauge_fused_reconstruction",
}
CANDIDATE = "page_gauge_factorized"
EXPECTED_POLICY = {
    "page_size_tokens": 16,
    "historical_kv_dtype": "int8",
    "metadata_dtype": "fp16",
    "exact_prefix_pages": 4,
    "static_exact_prefill_suffix_pages": 128,
    "exact_recent_tail_tokens": 768,
    "policy_name": "S4/A128/T768",
}


class ProtocolError(ValueError):
    """Raised when the controlled experiment fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return value


def _string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} is empty")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def _hex_sha(value: Any, label: str, lengths: tuple[int, ...] = (40, 64)) -> str:
    rendered = _string(value, label).lower()
    require(
        len(rendered) in lengths and all(char in "0123456789abcdef" for char in rendered),
        f"{label} is not an immutable hexadecimal revision/hash",
    )
    return rendered


def _cell_definition(
    matrix: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    hardware = _mapping(matrix["hardware"], "hardware")
    hardware_id = str(cell["hardware_id"])
    return {
        "experiment": EXPERIMENT,
        "design": matrix["design"],
        "policy": matrix["policy"],
        "model": matrix["model"],
        "hardware_id": hardware_id,
        "expected_hardware": hardware[hardware_id],
        "contrast": {
            "baseline": cell["baseline"],
            "candidate": cell["candidate"],
        },
        "workload": matrix["workload"],
        "gates": matrix["gates"],
    }


def validate_matrix(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    require(matrix.get("schema_version") == SCHEMA_VERSION, "wrong matrix schema")
    require(matrix.get("experiment") == EXPERIMENT, "wrong matrix experiment")
    require(matrix.get("status") in {"design", "frozen"}, "invalid matrix status")

    design = _mapping(matrix.get("design"), "design")
    require(design.get("seed_orders") == ["ABBA", "BAAB"], "noncanonical seed orders")
    require(
        design.get("experimental_unit") == "adjacent fresh-process treatment pair",
        "wrong experimental unit",
    )
    require(
        design.get("bootstrap_hierarchy") == ["seed_fixture", "adjacent_pair"],
        "wrong bootstrap hierarchy",
    )
    require(design.get("warmup_full_sequences") == 1, "wrong warmup count")
    require(design.get("measured_full_sequences") == 3, "wrong repeat count")
    require(
        design.get("timing_modes") == ["cache_neutral", "cache_hot"],
        "wrong timing modes",
    )
    require(
        design.get("timing_metrics") == ["wall_ms", "cuda_ms"],
        "wrong timing metrics",
    )
    require(dict(_mapping(matrix.get("policy"), "policy")) == EXPECTED_POLICY, "policy drift")

    model = _mapping(matrix.get("model"), "model")
    require(model.get("revision_status") == "pinned", "model revision is mutable")
    _hex_sha(model.get("revision"), "model revision", (40,))
    _string(model.get("repo_id"), "model repo_id")

    hardware = _mapping(matrix.get("hardware"), "hardware")
    require(bool(hardware), "hardware is empty")
    for hardware_id, raw in hardware.items():
        _string(hardware_id, "hardware id")
        item = _mapping(raw, f"hardware {hardware_id}")
        capability = _sequence(
            item.get("compute_capability"), f"hardware {hardware_id} capability"
        )
        require(
            len(capability) == 2
            and all(isinstance(value, int) and value >= 0 for value in capability),
            f"hardware {hardware_id} capability is invalid",
        )
        memory = item.get("minimum_memory_gib")
        require(
            isinstance(memory, (int, float))
            and not isinstance(memory, bool)
            and math.isfinite(float(memory))
            and float(memory) > 0.0,
            f"hardware {hardware_id} memory is invalid",
        )
        require(
            item.get("availability_status") in {"available", "to_secure"},
            f"hardware {hardware_id} availability is invalid",
        )

    methods = _mapping(matrix.get("methods"), "methods")
    require(set(methods) == EXPECTED_BASELINES | {CANDIDATE}, "method set drift")
    for method_id, raw in methods.items():
        method = _mapping(raw, f"method {method_id}")
        require(
            method.get("implementation_status") in {"ready", "planned"},
            f"method {method_id} implementation status is invalid",
        )

    workload = _mapping(matrix.get("workload"), "workload")
    page_size = EXPECTED_POLICY["page_size_tokens"]
    prefix = _positive_int(workload.get("prefix_tokens"), "prefix_tokens")
    decode = _positive_int(workload.get("decode_tokens"), "decode_tokens")
    require(prefix % page_size == 0 and decode % page_size == 0, "workload is not page aligned")
    require(workload.get("batch_size") == 4, "anchor batch size drift")
    require(workload.get("forced_decode") is True, "anchor must use forced decode")
    seeds = list(_sequence(workload.get("seeds"), "workload seeds"))
    require(
        len(seeds) == 2
        and len(set(seeds)) == 2
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
        "anchor requires two distinct integer seed fixtures",
    )
    require(workload.get("pairs_per_seed") == 2, "pairs_per_seed drift")
    require(
        workload.get("minimum_closed_generated_pages") == decode // page_size,
        "closed-page recurrence count drift",
    )
    consumed = _positive_int(
        workload.get("minimum_runtime_created_pages_later_consumed_int8"),
        "minimum runtime-created INT8 pages",
    )
    require(consumed <= decode // page_size, "runtime INT8 recurrence gate is impossible")

    cells: dict[str, Mapping[str, Any]] = {}
    observed_contrasts: set[tuple[str, str]] = set()
    for index, raw in enumerate(_sequence(matrix.get("cells"), "cells")):
        cell = _mapping(raw, f"cells[{index}]")
        cell_id = _string(cell.get("cell_id"), f"cells[{index}].cell_id")
        require(re.fullmatch(r"[a-z0-9_]+", cell_id) is not None, f"invalid cell id {cell_id}")
        require(cell_id not in cells, f"duplicate cell id {cell_id}")
        hardware_id = _string(cell.get("hardware_id"), f"cell {cell_id}.hardware_id")
        require(hardware_id in hardware, f"cell {cell_id} has unknown hardware")
        baseline = _string(cell.get("baseline"), f"cell {cell_id}.baseline")
        candidate = _string(cell.get("candidate"), f"cell {cell_id}.candidate")
        require(baseline in EXPECTED_BASELINES, f"cell {cell_id} has invalid baseline")
        require(candidate == CANDIDATE, f"cell {cell_id} has invalid candidate")
        key = (hardware_id, baseline)
        require(key not in observed_contrasts, f"duplicate contrast {key}")
        observed_contrasts.add(key)
        cells[cell_id] = cell
    expected_contrasts = {
        (hardware_id, baseline)
        for hardware_id in hardware
        for baseline in EXPECTED_BASELINES
    }
    require(observed_contrasts == expected_contrasts, "controlled contrast matrix is incomplete")
    return cells


def unresolved_requirements(matrix: Mapping[str, Any]) -> list[str]:
    validate_matrix(matrix)
    unresolved: set[str] = set(DESIGN_ONLY_REQUIREMENTS)
    for hardware_id, raw in matrix["hardware"].items():
        if raw["availability_status"] != "available":
            unresolved.add(f"hardware:{hardware_id}:access")
    for method_id, raw in matrix["methods"].items():
        if raw["implementation_status"] != "ready":
            unresolved.add(f"method:{method_id}:implementation")
    return sorted(unresolved)


def cell_config_sha256(matrix: Mapping[str, Any], cell_id: str) -> str:
    cells = validate_matrix(matrix)
    require(cell_id in cells, f"unknown cell {cell_id}")
    return canonical_sha256(_cell_definition(matrix, cells[cell_id]))


def expand_blocks(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = validate_matrix(matrix)
    seeds = list(matrix["workload"]["seeds"])
    orders = list(matrix["design"]["seed_orders"])
    blocks: list[dict[str, Any]] = []
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        cell_hash = canonical_sha256(_cell_definition(matrix, cell))
        for seed_index, seed in enumerate(seeds):
            order = orders[seed_index]
            for slot, treatment in enumerate(order):
                pair_index = slot // 2
                pair_id = f"{cell_id}-seed{seed}-pair{pair_index}"
                backend = cell["baseline"] if treatment == "A" else cell["candidate"]
                identity = {
                    "cell_id": cell_id,
                    "cell_config_sha256": cell_hash,
                    "seed": seed,
                    "seed_order": order,
                    "pair_id": pair_id,
                    "pair_order": order[pair_index * 2 : pair_index * 2 + 2],
                    "slot": slot,
                    "treatment": treatment,
                    "backend": backend,
                }
                digest = canonical_sha256(identity)
                blocks.append(
                    {
                        **identity,
                        "block_id": f"{cell_id}-{digest[:16]}",
                        "block_identity_sha256": digest,
                        "fresh_process_ordinal": seed_index * 4 + slot,
                        "status": "planned",
                    }
                )
    require(len({block["block_id"] for block in blocks}) == len(blocks), "duplicate block id")
    return blocks


def build_manifest(
    matrix: Mapping[str, Any], *, matrix_path: Path, source_paths: Sequence[Path]
) -> dict[str, Any]:
    validate_matrix(matrix)
    unresolved = unresolved_requirements(matrix)
    closure: dict[str, str] = {}
    for raw_path in source_paths:
        path = raw_path.resolve()
        require(path.is_file(), f"source-closure file is missing: {path}")
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        require(key not in closure, f"duplicate source-closure path {key}")
        closure[key] = sha256_file(path)
    blocks = expand_blocks(matrix)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "design",
        "runnable": False,
        "source_closure_status": "design_only_not_execution_locked",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_file_sha256": sha256_file(matrix_path.resolve()),
        "matrix_canonical_sha256": canonical_sha256(matrix),
        "source_sha256": closure,
        "unresolved_requirements": unresolved,
        "cell_count": len(matrix["cells"]),
        "physical_pair_count": len(blocks) // 2,
        "fresh_process_block_count": len(blocks),
        "blocks": blocks,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION, "manifest schema mismatch")
    require(manifest.get("experiment") == EXPERIMENT, "manifest experiment mismatch")
    digest = _hex_sha(manifest.get("manifest_sha256"), "manifest SHA256", (64,))
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    require(canonical_sha256(unhashed) == digest, "manifest hash integrity failure")
    require(
        manifest.get("status") == "design" and manifest.get("runnable") is False,
        "controlled runner is not integrated; manifest must remain non-runnable",
    )
    require(
        manifest.get("source_closure_status") == "design_only_not_execution_locked",
        "manifest cannot claim an execution source lock",
    )
    require(
        DESIGN_ONLY_REQUIREMENTS.issubset(
            _sequence(manifest.get("unresolved_requirements"), "unresolved_requirements")
        ),
        "manifest omits controlled runner/source-lock blockers",
    )
    blocks = list(_sequence(manifest.get("blocks"), "manifest blocks"))
    require(len(blocks) == manifest.get("fresh_process_block_count"), "block count mismatch")
    require(len(blocks) % 2 == 0, "block schedule cannot form adjacent pairs")
    require(len(blocks) // 2 == manifest.get("physical_pair_count"), "pair count mismatch")
    block_ids = [str(_mapping(block, "block").get("block_id")) for block in blocks]
    require(len(block_ids) == len(set(block_ids)), "manifest block IDs are not unique")


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
