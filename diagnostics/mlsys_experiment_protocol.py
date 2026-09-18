#!/usr/bin/env python3
"""CPU-only manifest protocol for the PageGauge MLSys 2027 experiments.

This module deliberately does not import torch, CUDA, FlashInfer, or a serving
runtime.  It validates the preregistered experiment design and expands it into
deterministic fresh-process run cells.  Execution workers are added separately
only after their model/hardware/method contracts are ready.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
EXPERIMENT = "page_gauge_mlsys_2027_experiment_matrix"

ALLOWED_PHASES = {
    "controlled_factorization",
    "serving",
    "quality",
    "hardware_profile",
}
ALLOWED_PRIORITIES = {"P0", "P1", "P2"}
ALLOWED_REVISION_STATUS = {"pinned", "to_pin"}
ALLOWED_AVAILABILITY_STATUS = {"available", "to_secure"}
ALLOWED_IMPLEMENTATION_STATUS = {"ready", "planned"}
ALLOWED_RUNNER_STATUS = {"ready", "planned"}


class ProtocolError(ValueError):
    """Raised when an experiment design or manifest fails closed."""


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


def _nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} is empty")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def _unique_strings(value: Any, label: str) -> tuple[str, ...]:
    rows = tuple(
        _nonempty_string(item, f"{label} item") for item in _sequence(value, label)
    )
    require(bool(rows), f"{label} cannot be empty")
    require(len(rows) == len(set(rows)), f"{label} contains duplicates")
    return rows


def _unique_positive_ints(value: Any, label: str) -> tuple[int, ...]:
    rows = tuple(
        _positive_int(item, f"{label} item") for item in _sequence(value, label)
    )
    require(bool(rows), f"{label} cannot be empty")
    require(len(rows) == len(set(rows)), f"{label} contains duplicates")
    return rows


def _index_by_id(
    rows: Any, id_field: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_sequence(rows, label)):
        row = _mapping(raw, f"{label}[{index}]")
        row_id = _nonempty_string(row.get(id_field), f"{label}[{index}].{id_field}")
        require(row_id not in result, f"duplicate {label} id: {row_id}")
        result[row_id] = row
    require(bool(result), f"{label} cannot be empty")
    return result


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a design spec and return indexed entities used for expansion."""

    require(spec.get("schema_version") == SCHEMA_VERSION, "wrong spec schema_version")
    require(spec.get("experiment") == EXPERIMENT, "wrong experiment name")

    models = _index_by_id(spec.get("models"), "model_id", "models")
    hardware = _index_by_id(spec.get("hardware"), "hardware_id", "hardware")
    methods = _index_by_id(spec.get("methods"), "method_id", "methods")
    suites = _index_by_id(spec.get("suites"), "suite_id", "suites")

    for model_id, model in models.items():
        _nonempty_string(model.get("hf_id"), f"model {model_id}.hf_id")
        status = _nonempty_string(
            model.get("revision_status"), f"model {model_id}.revision_status"
        )
        require(status in ALLOWED_REVISION_STATUS, f"model {model_id} revision status")
        revision = _nonempty_string(model.get("revision"), f"model {model_id}.revision")
        if status == "pinned":
            require(
                revision.upper() != "TO_PIN_BEFORE_RUN",
                f"model {model_id} is marked pinned but has a placeholder revision",
            )
        _positive_int(
            model.get("maximum_context_tokens"),
            f"model {model_id}.maximum_context_tokens",
        )
        _nonempty_string(model.get("family"), f"model {model_id}.family")

    for hardware_id, device in hardware.items():
        status = _nonempty_string(
            device.get("availability_status"),
            f"hardware {hardware_id}.availability_status",
        )
        require(
            status in ALLOWED_AVAILABILITY_STATUS,
            f"hardware {hardware_id} availability status",
        )
        capability = _sequence(
            device.get("compute_capability"),
            f"hardware {hardware_id}.compute_capability",
        )
        require(
            len(capability) == 2
            and all(isinstance(item, int) and item >= 0 for item in capability),
            f"hardware {hardware_id} compute capability is invalid",
        )
        memory_gib = device.get("minimum_memory_gib")
        require(
            isinstance(memory_gib, (int, float))
            and not isinstance(memory_gib, bool)
            and math.isfinite(float(memory_gib))
            and float(memory_gib) > 0.0,
            f"hardware {hardware_id} minimum_memory_gib is invalid",
        )

    for method_id, method in methods.items():
        status = _nonempty_string(
            method.get("implementation_status"),
            f"method {method_id}.implementation_status",
        )
        require(
            status in ALLOWED_IMPLEMENTATION_STATUS,
            f"method {method_id} implementation status",
        )
        comparison_class = _nonempty_string(
            method.get("comparison_class"),
            f"method {method_id}.comparison_class",
        )
        require(
            comparison_class in {"controlled", "published_system"},
            f"method {method_id} comparison class is invalid",
        )

    for suite_id, suite in suites.items():
        phase = _nonempty_string(suite.get("phase"), f"suite {suite_id}.phase")
        priority = _nonempty_string(
            suite.get("priority"), f"suite {suite_id}.priority"
        )
        runner_status = _nonempty_string(
            suite.get("runner_status"), f"suite {suite_id}.runner_status"
        )
        require(phase in ALLOWED_PHASES, f"suite {suite_id} phase is invalid")
        require(priority in ALLOWED_PRIORITIES, f"suite {suite_id} priority is invalid")
        require(
            runner_status in ALLOWED_RUNNER_STATUS,
            f"suite {suite_id} runner status is invalid",
        )
        model_ids = _unique_strings(suite.get("models"), f"suite {suite_id}.models")
        hardware_ids = _unique_strings(
            suite.get("hardware"), f"suite {suite_id}.hardware"
        )
        method_ids = _unique_strings(
            suite.get("methods"), f"suite {suite_id}.methods"
        )
        contexts = _unique_positive_ints(
            suite.get("contexts"), f"suite {suite_id}.contexts"
        )
        batch_sizes = _unique_positive_ints(
            suite.get("batch_sizes"), f"suite {suite_id}.batch_sizes"
        )
        decode_steps = _unique_positive_ints(
            suite.get("decode_steps"), f"suite {suite_id}.decode_steps"
        )
        workloads = _unique_strings(
            suite.get("workloads"), f"suite {suite_id}.workloads"
        )
        replicates = _positive_int(
            suite.get("fresh_process_replicates"),
            f"suite {suite_id}.fresh_process_replicates",
        )

        for model_id in model_ids:
            require(model_id in models, f"suite {suite_id} references unknown model")
            maximum = int(models[model_id]["maximum_context_tokens"])
            require(
                max(contexts) <= maximum,
                f"suite {suite_id} context exceeds model {model_id} limit",
            )
        for hardware_id in hardware_ids:
            require(
                hardware_id in hardware,
                f"suite {suite_id} references unknown hardware",
            )
        for method_id in method_ids:
            require(method_id in methods, f"suite {suite_id} references unknown method")
        require(
            all(context % 16 == 0 for context in contexts),
            f"suite {suite_id} contexts must be page aligned",
        )
        require(
            all(step % 16 == 0 for step in decode_steps),
            f"suite {suite_id} decode_steps must be page aligned",
        )
        require(replicates >= 2 or phase == "quality", f"suite {suite_id} lacks replication")
        require(bool(workloads), f"suite {suite_id} workloads cannot be empty")
        require(bool(batch_sizes), f"suite {suite_id} batches cannot be empty")

    return {
        "models": models,
        "hardware": hardware,
        "methods": methods,
        "suites": suites,
    }


def unresolved_requirements(spec: Mapping[str, Any]) -> list[str]:
    indexed = validate_spec(spec)
    unresolved: set[str] = set()
    for suite_id, suite in indexed["suites"].items():
        if suite["runner_status"] != "ready":
            unresolved.add(f"suite:{suite_id}:runner")
        for model_id in suite["models"]:
            if indexed["models"][model_id]["revision_status"] != "pinned":
                unresolved.add(f"model:{model_id}:revision")
        for hardware_id in suite["hardware"]:
            if indexed["hardware"][hardware_id]["availability_status"] != "available":
                unresolved.add(f"hardware:{hardware_id}:access")
        for method_id in suite["methods"]:
            if indexed["methods"][method_id]["implementation_status"] != "ready":
                unresolved.add(f"method:{method_id}:implementation")
    return sorted(unresolved)


def expand_matrix(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = validate_spec(spec)
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for suite_id in sorted(indexed["suites"]):
        suite = indexed["suites"][suite_id]
        axes = itertools.product(
            suite["models"],
            suite["hardware"],
            suite["methods"],
            suite["contexts"],
            suite["batch_sizes"],
            suite["decode_steps"],
            suite["workloads"],
            range(int(suite["fresh_process_replicates"])),
        )
        for (
            model_id,
            hardware_id,
            method_id,
            context,
            batch_size,
            decode_steps,
            workload,
            replicate,
        ) in axes:
            identity = {
                "suite_id": suite_id,
                "model_id": model_id,
                "hardware_id": hardware_id,
                "method_id": method_id,
                "context": int(context),
                "batch_size": int(batch_size),
                "decode_steps": int(decode_steps),
                "workload": workload,
                "fresh_process_replicate": replicate,
            }
            digest = canonical_sha256(identity)
            run_id = f"{suite_id}-{digest[:16]}"
            require(run_id not in seen, f"duplicate expanded run id: {run_id}")
            seen.add(run_id)
            pair_identity = {
                key: value for key, value in identity.items() if key != "method_id"
            }
            runs.append(
                {
                    "run_id": run_id,
                    "run_identity_sha256": digest,
                    "pair_group_sha256": canonical_sha256(pair_identity),
                    "priority": suite["priority"],
                    "phase": suite["phase"],
                    "runner_status": suite["runner_status"],
                    "identity": identity,
                    "result_path": f"raw/{suite_id}/{run_id}.json",
                    "status": "planned",
                }
            )
    require(bool(runs), "expanded matrix is empty")
    return runs


def _source_closure(source_paths: Sequence[Path]) -> dict[str, str]:
    closure: dict[str, str] = {}
    for raw_path in source_paths:
        path = raw_path.resolve()
        require(path.is_file(), f"source-closure file is missing: {path}")
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        require(key not in closure, f"duplicate source-closure key: {key}")
        closure[key] = sha256_file(path)
    return closure


def build_manifest(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    validate_spec(spec)
    runs = expand_matrix(spec)
    unresolved = unresolved_requirements(spec)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "design" if unresolved else "ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_file(spec_path.resolve()),
        "spec_canonical_sha256": canonical_sha256(spec),
        "source_sha256": _source_closure(source_paths),
        "unresolved_requirements": unresolved,
        "runnable": not unresolved,
        "result_contract": {
            "fresh_process_identity_required": True,
            "raw_samples_required": True,
            "model_revision_and_config_hash_required": True,
            "dataset_and_workload_hashes_required": True,
            "cache_tensor_manifest_required": True,
            "correctness_and_canary_gates_required": True,
            "wall_and_device_timing_required": True,
            "memory_bytes_required": True,
            "output_length_and_task_quality_required_for_quality_suites": True,
            "source_closure_required": True,
        },
        "run_count": len(runs),
        "runs": runs,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    require(
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "manifest schema mismatch",
    )
    require(manifest.get("experiment") == EXPERIMENT, "manifest experiment mismatch")
    expected = manifest.get("manifest_sha256")
    require(isinstance(expected, str) and len(expected) == 64, "manifest hash missing")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    require(canonical_sha256(unhashed) == expected, "manifest hash integrity failure")
    runs = _sequence(manifest.get("runs"), "manifest runs")
    require(len(runs) == manifest.get("run_count"), "manifest run count mismatch")
    ids = [
        _nonempty_string(_mapping(run, "manifest run").get("run_id"), "run id")
        for run in runs
    ]
    require(len(ids) == len(set(ids)), "manifest run ids are not unique")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    verify_manifest(manifest)
    phase_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for raw in manifest["runs"]:
        run = _mapping(raw, "manifest run")
        phase = str(run["phase"])
        priority = str(run["priority"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    return {
        "status": manifest["status"],
        "runnable": manifest["runnable"],
        "run_count": manifest["run_count"],
        "phase_counts": dict(sorted(phase_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "unresolved_requirements": manifest["unresolved_requirements"],
        "manifest_sha256": manifest["manifest_sha256"],
    }

