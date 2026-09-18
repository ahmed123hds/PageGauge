#!/usr/bin/env python3
"""Pure-CPU, provisional reducer for one MLSys controlled-decoder contrast cell."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import mlsys_controlled_protocol as protocol


MODES = ("cache_neutral", "cache_hot")
METRICS = ("wall_ms", "cuda_ms")
REQUIRED_ARTIFACT_HASHES = {"stdout", "stderr", "telemetry"}
REQUIRED_EXECUTION_GATES = {
    "exclusive_device", "correctness", "cache_accounting", "no_fallback"
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise protocol.ProtocolError(f"{label} must be an object")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise protocol.ProtocolError(f"{label} must be a SHA-256 hex digest")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise protocol.ProtocolError(f"{label} must be a SHA-256 hex digest")
    return lowered


def _nonnegative_int(value: Any, label: str) -> int:
    # Integer type checks reject NaN/Inf, fractions, booleans, and numeric strings.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise protocol.ProtocolError(f"{label} must be a nonnegative integer")
    return value


def _verify_evidence_file(root: Path, relative_path: Any, expected: str, label: str) -> None:
    if not isinstance(relative_path, str) or not relative_path:
        raise protocol.ProtocolError(f"{label} needs a relative evidence path")
    portable = PurePosixPath(relative_path.replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts or ":" in relative_path:
        raise protocol.ProtocolError(f"{label} evidence path must remain within evidence_root")
    path = (root / portable).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise protocol.ProtocolError(f"{label} evidence file missing or outside evidence_root")
    try:
        observed = protocol.sha256_file(path)
    except OSError as error:
        raise protocol.ProtocolError(f"{label} evidence file is unreadable") from error
    if observed != expected:
        raise protocol.ProtocolError(f"{label} evidence SHA256 mismatch")


def verify_cell_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    evidence_root: Path,
    frozen_source_sha256: Mapping[str, str],
) -> None:
    """Verify actual bytes against an independently supplied frozen closure.

    Source paths and terminal.artifact_paths are relative to evidence_root.
    The caller must supply the expected closure independently of result records.
    This proves byte identity, not that the closure covers an integrated runner.
    """
    root = evidence_root.resolve()
    if not root.is_dir():
        raise protocol.ProtocolError("evidence_root must be an existing directory")
    closure = _mapping(frozen_source_sha256, "frozen_source_sha256")
    if not closure:
        raise protocol.ProtocolError("frozen source closure is empty")
    expected = {
        path: _sha256(digest, f"frozen_source_sha256.{path}")
        for path, digest in closure.items()
    }
    for path, digest in expected.items():
        _verify_evidence_file(root, path, digest, f"source {path}")
    for index, record in enumerate(records):
        source_hashes = _mapping(record.get("source_sha256"), "source_sha256")
        observed = {
            path: _sha256(digest, f"source_sha256.{path}")
            for path, digest in source_hashes.items()
        }
        if observed != expected:
            raise protocol.ProtocolError(f"records[{index}] source closure differs from frozen closure")
        terminal = _mapping(record.get("terminal"), "terminal")
        hashes = _mapping(terminal.get("artifact_sha256s"), "terminal.artifact_sha256s")
        paths = _mapping(terminal.get("artifact_paths"), "terminal.artifact_paths")
        if not REQUIRED_ARTIFACT_HASHES.issubset(hashes) or set(paths) != set(hashes):
            raise protocol.ProtocolError(f"records[{index}] artifact paths/hashes are incomplete")
        for label, digest in hashes.items():
            _verify_evidence_file(
                root, paths[label], _sha256(digest, f"artifact {label}"),
                f"records[{index}] artifact {label}",
            )


def _positive_finite(values: Iterable[Any], label: str) -> list[float]:
    checked: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise protocol.ProtocolError(f"{label}[{index}] is boolean")
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise protocol.ProtocolError(f"{label}[{index}] is not numeric") from error
        if not math.isfinite(converted) or converted <= 0.0:
            raise protocol.ProtocolError(f"{label}[{index}] must be positive and finite")
        checked.append(converted)
    if not checked:
        raise protocol.ProtocolError(f"{label} is empty")
    return checked


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear quantile, unchanged from the audited publication reducer."""

    if not sorted_values:
        raise protocol.ProtocolError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _hierarchical_bootstrap_ci(
    pair_records: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    """Seed-fixture then adjacent-pair bootstrap from the audited reducer."""

    by_seed: dict[int, list[float]] = defaultdict(list)
    for record in pair_records:
        by_seed[int(record["seed"])].append(float(record["log_speedup"]))
    seeds = sorted(by_seed)
    if not seeds:
        raise protocol.ProtocolError("hierarchical bootstrap has no seed clusters")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled_logs: list[float] = []
        for _cluster_index in range(len(seeds)):
            selected_seed = seeds[rng.randrange(len(seeds))]
            cluster = by_seed[selected_seed]
            sampled_logs.extend(
                cluster[rng.randrange(len(cluster))] for _ in range(len(cluster))
            )
        draws.append(statistics.fmean(sampled_logs))
    draws.sort()
    lower_log = quantile(draws, 0.025)
    upper_log = quantile(draws, 0.975)
    return {
        "method": "hierarchical percentile bootstrap: seed then paired block",
        "cluster_unit": "seed/corpus-window fixture",
        "within_cluster_unit": "adjacent backend pair",
        "seed_cluster_count": len(seeds),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "log_speedup_95_ci": [lower_log, upper_log],
        "speedup_95_ci": [math.exp(lower_log), math.exp(upper_log)],
        "pilot_warning": (
            "one seed cannot estimate between-seed variation"
            if len(seeds) == 1
            else None
        ),
    }


def _extract_samples(result: Mapping[str, Any], repeats: int) -> dict[str, dict[str, list[float]]]:
    modes = _mapping(result.get("timing_modes"), "timing_modes")
    extracted: dict[str, dict[str, list[float]]] = {}
    for mode in MODES:
        payload = _mapping(modes.get(mode), f"timing_modes.{mode}")
        raw = payload.get("raw_samples")
        if not isinstance(raw, list) or not raw:
            raise protocol.ProtocolError(
                f"timing_modes.{mode} requires raw_samples; aggregates are invalid"
            )
        ordered: list[tuple[int, float, float]] = []
        for fallback_index, item in enumerate(raw):
            record = _mapping(item, f"timing_modes.{mode}.raw_samples[{fallback_index}]")
            sample_index = record.get("sample_index", fallback_index)
            if not isinstance(sample_index, int) or isinstance(sample_index, bool):
                raise protocol.ProtocolError(f"invalid sample_index {sample_index!r}")
            wall = _positive_finite([record.get("wall_ms")], f"{mode}.wall_ms")[0]
            cuda = _positive_finite([record.get("cuda_ms")], f"{mode}.cuda_ms")[0]
            ordered.append((sample_index, wall, cuda))
        ordered.sort(key=lambda item: item[0])
        if [item[0] for item in ordered] != list(range(repeats)):
            raise protocol.ProtocolError(
                f"timing_modes.{mode} samples must be contiguous and total {repeats}"
            )
        extracted[mode] = {
            "wall_ms": [item[1] for item in ordered],
            "cuda_ms": [item[2] for item in ordered],
        }
    return extracted


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise protocol.ProtocolError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise protocol.ProtocolError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise protocol.ProtocolError(f"{label} must include a timezone")
    return parsed


def validate_cell_records(
    matrix: Mapping[str, Any], cell_id: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Validate structure and reported assertions, not source/artifact file bytes."""
    cells = protocol.validate_matrix(matrix)
    if cell_id not in cells:
        raise protocol.ProtocolError(f"unknown cell {cell_id}")
    expected = [block for block in protocol.expand_blocks(matrix) if block["cell_id"] == cell_id]
    if len(records) != len(expected):
        raise protocol.ProtocolError(
            f"cell {cell_id} requires {len(expected)} fresh-process blocks, got {len(records)}"
        )
    expected_by_id = {block["block_id"]: block for block in expected}
    matrix_hash = protocol.canonical_sha256(matrix)
    repeats = int(matrix["design"]["measured_full_sequences"])
    hardware = matrix["hardware"][cells[cell_id]["hardware_id"]]
    normalized: dict[str, dict[str, Any]] = {}
    pids: set[int] = set()
    for index, raw in enumerate(records):
        result = _mapping(raw, f"records[{index}]")
        block_id = result.get("block_id")
        if block_id not in expected_by_id:
            raise protocol.ProtocolError(f"unexpected block_id {block_id!r}")
        if block_id in normalized:
            raise protocol.ProtocolError(f"duplicate block_id {block_id}")
        block = expected_by_id[str(block_id)]
        for key in (
            "cell_id",
            "cell_config_sha256",
            "seed",
            "pair_id",
            "pair_order",
            "slot",
            "treatment",
            "backend",
        ):
            if result.get(key) != block[key]:
                raise protocol.ProtocolError(f"block {block_id} identity mismatch for {key}")
        if result.get("matrix_canonical_sha256") != matrix_hash:
            raise protocol.ProtocolError(f"block {block_id} matrix identity mismatch")
        pid = result.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise protocol.ProtocolError(f"block {block_id} has invalid PID")
        if pid in pids:
            raise protocol.ProtocolError(f"fresh-process PID reused: {pid}")
        pids.add(pid)

        observed = _mapping(result.get("observed_hardware"), "observed_hardware")
        if observed.get("family") != hardware["family"]:
            raise protocol.ProtocolError(f"block {block_id} hardware family mismatch")
        if observed.get("compute_capability") != hardware["compute_capability"]:
            raise protocol.ProtocolError(f"block {block_id} compute capability mismatch")

        source_hashes = _mapping(result.get("source_sha256"), "source_sha256")
        if not source_hashes:
            raise protocol.ProtocolError(f"block {block_id} source closure is empty")
        for label, digest in source_hashes.items():
            _sha256(digest, f"source_sha256.{label}")

        gates = _mapping(result.get("execution_gates"), "execution_gates")
        missing_gates = REQUIRED_EXECUTION_GATES.difference(gates)
        if missing_gates:
            raise protocol.ProtocolError(
                f"block {block_id} lacks required execution gates: {sorted(missing_gates)}"
            )
        if any(value is not True for value in gates.values()):
            raise protocol.ProtocolError(f"block {block_id} execution gate failed")
        recurrence = _mapping(result.get("recurrence"), "recurrence")
        for label, value in recurrence.items():
            _nonnegative_int(value, f"recurrence.{label}")
        closed_pages = _nonnegative_int(
            recurrence.get("closed_generated_pages"), "recurrence.closed_generated_pages"
        )
        int8_pages = _nonnegative_int(
            recurrence.get("runtime_created_pages_later_consumed_int8"),
            "recurrence.runtime_created_pages_later_consumed_int8",
        )
        if closed_pages < matrix["workload"]["minimum_closed_generated_pages"]:
            raise protocol.ProtocolError(f"block {block_id} closed-page recurrence gate failed")
        if int8_pages > closed_pages:
            raise protocol.ProtocolError(f"block {block_id} consumed INT8 pages exceed closed pages")
        if (
            block["backend"] != "flashinfer_fp16"
            and int8_pages < matrix["workload"]["minimum_runtime_created_pages_later_consumed_int8"]
        ):
            raise protocol.ProtocolError(f"block {block_id} runtime INT8 recurrence gate failed")

        terminal = _mapping(result.get("terminal"), "terminal")
        if terminal.get("status") != "completed" or terminal.get("return_code") != 0:
            raise protocol.ProtocolError(f"block {block_id} is not a valid completed attempt")
        hashes = _mapping(terminal.get("artifact_sha256s"), "terminal.artifact_sha256s")
        if not REQUIRED_ARTIFACT_HASHES.issubset(hashes):
            raise protocol.ProtocolError(f"block {block_id} lacks required artifact hashes")
        for label, digest in hashes.items():
            _sha256(digest, f"terminal.artifact_sha256s.{label}")
        started = _parse_time(terminal.get("started_utc"), "terminal.started_utc")
        ended = _parse_time(terminal.get("ended_utc"), "terminal.ended_utc")
        if ended <= started:
            raise protocol.ProtocolError(f"block {block_id} terminal timestamps are inverted")

        pairing_key = _sha256(result.get("pairing_key_sha256"), "pairing_key_sha256")
        normalized[str(block_id)] = {
            **block,
            "pid": pid,
            "started": started,
            "ended": ended,
            "pairing_key_sha256": pairing_key,
            "samples": _extract_samples(result, repeats),
        }

    ordered = [normalized[block["block_id"]] for block in expected]
    for previous, current in zip(ordered, ordered[1:]):
        if current["started"] < previous["ended"]:
            raise protocol.ProtocolError("fresh-process chronology is noncanonical or overlapping")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        by_pair[record["pair_id"]].append(record)
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {row["treatment"] for row in pair} != {"A", "B"}:
            raise protocol.ProtocolError(f"pair {pair_id} lacks one baseline and one candidate")
        if len({row["pairing_key_sha256"] for row in pair}) != 1:
            raise protocol.ProtocolError(f"pair {pair_id} pairing-key mismatch")
    return ordered


def _stratified(pair_records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in pair_records:
        grouped[str(record[key])].append(float(record["log_speedup"]))
    return {
        value: {
            "physical_pairs": len(logs),
            "mean_log_speedup": statistics.fmean(logs),
            "speedup_geomean": math.exp(statistics.fmean(logs)),
        }
        for value, logs in sorted(grouped.items())
    }


def reduce_cell_records(
    matrix: Mapping[str, Any],
    cell_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 5090,
    evidence_root: Path | None = None,
    frozen_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    ordered = validate_cell_records(matrix, cell_id, records)
    if (evidence_root is None) != (frozen_source_sha256 is None):
        raise protocol.ProtocolError("evidence_root and frozen_source_sha256 must be supplied together")
    evidence_verified = evidence_root is not None
    if evidence_verified:
        verify_cell_evidence(
            records, evidence_root=evidence_root, frozen_source_sha256=frozen_source_sha256
        )
    cell = protocol.validate_matrix(matrix)[cell_id]
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        by_pair[record["pair_id"]].append(record)
    endpoints: dict[str, Any] = {}
    for mode in MODES:
        for metric in METRICS:
            pair_records: list[dict[str, Any]] = []
            for pair_id in sorted(by_pair):
                pair = by_pair[pair_id]
                baseline = next(row for row in pair if row["backend"] == cell["baseline"])
                candidate = next(row for row in pair if row["backend"] == cell["candidate"])
                baseline_log = statistics.fmean(
                    math.log(value) for value in baseline["samples"][mode][metric]
                )
                candidate_log = statistics.fmean(
                    math.log(value) for value in candidate["samples"][mode][metric]
                )
                pair_records.append(
                    {
                        "pair_id": pair_id,
                        "seed": baseline["seed"],
                        "pair_order": baseline["pair_order"],
                        "chronology_half": (
                            "first" if baseline["fresh_process_ordinal"] < 4 else "second"
                        ),
                        "log_speedup": baseline_log - candidate_log,
                    }
                )
            mean_log = statistics.fmean(
                record["log_speedup"] for record in pair_records
            )
            interval = _hierarchical_bootstrap_ci(
                pair_records, bootstrap_samples, bootstrap_seed
            )
            threshold = float(
                matrix["gates"][
                    "factorized_vs_fp16_speedup_lower_95"
                    if cell["baseline"] == "flashinfer_fp16"
                    else "factorized_vs_reconstruction_speedup_lower_95"
                ]
            )
            endpoints[f"{mode}.{metric}"] = {
                "physical_pair_count": len(pair_records),
                "point_speedup": math.exp(mean_log),
                "mean_log_speedup": mean_log,
                **interval,
                "strata": {
                    "seed": _stratified(pair_records, "seed"),
                    "pair_order": _stratified(pair_records, "pair_order"),
                    "chronology_half": _stratified(pair_records, "chronology_half"),
                },
                "gate": {
                    "lower_95_must_exceed": threshold,
                    "passed": interval["speedup_95_ci"][0] > threshold,
                },
            }
    return {
        "schema_version": 1,
        "experiment": protocol.EXPERIMENT,
        "status": "provisional",
        "publication_eligible": False,
        "evidence_verification": {
            "status": "verified_file_hashes" if evidence_verified else "not_verified",
            "source_closure_verified_against_frozen_hashes": evidence_verified,
            "frozen_source_closure_sha256": (
                protocol.canonical_sha256(
                    {path: digest.lower() for path, digest in frozen_source_sha256.items()}
                )
                if evidence_verified else None
            ),
            "artifact_files_verified": evidence_verified,
            "execution_source_lock_verified": False,
            "limitations": (
                ["Controlled runner and complete execution source lock are not integrated."]
                if evidence_verified
                else [
                    "Source/artifact hashes are format-checked only; actual files were not verified.",
                    "Controlled runner and complete execution source lock are not integrated.",
                ]
            ),
        },
        "cell_id": cell_id,
        "cell_config_sha256": protocol.cell_config_sha256(matrix, cell_id),
        "contrast": {"baseline": cell["baseline"], "candidate": cell["candidate"]},
        "physical_pair_count": len(by_pair),
        "fresh_process_block_count": len(ordered),
        "metric_record_count": len(by_pair) * len(MODES) * len(METRICS),
        "endpoints": endpoints,
        "all_endpoint_gates_passed": all(
            endpoint["gate"]["passed"] for endpoint in endpoints.values()
        ),
    }
