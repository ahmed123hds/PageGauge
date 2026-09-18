#!/usr/bin/env python3
"""Analyze paired, backend-exclusive PageGauge publication runs.

The orchestration manifest is the source of pairing and chronological order.
Worker JSON is accepted only when it contains raw wall-clock and CUDA-event
latencies for both cache-neutral and cache-hot modes.  The primary estimand is
the mean paired log latency ratio, log(FI / PageGauge), exponentiated to a
geometric-mean speedup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
BASELINE = "flashinfer_fp16"
CANDIDATE = "page_gauge"
BACKENDS = (BASELINE, CANDIDATE)
MODES = ("cache_neutral", "cache_hot")
METRICS = ("wall_ms", "cuda_ms")


class ProtocolError(RuntimeError):
    """Raised when an input cannot support the preregistered paired analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="defaults to RUN_DIR/orchestration_manifest.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=5090)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to RUN_DIR/publication_analysis.json",
    )
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ProtocolError("cannot compute a quantile of an empty sequence")
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


def geometric_mean(values: Sequence[float]) -> float:
    checked = _positive_finite(values, "geometric-mean values")
    return math.exp(statistics.fmean(math.log(value) for value in checked))


def _positive_finite(values: Iterable[Any], label: str) -> list[float]:
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ProtocolError(f"{label}[{index}] is boolean, not a latency")
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise ProtocolError(f"{label}[{index}] is not numeric: {value!r}") from error
        if not math.isfinite(converted) or converted <= 0.0:
            raise ProtocolError(
                f"{label}[{index}] must be positive and finite, got {converted!r}"
            )
        result.append(converted)
    if not result:
        raise ProtocolError(f"{label} is empty")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ProtocolError(f"{label} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _lookup_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_path(
    value: Mapping[str, Any], paths: Sequence[Sequence[str]], label: str
) -> Any:
    present: list[tuple[Sequence[str], Any]] = []
    for path in paths:
        candidate = _lookup_path(value, path)
        if candidate is not None:
            present.append((path, candidate))
    if not present:
        rendered = ", ".join(".".join(path) for path in paths)
        raise ProtocolError(f"missing {label}; expected one of: {rendered}")
    if len(present) > 1:
        first = present[0][1]
        if any(candidate != first for _, candidate in present[1:]):
            rendered = ", ".join(".".join(path) for path, _ in present)
            raise ProtocolError(f"ambiguous conflicting {label} at {rendered}")
    return present[0][1]


def _mode_payload(result: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    aliases = (mode, "cache_neutral_start") if mode == "cache_neutral" else (mode,)
    modes = _mapping(result.get("timing_modes"), "worker timing_modes")
    found = [(alias, modes[alias]) for alias in aliases if alias in modes]
    if not found:
        raise ProtocolError(f"worker result lacks timing mode {mode!r}")
    if len(found) > 1 and found[0][1] != found[1][1]:
        raise ProtocolError(
            f"worker result has conflicting aliases for timing mode {mode!r}"
        )
    return _mapping(found[0][1], f"timing_modes.{found[0][0]}")


def extract_mode_samples(
    result: Mapping[str, Any], mode: str
) -> dict[str, list[float]]:
    """Extract the strict publication worker raw-sample contract.

    The preferred schema is::

        timing_modes.MODE.raw_samples = [
          {"sample_index": 0, "wall_ms": ..., "cuda_ms": ...}, ...
        ]

    For compatibility during worker integration, parallel raw_wall_ms and
    raw_cuda_ms vectors are also accepted.  Aggregates without raw samples are
    deliberately rejected.
    """

    payload = _mode_payload(result, mode)
    if "raw_samples" in payload:
        raw = payload["raw_samples"]
        if not isinstance(raw, list) or not raw:
            raise ProtocolError(f"timing_modes.{mode}.raw_samples must be non-empty")
        ordered: list[tuple[int, float, float]] = []
        for fallback_index, item in enumerate(raw):
            record = _mapping(item, f"timing_modes.{mode}.raw_samples[{fallback_index}]")
            sample_index = record.get("sample_index", fallback_index)
            if not isinstance(sample_index, int) or isinstance(sample_index, bool):
                raise ProtocolError(f"invalid sample_index {sample_index!r}")
            wall = _positive_finite(
                [record.get("wall_ms")], f"{mode}.raw_samples[{fallback_index}].wall_ms"
            )[0]
            cuda = _positive_finite(
                [record.get("cuda_ms")], f"{mode}.raw_samples[{fallback_index}].cuda_ms"
            )[0]
            ordered.append((sample_index, wall, cuda))
        ordered.sort(key=lambda item: item[0])
        indices = [item[0] for item in ordered]
        if indices != list(range(len(indices))):
            raise ProtocolError(
                f"timing_modes.{mode}.raw_samples indices must be contiguous from zero"
            )
        return {
            "wall_ms": [item[1] for item in ordered],
            "cuda_ms": [item[2] for item in ordered],
        }

    wall = _first_path(
        payload,
        (("raw_wall_ms",), ("wall_ms",)),
        f"raw wall samples for {mode}",
    )
    cuda = _first_path(
        payload,
        (("raw_cuda_ms",), ("cuda_ms",), ("raw_gpu_ms",)),
        f"raw CUDA samples for {mode}",
    )
    if not isinstance(wall, list) or not isinstance(cuda, list):
        raise ProtocolError(f"timing_modes.{mode} raw latency fields must be arrays")
    checked_wall = _positive_finite(wall, f"{mode}.raw_wall_ms")
    checked_cuda = _positive_finite(cuda, f"{mode}.raw_cuda_ms")
    if len(checked_wall) != len(checked_cuda):
        raise ProtocolError(
            f"timing_modes.{mode} has {len(checked_wall)} wall and "
            f"{len(checked_cuda)} CUDA samples"
        )
    return {"wall_ms": checked_wall, "cuda_ms": checked_cuda}


def worker_backend(result: Mapping[str, Any]) -> str:
    backend = _first_path(
        result,
        (("backend",), ("config", "backend")),
        "worker backend",
    )
    if backend not in BACKENDS:
        raise ProtocolError(f"unknown worker backend {backend!r}")
    return str(backend)


def worker_passed(result: Mapping[str, Any]) -> bool:
    value = result.get("passed")
    if value is None:
        value = _lookup_path(result, ("correctness", "passed"))
    if not isinstance(value, bool):
        raise ProtocolError("worker pass/fail gate must be boolean")
    return value


def worker_seed(result: Mapping[str, Any]) -> int:
    value = _first_path(result, (("seed",), ("config", "seed")), "worker seed")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError("worker seed must be an integer")
    return value


def worker_provenance(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract backend-independent identity used to reject mismatched pairs."""

    token_hash = _first_path(
        result,
        (
            ("token_ids_sha256",),
            ("token_source", "token_ids_sha256"),
            ("fixture_provenance", "token_ids_sha256"),
        ),
        "token IDs SHA256",
    )
    fields: dict[str, Any] = {
        "token_ids_sha256": _sha256(token_hash, "token IDs SHA256")
    }
    required_paths: dict[str, tuple[tuple[str, ...], ...]] = {
        "model": (("model",), ("config", "model")),
        "model_revision": (
            ("model_revision",),
            ("config", "model_revision"),
        ),
        "model_config_sha256": (
            ("model_config_sha256",),
            ("config", "model_config_sha256"),
        ),
        "batch_size": (("batch_size",), ("config", "batch_size")),
        "context": (("context",), ("config", "context")),
        "decode_steps": (("decode_steps",), ("config", "decode_steps")),
        "exact_tail_tokens": (
            ("exact_tail_tokens",),
            ("exact_tail",),
            ("config", "exact_tail_tokens"),
            ("config", "exact_tail"),
        ),
    }
    for name, paths in required_paths.items():
        fields[name] = _first_path(result, paths, name)
    fixture = _mapping(result.get("fixture_provenance"), "fixture_provenance")
    fields["sampled_boundary_kv_sha256"] = _sha256(
        _first_path(
            fixture,
            (("sampled_boundary_kv_sha256",),),
            "sampled boundary K/V SHA256",
        ),
        "sampled boundary K/V SHA256",
    )
    correctness = _mapping(result.get("correctness"), "worker correctness")
    correctness_hashes = _mapping(
        correctness.get("hashes"), "worker correctness.hashes"
    )
    trajectory_hashes = {
        name: _sha256(
            correctness_hashes.get(name), f"worker correctness.hashes.{name}"
        )
        for name in (
            "hf_generated_tokens_sha256",
            "graph_generated_tokens_sha256",
        )
    }
    if (
        trajectory_hashes["hf_generated_tokens_sha256"]
        != trajectory_hashes["graph_generated_tokens_sha256"]
    ):
        raise ProtocolError(
            "worker HF and graph generated-token trajectories are not identical"
        )
    fields["generated_trajectory_sha256"] = trajectory_hashes
    config = _mapping(result.get("config"), "worker config")
    fields["timing_configuration"] = {
        key: config.get(key)
        for key in (
            "prefill_chunk_tokens",
            "baseline_split_pages",
            "candidate_split_pages",
            "tail_attention",
            "cuda_graph_scope",
            "token_offset",
            "token_stride",
            "warmups",
            "repeats",
            "cache_scrub_mib",
        )
    }
    if any(value is None for value in fields["timing_configuration"].values()):
        raise ProtocolError("worker config lacks a required timing identity field")
    source_hashes = _mapping(result.get("source_sha256"), "worker source_sha256")
    if not source_hashes:
        raise ProtocolError("worker source_sha256 must not be empty")
    for path, digest in source_hashes.items():
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ProtocolError(f"invalid source hash record {path!r}: {digest!r}")
    fields["source_sha256"] = dict(sorted(source_hashes.items()))
    return fields


def validate_worker_evidence(
    result: Mapping[str, Any],
    expected_backend: str,
    expected_orchestration: Mapping[str, str] | None,
) -> None:
    exclusivity = _mapping(result.get("exclusivity"), "worker exclusivity evidence")
    required_exclusivity = {
        "fresh_process_required": True,
        "opposite_backend_full_gpu_cache_allocated": False,
        "hf_dynamic_caches_released_before_decoder_construction": True,
        "direct_destination_construction": True,
    }
    for key, expected in required_exclusivity.items():
        if exclusivity.get(key) is not expected:
            raise ProtocolError(
                f"worker exclusivity.{key} must be {expected!r}, "
                f"got {exclusivity.get(key)!r}"
            )
    if exclusivity.get("selected_persistent_backend") != expected_backend:
        raise ProtocolError(
            "worker selected persistent backend does not match its scheduled backend"
        )

    residency = _mapping(result.get("residency_gate"), "worker residency_gate")
    if residency.get("passed") is not True:
        raise ProtocolError("worker per-layer residency gate did not pass")
    selected_layers = residency.get("selected_layers")
    try:
        selected_layer_set = (
            {int(value) for value in selected_layers}
            if isinstance(selected_layers, list)
            else set()
        )
    except (TypeError, ValueError) as error:
        raise ProtocolError("worker residency selected_layers is malformed") from error
    if not {0, 15, 20, 23, 31}.issubset(selected_layer_set):
        raise ProtocolError(
            "worker residency gate must cover layers 0,15,20,23,31"
        )

    memory = _mapping(result.get("memory"), "worker memory evidence")
    required_snapshots = (
        "process_start",
        "model_loaded",
        "selected_backend_cache_allocated",
        "after_hf_dynamic_caches_released",
        "decoder_constructed",
        "attention_graphs_captured",
        "after_residency_gate",
        "before_timing",
        "after_timing",
    )
    required_memory_fields = (
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "cuda_mem_get_info_free_bytes",
        "cuda_mem_get_info_total_bytes",
    )
    for snapshot_name in required_snapshots:
        snapshot = _mapping(
            memory.get(snapshot_name), f"worker memory.{snapshot_name}"
        )
        for field in required_memory_fields:
            value = snapshot.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or (field == "cuda_mem_get_info_total_bytes" and value == 0)
            ):
                raise ProtocolError(
                    f"worker memory.{snapshot_name}.{field} is invalid: {value!r}"
                )

    if expected_orchestration is not None:
        environment = _mapping(result.get("environment"), "worker environment")
        recorded = _mapping(
            environment.get("orchestration_environment"),
            "worker environment.orchestration_environment",
        )
        for key, expected in expected_orchestration.items():
            if recorded.get(key) != expected:
                raise ProtocolError(
                    f"worker orchestration environment mismatch for {key}: "
                    f"expected {expected!r}, got {recorded.get(key)!r}"
                )


def validate_worker_result(
    result: Mapping[str, Any],
    expected_backend: str,
    expected_seed: int,
    expected_orchestration: Mapping[str, str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    if worker_backend(result) != expected_backend:
        raise ProtocolError(
            f"worker backend mismatch: expected {expected_backend}, "
            f"got {worker_backend(result)}"
        )
    if worker_seed(result) != expected_seed:
        raise ProtocolError(
            f"worker seed mismatch: expected {expected_seed}, got {worker_seed(result)}"
        )
    if not worker_passed(result):
        raise ProtocolError(f"worker correctness/residency gate failed for {expected_backend}")
    validate_worker_evidence(result, expected_backend, expected_orchestration)
    extracted = {mode: extract_mode_samples(result, mode) for mode in MODES}
    config = _mapping(result.get("config"), "worker config")
    repeats = config.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ProtocolError("worker config.repeats must be a positive integer")
    for mode in MODES:
        if len(extracted[mode]["wall_ms"]) != len(extracted[mode]["cuda_ms"]):
            raise ProtocolError(f"raw sample count mismatch in {mode}")
        if len(extracted[mode]["wall_ms"]) != repeats:
            raise ProtocolError(
                f"{mode} has {len(extracted[mode]['wall_ms'])} raw samples, "
                f"but config.repeats is {repeats}"
            )
    worker_provenance(result)
    return extracted


def expected_orchestration_environment(
    config_sha256: str, block: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "PAGE_GAUGE_PUBLICATION_CONFIG_SHA256": config_sha256,
        "PAGE_GAUGE_PUBLICATION_BLOCK_ID": str(block["block_id"]),
        "PAGE_GAUGE_PUBLICATION_PAIR_ID": str(block["pair_id"]),
        "PAGE_GAUGE_PUBLICATION_PAIR_ORDER": str(block["pair_order"]),
        "PAGE_GAUGE_PUBLICATION_SLOT": str(block["slot"]),
        "PAGE_GAUGE_PUBLICATION_BACKEND": str(block["backend"]),
    }


def _bootstrap_ci(
    logs: Sequence[float], samples: int, seed: int
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not logs:
        raise ProtocolError("bootstrap requires paired log-speedup values")
    rng = random.Random(seed)
    count = len(logs)
    draws = [
        statistics.fmean(logs[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    draws.sort()
    lower_log = quantile(draws, 0.025)
    upper_log = quantile(draws, 0.975)
    return {
        "method": "paired-block nonparametric percentile bootstrap",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "log_speedup_95_ci": [lower_log, upper_log],
        "speedup_95_ci": [math.exp(lower_log), math.exp(upper_log)],
    }


def _hierarchical_bootstrap_ci(
    pair_records: Sequence[Mapping[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for record in pair_records:
        by_seed[int(record["seed"])].append(float(record["log_speedup"]))
    seeds = sorted(by_seed)
    if not seeds:
        raise ProtocolError("hierarchical bootstrap has no seed clusters")
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


def _stratified_geomean(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(float(record["log_speedup"]))
    return {
        value: {
            "pairs": len(logs),
            "mean_log_speedup": statistics.fmean(logs),
            "speedup_geomean": math.exp(statistics.fmean(logs)),
        }
        for value, logs in sorted(grouped.items())
    }


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProtocolError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON in {label} {path}: {error}") from error
    return _mapping(value, label)


def _completed_result_path(
    run_dir: Path, manifest: Mapping[str, Any], block: Mapping[str, Any]
) -> Path:
    executions = manifest.get("executions")
    if not isinstance(executions, Mapping):
        raise ProtocolError("manifest lacks executions mapping")
    execution = _mapping(
        executions.get(str(block["block_id"])),
        f"execution for {block['block_id']}",
    )
    if execution.get("status") != "completed":
        raise ProtocolError(
            f"block {block['block_id']} status is {execution.get('status')!r}, not completed"
        )
    relative = execution.get("worker_result_path")
    if not isinstance(relative, str) or not relative:
        raise ProtocolError(f"block {block['block_id']} lacks worker_result_path")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ProtocolError(f"worker result escapes run directory: {path}") from error
    expected_hash = execution.get("worker_result_sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ProtocolError(f"worker result hash mismatch for block {block['block_id']}")
    return path


def analyze_run_directory(
    run_dir: Path,
    manifest_path: Path | None = None,
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 5090,
    write_outputs: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else run_dir / "orchestration_manifest.json"
    )
    manifest = _load_json(manifest_path, "orchestration manifest")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ProtocolError("manifest schedule must be a non-empty array")

    block_records: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    provenance_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for chronological_index, raw_block in enumerate(schedule):
        block = _mapping(raw_block, f"schedule[{chronological_index}]")
        block_id = str(block.get("block_id"))
        backend = str(block.get("backend"))
        seed = block.get("seed")
        if backend not in BACKENDS:
            raise ProtocolError(f"schedule block {block_id} has invalid backend {backend!r}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ProtocolError(f"schedule block {block_id} has invalid seed")
        result_path = _completed_result_path(run_dir, manifest, block)
        result = _load_json(result_path, f"worker result for {block_id}")
        config_sha256 = manifest.get("config_sha256")
        if not isinstance(config_sha256, str):
            raise ProtocolError("manifest lacks config_sha256")
        samples = validate_worker_result(
            result,
            backend,
            seed,
            expected_orchestration_environment(config_sha256, block),
        )
        pair_id = str(block.get("pair_id"))
        provenance_by_pair[pair_id][backend] = worker_provenance(result)
        record = {
            "block_id": block_id,
            "chronological_index": chronological_index,
            "pair_id": pair_id,
            "pair_index": int(block["pair_index"]),
            "seed": seed,
            "seed_index": int(block["seed_index"]),
            "pair_order": str(block["pair_order"]),
            "slot": int(block["slot"]),
            "backend": backend,
            "token_offset": int(block["token_offset"]),
            "worker_result_path": str(result_path.relative_to(run_dir)),
            "worker_result_sha256": sha256_file(result_path),
            "samples": samples,
        }
        block_records[block_id] = record
        for mode in MODES:
            for metric in METRICS:
                for sample_index, latency in enumerate(samples[mode][metric]):
                    raw_rows.append(
                        {
                            "block_id": block_id,
                            "chronological_index": chronological_index,
                            "pair_id": pair_id,
                            "pair_index": int(block["pair_index"]),
                            "seed": seed,
                            "seed_index": int(block["seed_index"]),
                            "pair_order": str(block["pair_order"]),
                            "slot": int(block["slot"]),
                            "backend": backend,
                            "token_offset": int(block["token_offset"]),
                            "mode": mode,
                            "metric": metric,
                            "sample_index": sample_index,
                            "latency_ms": latency,
                        }
                    )

    for pair_id, provenance in provenance_by_pair.items():
        if set(provenance) != set(BACKENDS):
            raise ProtocolError(f"pair {pair_id} does not contain both backends")
        if provenance[BASELINE] != provenance[CANDIDATE]:
            raise ProtocolError(
                f"pair {pair_id} has mismatched model/token/config provenance: "
                f"{provenance}"
            )

    scheduled_pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw_block in schedule:
        block = _mapping(raw_block, "schedule block")
        scheduled_pairs[str(block["pair_id"])].append(block)

    pair_rows: list[dict[str, Any]] = []
    for pair_id, blocks in sorted(
        scheduled_pairs.items(), key=lambda item: min(int(x["chronological_index"]) for x in item[1])
    ):
        if len(blocks) != 2 or {str(block["backend"]) for block in blocks} != set(BACKENDS):
            raise ProtocolError(f"pair {pair_id} is not one FI/PG adjacent pair")
        chronological = sorted(blocks, key=lambda block: int(block["chronological_index"]))
        if int(chronological[1]["chronological_index"]) != int(
            chronological[0]["chronological_index"]
        ) + 1:
            raise ProtocolError(f"pair {pair_id} blocks are not temporally adjacent")
        order = "".join(
            "A" if str(block["backend"]) == BASELINE else "B"
            for block in chronological
        )
        if order != str(chronological[0]["pair_order"]):
            raise ProtocolError(f"pair {pair_id} order metadata mismatch")
        by_backend = {
            str(block["backend"]): block_records[str(block["block_id"])]
            for block in blocks
        }
        for mode in MODES:
            for metric in METRICS:
                baseline_values = by_backend[BASELINE]["samples"][mode][metric]
                candidate_values = by_backend[CANDIDATE]["samples"][mode][metric]
                if len(baseline_values) != len(candidate_values):
                    raise ProtocolError(
                        f"pair {pair_id} has unequal {mode}/{metric} raw sample "
                        f"counts: FI={len(baseline_values)}, "
                        f"PageGauge={len(candidate_values)}"
                    )
                baseline_log = statistics.fmean(math.log(x) for x in baseline_values)
                candidate_log = statistics.fmean(math.log(x) for x in candidate_values)
                log_speedup = baseline_log - candidate_log
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "pair_index": int(chronological[0]["pair_index"]),
                        "seed": int(chronological[0]["seed"]),
                        "seed_index": int(chronological[0]["seed_index"]),
                        "token_offset": int(chronological[0]["token_offset"]),
                        "pair_order": order,
                        "williams_sequence": str(
                            chronological[0]["williams_sequence"]
                        ),
                        "quartet_index": int(chronological[0]["quartet_index"]),
                        "first_chronological_index": int(
                            chronological[0]["chronological_index"]
                        ),
                        "baseline_block_id": by_backend[BASELINE]["block_id"],
                        "candidate_block_id": by_backend[CANDIDATE]["block_id"],
                        "mode": mode,
                        "metric": metric,
                        "baseline_sample_count": len(baseline_values),
                        "candidate_sample_count": len(candidate_values),
                        "baseline_latency_geomean_ms": math.exp(baseline_log),
                        "candidate_latency_geomean_ms": math.exp(candidate_log),
                        "log_speedup": log_speedup,
                        "speedup": math.exp(log_speedup),
                    }
                )

    pair_chronology = sorted(
        {
            (str(row["pair_id"]), int(row["first_chronological_index"]))
            for row in pair_rows
        },
        key=lambda item: item[1],
    )
    chronology_labels = {
        pair_id: ("first_half" if index < len(pair_chronology) / 2 else "second_half")
        for index, (pair_id, _chronological_index) in enumerate(pair_chronology)
    }
    for row in pair_rows:
        row["chronology_half"] = chronology_labels[str(row["pair_id"])]

    aggregates: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        aggregates[mode] = {}
        for metric_index, metric in enumerate(METRICS):
            selected = [
                row
                for row in pair_rows
                if row["mode"] == mode and row["metric"] == metric
            ]
            logs = [float(row["log_speedup"]) for row in selected]
            mean_log = statistics.fmean(logs)
            seed_offset = mode_index * 1000 + metric_index * 100
            aggregates[mode][metric] = {
                "estimand": "exp(mean adjacent-pair log(FI_latency/PG_latency))",
                "pair_count": len(selected),
                "seed_cluster_count": len({int(row["seed"]) for row in selected}),
                "mean_log_speedup": mean_log,
                "speedup_geomean": math.exp(mean_log),
                "paired_block_bootstrap": _bootstrap_ci(
                    logs, bootstrap_samples, bootstrap_seed + seed_offset
                ),
                "hierarchical_seed_pair_bootstrap": _hierarchical_bootstrap_ci(
                    selected,
                    bootstrap_samples,
                    bootstrap_seed + seed_offset + 50,
                ),
                "order_stratified": _stratified_geomean(selected, "pair_order"),
                "williams_sequence_stratified": _stratified_geomean(
                    selected, "williams_sequence"
                ),
                "chronology_half_stratified": _stratified_geomean(
                    selected, "chronology_half"
                ),
                "seed_stratified": _stratified_geomean(selected, "seed"),
            }

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "page_gauge_backend_exclusive_paired_analysis",
        "claim_scope": (
            "backend-exclusive fresh-process fixed-batch decode timing; paired "
            "log latency ratio with Williams-order control"
        ),
        "primary_mode": "cache_neutral",
        "primary_metric": "wall_ms",
        "backend_symbols": {"A": BASELINE, "B": CANDIDATE},
        "manifest_path": str(manifest_path.relative_to(run_dir)),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_config_sha256": manifest.get("config_sha256"),
        "raw_sample_count": len(raw_rows),
        "paired_record_count": len(pair_rows),
        "aggregates": aggregates,
        "pair_records": pair_rows,
        "worker_result_records": [
            {
                key: value
                for key, value in record.items()
                if key != "samples"
            }
            for record in sorted(
                block_records.values(), key=lambda item: item["chronological_index"]
            )
        ],
        "protocol_notes": {
            "latency_reduction": (
                "each backend block is reduced by the arithmetic mean of log raw "
                "latencies; adjacent backend-block differences are the paired units"
            ),
            "bootstrap": (
                "both a paired-block bootstrap and a hierarchical seed-then-pair "
                "bootstrap are reported; neither treats within-block repeats as "
                "independent experimental units"
            ),
            "positive_speedup_direction": "greater than one favors PageGauge",
        },
    }
    analysis["analysis_payload_sha256"] = canonical_json_sha256(analysis)

    if write_outputs:
        output_path = output_path or run_dir / "publication_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_csv(run_dir / "raw_latency_samples.csv", raw_rows)
        _write_csv(run_dir / "paired_log_speedups.csv", pair_rows)
    return analysis


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ProtocolError(f"refusing to write empty CSV {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output is not None else None
    try:
        analysis = analyze_run_directory(
            run_dir,
            args.manifest,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            output_path=output,
        )
    except ProtocolError as error:
        raise SystemExit(f"publication analysis failed closed: {error}") from error
    primary = analysis["aggregates"]["cache_neutral"]["wall_ms"]
    print(
        "Backend-exclusive cache-neutral wall speedup: "
        f"{primary['speedup_geomean']:.4f}x; hierarchical 95% CI "
        f"[{primary['hierarchical_seed_pair_bootstrap']['speedup_95_ci'][0]:.4f}, "
        f"{primary['hierarchical_seed_pair_bootstrap']['speedup_95_ci'][1]:.4f}]"
    )


if __name__ == "__main__":
    main()
