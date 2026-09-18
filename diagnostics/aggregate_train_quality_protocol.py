#!/usr/bin/env python3
"""Aggregate the frozen PageGauge TRAIN selection/confirmation protocol.

The confirmation decision uses only the 20 pre-registered TRAIN windows.  The
four earlier discovery windows are reported separately as selection evidence.
Bootstrap resampling is over complete windows, never over correlated token
rows.  KL, JS, NLL, and perplexity are not inferred from the sustained worker's
reduced diagnostics; those metrics remain the job of the untouched held-out
full-distribution protocol.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import train_quality_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "diagnostics/run_train_quality_protocol.py"
SHARED = ROOT / "diagnostics/train_quality_protocol.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_status(payload: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    protocol.require(payload.get("schema_version") == 1, "status schema")
    protocol.require(
        payload.get("experiment") == "page_gauge_train_confirmation_run_status",
        "status experiment",
    )
    expected = dict(payload)
    observed = expected.pop("status_sha256", None)
    protocol.require(protocol.is_sha256(observed), "status hash missing")
    protocol.require(
        protocol.canonical_sha256(expected) == observed, "status hash integrity failure"
    )
    protocol.require(
        payload.get("manifest_sha256") == manifest.get("manifest_sha256"),
        "status belongs to another manifest",
    )
    protocol.require(payload.get("complete") is True, "confirmation run is incomplete")
    states = [item.get("state") for item in payload.get("jobs", {}).values()]
    protocol.require(
        len(states) == len(protocol.CONFIRMATION_GROUPS), "status job count"
    )
    protocol.require(
        all(state in {"passed", "failed_quality_gate"} for state in states),
        "status contains a non-terminal job",
    )
    protocol.require(
        payload.get("passed") is all(state == "passed" for state in states),
        "status pass decision mismatch",
    )


def verify_source_locks(manifest: Mapping[str, Any]) -> None:
    raw_lock = manifest["source_lock"]["raw_worker_source_sha256"]
    for relative, expected in raw_lock.items():
        path = ROOT / relative
        protocol.require(path.is_file(), f"locked raw source missing: {relative}")
        protocol.require(
            protocol.sha256_file(path) == expected,
            f"locked raw source changed: {relative}",
        )
    expected_tools = manifest["source_lock"]["protocol_tool_source_sha256"]
    current_tools = {}
    for path in (RUNNER.resolve(), SHARED.resolve(), Path(__file__).resolve()):
        protocol.require(path.is_file(), f"protocol tool missing: {path}")
        current_tools[str(path.relative_to(ROOT)).replace("\\", "/")] = (
            protocol.sha256_file(path)
        )
    protocol.require(current_tools == expected_tools, "protocol tool source lock mismatch")


def validate_inputs(
    manifest: Mapping[str, Any], status: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_lock = manifest["source_lock"]["raw_worker_source_sha256"]
    selection_path = Path(manifest["selection_artifact"]["path"])
    protocol.require(selection_path.is_file(), "selection artifact is missing")
    protocol.require(
        protocol.sha256_file(selection_path)
        == manifest["selection_artifact"]["sha256"],
        "selection artifact hash mismatch",
    )
    selection = protocol.validate_raw_artifact(
        protocol.load_json(selection_path),
        expected_starts=protocol.SELECTION_STARTS,
        cohort="selection",
        label=str(selection_path),
        locked_source_sha256=source_lock,
    )
    protocol.require(
        selection["content_hashes"] == manifest["selection_artifact"]["content_hashes"],
        "selection content hash mismatch",
    )
    protocol.require(
        selection["policy_sha256"] == manifest["policy_sha256"],
        "selection policy hash mismatch",
    )

    confirmations = []
    input_manifest = [
        {
            "cohort": "selection",
            "path": str(selection_path),
            "sha256": protocol.sha256_file(selection_path),
            "window_starts": selection["starts"],
            "rows": selection["rows"],
            "policy_sha256": selection["policy_sha256"],
            "content_hashes": selection["content_hashes"],
            "source_sha256": selection["source_sha256"],
        }
    ]
    for job, expected_starts in zip(
        manifest["jobs"], protocol.CONFIRMATION_GROUPS
    ):
        job_id = job["job_id"]
        job_status = status["jobs"].get(job_id)
        protocol.require(isinstance(job_status, dict), f"missing status for {job_id}")
        protocol.require(
            job_status.get("fresh_worker_process") is True,
            f"{job_id}: fresh process not attested",
        )
        pid = job_status.get("worker_pid")
        protocol.require(isinstance(pid, int) and pid > 0, f"{job_id}: worker PID")
        output = Path(job["output_path"])
        protocol.require(output.is_file(), f"missing confirmation artifact: {output}")
        result_hash = protocol.sha256_file(output)
        protocol.require(
            result_hash == job_status.get("result_sha256"),
            f"{job_id}: result/status hash mismatch",
        )
        validated = protocol.validate_raw_artifact(
            protocol.load_json(output),
            expected_starts=expected_starts,
            cohort="confirmation",
            label=str(output),
            locked_source_sha256=source_lock,
        )
        protocol.require(
            validated["common_signature"] == selection["common_signature"],
            f"{job_id}: selection/confirmation signature mismatch",
        )
        protocol.require(
            validated["content_hashes"] == job_status.get("content_hashes"),
            f"{job_id}: content/status hash mismatch",
        )
        protocol.require(
            validated["policy_sha256"] == manifest["policy_sha256"],
            f"{job_id}: policy hash mismatch",
        )
        expected_state = (
            "passed"
            if validated["quality"]["worker_quality_gate_passed"]
            else "failed_quality_gate"
        )
        protocol.require(
            job_status.get("state") == expected_state,
            f"{job_id}: quality/status decision mismatch",
        )
        confirmations.append(validated)
        input_manifest.append(
            {
                "cohort": "confirmation",
                "job_id": job_id,
                "path": str(output),
                "sha256": result_hash,
                "log_sha256": job_status.get("log_sha256"),
                "fresh_worker_pid": pid,
                "window_starts": validated["starts"],
                "rows": validated["rows"],
                "policy_sha256": validated["policy_sha256"],
                "content_hashes": validated["content_hashes"],
                "source_sha256": validated["source_sha256"],
                "system_gates": validated["system_gates"],
                "worker_quality": validated["quality"],
            }
        )
    protocol.require(len(confirmations) == 5, "confirmation shard count")
    content_manifests = [selection["content_hashes"], *[
        shard["content_hashes"] for shard in confirmations
    ]]
    for hash_name in (
        "token_ids_sha256",
        "teacher_inputs_sha256",
        "token_matrix_sha256",
        "token_source_provenance_sha256",
    ):
        observed = [item[hash_name] for item in content_manifests]
        protocol.require(
            len(set(observed)) == len(observed),
            f"duplicate selection/confirmation content hash: {hash_name}",
        )
    all_starts = selection["starts"] + [
        start for shard in confirmations for start in shard["starts"]
    ]
    protocol.require(len(all_starts) == 24, "combined window count")
    protocol.require(len(set(all_starts)) == len(all_starts), "duplicate corpus window")
    combined_intervals = protocol.intervals(all_starts)
    protocol.require(
        all(
            right_start >= left_end
            for (_, left_end), (right_start, _) in zip(
                combined_intervals, combined_intervals[1:]
            )
        ),
        "selection/confirmation corpus windows overlap",
    )
    return selection, confirmations, input_manifest


def cohort_result(
    name: str,
    shards: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    clusters = [cluster for shard in shards for cluster in shard["clusters"]]
    point = protocol.cohort_point_summary(clusters)
    bootstrap = protocol.whole_window_bootstrap(
        clusters, bootstrap_samples, bootstrap_seed
    )
    gate = protocol.strict_cohort_gate(
        point, [shard["quality"] for shard in shards]
    )
    return {
        "cohort": name,
        "used_for_fixed_policy_selection": name == "selection",
        "used_as_untouched_confirmation": name == "confirmation",
        "shard_count": len(shards),
        "window_count": len(clusters),
        "point": point,
        "whole_window_cluster_bootstrap": bootstrap,
        "strict_gate": gate,
    }


def aggregate(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    bootstrap_samples: int = protocol.BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = protocol.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    protocol.verify_manifest(manifest)
    verify_status(status, manifest)
    verify_source_locks(manifest)
    selection, confirmations, inputs = validate_inputs(manifest, status)
    selection_result = cohort_result(
        "selection",
        [selection],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    confirmation_result = cohort_result(
        "confirmation",
        confirmations,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    passed = bool(
        selection_result["strict_gate"]["passed"]
        and confirmation_result["strict_gate"]["passed"]
    )
    failures = []
    for cohort_name, result in (
        ("selection", selection_result),
        ("confirmation", confirmation_result),
    ):
        failures.extend(
            f"{cohort_name}.{name}"
            for name, check in result["strict_gate"]["checks"].items()
            if not check["passed"]
        )
    return {
        "schema_version": protocol.AGGREGATE_SCHEMA_VERSION,
        "experiment": "page_gauge_wikitext2_train_selection_confirmation_quality",
        "passed": passed,
        "failures": failures,
        "primary_publication_quality_decision_cohort": "confirmation",
        "policy_sha256": manifest["policy_sha256"],
        "frozen_policy": manifest["frozen_policy"],
        "thresholds": manifest["gates"],
        "protocol": {
            "dataset": "WikiText-2 raw",
            "split": "train",
            "test_split_used": False,
            "selection_and_confirmation_separate": True,
            "selection_windows": len(protocol.SELECTION_STARTS),
            "confirmation_windows": len(protocol.CONFIRMATION_STARTS),
            "labels_per_window": protocol.ROWS_PER_WINDOW,
            "rows_per_b4_shard": protocol.ROWS_PER_SHARD,
            "confirmation_total_rows": len(protocol.CONFIRMATION_STARTS)
            * protocol.ROWS_PER_WINDOW,
            "all_windows_disjoint": True,
            "duplicate_window_count": 0,
            "overlapping_window_count": 0,
            "fresh_worker_subprocess_per_confirmation_shard": True,
            "confirmation_shards_run_even_after_a_quality_failure": True,
            "fallback_or_policy_tuning_after_confirmation": False,
            "manifest_sha256": manifest["manifest_sha256"],
            "status_sha256": status["status_sha256"],
        },
        "selection": selection_result,
        "confirmation": confirmation_result,
        "distribution_metric_availability": {
            "available_from_frozen_sustained_worker": [
                "full-vocabulary logits cosine",
                "relative L2 logit error",
                "maximum absolute logit error",
                "top1 agreement",
            ],
            "nll_kl_js_perplexity_available": False,
            "invented_or_reconstructed_from_top_vocab": False,
            "deferred_to": "untouched held-out full-distribution quality protocol",
            "reason": "the sustained artifact stores row scalars and a small diagnostic top-vocabulary subset, not complete distributions",
        },
        "inputs": inputs,
        "source_lock": manifest["source_lock"],
        "environment": {
            "aggregator_gpu_used": False,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "invocation": {
            "argv": sys.argv,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"): protocol.sha256_file(
                Path(__file__).resolve()
            ),
            str(SHARED.relative_to(ROOT)).replace("\\", "/"): protocol.sha256_file(
                SHARED
            ),
            str(RUNNER.relative_to(ROOT)).replace("\\", "/"): protocol.sha256_file(
                RUNNER
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = aggregate(
            protocol.load_json(args.manifest), protocol.load_json(args.status)
        )
    except Exception as error:
        result = {
            "schema_version": protocol.AGGREGATE_SCHEMA_VERSION,
            "experiment": "page_gauge_wikitext2_train_selection_confirmation_quality",
            "passed": False,
            "validation_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "invocation": {
                "argv": sys.argv,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failures": result.get("failures"),
                "validation_error": result.get("validation_error"),
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    if not result["passed"]:
        raise SystemExit("TRAIN confirmation quality acceptance failed")


if __name__ == "__main__":
    main()
