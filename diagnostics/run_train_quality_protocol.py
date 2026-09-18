#!/usr/bin/env python3
"""Pre-register and run untouched PageGauge TRAIN confirmation shards.

The ``initialize`` action seals the already-observed selection artifact, fixed
policy, untouched confirmation coordinates, gates, and every relevant source
hash before confirmation is run.  The ``run`` action then launches one fresh
worker subprocess per B4 shard.  Resume is fail-closed: completed artifacts
must match their recorded SHA-256, and untracked files are never adopted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import train_quality_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py"
AGGREGATOR = ROOT / "diagnostics/aggregate_train_quality_protocol.py"
SHARED = ROOT / "diagnostics/train_quality_protocol.py"
MANIFEST_NAME = "TRAIN_QUALITY_PREREGISTRATION.json"
STATUS_NAME = "TRAIN_QUALITY_RUN_STATUS.json"
LOCK_NAME = ".train_quality_protocol.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    initialize = subparsers.add_parser(
        "initialize", help="seal the protocol before confirmation is run"
    )
    initialize.add_argument("--selection-artifact", type=Path, required=True)
    initialize.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser(
        "run", help="run or integrity-check the sealed confirmation jobs"
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def seal_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("status_sha256", None)
    result["status_sha256"] = protocol.canonical_sha256(result)
    return result


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


def verify_current_raw_sources(source_lock: Mapping[str, str]) -> None:
    for relative, expected in source_lock.items():
        path = ROOT / relative
        protocol.require(path.is_file(), f"locked raw source missing: {relative}")
        protocol.require(
            protocol.sha256_file(path) == expected,
            f"locked raw source changed: {relative}",
        )


def protocol_tool_hashes() -> dict[str, str]:
    paths = (Path(__file__).resolve(), SHARED.resolve(), AGGREGATOR.resolve())
    result = {}
    for path in paths:
        protocol.require(path.is_file(), f"protocol tool missing: {path}")
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = protocol.sha256_file(
            path
        )
    return result


def verify_current_protocol_tools(source_lock: Mapping[str, str]) -> None:
    observed = protocol_tool_hashes()
    protocol.require(observed == dict(source_lock), "protocol tool source lock mismatch")


def build_manifest(selection_path: Path, output_dir: Path) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    protocol.require(selection_path.is_file(), f"selection artifact missing: {selection_path}")
    selection_payload = protocol.load_json(selection_path)
    selection = protocol.validate_raw_artifact(
        selection_payload,
        expected_starts=protocol.SELECTION_STARTS,
        cohort="selection",
        label=str(selection_path),
    )
    protocol.require(
        selection["quality"]["worker_quality_gate_passed"] is True,
        "the frozen selection policy did not pass its strict quality gate",
    )
    verify_current_raw_sources(selection["source_sha256"])
    archive_path = ROOT / "data/wikitext-2-raw-v1.zip"
    protocol.require(archive_path.is_file(), f"WikiText archive missing: {archive_path}")
    protocol.require(
        protocol.sha256_file(archive_path) == protocol.ARCHIVE_SHA256,
        "WikiText archive hash mismatch",
    )
    created = utc_now()
    jobs = []
    for index, starts in enumerate(protocol.CONFIRMATION_GROUPS):
        job_id = f"confirmation_{index:02d}"
        output = (output_dir / "confirmation" / f"{job_id}.json").resolve()
        log = (output_dir / "confirmation" / f"{job_id}.log").resolve()
        arguments = protocol.worker_arguments(starts, output)
        jobs.append(
            {
                "job_id": job_id,
                "cohort": "confirmation",
                "fresh_worker_process_required": True,
                "window_starts": list(starts),
                "window_ends_exclusive": [
                    start + protocol.WINDOW_CORPUS_TOKENS for start in starts
                ],
                "token_offset": starts[0],
                "token_stride": starts[1] - starts[0],
                "expected_rows": protocol.ROWS_PER_SHARD,
                "output_path": str(output),
                "log_path": str(log),
                "worker_path": str(WORKER.resolve()),
                "worker_arguments": arguments,
                "command_sha256": protocol.canonical_sha256(
                    [str(WORKER.resolve()), *arguments]
                ),
            }
        )
    manifest = {
        "schema_version": protocol.PROTOCOL_SCHEMA_VERSION,
        "experiment": "page_gauge_train_selection_confirmation_preregistration",
        "created_utc": created,
        "created_before_any_confirmation_job": True,
        "selection_was_observed_before_preregistration": True,
        "selection_role": "policy selection/localization only; never labeled untouched confirmation",
        "confirmation_role": "untouched fixed-policy evidence; no policy or threshold adaptation permitted",
        "one_shot_rule": {
            "run_all_five_preregistered_shards_even_if_an_earlier_quality_gate_fails": True,
            "failed_quality_artifact_is_terminal": True,
            "rerun_failed_quality_artifact": False,
            "fallback_policy": None,
            "retune_after_confirmation": False,
        },
        "frozen_policy": protocol.FROZEN_POLICY,
        "policy_sha256": protocol.frozen_policy_sha256(),
        "gates": protocol.FROZEN_GATES,
        "bootstrap": {
            "unit": "one complete disjoint 1024-label WikiText TRAIN window",
            "within_window_rows_resampled_independently": False,
            "samples": protocol.BOOTSTRAP_SAMPLES,
            "seed": protocol.BOOTSTRAP_SEED,
        },
        "cohort_design": {
            "dataset": "WikiText-2 raw",
            "split": "train",
            "test_split_used": False,
            "archive_sha256": protocol.ARCHIVE_SHA256,
            "archive_member": protocol.TRAIN_MEMBER,
            "archive_member_sha256": protocol.TRAIN_MEMBER_SHA256,
            "available_train_tokens": protocol.TRAIN_TOKEN_COUNT,
            "corpus_tokens_per_window": protocol.WINDOW_CORPUS_TOKENS,
            "labels_per_window": protocol.ROWS_PER_WINDOW,
            "selection_window_starts": list(protocol.SELECTION_STARTS),
            "selection_window_count": len(protocol.SELECTION_STARTS),
            "confirmation_window_starts": list(protocol.CONFIRMATION_STARTS),
            "confirmation_window_count": len(protocol.CONFIRMATION_STARTS),
            "confirmation_worker_shards": len(protocol.CONFIRMATION_GROUPS),
            "rows_per_b4_shard": protocol.ROWS_PER_SHARD,
            "confirmation_total_rows": len(protocol.CONFIRMATION_STARTS)
            * protocol.ROWS_PER_WINDOW,
            "theoretical_maximum_nonoverlapping_train_windows": protocol.THEORETICAL_MAX_NONOVERLAPPING_TRAIN_WINDOWS,
            "selection_confirmation_duplicate_count": 0,
            "selection_confirmation_overlap_count": 0,
            "confirmation_sampling_rule": "start_n = 300000 + n*125000 for n=0..19, frozen before confirmation",
        },
        "selection_artifact": {
            "path": str(selection_path),
            "sha256": protocol.sha256_file(selection_path),
            "invocation_utc_timestamp": selection["invocation_utc_timestamp"],
            "rows": selection["rows"],
            "window_starts": selection["starts"],
            "content_hashes": selection["content_hashes"],
            "policy_sha256": selection["policy_sha256"],
            "worker_quality": selection["quality"],
        },
        "source_lock": {
            "raw_worker_source_sha256": selection["source_sha256"],
            "protocol_tool_source_sha256": protocol_tool_hashes(),
            "worker_entrypoint_sha256": protocol.sha256_file(WORKER),
        },
        "required_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0",
            "fresh_process_per_shard": True,
        },
        "jobs": jobs,
        "resume_policy": {
            "completed_result_sha256_must_match_status": True,
            "completed_result_must_revalidate": True,
            "untracked_existing_artifacts_are_rejected": True,
            "failed_quality_artifacts_are_terminal_and_not_rerun": True,
        },
    }
    return protocol.seal_manifest(manifest)


def initial_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return seal_status(
        {
            "schema_version": 1,
            "experiment": "page_gauge_train_confirmation_run_status",
            "manifest_sha256": manifest["manifest_sha256"],
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "passed": False,
            "complete": False,
            "selection_artifact_verified": True,
            "jobs": {
                job["job_id"]: {
                    "state": "pending",
                    "attempts": 0,
                    "output_path": job["output_path"],
                    "expected_rows": protocol.ROWS_PER_SHARD,
                }
                for job in manifest["jobs"]
            },
        }
    )


def initialize(selection: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    status_path = output_dir / STATUS_NAME
    protocol.require(not manifest_path.exists(), f"refusing to overwrite {manifest_path}")
    protocol.require(not status_path.exists(), f"refusing to overwrite {status_path}")
    confirmation_dir = output_dir / "confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    protocol.require(
        not any(confirmation_dir.iterdir()),
        f"confirmation directory is not empty: {confirmation_dir}",
    )
    manifest = build_manifest(selection, output_dir)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(status_path, initial_status(manifest))
    print(json.dumps({
        "initialized": True,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "confirmation_jobs": len(manifest["jobs"]),
        "confirmation_windows": len(protocol.CONFIRMATION_STARTS),
        "confirmation_rows": len(protocol.CONFIRMATION_STARTS) * protocol.ROWS_PER_WINDOW,
    }, indent=2))


def parse_timestamp(value: Any, label: str) -> datetime:
    protocol.require(isinstance(value, str) and value, f"{label}: timestamp")
    parsed = datetime.fromisoformat(value)
    protocol.require(parsed.tzinfo is not None, f"{label}: naive timestamp")
    return parsed


def validate_completed_job(
    job: Mapping[str, Any],
    job_status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selection_signature: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(job["output_path"])
    protocol.require(output.is_file(), f"completed result missing: {output}")
    observed_hash = protocol.sha256_file(output)
    protocol.require(
        observed_hash == job_status.get("result_sha256"),
        f"resume hash mismatch: {output}",
    )
    payload = protocol.load_json(output)
    validated = protocol.validate_raw_artifact(
        payload,
        expected_starts=job["window_starts"],
        cohort="confirmation",
        label=str(output),
        locked_source_sha256=manifest["source_lock"]["raw_worker_source_sha256"],
    )
    protocol.require(
        validated["common_signature"] == dict(selection_signature),
        f"{output}: selection/confirmation signature mismatch",
    )
    preregistered = parse_timestamp(manifest["created_utc"], "manifest")
    invoked = parse_timestamp(validated["invocation_utc_timestamp"], str(output))
    protocol.require(invoked >= preregistered, f"{output}: predates preregistration")
    return validated


def run_jobs(output_dir: Path, python: Path) -> None:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / MANIFEST_NAME
    status_path = output_dir / STATUS_NAME
    manifest = protocol.load_json(manifest_path)
    protocol.verify_manifest(manifest)
    status = protocol.load_json(status_path)
    verify_status(status, manifest)
    verify_current_raw_sources(manifest["source_lock"]["raw_worker_source_sha256"])
    verify_current_protocol_tools(manifest["source_lock"]["protocol_tool_source_sha256"])
    protocol.require(
        protocol.sha256_file(WORKER)
        == manifest["source_lock"]["worker_entrypoint_sha256"],
        "worker entrypoint source changed",
    )
    selection_path = Path(manifest["selection_artifact"]["path"])
    protocol.require(selection_path.is_file(), "selection artifact disappeared")
    protocol.require(
        protocol.sha256_file(selection_path)
        == manifest["selection_artifact"]["sha256"],
        "selection artifact hash changed after preregistration",
    )
    selection_validated = protocol.validate_raw_artifact(
        protocol.load_json(selection_path),
        expected_starts=protocol.SELECTION_STARTS,
        cohort="selection",
        label=str(selection_path),
        locked_source_sha256=manifest["source_lock"]["raw_worker_source_sha256"],
    )
    python = python.resolve()
    protocol.require(python.is_file(), f"Python executable missing: {python}")

    lock_path = output_dir / LOCK_NAME
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError(f"another run holds the protocol lock: {lock_path}") from error
    try:
        os.write(lock_fd, f"pid={os.getpid()} utc={utc_now()}\n".encode("utf-8"))
        os.close(lock_fd)
        for job in manifest["jobs"]:
            job_id = job["job_id"]
            current = status["jobs"][job_id]
            state = current["state"]
            if state in {"passed", "failed_quality_gate"}:
                validated = validate_completed_job(
                    job, current, manifest, selection_validated["common_signature"]
                )
                expected_state = (
                    "passed"
                    if validated["quality"]["worker_quality_gate_passed"]
                    else "failed_quality_gate"
                )
                protocol.require(state == expected_state, f"{job_id}: status decision mismatch")
                print(f"resume verified {job_id}: {current['result_sha256']}", flush=True)
                continue
            protocol.require(state == "pending", f"{job_id}: non-resumable state {state}")
            output = Path(job["output_path"])
            log = Path(job["log_path"])
            protocol.require(not output.exists(), f"untracked result exists: {output}")
            protocol.require(not log.exists(), f"untracked log exists: {log}")
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(python), str(WORKER.resolve()), *job["worker_arguments"]]
            protocol.require(
                protocol.canonical_sha256([str(WORKER.resolve()), *job["worker_arguments"]])
                == job["command_sha256"],
                f"{job_id}: command integrity failure",
            )
            environment = dict(os.environ)
            environment.update(manifest["required_environment"])
            environment.pop("fresh_process_per_shard", None)
            started = utc_now()
            with log.open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                pid = process.pid
                return_code = process.wait()
            ended = utc_now()
            if not output.is_file():
                current.update(
                    {
                        "state": "worker_failed_without_result",
                        "attempts": int(current["attempts"]) + 1,
                        "worker_pid": pid,
                        "return_code": return_code,
                        "started_utc": started,
                        "ended_utc": ended,
                        "log_sha256": protocol.sha256_file(log),
                    }
                )
                status["updated_utc"] = utc_now()
                atomic_write_json(status_path, seal_status(status))
                raise SystemExit(f"{job_id} failed without a result; see {log}")
            result_hash = protocol.sha256_file(output)
            try:
                validated = protocol.validate_raw_artifact(
                    protocol.load_json(output),
                    expected_starts=job["window_starts"],
                    cohort="confirmation",
                    label=str(output),
                    locked_source_sha256=manifest["source_lock"]["raw_worker_source_sha256"],
                )
                protocol.require(
                    validated["common_signature"]
                    == selection_validated["common_signature"],
                    f"{job_id}: selection/confirmation signature mismatch",
                )
                protocol.require(
                    parse_timestamp(validated["invocation_utc_timestamp"], job_id)
                    >= parse_timestamp(manifest["created_utc"], "manifest"),
                    f"{job_id}: result predates preregistration",
                )
            except Exception as error:
                current.update(
                    {
                        "state": "artifact_validation_failed",
                        "attempts": int(current["attempts"]) + 1,
                        "worker_pid": pid,
                        "return_code": return_code,
                        "started_utc": started,
                        "ended_utc": ended,
                        "result_sha256": result_hash,
                        "log_sha256": protocol.sha256_file(log),
                        "validation_error": f"{type(error).__name__}: {error}",
                    }
                )
                status["updated_utc"] = utc_now()
                atomic_write_json(status_path, seal_status(status))
                raise
            quality_passed = validated["quality"]["worker_quality_gate_passed"]
            expected_return = 0 if quality_passed else 2
            protocol.require(
                return_code == expected_return,
                f"{job_id}: worker return code {return_code}, expected {expected_return}",
            )
            current.update(
                {
                    "state": "passed" if quality_passed else "failed_quality_gate",
                    "attempts": int(current["attempts"]) + 1,
                    "fresh_worker_process": True,
                    "worker_pid": pid,
                    "return_code": return_code,
                    "started_utc": started,
                    "ended_utc": ended,
                    "result_sha256": result_hash,
                    "log_sha256": protocol.sha256_file(log),
                    "content_hashes": validated["content_hashes"],
                    "policy_sha256": validated["policy_sha256"],
                    "quality": validated["quality"],
                    "system_gates": validated["system_gates"],
                }
            )
            status["updated_utc"] = utc_now()
            status["complete"] = all(
                item["state"] in {"passed", "failed_quality_gate"}
                for item in status["jobs"].values()
            )
            status["passed"] = bool(
                status["complete"]
                and all(item["state"] == "passed" for item in status["jobs"].values())
            )
            atomic_write_json(status_path, seal_status(status))
            print(f"{job_id}: {current['state']} {result_hash}", flush=True)
        status["complete"] = True
        status["passed"] = all(
            item["state"] == "passed" for item in status["jobs"].values()
        )
        status["updated_utc"] = utc_now()
        atomic_write_json(status_path, seal_status(status))
        print(
            json.dumps(
                {
                    "complete": True,
                    "passed": status["passed"],
                    "status": str(status_path),
                },
                indent=2,
            )
        )
        if not status["passed"]:
            raise SystemExit(
                "all pre-registered shards completed; one or more failed the frozen strict quality gate"
            )
    finally:
        if lock_path.exists():
            lock_path.unlink()


def main() -> None:
    args = parse_args()
    if args.action == "initialize":
        initialize(args.selection_artifact, args.output_dir)
    else:
        run_jobs(args.output_dir, args.python)


if __name__ == "__main__":
    main()
