#!/usr/bin/env python3
"""CPU-only tests for the frozen TRAIN selection/confirmation protocol."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import aggregate_train_quality_protocol as aggregator  # noqa: E402
import run_train_quality_protocol as runner  # noqa: E402
import train_quality_protocol as protocol  # noqa: E402


SELECTION_PATH = (
    ROOT
    / "results/exact_prefix_s3_v1/train_structural/matched_d1024_t768.json"
)


@pytest.fixture(scope="module")
def selection_payload() -> dict:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def synthetic_confirmation_payload(
    selection: dict,
    starts: tuple[int, ...],
    index: int,
    invocation_timestamp: str,
) -> dict:
    payload = copy.deepcopy(selection)
    stride = starts[1] - starts[0]
    payload["configuration"]["token_offset"] = starts[0]
    payload["configuration"]["token_stride"] = stride
    payload["pairing"]["configuration"]["token_offset"] = starts[0]
    payload["pairing"]["configuration"]["token_stride"] = stride
    source = payload["token_source"]
    source["corpus_window_start_offsets"] = list(starts)
    source["corpus_window_end_offsets_exclusive"] = [
        start + protocol.WINDOW_CORPUS_TOKENS for start in starts
    ]
    source["corpus_window_stride"] = stride
    content_base = 100 + index * 10
    source["token_ids_sha256"] = f"{content_base:064x}"
    payload["trajectory"]["teacher_inputs_sha256"] = f"{content_base + 1:064x}"
    pairing = payload["pairing"]["configuration"]
    pairing["teacher_inputs_sha256"] = payload["trajectory"][
        "teacher_inputs_sha256"
    ]
    pairing["token_matrix_sha256"] = f"{content_base + 2:064x}"
    pairing["token_source_provenance_sha256"] = protocol.canonical_sha256(source)
    for request, (window, start) in enumerate(zip(payload["quality_windows"], starts)):
        window.update(
            {
                "request": request,
                "cluster_unit_id": f"fixture-confirmation-{index}-{request}-{start}",
                "corpus_window_start_offset": start,
                "corpus_window_end_offset_exclusive": start
                + protocol.WINDOW_CORPUS_TOKENS,
                "corpus_label_start_offset": start + protocol.CONTEXT,
                "corpus_label_end_offset_exclusive": start
                + protocol.CONTEXT
                + protocol.DECODE_STEPS,
            }
        )
    payload["pairing"]["pairing_key_sha256"] = protocol.canonical_sha256(pairing)
    payload["invocation"]["utc_timestamp"] = invocation_timestamp
    return payload


def write_confirmation_fixture(
    tmp_path: Path, selection: dict
) -> tuple[dict, dict]:
    manifest = runner.build_manifest(SELECTION_PATH, tmp_path)
    after_manifest = (
        datetime.fromisoformat(manifest["created_utc"]) + timedelta(seconds=1)
    ).isoformat()
    status = runner.initial_status(manifest)
    for index, (job, starts) in enumerate(
        zip(manifest["jobs"], protocol.CONFIRMATION_GROUPS)
    ):
        output = Path(job["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = synthetic_confirmation_payload(
            selection, starts, index, after_manifest
        )
        output.write_text(json.dumps(payload), encoding="utf-8")
        validated = protocol.validate_raw_artifact(
            payload,
            expected_starts=starts,
            cohort="confirmation",
            label=job["job_id"],
            locked_source_sha256=manifest["source_lock"][
                "raw_worker_source_sha256"
            ],
        )
        status["jobs"][job["job_id"]].update(
            {
                "state": "passed",
                "attempts": 1,
                "fresh_worker_process": True,
                "worker_pid": 10_000 + index,
                "return_code": 0,
                "started_utc": after_manifest,
                "ended_utc": after_manifest,
                "result_sha256": protocol.sha256_file(output),
                "log_sha256": f"{200 + index:064x}",
                "content_hashes": validated["content_hashes"],
                "policy_sha256": validated["policy_sha256"],
                "quality": validated["quality"],
                "system_gates": validated["system_gates"],
            }
        )
    status["complete"] = True
    status["passed"] = True
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    return manifest, runner.seal_status(status)


def test_frozen_cohorts_are_disjoint_train_only_and_feasible() -> None:
    protocol.validate_frozen_cohorts()
    assert len(protocol.SELECTION_STARTS) == 4
    assert len(protocol.CONFIRMATION_STARTS) == 20
    assert len(protocol.CONFIRMATION_GROUPS) == 5
    assert protocol.CONFIRMATION_STARTS[-1] == 2_675_000
    assert (
        protocol.CONFIRMATION_STARTS[-1] + protocol.WINDOW_CORPUS_TOKENS
        == 2_696_504
    )
    assert protocol.THEORETICAL_MAX_NONOVERLAPPING_TRAIN_WINDOWS == 128
    arguments = protocol.worker_arguments(
        protocol.CONFIRMATION_GROUPS[0], Path("fixture.json")
    )
    assert any("wiki.train.raw" in argument for argument in arguments)
    assert not any("test" in argument.lower() for argument in arguments)


def test_real_selection_artifact_passes_and_preserves_thin_worst_row(
    selection_payload: dict,
) -> None:
    validated = protocol.validate_raw_artifact(
        selection_payload,
        expected_starts=protocol.SELECTION_STARTS,
        cohort="selection",
        label="real selection",
    )
    point = protocol.cohort_point_summary(validated["clusters"])
    assert point["row_count"] == 4096
    assert point["rows_below_minimum_logits_cosine_gate"] == 0
    assert point["minimum_logits_cosine"] == pytest.approx(0.9952273368835449)
    assert point["worst_row_identity"]["request_within_shard"] == 3
    assert point["worst_row_identity"]["step"] == 907
    assert point["worst_row_identity"]["model_input_absolute_position"] == 21387


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["configuration"].__setitem__(
                "exact_tail_tokens", 1024
            ),
            "exact_tail_tokens",
        ),
        (
            lambda payload: payload["correctness"][
                "same_backend_eager_vs_graph"
            ]["immutable_exact_prefix_timed_sample_canary"].__setitem__(
                "passed", False
            ),
            "prefix timed canary",
        ),
        (
            lambda payload: payload["correctness"][
                "runtime_page_finalization_and_consumption"
            ].__setitem__("runtime_finalized_int8_pages_consumed_count", 0),
            "expected exactly 16 runtime-finalized INT8",
        ),
    ],
)
def test_artifact_validator_fails_closed_on_policy_canary_and_recurrence(
    selection_payload: dict, mutation, message: str
) -> None:
    payload = copy.deepcopy(selection_payload)
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        protocol.validate_raw_artifact(
            payload,
            expected_starts=protocol.SELECTION_STARTS,
            cohort="selection",
        )


def test_manifest_seals_one_shot_no_fallback_and_detects_tamper(
    tmp_path: Path,
) -> None:
    manifest = runner.build_manifest(SELECTION_PATH, tmp_path)
    protocol.verify_manifest(manifest)
    assert manifest["cohort_design"]["confirmation_window_count"] == 20
    assert manifest["cohort_design"]["confirmation_total_rows"] == 20480
    assert manifest["one_shot_rule"][
        "run_all_five_preregistered_shards_even_if_an_earlier_quality_gate_fails"
    ]
    assert manifest["one_shot_rule"]["fallback_policy"] is None
    tampered = copy.deepcopy(manifest)
    tampered["frozen_policy"]["exact_tail_tokens"] = 1024
    with pytest.raises(ValueError, match="manifest hash integrity"):
        protocol.verify_manifest(tampered)


def test_aggregate_accepts_exact_20_window_confirmation_deterministically(
    tmp_path: Path, selection_payload: dict
) -> None:
    manifest, status = write_confirmation_fixture(tmp_path, selection_payload)
    first = aggregator.aggregate(
        manifest, status, bootstrap_samples=200, bootstrap_seed=73
    )
    second = aggregator.aggregate(
        manifest, status, bootstrap_samples=200, bootstrap_seed=73
    )
    assert first["passed"]
    assert first["selection"]["point"]["row_count"] == 4096
    assert first["confirmation"]["point"]["row_count"] == 20480
    assert first["confirmation"]["point"][
        "rows_below_minimum_logits_cosine_gate"
    ] == 0
    assert first["confirmation"]["whole_window_cluster_bootstrap"] == second[
        "confirmation"
    ]["whole_window_cluster_bootstrap"]
    assert not first["distribution_metric_availability"][
        "nll_kl_js_perplexity_available"
    ]


def test_aggregate_rejects_duplicate_content_hash(
    tmp_path: Path, selection_payload: dict
) -> None:
    manifest, status = write_confirmation_fixture(tmp_path, selection_payload)
    first_job, second_job = manifest["jobs"][:2]
    first_payload = protocol.load_json(Path(first_job["output_path"]))
    second_path = Path(second_job["output_path"])
    second_payload = protocol.load_json(second_path)
    second_payload["token_source"]["token_ids_sha256"] = first_payload[
        "token_source"
    ]["token_ids_sha256"]
    second_payload["pairing"]["configuration"][
        "token_source_provenance_sha256"
    ] = protocol.canonical_sha256(second_payload["token_source"])
    second_payload["pairing"]["pairing_key_sha256"] = protocol.canonical_sha256(
        second_payload["pairing"]["configuration"]
    )
    second_path.write_text(json.dumps(second_payload), encoding="utf-8")
    validated = protocol.validate_raw_artifact(
        second_payload,
        expected_starts=second_job["window_starts"],
        cohort="confirmation",
        locked_source_sha256=manifest["source_lock"]["raw_worker_source_sha256"],
    )
    status_body = dict(status)
    status_body.pop("status_sha256")
    status_body["jobs"][second_job["job_id"]]["result_sha256"] = (
        protocol.sha256_file(second_path)
    )
    status_body["jobs"][second_job["job_id"]]["content_hashes"] = validated[
        "content_hashes"
    ]
    status = runner.seal_status(status_body)
    with pytest.raises(ValueError, match="duplicate.*content hash"):
        aggregator.aggregate(
            manifest, status, bootstrap_samples=100, bootstrap_seed=79
        )


def test_all_five_shards_remain_in_failed_one_shot_aggregate(
    tmp_path: Path, selection_payload: dict
) -> None:
    manifest, status = write_confirmation_fixture(tmp_path, selection_payload)
    failed_job = manifest["jobs"][0]
    failed_path = Path(failed_job["output_path"])
    payload = protocol.load_json(failed_path)
    comparison = payload["correctness"]["backend_vs_hf_sdpa_fp16"]
    matrices = comparison["outlier_diagnostics"]["row_matrices"]
    matrices["cosine_by_step_request"][0][0] = 0.994
    cosines = [
        value
        for row in matrices["cosine_by_step_request"]
        for value in row
    ]
    comparison["logits"]["minimum_cosine"] = min(cosines)
    comparison["logits"]["mean_cosine"] = sum(cosines) / len(cosines)
    summary = comparison["outlier_diagnostics"]["summary"]
    summary["rows_below_gate"] = 1
    summary["fraction_below_gate"] = 1 / protocol.ROWS_PER_SHARD
    summary["cosine_quantiles"]["minimum"] = 0.994
    comparison["passed"] = False
    payload["correctness"]["passed"] = False
    payload["passed"] = False
    failed_path.write_text(json.dumps(payload), encoding="utf-8")
    validated = protocol.validate_raw_artifact(
        payload,
        expected_starts=failed_job["window_starts"],
        cohort="confirmation",
        locked_source_sha256=manifest["source_lock"]["raw_worker_source_sha256"],
    )
    status_body = dict(status)
    status_body.pop("status_sha256")
    failed_status = status_body["jobs"][failed_job["job_id"]]
    failed_status.update(
        {
            "state": "failed_quality_gate",
            "return_code": 2,
            "result_sha256": protocol.sha256_file(failed_path),
            "quality": validated["quality"],
            "content_hashes": validated["content_hashes"],
        }
    )
    status_body["passed"] = False
    status = runner.seal_status(status_body)
    result = aggregator.aggregate(
        manifest, status, bootstrap_samples=100, bootstrap_seed=83
    )
    assert not result["passed"]
    assert result["confirmation"]["shard_count"] == 5
    assert result["confirmation"]["window_count"] == 20
    assert result["confirmation"]["point"][
        "rows_below_minimum_logits_cosine_gate"
    ] == 1
    assert "confirmation.minimum_logits_cosine" in result["failures"]


def test_resume_rejects_changed_completed_result_hash(
    tmp_path: Path, selection_payload: dict
) -> None:
    manifest, status = write_confirmation_fixture(tmp_path, selection_payload)
    job = manifest["jobs"][0]
    job_status = status["jobs"][job["job_id"]]
    job_status = dict(job_status)
    job_status["result_sha256"] = "0" * 64
    selection = protocol.validate_raw_artifact(
        selection_payload,
        expected_starts=protocol.SELECTION_STARTS,
        cohort="selection",
    )
    with pytest.raises(ValueError, match="resume hash mismatch"):
        runner.validate_completed_job(
            job, job_status, manifest, selection["common_signature"]
        )
