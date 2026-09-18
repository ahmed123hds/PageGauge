#!/usr/bin/env python3
"""CPU-only synthetic tests for the MLSys controlled decoder protocol."""

from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import mlsys_controlled_protocol as protocol  # noqa: E402
import prepare_mlsys_controlled_matrix as prepare  # noqa: E402
import reduce_mlsys_controlled as reducer  # noqa: E402


MATRIX_PATH = ROOT / "experiments/mlsys2027/decoder_matrix_v1.json"


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def synthetic_records(matrix: dict, cell_id: str) -> list[dict]:
    cell = protocol.validate_matrix(matrix)[cell_id]
    hardware = matrix["hardware"][cell["hardware_id"]]
    matrix_hash = protocol.canonical_sha256(matrix)
    epoch = datetime(2026, 9, 2, tzinfo=timezone.utc)
    records: list[dict] = []
    for ordinal, block in enumerate(
        row for row in protocol.expand_blocks(matrix) if row["cell_id"] == cell_id
    ):
        latency = 2.0 if block["backend"] == cell["baseline"] else 1.0
        started = epoch + timedelta(seconds=ordinal * 4)
        ended = started + timedelta(seconds=2)
        raw_samples = [
            {"sample_index": index, "wall_ms": latency, "cuda_ms": latency}
            for index in range(3)
        ]
        records.append(
            {
                **{
                    key: block[key]
                    for key in (
                        "block_id",
                        "cell_id",
                        "cell_config_sha256",
                        "seed",
                        "pair_id",
                        "pair_order",
                        "slot",
                        "treatment",
                        "backend",
                    )
                },
                "matrix_canonical_sha256": matrix_hash,
                "pairing_key_sha256": protocol.canonical_sha256(
                    {"pair_id": block["pair_id"], "fixture": block["seed"]}
                ),
                "pid": 50000 + ordinal,
                "observed_hardware": {
                    "family": hardware["family"],
                    "compute_capability": hardware["compute_capability"],
                },
                "source_sha256": {"worker.py": "1" * 64},
                "execution_gates": {
                    "exclusive_device": True,
                    "correctness": True,
                    "cache_accounting": True,
                    "no_fallback": True,
                },
                "recurrence": {
                    "closed_generated_pages": 96,
                    "runtime_created_pages_later_consumed_int8": (
                        0 if block["backend"] == "flashinfer_fp16" else 48
                    ),
                },
                "timing_modes": {
                    "cache_neutral": {"raw_samples": copy.deepcopy(raw_samples)},
                    "cache_hot": {"raw_samples": copy.deepcopy(raw_samples)},
                },
                "terminal": {
                    "status": "completed",
                    "return_code": 0,
                    "started_utc": started.isoformat(),
                    "ended_utc": ended.isoformat(),
                    "artifact_sha256s": {
                        "stdout": "2" * 64,
                        "stderr": "3" * 64,
                        "telemetry": "4" * 64,
                    },
                },
            }
        )
    return records


class MLSysControlledProtocolTests(unittest.TestCase):
    def test_explicit_pairwise_matrix_and_schedule_are_canonical(self) -> None:
        matrix = load_matrix()
        cells = protocol.validate_matrix(matrix)
        self.assertEqual(len(cells), 9)
        blocks = protocol.expand_blocks(matrix)
        self.assertEqual(len(blocks), 72)
        self.assertEqual(len({block["block_id"] for block in blocks}), 72)
        for cell_id, cell in cells.items():
            selected = [block for block in blocks if block["cell_id"] == cell_id]
            self.assertEqual(len(selected), 8)
            baseline, candidate = cell["baseline"], cell["candidate"]
            self.assertEqual(
                [block["backend"] for block in selected],
                [
                    baseline,
                    candidate,
                    candidate,
                    baseline,
                    candidate,
                    baseline,
                    baseline,
                    candidate,
                ],
            )
            self.assertEqual(
                [block["pair_order"] for block in selected],
                ["AB", "AB", "BA", "BA", "BA", "BA", "AB", "AB"],
            )

    def test_manifest_seals_design_sources_and_is_tamper_evident(self) -> None:
        manifest = prepare.prepare(MATRIX_PATH)
        protocol.verify_manifest(manifest)
        self.assertEqual(manifest["cell_count"], 9)
        self.assertEqual(manifest["physical_pair_count"], 36)
        self.assertEqual(manifest["fresh_process_block_count"], 72)
        self.assertFalse(manifest["runnable"])
        self.assertEqual(manifest["source_closure_status"], "design_only_not_execution_locked")
        self.assertIn("diagnostics/reduce_mlsys_controlled.py", manifest["source_sha256"])
        self.assertTrue(protocol.DESIGN_ONLY_REQUIREMENTS.issubset(manifest["unresolved_requirements"]))
        self.assertIn(
            "hardware:h100_sxm_80gb:access", manifest["unresolved_requirements"]
        )
        tampered = copy.deepcopy(manifest)
        tampered["blocks"][0]["slot"] = 99
        with self.assertRaisesRegex(protocol.ProtocolError, "manifest hash integrity"):
            protocol.verify_manifest(tampered)

    def test_readiness_flags_cannot_make_an_unintegrated_design_runnable(self) -> None:
        matrix = load_matrix()
        matrix["status"] = "frozen"
        for hardware in matrix["hardware"].values():
            hardware["availability_status"] = "available"
        for method in matrix["methods"].values():
            method["implementation_status"] = "ready"
        manifest = protocol.build_manifest(
            matrix, matrix_path=MATRIX_PATH, source_paths=(MATRIX_PATH,)
        )
        protocol.verify_manifest(manifest)
        self.assertEqual(manifest["status"], "design")
        self.assertFalse(manifest["runnable"])
        self.assertEqual(set(manifest["unresolved_requirements"]), protocol.DESIGN_ONLY_REQUIREMENTS)
        manifest["status"] = "ready"
        manifest["runnable"] = True
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = protocol.canonical_sha256(manifest)
        with self.assertRaisesRegex(protocol.ProtocolError, "must remain non-runnable"):
            protocol.verify_manifest(manifest)

    def test_known_two_x_reduction_and_bootstrap_are_deterministic(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"
        records = synthetic_records(matrix, cell_id)
        first = reducer.reduce_cell_records(
            matrix, cell_id, records, bootstrap_samples=500, bootstrap_seed=17
        )
        second = reducer.reduce_cell_records(
            matrix, cell_id, records, bootstrap_samples=500, bootstrap_seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["physical_pair_count"], 4)
        self.assertEqual(first["fresh_process_block_count"], 8)
        self.assertEqual(first["metric_record_count"], 16)
        self.assertEqual(first["status"], "provisional")
        self.assertFalse(first["publication_eligible"])
        self.assertEqual(first["evidence_verification"]["status"], "not_verified")
        for endpoint in first["endpoints"].values():
            self.assertAlmostEqual(endpoint["point_speedup"], 2.0, places=12)
            self.assertAlmostEqual(endpoint["speedup_95_ci"][0], 2.0, places=12)
            self.assertAlmostEqual(endpoint["speedup_95_ci"][1], 2.0, places=12)
            self.assertTrue(endpoint["gate"]["passed"])

    def test_recurrence_is_backend_aware_and_still_requires_closed_fp16_pages(self) -> None:
        matrix = load_matrix()
        for cell_id in protocol.validate_matrix(matrix):
            with self.subTest(cell_id=cell_id):
                records = synthetic_records(matrix, cell_id)
                reducer.validate_cell_records(matrix, cell_id, records)
                for record in records:
                    if record["backend"] != "flashinfer_fp16":
                        record["recurrence"]["runtime_created_pages_later_consumed_int8"] = 0
                        break
                with self.assertRaisesRegex(protocol.ProtocolError, "runtime INT8 recurrence gate"):
                    reducer.validate_cell_records(matrix, cell_id, records)
        cell_id = "anchor_a100_pg_vs_fp16"
        records = synthetic_records(matrix, cell_id)
        records[0]["recurrence"]["closed_generated_pages"] = 95
        with self.assertRaisesRegex(protocol.ProtocolError, "closed-page recurrence gate"):
            reducer.validate_cell_records(matrix, cell_id, records)

    def test_recurrence_counters_fail_closed_for_nonintegers_and_missing_values(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"
        for index in (0, 1):
            for counter in ("closed_generated_pages", "runtime_created_pages_later_consumed_int8"):
                for value in (float("nan"), float("inf"), -1, 96.0, True, "96", None):
                    with self.subTest(index=index, counter=counter, value=value):
                        records = synthetic_records(matrix, cell_id)
                        records[index]["recurrence"][counter] = value
                        with self.assertRaisesRegex(protocol.ProtocolError, "nonnegative integer"):
                            reducer.validate_cell_records(matrix, cell_id, records)
                records = synthetic_records(matrix, cell_id)
                del records[index]["recurrence"][counter]
                with self.assertRaisesRegex(protocol.ProtocolError, "nonnegative integer"):
                    reducer.validate_cell_records(matrix, cell_id, records)
        records = synthetic_records(matrix, cell_id)
        records[1]["recurrence"]["runtime_created_pages_later_consumed_int8"] = 97
        with self.assertRaisesRegex(protocol.ProtocolError, "exceed closed pages"):
            reducer.validate_cell_records(matrix, cell_id, records)

    def test_named_execution_gates_cannot_be_omitted_or_substituted(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"
        for gate in reducer.REQUIRED_EXECUTION_GATES:
            records = synthetic_records(matrix, cell_id)
            del records[0]["execution_gates"][gate]
            with self.assertRaisesRegex(protocol.ProtocolError, "lacks required execution gates"):
                reducer.validate_cell_records(matrix, cell_id, records)
            records = synthetic_records(matrix, cell_id)
            records[0]["execution_gates"][gate] = False
            with self.assertRaisesRegex(protocol.ProtocolError, "execution gate failed"):
                reducer.validate_cell_records(matrix, cell_id, records)
        records = synthetic_records(matrix, cell_id)
        records[0]["execution_gates"] = {"anything": True}
        with self.assertRaisesRegex(protocol.ProtocolError, "lacks required execution gates"):
            reducer.validate_cell_records(matrix, cell_id, records)

    def test_optional_evidence_verifier_hashes_actual_sources_and_artifacts(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"
        records = synthetic_records(matrix, cell_id)
        with tempfile.TemporaryDirectory() as temporary:
            evidence_root = Path(temporary)
            source = evidence_root / "worker.py"
            source.write_text("# synthetic CPU-only fixture\n", encoding="utf-8")
            frozen = {"worker.py": protocol.sha256_file(source)}
            for index, record in enumerate(records):
                record["source_sha256"] = dict(frozen)
                terminal = record["terminal"]
                terminal["artifact_paths"] = {}
                for label in sorted(reducer.REQUIRED_ARTIFACT_HASHES):
                    artifact = evidence_root / f"{index}.{label}.txt"
                    artifact.write_text(f"synthetic {index} {label}\n", encoding="utf-8")
                    terminal["artifact_paths"][label] = artifact.name
                    terminal["artifact_sha256s"][label] = protocol.sha256_file(artifact)
            summary = reducer.reduce_cell_records(
                matrix, cell_id, records, bootstrap_samples=100,
                evidence_root=evidence_root, frozen_source_sha256=frozen,
            )
            self.assertEqual(summary["evidence_verification"]["status"], "verified_file_hashes")
            self.assertEqual(
                summary["evidence_verification"]["frozen_source_closure_sha256"],
                protocol.canonical_sha256(frozen),
            )
            self.assertTrue(summary["evidence_verification"]["artifact_files_verified"])
            self.assertFalse(summary["publication_eligible"])
            self.assertFalse(summary["evidence_verification"]["execution_source_lock_verified"])

            changed_closure = copy.deepcopy(records)
            changed_closure[0]["source_sha256"]["worker.py"] = "0" * 64
            with self.assertRaisesRegex(protocol.ProtocolError, "differs from frozen closure"):
                reducer.verify_cell_evidence(
                    changed_closure, evidence_root=evidence_root, frozen_source_sha256=frozen
                )
            missing_source = copy.deepcopy(records)
            missing_source[0]["source_sha256"] = {}
            with self.assertRaisesRegex(protocol.ProtocolError, "differs from frozen closure"):
                reducer.verify_cell_evidence(
                    missing_source, evidence_root=evidence_root, frozen_source_sha256=frozen
                )
            missing_artifact = copy.deepcopy(records)
            missing_artifact[0]["terminal"]["artifact_paths"]["stdout"] = "missing.txt"
            with self.assertRaisesRegex(protocol.ProtocolError, "evidence file missing"):
                reducer.verify_cell_evidence(
                    missing_artifact, evidence_root=evidence_root, frozen_source_sha256=frozen
                )
            for unsafe in ("../outside.txt", str(source.resolve()), "C:/outside.txt"):
                unsafe_path = copy.deepcopy(records)
                unsafe_path[0]["terminal"]["artifact_paths"]["stdout"] = unsafe
                with self.assertRaisesRegex(protocol.ProtocolError, "within evidence_root"):
                    reducer.verify_cell_evidence(
                        unsafe_path, evidence_root=evidence_root, frozen_source_sha256=frozen
                    )
            artifact = evidence_root / records[0]["terminal"]["artifact_paths"]["stdout"]
            artifact.write_text("changed artifact\n", encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "artifact stdout evidence SHA256 mismatch"):
                reducer.verify_cell_evidence(
                    records, evidence_root=evidence_root, frozen_source_sha256=frozen
                )
            source.write_text("# changed source\n", encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "source worker.py evidence SHA256 mismatch"):
                reducer.verify_cell_evidence(
                    records, evidence_root=evidence_root, frozen_source_sha256=frozen
                )

    def test_evidence_verification_requires_root_and_independent_frozen_closure(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"
        records = synthetic_records(matrix, cell_id)
        for kwargs in ({"evidence_root": ROOT}, {"frozen_source_sha256": {"worker.py": "1" * 64}}):
            with self.assertRaisesRegex(protocol.ProtocolError, "must be supplied together"):
                reducer.reduce_cell_records(matrix, cell_id, records, **kwargs)

    def test_raw_aggregate_nonfinite_pid_and_pairing_fail_closed(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"

        aggregate_only = synthetic_records(matrix, cell_id)
        aggregate_only[0]["timing_modes"]["cache_neutral"] = {"p50_ms": 1.0}
        with self.assertRaisesRegex(protocol.ProtocolError, "requires raw_samples"):
            reducer.validate_cell_records(matrix, cell_id, aggregate_only)

        nonfinite = synthetic_records(matrix, cell_id)
        nonfinite[0]["timing_modes"]["cache_neutral"]["raw_samples"][0][
            "wall_ms"
        ] = 0.0
        with self.assertRaisesRegex(protocol.ProtocolError, "positive and finite"):
            reducer.validate_cell_records(matrix, cell_id, nonfinite)

        reused_pid = synthetic_records(matrix, cell_id)
        reused_pid[1]["pid"] = reused_pid[0]["pid"]
        with self.assertRaisesRegex(protocol.ProtocolError, "PID reused"):
            reducer.validate_cell_records(matrix, cell_id, reused_pid)

        mismatched_pair = synthetic_records(matrix, cell_id)
        mismatched_pair[1]["pairing_key_sha256"] = "f" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "pairing-key mismatch"):
            reducer.validate_cell_records(matrix, cell_id, mismatched_pair)

    def test_noncanonical_schedule_chronology_and_policy_drift_fail_closed(self) -> None:
        matrix = load_matrix()
        cell_id = "anchor_a100_pg_vs_fp16"

        wrong_slot = synthetic_records(matrix, cell_id)
        wrong_slot[0]["slot"] = 1
        with self.assertRaisesRegex(protocol.ProtocolError, "identity mismatch for slot"):
            reducer.validate_cell_records(matrix, cell_id, wrong_slot)

        overlap = synthetic_records(matrix, cell_id)
        overlap[1]["terminal"]["started_utc"] = overlap[0]["terminal"]["started_utc"]
        with self.assertRaisesRegex(protocol.ProtocolError, "chronology"):
            reducer.validate_cell_records(matrix, cell_id, overlap)

        drift = load_matrix()
        drift["policy"]["exact_prefix_pages"] = 3
        with self.assertRaisesRegex(protocol.ProtocolError, "policy drift"):
            protocol.validate_matrix(drift)

    def test_cpu_only_design_tools_cannot_launch_or_initialize_gpu(self) -> None:
        modules = (protocol, prepare, reducer)
        forbidden = {"torch", "flashinfer", "subprocess"}
        for module in modules:
            source = Path(module.__file__).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            self.assertTrue(forbidden.isdisjoint(imported), module.__file__)
            self.assertNotIn("nvidia-smi", source.lower())


if __name__ == "__main__":
    unittest.main()
