#!/usr/bin/env python3
"""CPU-only tests for the PageGauge MLSys 2027 experiment contract."""

from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import mlsys_experiment_protocol as protocol  # noqa: E402
import prepare_mlsys_experiment_matrix as prepare  # noqa: E402


SPEC_PATH = ROOT / "docs/mlsys_2027_experiment_matrix.json"


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


class MLSysExperimentProtocolTests(unittest.TestCase):
    def test_design_spec_validates_and_records_every_acceptance_gap(self) -> None:
        spec = load_spec()
        protocol.validate_spec(spec)
        unresolved = protocol.unresolved_requirements(spec)
        self.assertIn("hardware:h100_sxm_80gb:access", unresolved)
        self.assertIn("model:llama_31_8b_instruct:revision", unresolved)
        self.assertIn("model:qwen_25_7b_instruct:revision", unresolved)
        self.assertIn(
            "method:page_gauge_explicit_reconstruction:implementation", unresolved
        )
        self.assertIn(
            "method:page_gauge_fused_reconstruction:implementation", unresolved
        )
        self.assertIn("method:bitdecoding:implementation", unresolved)
        self.assertIn("suite:serving_slo_common:runner", unresolved)

    def test_matrix_expansion_is_deterministic_unique_and_paired(self) -> None:
        spec = load_spec()
        first = protocol.expand_matrix(spec)
        second = protocol.expand_matrix(spec)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1_000)
        self.assertEqual(len({row["run_id"] for row in first}), len(first))
        self.assertEqual(
            len({row["run_identity_sha256"] for row in first}), len(first)
        )
        target = [
            row
            for row in first
            if row["identity"]["suite_id"] == "controlled_factorization_common"
            and row["identity"]["model_id"] == "mistral_7b_v03"
            and row["identity"]["hardware_id"] == "a100_sxm4_40gb"
            and row["identity"]["context"] == 16384
            and row["identity"]["batch_size"] == 4
            and row["identity"]["decode_steps"] == 1024
            and row["identity"]["fresh_process_replicate"] == 0
        ]
        self.assertEqual(len(target), 4)
        self.assertEqual(len({row["pair_group_sha256"] for row in target}), 1)
        self.assertEqual(len({row["identity"]["method_id"] for row in target}), 4)

    def test_context_above_model_limit_fails_closed(self) -> None:
        spec = load_spec()
        suite = next(
            row
            for row in spec["suites"]
            if row["suite_id"] == "controlled_factorization_common"
        )
        suite["contexts"].append(65536)
        with self.assertRaisesRegex(
            protocol.ProtocolError, "exceeds model mistral_7b_v03"
        ):
            protocol.validate_spec(spec)

    def test_duplicate_and_non_page_aligned_axes_fail_closed(self) -> None:
        duplicate = load_spec()
        duplicate["models"].append(copy.deepcopy(duplicate["models"][0]))
        with self.assertRaisesRegex(protocol.ProtocolError, "duplicate models id"):
            protocol.validate_spec(duplicate)

        unaligned = load_spec()
        unaligned["suites"][0]["decode_steps"] = [257]
        with self.assertRaisesRegex(
            protocol.ProtocolError, "decode_steps must be page aligned"
        ):
            protocol.validate_spec(unaligned)

    def test_manifest_is_source_closed_and_tamper_evident(self) -> None:
        manifest = prepare.prepare(SPEC_PATH)
        protocol.verify_manifest(manifest)
        self.assertEqual(manifest["status"], "design")
        self.assertIs(manifest["runnable"], False)
        self.assertEqual(manifest["run_count"], len(manifest["runs"]))
        self.assertEqual(
            set(manifest["source_sha256"]),
            {
                "diagnostics/mlsys_experiment_protocol.py",
                "diagnostics/prepare_mlsys_experiment_matrix.py",
                "docs/mlsys_2027_experiment_matrix.json",
            },
        )
        self.assertIs(manifest["result_contract"]["raw_samples_required"], True)
        tampered = copy.deepcopy(manifest)
        tampered["runs"][0]["identity"]["batch_size"] = 999
        with self.assertRaisesRegex(
            protocol.ProtocolError, "manifest hash integrity"
        ):
            protocol.verify_manifest(tampered)

    def test_protocol_is_cpu_only_and_does_not_import_execution_stacks(self) -> None:
        source = Path(protocol.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue(
            {"torch", "flashinfer", "subprocess"}.isdisjoint(imported_roots)
        )
        self.assertNotIn("nvidia-smi", source.lower())


if __name__ == "__main__":
    unittest.main()
