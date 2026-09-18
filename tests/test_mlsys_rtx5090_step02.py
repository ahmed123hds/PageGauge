"""CPU-only protocol regression tests; historical results are fixtures, not new evidence."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
import mlsys_rtx5090_step02 as step

OLD = ROOT / "results/final_fixed_suffix_s4_a128_t768/performance/fresh_process_williams_split128_v2"


def fixtures():
    """Adapt metadata in memory only; never write or relabel historical raw files."""
    rows = []
    for block in step.schedule():
        path = OLD / f"block_{block['index']}_{block['backend']}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["configuration"].update(step.worker_config(block, "/fixture/model/" + step.REVISION))
        rows.append(row)
    first = rows[0]
    archive = "/fixture/archive.zip"
    abi = {name: "/fixture/" + name for name in first["environment"]["flashinfer_abi"]["source_sha256"]}
    manifest = {"inputs": {"model_path": "/fixture/model/" + step.REVISION, "archive": archive, "abi_files": abi},
                "environment": {"torch": first["environment"]["torch"], "flashinfer-python": "0.6.17",
                                "transformers": first["environment"]["transformers"]},
                "source_sha256": first["source_sha256"], "schedule": step.schedule(),
                "input_file_evidence": {archive: {"sha256": first["configuration"]["wikitext_archive_sha256"]}},
                "manifest_sha256": "0" * 64,
                "statistics": {"bootstrap_samples": 200, "bootstrap_seed": 5140}}
    for key, path in abi.items():
        manifest["input_file_evidence"][path] = {"sha256": first["environment"]["flashinfer_abi"]["source_sha256"][key]}
    return rows, manifest


class Step02UnitTests(unittest.TestCase):
    def test_schedule_and_worker_flags(self):
        self.assertEqual([x["backend"] for x in step.schedule()],
                         ["flashinfer_fp16", "page_gauge", "page_gauge", "flashinfer_fp16",
                          "page_gauge", "flashinfer_fp16", "flashinfer_fp16", "page_gauge"])
        for block in step.schedule():
            command = step.worker_command(block, "/model", "/output/result.json")
            self.assertIn("--exact-tail", command)
            self.assertNotIn("--exact-tail-tokens", command)
            self.assertNotIn("--model-revision", command)
            self.assertIn("wikitext-2-raw/wiki.train.raw", command)
            self.assertNotIn("wikitext-2-raw/wiki.test.raw", command)

    def test_deterministic_cache_accounting(self):
        self.assertEqual(step.expected_cache("flashinfer_fp16")["served_bytes"], 11542724608)
        self.assertEqual(step.expected_cache("page_gauge")["served_bytes"], 7288520704)
        self.assertEqual(step.expected_cache("page_gauge")["total_bytes"], 7292719104)

    def test_busy_preflight_never_launches_worker(self):
        with mock.patch.object(step, "prerequisites", return_value={}), \
                mock.patch.object(step.entry, "idle_preflight", side_effect=RuntimeError("GPU BUSY")), \
                mock.patch.object(step, "run_process") as launch:
            with self.assertRaisesRegex(RuntimeError, "GPU BUSY"):
                step.launch({}, 0)
            launch.assert_not_called()

    def test_process_monitor_allows_only_own_pid_on_selected_gpu(self):
        with mock.patch.object(step.subprocess, "run", return_value=mock.Mock(stdout="GPU-test, 123\nGPU-other, 456\n")):
            self.assertEqual(step.process_snapshot("GPU-test", 123)["unexpected_pids"], [])
        with mock.patch.object(step.subprocess, "run", return_value=mock.Mock(stdout="GPU-test, 123\nGPU-test, 456\n")):
            self.assertEqual(step.process_snapshot("GPU-test", 123)["unexpected_pids"], [456])
        with mock.patch.object(step.subprocess, "run", return_value=mock.Mock(stdout="GPU-test, N/A\n")):
            with self.assertRaises(RuntimeError):
                step.process_snapshot("GPU-test", 123)

    def test_real_cpu_subprocess_log_and_telemetry(self):
        # Exercises actual Popen/tee/join, but no GPU query or GPU worker.
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(step, "process_snapshot", side_effect=lambda gpu, pid: {"pids": [pid], "unexpected_pids": []}):
            result = step.run_process([sys.executable, "-c", "print('cpu fixture')"], Path(temp), {"index": 0}, "GPU-fake")
            self.assertEqual(result["return_code"], 0)
            self.assertTrue(result["sampled_exclusivity_passed"])
            self.assertIn("cpu fixture", (Path(temp) / "block_0.log").read_text())

    def test_invisible_worker_is_not_attested_as_exclusive(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(step, "process_snapshot", return_value={"pids": [], "unexpected_pids": []}):
            result = step.run_process([sys.executable, "-c", "pass"], Path(temp), {"index": 0}, "GPU-fake")
            self.assertFalse(result["own_pid_seen"])
            self.assertFalse(result["sampled_exclusivity_passed"])


@unittest.skipUnless(OLD.is_dir(), "Historical raw fixtures unavailable")
class Step02PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = fixtures()

    def validate(self, row, index=0, return_code=None):
        if return_code is None:
            return_code = 0 if row["correctness"]["backend_vs_hf_sdpa_fp16"]["passed"] else 2
        return step.validate_worker(row, step.schedule()[index], self.manifest, return_code)

    def test_all_eight_historical_payload_schemas_and_backend_aware_recurrence(self):
        for i, row in enumerate(self.rows):
            with self.subTest(index=i):
                self.assertTrue(self.validate(row, i)["execution_passed"])

    def test_hf_only_failure_stays_visible_but_does_not_fail_execution(self):
        row = copy.deepcopy(self.rows[0])
        row["passed"] = False
        row["correctness"]["backend_vs_hf_sdpa_fp16"]["passed"] = False
        result = self.validate(row, return_code=2)
        self.assertTrue(result["execution_passed"])
        self.assertFalse(result["hf_quality_passed"])
        with self.assertRaisesRegex(RuntimeError, "exit code"):
            self.validate(row, return_code=1)

    def test_failures_cannot_be_hidden_by_overall_passed(self):
        changes = [
            lambda r: r["configuration"].update(decode_steps=768),
            lambda r: r["correctness"]["same_backend_eager_vs_graph"].update(passed=False),
            lambda r: r["correctness"]["same_backend_eager_vs_graph"]["every_page_close_cache_digest"].update(checked_page_closes=95),
            lambda r: r["correctness"]["runtime_page_finalization_and_consumption"].update(runtime_finalized_int8_pages_consumed_count=1),
            lambda r: r["timing_modes"]["cache_neutral"]["raw_samples"][0].update(wall_ms=math.nan),
            lambda r: r["timing_modes"]["cache_hot"]["raw_samples"][0].update(cuda_ms=0),
            lambda r: r["timing_modes"]["cache_hot"]["raw_samples"][0]["runtime_gate"]["observed_dispatch"].update(eager_calls=1),
            lambda r: r["cache_build"].update(selected_backend_cache_served_bytes_excluding_following_canary=1),
            lambda r: r["environment"].update(compute_capability=[8, 0]),
            lambda r: r["source_sha256"].update({step.WORKER: "0" * 64}),
        ]
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                row = copy.deepcopy(self.rows[0])
                change(row)
                with self.assertRaises(RuntimeError):
                    self.validate(row)

    def test_input_mismatch_and_partial_results_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Eight"):
            step.reduce_results(self.rows[:2], self.manifest)
        rows = copy.deepcopy(self.rows)
        rows[1]["pairing"]["configuration"]["teacher_inputs_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "input mismatch"):
            step.reduce_results(rows, self.manifest)

    def test_hierarchical_reducer_known_two_x_and_honest_scope(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for mode in step.MODES:
                for sample in row["timing_modes"][mode]["raw_samples"]:
                    for metric in step.METRICS:
                        sample[metric] = 2.0 if row["backend"] == "flashinfer_fp16" else 1.0
        result = step.reduce_results(rows, self.manifest)
        self.assertEqual(result["adjacent_pairs"], 4)
        self.assertEqual(result["fixture_clusters"], 2)
        self.assertTrue(result["performance_gate_passed"])
        for endpoint in result["endpoints"].values():
            self.assertAlmostEqual(endpoint["point_speedup"], 2.0)
            self.assertEqual(endpoint["speedup_95_ci"], [2.0, 2.0])
        self.assertFalse(result["new_heldout_quality_claim"])
        self.assertFalse(result["reconstruction_control_implemented"])


if __name__ == "__main__":
    unittest.main()
