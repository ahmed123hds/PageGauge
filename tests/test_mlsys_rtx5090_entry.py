"""CPU-only checks. All GPU telemetry is fake; no accelerator modules imported."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
import mlsys_rtx5090_entry as entry
import mlsys_rtx5090_step01 as worker


class EntryTests(unittest.TestCase):
    IDLE = "NVIDIA GeForce RTX 5090, GPU-test, 32607, 700, 0\n"

    def test_idle_and_other_gpu_process(self):
        for processes in ("", "GPU-other, 400\n"):
            with self.subTest(processes=processes):
                row = entry.parse_device_snapshot(self.IDLE, processes)
                self.assertEqual(row["compute_process_count"], 0)
                self.assertEqual(row["uuid"], "GPU-test")

    def test_existing_job_is_rejected_even_when_utilization_zero(self):
        with self.assertRaisesRegex(RuntimeError, "GPU BUSY"):
            entry.parse_device_snapshot(self.IDLE, "GPU-test, 1234\n")

    def test_uncertain_telemetry_is_rejected(self):
        for device, processes in (
            (self.IDLE.replace(", 700,", ", N/A,"), ""),
            (self.IDLE.replace(", 700,", ", nan,"), ""),
            (self.IDLE.replace(", 700,", ", -1,"), ""),
            (self.IDLE.replace("5090", "4090"), ""),
            (self.IDLE.replace("32607", "16000"), ""),
            (self.IDLE, "GPU-test, N/A"),
            (self.IDLE, "Not Supported"),
            (self.IDLE + self.IDLE, ""),
        ):
            with self.subTest(device=device, processes=processes):
                with self.assertRaises(RuntimeError):
                    entry.parse_device_snapshot(device, processes)

    def test_busy_memory_or_utilization_is_rejected(self):
        for device in (
            self.IDLE.replace(", 700,", ", 3000,"),
            self.IDLE.replace(", 0\n", ", 20\n"),
        ):
            with self.subTest(device=device):
                with self.assertRaisesRegex(RuntimeError, "GPU BUSY"):
                    entry.parse_device_snapshot(device, "")

    def test_busy_preflight_cannot_start_worker(self):
        with mock.patch.object(entry, "idle_preflight", side_effect=RuntimeError("GPU BUSY")), \
                mock.patch.object(entry.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "GPU BUSY"):
                entry.launch_step01({}, 0)
            popen.assert_not_called()

    def test_environment_check_never_queries_gpu_or_runs_subprocess(self):
        with mock.patch.object(entry.subprocess, "run", side_effect=AssertionError("No subprocess permitted")), \
                mock.patch.object(entry, "idle_preflight", side_effect=AssertionError("No GPU query permitted")):
            checked = entry.check_environment()
        self.assertIs(checked["cpu_only"], True)
        self.assertIs(checked["gpu_queried"], False)

    def test_entry_and_worker_top_level_imports_are_cpu_only(self):
        forbidden = {"torch", "flashinfer", "numpy", "transformers", "benchmark_page_gauge_transformer"}
        for module in (entry, worker):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported = {(node.module or "").split(".")[0]}
                else:
                    continue
                self.assertFalse(imported & forbidden)

    def test_missing_ninja_fails_before_gpu_launch(self):
        with mock.patch.object(entry.shutil, "which", return_value=None):
            checked = entry.check_environment()
        self.assertFalse(checked["passed"])
        self.assertTrue(any("ninja is not on PATH" in item for item in checked["failures"]))

    def test_ninja_path_is_reported(self):
        with mock.patch.object(entry.shutil, "which", return_value="/test/env/bin/ninja"):
            checked = entry.check_environment()
        self.assertEqual(checked["build_tools"]["ninja"], "/test/env/bin/ninja")
        self.assertFalse(any("ninja is not on PATH" in item for item in checked["failures"]))


class SnapshotTests(unittest.TestCase):
    def test_all_predeclared_snapshots_cover_context_once(self):
        for generated in worker.CONFIG["generated_token_snapshots"]:
            with self.subTest(generated=generated):
                row = worker.expected_partition(generated)
                old, exact = set(row["old"]), set(row["exact"])
                self.assertFalse(old & exact)
                self.assertEqual(old | exact, set(range(row["total_pages"])))
                self.assertLessEqual(len(exact), 180)
                self.assertEqual(row["total_pages"] - 1, max(exact))
                self.assertEqual((row["total_pages"] - 1) * 16 + row["last_page_len"], 20480 + generated)

    def test_initial_snapshot_does_not_double_count_suffix_and_tail(self):
        row = worker.expected_partition(0)
        self.assertEqual(len(row["old"]), 1148)
        self.assertEqual(len(row["exact"]), 132)
        self.assertEqual(row["generated_old_pages_in_snapshot"], 0)

    def test_page_boundaries(self):
        for generated, expected_len in ((1, 1), (15, 15), (16, 16)):
            self.assertEqual(worker.expected_partition(generated)["last_page_len"], expected_len)
        self.assertEqual(worker.expected_partition(768)["generated_old_pages_in_snapshot"], 0)
        self.assertEqual(worker.expected_partition(784)["generated_old_pages_in_snapshot"], 1)

    def test_final_snapshot_contains_48_generated_history_pages(self):
        row = worker.expected_partition(1536)
        self.assertEqual(len(row["old"]), 1196)
        self.assertEqual(len(row["exact"]), 180)
        self.assertEqual(row["generated_old_pages_in_snapshot"], 48)


if __name__ == "__main__":
    unittest.main()
