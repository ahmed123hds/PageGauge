#!/usr/bin/env python3
"""CPU-only tests for the backend-exclusive publication protocol."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

import analyze_backend_exclusive as analysis  # noqa: E402
import benchmark_backend_exclusive as worker  # noqa: E402
import orchestrate_backend_exclusive as orchestration  # noqa: E402

import torch  # noqa: E402


def worker_payload(backend: str, seed: int, latency: float) -> dict:
    memory_snapshot = {
        "torch_allocated_bytes": 10,
        "torch_reserved_bytes": 20,
        "cuda_mem_get_info_free_bytes": 1_000,
        "cuda_mem_get_info_total_bytes": 2_000,
    }
    config = {
        "backend": backend,
        "model": "fixture/model",
        "model_revision": "fixture-revision",
        "model_config_sha256": "a" * 64,
        "batch_size": 4,
        "context": 20480,
        "decode_steps": 16,
        "exact_tail_tokens": 256,
        "prefill_chunk_tokens": 1024,
        "baseline_split_pages": 256,
        "candidate_split_pages": 128,
        "tail_attention": "flashinfer_merge",
        "cuda_graph_scope": "attention",
        "seed": seed,
        "token_offset": 0,
        "token_stride": 0,
        "warmups": 3,
        "repeats": 4,
        "cache_scrub_mib": 256,
    }
    return {
        "schema_version": 1,
        "experiment": "synthetic_backend_exclusive_worker",
        "passed": True,
        "backend": backend,
        "seed": seed,
        "model": "fixture/model",
        "model_revision": "fixture-revision",
        "model_config_sha256": "a" * 64,
        "batch_size": 4,
        "context": 20480,
        "decode_steps": 16,
        "exact_tail_tokens": 256,
        "config": config,
        "token_source": {"token_ids_sha256": "b" * 64},
        "fixture_provenance": {"sampled_boundary_kv_sha256": "e" * 64},
        "exclusivity": {
            "fresh_process_required": True,
            "selected_persistent_backend": backend,
            "opposite_backend_full_gpu_cache_allocated": False,
            "hf_dynamic_caches_released_before_decoder_construction": True,
            "direct_destination_construction": True,
        },
        "residency_gate": {
            "passed": True,
            "selected_layers": [0, 15, 20, 23, 31],
        },
        "correctness": {
            "passed": True,
            "hashes": {
                "hf_generated_tokens_sha256": "1" * 64,
                "graph_generated_tokens_sha256": "1" * 64,
            },
        },
        "memory": {
            name: dict(memory_snapshot)
            for name in (
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
        },
        "environment": {"orchestration_environment": {}},
        "source_sha256": {"diagnostics/fake_worker.py": "f" * 64},
        "timing_modes": {
            mode: {
                "raw_samples": [
                    {
                        "sample_index": index,
                        "wall_ms": latency * factor,
                        "cuda_ms": latency * factor,
                    }
                    for index, factor in enumerate((1.0, 1.1, 0.9, 1.0))
                ]
            }
            for mode in analysis.MODES
        },
    }


def write_synthetic_run(
    run_dir: Path,
    schedule: list[dict],
    payload_mutator=None,
) -> Path:
    executions = {}
    for block in schedule:
        block_dir = run_dir / "blocks" / block["block_id"]
        block_dir.mkdir(parents=True)
        result_path = block_dir / "worker_result.json"
        latency = 2.0 if block["backend"] == analysis.BASELINE else 1.0
        payload = worker_payload(block["backend"], block["seed"], latency)
        payload["config"]["token_offset"] = block["token_offset"]
        payload["environment"]["orchestration_environment"] = (
            analysis.expected_orchestration_environment("c" * 64, block)
        )
        if payload_mutator is not None:
            payload_mutator(payload, block)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        executions[block["block_id"]] = {
            "status": "completed",
            "worker_result_path": str(result_path.relative_to(run_dir)),
            "worker_result_sha256": analysis.sha256_file(result_path),
        }
    manifest = {
        "schema_version": 1,
        "config_sha256": "c" * 64,
        "schedule": schedule,
        "executions": executions,
    }
    manifest_path = run_dir / "orchestration_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class BackendExclusiveProtocolTest(unittest.TestCase):
    def test_hot_sample_has_no_restore_or_canary_between_precondition_and_sample(
        self,
    ) -> None:
        events: list[str] = []
        cache_scrub = SimpleNamespace(add_=lambda _value: events.append("scrub"))

        def timed(*_args, **_kwargs):
            events.append("timed")
            return 1.0, 1.0

        def compare(*_args, **_kwargs):
            events.append("digest")
            return {"passed": True}

        with (
            mock.patch.object(
                worker,
                "restore_mutated_page",
                side_effect=lambda *_args: events.append("restore"),
            ),
            mock.patch.object(worker, "timed_generated_block", side_effect=timed),
            mock.patch.object(
                worker, "compare_exact_prefix_digest", side_effect=compare
            ),
            mock.patch.object(worker.torch.cuda, "synchronize", return_value=None),
        ):
            result = worker.measure_timing_modes(
                decoder=SimpleNamespace(
                    batch_size=1,
                    plan=lambda _position: events.append("plan"),
                ),
                initial_token=None,
                dynamic_token=None,
                initial_cache={},
                start_position=16,
                cache_scrub=cache_scrub,
                warmups=0,
                repeats=1,
                expected_prefix_digest={"enabled": True},
            )

        self.assertEqual(
            events,
            [
                "restore",
                "scrub",
                "timed",
                "digest",
                "restore",
                "timed",
                "plan",
                "timed",
                "digest",
                "restore",
            ],
        )
        hot_sample = result["cache_hot"]["raw_samples"][0]
        self.assertTrue(hot_sample["precondition"]["no_restore_before_timed_sample"])
        self.assertEqual(hot_sample["precondition"]["precondition_steps"], 16)
        self.assertTrue(
            hot_sample["precondition"]["planner_reset_without_cache_restore"]
        )
        self.assertTrue(
            result["cache_hot"][
                "no_restore_between_hot_precondition_and_sample"
            ]
        )

    def test_worker_accepts_generic_exact_prefix_length(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "worker",
                "--backend",
                "page_gauge",
                "--exact-sink-pages",
                "3",
                "--output",
                "fixture.json",
            ],
        ):
            args = worker.parse_args()
        self.assertEqual(args.exact_sink_pages, 3)

    def test_s3_snapshot_restore_is_request_major_and_never_touches_prefix(self) -> None:
        batch_size = 2
        prefix_pages = 3
        tail_pages = 2
        exact_storage_pages = prefix_pages + tail_pages
        max_pages = 8
        layers = 2
        exact_total_pages = batch_size * exact_storage_pages
        code_total_pages = batch_size * max_pages

        exact_key = torch.arange(
            layers * exact_total_pages * 2, dtype=torch.float32
        ).reshape(layers, exact_total_pages, 2)
        exact_value = exact_key.clone() + 1000
        key_codes = torch.arange(
            layers * code_total_pages * 2, dtype=torch.float32
        ).reshape(layers, code_total_pages, 2)
        value_codes = key_codes.clone() + 2000
        key_scales = key_codes.clone() + 3000
        value_scales = key_codes.clone() + 4000
        cache = SimpleNamespace(
            exact_key=exact_key,
            exact_value=exact_value,
            key_codes=key_codes,
            value_codes=value_codes,
            key_scales=key_scales,
            value_scales=value_scales,
        )

        def exact_physical_page(request: int, logical_page: int) -> int:
            local_page = (
                logical_page
                if logical_page < prefix_pages
                else prefix_pages + logical_page % tail_pages
            )
            return request * exact_storage_pages + local_page

        decoder = SimpleNamespace(
            backend="page_gauge",
            batch_size=batch_size,
            exact_prefix_pages=prefix_pages,
            exact_sink_pages=prefix_pages,
            exact_tail_pages=tail_pages,
            max_pages=max_pages,
            exact_physical_page=exact_physical_page,
            cache=cache,
        )
        prefix_digest = worker.immutable_exact_prefix_digest(decoder)
        self.assertEqual(prefix_digest["physical_pages"], [0, 1, 2, 5, 6, 7])
        decoder.exact_physical_page = (
            lambda request, logical_page: request * exact_storage_pages
            + (logical_page + 1) % exact_storage_pages
        )
        with self.assertRaisesRegex(RuntimeError, "fixed slots"):
            worker.immutable_exact_prefix_digest(decoder)
        decoder.exact_physical_page = exact_physical_page
        logical_page = 7
        snapshot = worker.snapshot_mutated_page(decoder, logical_page * worker.PG.PAGE)
        self.assertEqual(snapshot["physical_pages"]["exact"], (4, 9))
        self.assertEqual(snapshot["physical_pages"]["codes"], (7, 15))
        with self.assertRaisesRegex(ValueError, "immutable exact prefix"):
            worker.snapshot_mutated_page(decoder, worker.PG.PAGE)

        selected_exact = snapshot["physical_pages"]["exact"]
        selected_codes = snapshot["physical_pages"]["codes"]
        original_exact = exact_key[:, list(selected_exact)].clone()
        original_codes = key_codes[:, list(selected_codes)].clone()
        prefix_physical = (0, 1, 2, 5, 6, 7)
        untouched_tail_physical = (3, 8)
        exact_key[:, list(selected_exact)].fill_(-1)
        key_codes[:, list(selected_codes)].fill_(-1)
        exact_key[:, list(prefix_physical)].add_(100_000)
        exact_key[:, list(untouched_tail_physical)].add_(200_000)
        mutated_prefix = exact_key[:, list(prefix_physical)].clone()
        mutated_other_tail = exact_key[:, list(untouched_tail_physical)].clone()

        worker.restore_mutated_page(decoder, snapshot)
        self.assertTrue(torch.equal(exact_key[:, list(selected_exact)], original_exact))
        self.assertTrue(torch.equal(key_codes[:, list(selected_codes)], original_codes))
        self.assertTrue(
            torch.equal(exact_key[:, list(prefix_physical)], mutated_prefix)
        )
        self.assertTrue(
            torch.equal(
                exact_key[:, list(untouched_tail_physical)], mutated_other_tail
            )
        )

    def test_worker_accepts_opt_in_heterogeneous_fa2(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "worker",
                "--backend",
                "page_gauge",
                "--tail-attention",
                "heterogeneous_fa2",
                "--output",
                "fixture.json",
            ],
        ):
            args = worker.parse_args()
        self.assertEqual(args.tail_attention, "heterogeneous_fa2")

    def test_full_decoder_layer_graph_scope_is_explicitly_selectable(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "worker",
                "--backend",
                "page_gauge",
                "--cuda-graph-scope",
                "decoder_layer",
                "--output",
                "fixture.json",
            ],
        ):
            args = worker.parse_args()
        self.assertEqual(args.cuda_graph_scope, "decoder_layer")

        with mock.patch.object(
            sys,
            "argv",
            [
                "orchestrator",
                "--output-dir",
                "fixture-run",
                "--cuda-graph-scope",
                "decoder_layer",
                "--dry-run",
            ],
        ):
            orchestrator_args = orchestration.parse_args()
        profile = orchestration.resolved_profile(orchestrator_args)
        block = orchestration.build_schedule(
            profile["seeds"],
            profile["pairs_per_seed"],
            0,
            profile["seed_token_offset_stride"],
        )[0]
        command = orchestration.worker_command(
            orchestrator_args,
            profile,
            block,
            Path("worker-result.json"),
        )
        scope_index = command.index("--cuda-graph-scope")
        self.assertEqual(command[scope_index + 1], "decoder_layer")
        config = orchestration.build_config(orchestrator_args, profile, {})
        self.assertEqual(config["cuda_graph_scope"], "decoder_layer")

    def test_worker_extra_args_cannot_override_recorded_graph_scope(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "orchestrator",
                "--output-dir",
                "fixture-run",
                "--worker-extra-arg=--cuda-graph-scope=decoder_layer",
                "--dry-run",
            ],
        ):
            args = orchestration.parse_args()
        profile = orchestration.resolved_profile(args)
        with self.assertRaisesRegex(
            analysis.ProtocolError, "cannot override controlled option"
        ):
            orchestration.validate_args(args, profile)

    def test_token_major_centers_preserve_every_kv_head(self) -> None:
        tokens, heads, dimension = 7, 8, 128
        token_axis = torch.arange(tokens, dtype=torch.float32)[:, None, None]
        head_axis = torch.arange(heads, dtype=torch.float32)[None, :, None]
        channel_axis = torch.arange(dimension, dtype=torch.float32)[None, None, :]
        key = (100.0 * head_axis + token_axis + channel_axis / 256.0).half()
        value = (-50.0 * head_axis + 2.0 * token_axis - channel_axis / 512.0).half()

        key_center, value_center = worker.token_major_centers(key, value)

        self.assertEqual(tuple(key_center.shape), (heads, dimension))
        self.assertEqual(tuple(value_center.shape), (heads, dimension))
        self.assertTrue(torch.equal(key_center, key.float().mean(dim=0).half()))
        self.assertTrue(torch.equal(value_center, value.float().mean(dim=0).half()))
        self.assertFalse(torch.equal(key_center[0], key_center[1]))
        self.assertFalse(torch.equal(value_center[0], value_center[1]))
        self.assertEqual(
            tuple((key.float() - key_center[None].float()).shape),
            (tokens, heads, dimension),
        )

    def test_williams_schedule_is_adjacent_and_balanced(self) -> None:
        schedule = orchestration.build_schedule(
            [11, 12], pairs_per_seed=4, base_token_offset=5, seed_token_offset_stride=100
        )
        self.assertEqual(len(schedule), 16)
        by_seed: dict[int, str] = {}
        for seed_index in (0, 1):
            blocks = [item for item in schedule if item["seed_index"] == seed_index]
            by_seed[seed_index] = "".join(item["symbol"] for item in blocks)
            for pair_index in range(4):
                pair = [item for item in blocks if item["pair_index"] == pair_index]
                self.assertEqual(len(pair), 2)
                self.assertEqual(
                    pair[1]["chronological_index"],
                    pair[0]["chronological_index"] + 1,
                )
                self.assertEqual({item["symbol"] for item in pair}, {"A", "B"})
        self.assertEqual(by_seed[0], "ABBABAAB")
        self.assertEqual(by_seed[1], "BAABABBA")
        self.assertEqual(
            {item["token_offset"] for item in schedule if item["seed_index"] == 0},
            {5},
        )
        self.assertEqual(
            {item["token_offset"] for item in schedule if item["seed_index"] == 1},
            {105},
        )

    def test_raw_sample_contract_rejects_aggregate_only_payload(self) -> None:
        payload = worker_payload(analysis.BASELINE, 7, 2.0)
        payload["timing_modes"]["cache_hot"] = {"p50_ms": 2.0}
        with self.assertRaises(analysis.ProtocolError):
            analysis.validate_worker_result(payload, analysis.BASELINE, 7)

    def test_raw_sample_count_must_equal_config_repeats(self) -> None:
        payload = worker_payload(analysis.BASELINE, 7, 2.0)
        payload["timing_modes"]["cache_neutral"]["raw_samples"].pop()
        with self.assertRaisesRegex(analysis.ProtocolError, "config.repeats"):
            analysis.validate_worker_result(payload, analysis.BASELINE, 7)

    def test_end_to_end_synthetic_analysis_is_exactly_two_x(self) -> None:
        schedule = orchestration.build_schedule(
            [101, 102],
            pairs_per_seed=2,
            base_token_offset=0,
            seed_token_offset_stride=1000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_synthetic_run(run_dir, schedule)
            result = analysis.analyze_run_directory(
                run_dir,
                bootstrap_samples=200,
                bootstrap_seed=17,
                write_outputs=False,
            )
            for mode in analysis.MODES:
                for metric in analysis.METRICS:
                    aggregate = result["aggregates"][mode][metric]
                    self.assertTrue(
                        math.isclose(aggregate["speedup_geomean"], 2.0, rel_tol=1e-12)
                    )
                    self.assertEqual(
                        aggregate["hierarchical_seed_pair_bootstrap"]["speedup_95_ci"],
                        [2.0, 2.0],
                    )
            self.assertEqual(result["raw_sample_count"], 128)
            self.assertEqual(result["paired_record_count"], 16)

    def test_paired_generated_trajectory_hash_mismatch_fails_closed(self) -> None:
        schedule = orchestration.build_schedule(
            [303],
            pairs_per_seed=2,
            base_token_offset=0,
            seed_token_offset_stride=0,
        )

        def mutate(payload: dict, block: dict) -> None:
            if block["backend"] == analysis.CANDIDATE:
                hashes = payload["correctness"]["hashes"]
                hashes["hf_generated_tokens_sha256"] = "9" * 64
                hashes["graph_generated_tokens_sha256"] = "9" * 64

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_synthetic_run(run_dir, schedule, mutate)
            with self.assertRaisesRegex(
                analysis.ProtocolError, "mismatched model/token/config provenance"
            ):
                analysis.analyze_run_directory(
                    run_dir,
                    bootstrap_samples=50,
                    bootstrap_seed=19,
                    write_outputs=False,
                )

    def test_pair_provenance_mismatch_fails_closed(self) -> None:
        baseline = worker_payload(analysis.BASELINE, 9, 2.0)
        candidate = worker_payload(analysis.CANDIDATE, 9, 1.0)
        candidate["token_source"]["token_ids_sha256"] = "d" * 64
        self.assertNotEqual(
            analysis.worker_provenance(baseline),
            analysis.worker_provenance(candidate),
        )

    def test_cuda_graph_scope_provenance_mismatch_fails_closed(self) -> None:
        baseline = worker_payload(analysis.BASELINE, 9, 2.0)
        candidate = worker_payload(analysis.CANDIDATE, 9, 1.0)
        candidate["config"]["cuda_graph_scope"] = "decoder_layer"
        baseline_provenance = analysis.worker_provenance(baseline)
        candidate_provenance = analysis.worker_provenance(candidate)
        self.assertEqual(
            baseline_provenance["timing_configuration"]["cuda_graph_scope"],
            "attention",
        )
        self.assertEqual(
            candidate_provenance["timing_configuration"]["cuda_graph_scope"],
            "decoder_layer",
        )
        self.assertNotEqual(baseline_provenance, candidate_provenance)


if __name__ == "__main__":
    unittest.main()
