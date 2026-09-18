#!/usr/bin/env python3
"""Fresh-process FP16/PageGauge full-decoder performance; not reconstruction control.

Standard library only. The unchanged GPU worker runs in a separate process.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlsys_rtx5090_entry as entry
from reduce_mlsys_controlled import _hierarchical_bootstrap_ci

ROOT = entry.ROOT
EXPERIMENT = "pagegauge_mlsys2027_rtx5090_fp16_contrast_v1"
REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
WORKER = "diagnostics/benchmark_sustained_dynamic_graphs.py"
EXTRA_SOURCES = (
    "diagnostics/mlsys_rtx5090_step02.py", "diagnostics/reduce_mlsys_controlled.py",
    "diagnostics/mlsys_controlled_protocol.py", WORKER,
    "diagnostics/benchmark_backend_exclusive.py", "diagnostics/benchmark_full_sequence_graph.py",
    "diagnostics/benchmark_model_prefill_correctness.py", "diagnostics/benchmark_token_step_graphs.py",
    "scripts/prepare_flashinfer_page_gauge.py", "scripts/prepare_flashinfer_page_gauge_heterogeneous.py",
    "tests/page_gauge_append_extension.cu", "patches/flashinfer-0.6.17-page-gauge-heterogeneous.patch",
    "docs/decoder_layer_cuda_graphs.md",
)
CONFIG = {
    "batch_size": 4, "context": 20480, "decode_steps": 1536,
    "exact_tail_tokens": 768, "prefill_chunk_tokens": 1024,
    "baseline_split_pages": 256, "candidate_split_pages": 128,
    "tail_attention": "flashinfer_merge", "old_value_scale_placement": "probability",
    "trajectory_mode": "frozen_hf_teacher_forced", "cuda_graph_scope": "decoder_layer_device_dynamic",
    "token_source": "wikitext2", "wikitext_member": "wikitext-2-raw/wiki.train.raw",
    "token_stride": 23600, "capture_warmups": 1, "maximum_graph_banks": 16,
    "warmups": 1, "repeats": 3, "cache_scrub_mib": 256,
    "min_logits_cosine": 0.995, "min_top1_agreement": 0.99,
    "quality_diagnostics_top_k": 0, "quality_diagnostics_top_vocab": 8,
}
MODES = ("cache_neutral", "cache_hot")
METRICS = ("wall_ms", "cuda_ms")
MATCHED_FIELDS = (
    "teacher_inputs_sha256", "token_matrix_sha256", "token_source_provenance_sha256",
    "model_config_sha256", "sampled_model_parameters_sha256", "flashinfer_abi_source_sha256",
)
SCOPE = (
    "Matched fixed-B4 GPU-resident full-decoder loop, including planner, cache append/finalize, "
    "embedding, all layers, final norm, LM head and argmax. Excludes load, prefill, graph capture, "
    "cache restore/scrub and preconditioning. Not online serving, not a reconstruction control, "
    "not independent held-out quality confirmation. Only two fixture clusters."
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def schedule():
    rows = []
    for cluster, order in enumerate(("ABBA", "BAAB")):
        for slot, treatment in enumerate(order):
            rows.append({"index": len(rows), "backend": "flashinfer_fp16" if treatment == "A" else "page_gauge",
                         "seed": 20260861 + cluster, "token_offset": 94400 * cluster,
                         "pair_id": 2 * cluster + slot // 2,
                         "pair_order": order[slot // 2 * 2:slot // 2 * 2 + 2]})
    return rows


def worker_config(block, model_path):
    pg = block["backend"] == "page_gauge"
    return {**CONFIG, "backend": block["backend"], "model": str(model_path),
            "model_revision": REVISION, "seed": block["seed"], "token_offset": block["token_offset"],
            "exact_sink_pages": 4 if pg else 0, "exact_prefix_pages": 4 if pg else 0,
            "exact_static_suffix_pages": 128 if pg else 0}


def worker_command(block, model_path, output):
    config = worker_config(block, model_path)
    aliases = {"exact_tail_tokens": "exact-tail"}
    command = [sys.executable, "-u", str(ROOT / WORKER)]
    for key, value in config.items():
        if key not in {"model_revision", "cuda_graph_scope", "exact_prefix_pages"}:
            command.extend(["--" + aliases.get(key, key.replace("_", "-")), str(value)])
    command.extend(["--wikitext-zip", str(ROOT / "data/wikitext-2-raw-v1.zip"), "--output", str(output)])
    return command


def passing_step01():
    directory = ROOT / "results/mlsys2027_rtx5090/01_correctness"
    for path in sorted(directory.glob("*/completion.json"), reverse=True):
        completion = json.loads(path.read_text(encoding="utf-8"))
        result_path = path.with_name("result.json")
        if completion.get("passed") is not True or not result_path.is_file():
            continue
        if entry.sha256_file(result_path) != completion.get("result_sha256"):
            continue
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        production = {name: digest for name, digest in manifest["source_sha256"].items()
                      if name.startswith(("scripts/", "patches/", "build/"))}
        if production and all((ROOT / name).is_file() and entry.sha256_file(ROOT / name) == digest
                              for name, digest in production.items()):
            return {"path": str(result_path), "sha256": completion["result_sha256"]}
    raise RuntimeError("No passing Step 01 with matching production sources; run 01 first.")


def prerequisites():
    model_path = Path(os.environ.get("PG_MODEL_SNAPSHOT", str(
        Path.home() / ".cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots" / REVISION))).expanduser().resolve()
    require(model_path.name == REVISION and model_path.parent.name == "snapshots" and model_path.is_dir(),
            "Pinned local Mistral snapshot is missing or not under snapshots/<revision>.")
    index_path = model_path / "model.safetensors.index.json"
    require(index_path.is_file(), "Mistral shard index is missing.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    require(bool(shards), "Mistral shard index is empty.")
    names = shards | {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    names |= {p.name for p in model_path.iterdir() if p.suffix in {".json", ".model"}}
    require(all(Path(name).name == name for name in names), "Model filename outside pinned snapshot.")
    files = [model_path / name for name in sorted(names)]
    require(all(path.is_file() and path.stat().st_size > 0 for path in files), "Incomplete local model/tokenizer files.")
    archive = ROOT / "data/wikitext-2-raw-v1.zip"
    require(archive.is_file(), "Local WikiText archive is missing.")
    package = Path(importlib.metadata.distribution("flashinfer-python").locate_file("flashinfer"))
    abi = {"decode.py": package / "decode.py",
           "scheduler.cuh": package / "data/include/flashinfer/attention/scheduler.cuh"}
    require(all(path.is_file() for path in abi.values()), "Installed FlashInfer ABI source files missing.")
    return {"model_path": str(model_path), "model_files": [str(p) for p in files],
            "archive": str(archive), "abi_files": {k: str(v) for k, v in abi.items()},
            "step01": passing_step01(), "gpu_used": False}


def file_evidence(path):
    path = Path(path)
    stat = path.stat()
    return {"sha256": entry.sha256_file(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def freeze_manifest(checked, required, gpu_uuid):
    print("Freezing source, corpus archive and local model file hashes (CPU only)...", flush=True)
    external = {path: file_evidence(path) for path in
                required["model_files"] + [required["archive"]] + list(required["abi_files"].values())}
    value = {"schema_version": 1, "experiment": EXPERIMENT, "created_utc": now(),
             "config": CONFIG, "schedule": schedule(), "source_sha256": checked["source_sha256"],
             "environment": checked["versions"], "inputs": required, "input_file_evidence": external,
             "gpu_uuid": gpu_uuid, "scope": SCOPE,
             "statistics": {"primary": "cache_neutral.wall_ms", "bootstrap_samples": 50000,
                            "bootstrap_seed": 5140, "minimum_primary_point_speedup_exclusive": 1.10,
                            "minimum_primary_lower95_exclusive": 1.10},
             "quality_policy": "Record all HF checks, including failures, separately; never infer new quality confirmation."}
    value["manifest_sha256"] = entry.canonical_hash(value)
    return value


def verify_frozen(manifest, full_inputs=False):
    require(entry.canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            == manifest["manifest_sha256"], "Manifest hash mismatch.")
    for name, digest in manifest["source_sha256"].items():
        require(entry.sha256_file(ROOT / name) == digest, f"Source changed after freeze: {name}")
    for name, evidence in manifest["input_file_evidence"].items():
        stat = Path(name).stat()
        require(stat.st_size == evidence["size"] and stat.st_mtime_ns == evidence["mtime_ns"], f"Input changed: {name}")
        if full_inputs:
            require(entry.sha256_file(Path(name)) == evidence["sha256"], f"Input hash changed: {name}")


def expected_cache(backend):
    layers, batch, pages, page, kv, dim = 32, 4, 1377, 16, 8, 128
    if backend == "flashinfer_fp16":
        shapes = {name: ([layers, batch * pages, page, kv, dim], "torch.float16") for name in ("key", "value")}
        canary = layers * batch * page * kv * dim * 4
    else:
        shapes = {name: ([layers, batch * 180, page, kv, dim], "torch.float16") for name in ("exact_key", "exact_value")}
        shapes.update({name: ([layers, batch * pages, page, kv, dim], "torch.int8") for name in ("key_codes", "value_codes")})
        shapes.update({name: ([layers, batch * pages, kv], "torch.float16") for name in ("key_scales", "value_scales")})
        shapes.update({name: ([layers, batch, kv, dim], "torch.float16") for name in ("key_center", "value_center")})
        shapes["output_center"] = ([layers, batch, 32, dim], "torch.float16")
        canary = layers * batch * (page * kv * dim * 2 + kv * 4)
    tensors = {key: {"shape": shape, "dtype": dtype, "bytes": math.prod(shape) * (1 if dtype == "torch.int8" else 2)}
               for key, (shape, dtype) in shapes.items()}
    total = sum(row["bytes"] for row in tensors.values())
    return {"tensors": tensors, "total_bytes": total, "canary_bytes": canary, "served_bytes": total - canary}


def validate_runtime(gate, backend):
    require(gate["passed"] is True, "Runtime gate failed.")
    observed = gate["observed_dispatch"]
    for key, expected in {"graph_replays": 49152, "eager_calls": 0, "graph_bank_misses": 0}.items():
        require(observed[key] == expected, f"Runtime dispatch: {key}")
    require(gate["observed_nested_attention_dispatch"]["total_calls"] == 0, "Nested attention fallback.")
    operations = gate["observed_operations"]
    for key in ("decoder_plan_calls", "device_position_fills"):
        require(operations[key] == 1536, f"Runtime operation: {key}")
    require(operations["heterogeneous_page_table_updates"] == 0, "Unexpected heterogeneous path.")
    require(operations["exact_page_table_updates"] == (96 if backend == "page_gauge" else 0), "Exact table updates.")
    wrappers = {"baseline": 96} if backend == "flashinfer_fp16" else {"old_int8": 48, "exact_fp16": 96}
    require(set(operations["wrappers"]) == set(wrappers), "Unexpected attention wrappers.")
    for name, rebuilds in wrappers.items():
        counts = operations["wrappers"][name]
        require(counts == {"plan_invocations": 1536, "plan_rebuilds": rebuilds,
                           "last_page_len_device_fills": 1536}, f"Wrapper operation count: {name}")


def validate_worker(payload, block, manifest, return_code):
    require(payload.get("schema_version") == 3 and payload.get("experiment") ==
            "page_gauge_backend_exclusive_sustained_dynamic_graphs", "Incomplete/incorrect worker result.")
    backend = block["backend"]
    require(payload["backend"] == backend, "Worker backend mismatch.")
    expected = worker_config(block, manifest["inputs"]["model_path"])
    expected["wikitext_archive_sha256"] = manifest["input_file_evidence"][manifest["inputs"]["archive"]]["sha256"]
    for name, value in expected.items():
        require(payload["configuration"].get(name) == value, f"Worker configuration mismatch: {name}")
    environment = payload["environment"]
    require("RTX 5090" in environment["gpu"] and environment["compute_capability"] == [12, 0], "Wrong GPU.")
    for worker_key, package_key in (("torch", "torch"), ("flashinfer", "flashinfer-python"), ("transformers", "transformers")):
        require(environment[worker_key] == manifest["environment"][package_key], f"Version drift: {package_key}")
    abi_hashes = {k: manifest["input_file_evidence"][v]["sha256"] for k, v in manifest["inputs"]["abi_files"].items()}
    require(environment["flashinfer_abi"]["source_sha256"] == abi_hashes, "FlashInfer ABI drift.")
    require(bool(payload["source_sha256"]), "Missing worker source hashes.")
    for name, digest in payload["source_sha256"].items():
        require(manifest["source_sha256"].get(name) == digest, f"Unfrozen/changed worker source: {name}")
    same = payload["correctness"]["same_backend_eager_vs_graph"]
    require(same["passed"] is True and same["eager_gate_passed"] is True, "Same-backend correctness failed.")
    for comparison in (same, same["restored_graph_repeat"]):
        require(comparison["passed"] is True, "Restored repeat failed.")
        for name in ("full_mutated_cache_range", "every_page_close_cache_digest", "final_serving_metadata"):
            require(comparison[name]["passed"] is True, f"Recurrence correctness: {name}")
        require(comparison["every_page_close_cache_digest"]["checked_page_closes"] == 96, "Not all 96 page closes checked.")
    recurrence = payload["correctness"]["runtime_page_finalization_and_consumption"]
    for key, value in {"passed": True, "generated_pages": 96, "first_generated_logical_page": 1280,
                       "last_generated_logical_page": 1375, "final_attention_page_table_gate_passed": True}.items():
        require(recurrence[key] == value, f"Recurrence: {key}")
    consumed = list(range(1280, 1328)) if backend == "page_gauge" else []
    require(recurrence["runtime_finalized_pages_consumed_as_int8"] == consumed and
            recurrence["runtime_finalized_int8_pages_consumed_count"] == len(consumed), "INT8 recurrence count/list.")
    if backend == "page_gauge":
        for name in ("logical_token_coverage_exactly_once", "logical_page_sets_disjoint", "prefix_exclusion_gate_passed",
                     "static_suffix_exclusion_gate_passed", "old_attention_page_table_gate_passed", "exact_attention_page_table_gate_passed"):
            require(recurrence[name] is True, f"PageGauge partition: {name}")
        require(payload["attention_implementation"]["custom_module_source_hashes"]["header_sha256"]
                == entry.EXPECTED_HEADER_SHA256, "PageGauge header drift.")
    graph = payload["cuda_graph_provenance"]
    require(graph["enabled"] is True and graph["structure_gate_passed"] is True
            and graph["strict_missing_bucket_failure"] is True and graph["graph_bank_misses"] == 0
            and graph["preflight_position_count"] == 1536 and graph["graphs_per_bank"] == 32
            and graph["total_graphs"] == 32 * graph["graph_bank_count"] and graph["nested_attention_graphs"] is False,
            "Graph structure/no-fallback gate.")
    require(payload["scheduler_capacity"]["all_wrappers_analytically_within_capacity"] is True, "Scheduler capacity.")
    for mode in MODES:
        samples = payload["timing_modes"][mode]["raw_samples"]
        require(len(samples) == 3 and [x["sample_index"] for x in samples] == [0, 1, 2], "Missing/duplicate timing samples.")
        for sample in samples:
            validate_runtime(sample["runtime_gate"], backend)
            require(sample["exact_prefix_canary"]["passed"] is True, "Timed exact-prefix canary.")
            for metric in METRICS:
                value = sample[metric]
                require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid timing value.")
    cache = payload["cache_build"]
    accounting = expected_cache(backend)
    actual = cache["selected_backend_cache"]
    require(set(actual["tensors"]) == set(accounting["tensors"]), "Unexpected cache tensors.")
    for name, fields in accounting["tensors"].items():
        require(all(actual["tensors"][name][k] == v for k, v in fields.items()), f"Cache accounting: {name}")
    require(actual["total_bytes"] == accounting["total_bytes"] and actual["all_storage_pointers_distinct"] is True
            and cache["following_canary_storage_bytes"] == accounting["canary_bytes"]
            and cache["selected_backend_cache_served_bytes_excluding_following_canary"] == accounting["served_bytes"], "Total cache accounting.")
    require(cache["opposite_backend_full_gpu_cache_allocated"] is False
            and payload["exclusivity"]["opposite_backend_full_gpu_cache_allocated"] is False
            and payload["exclusivity"]["hf_dynamic_caches_released_before_decoder_construction"] is True, "Opposite cache residency.")
    hf = payload["correctness"]["backend_vs_hf_sdpa_fp16"]
    require(type(hf["passed"]) is bool and payload["passed"] is hf["passed"], "Inconsistent overall/HF status.")
    require(return_code == (0 if hf["passed"] else 2), "Unexpected worker exit code.")
    return {"execution_passed": True, "hf_quality_passed": hf["passed"], "hf_min_cosine": hf["logits"]["minimum_cosine"],
            "hf_top1_agreement": hf["logits"]["top1_agreement_fraction"], "served_cache_bytes": accounting["served_bytes"]}


def reduce_results(payloads, manifest):
    require(len(payloads) == 8, "Eight fresh-process blocks required; no partial speed claim.")
    pairs = []
    for first in (0, 2, 4, 6):
        indexes = (first, first + 1)
        fi = next(i for i in indexes if manifest["schedule"][i]["backend"] == "flashinfer_fp16")
        pg = next(i for i in indexes if manifest["schedule"][i]["backend"] == "page_gauge")
        for key in MATCHED_FIELDS:
            left, right = (payloads[i]["pairing"]["configuration"][key] for i in (fi, pg))
            require(left == right and bool(left), f"Pair {first // 2} input mismatch: {key}")
        pairs.append((fi, pg, manifest["schedule"][first]))
    endpoints = {}
    for mode in MODES:
        for metric in METRICS:
            rows = []
            for fi, pg, block in pairs:
                means = [statistics.fmean(math.log(x[metric]) for x in payloads[i]["timing_modes"][mode]["raw_samples"])
                         for i in (fi, pg)]
                rows.append({"pair_id": block["pair_id"], "seed": block["seed"], "pair_order": block["pair_order"],
                             "log_speedup": means[0] - means[1], "fi_ms_per_step": math.exp(means[0]) / 1536,
                             "pg_ms_per_step": math.exp(means[1]) / 1536})
            point = math.exp(statistics.fmean(row["log_speedup"] for row in rows))
            interval = _hierarchical_bootstrap_ci(rows, manifest["statistics"]["bootstrap_samples"], manifest["statistics"]["bootstrap_seed"])
            endpoints[f"{mode}.{metric}"] = {"point_speedup": point, **interval, "pairs": rows,
                "point_gt_1_10": point > 1.10, "lower95_gt_1_10": interval["speedup_95_ci"][0] > 1.10}
    primary = endpoints["cache_neutral.wall_ms"]
    fi_bytes, pg_bytes = (expected_cache(b)["served_bytes"] for b in ("flashinfer_fp16", "page_gauge"))
    return {"schema_version": 1, "experiment": EXPERIMENT, "manifest_sha256": manifest["manifest_sha256"],
            "status": "complete", "execution_gates_passed": True,
            "performance_gate_passed": primary["point_gt_1_10"] and primary["lower95_gt_1_10"],
            "primary_endpoint": "cache_neutral.wall_ms", "endpoints": endpoints,
            "fresh_process_blocks": 8, "adjacent_pairs": 4, "fixture_clusters": 2,
            "served_cache": {"flashinfer_fp16_bytes": fi_bytes, "page_gauge_bytes": pg_bytes,
                             "reduction_fraction": 1 - pg_bytes / fi_bytes},
            "hf_quality": [{"block": i, "backend": p["backend"],
                            "passed": p["correctness"]["backend_vs_hf_sdpa_fp16"]["passed"],
                            "min_cosine": p["correctness"]["backend_vs_hf_sdpa_fp16"]["logits"]["minimum_cosine"],
                            "top1_agreement": p["correctness"]["backend_vs_hf_sdpa_fp16"]["logits"]["top1_agreement_fraction"]}
                           for i, p in enumerate(payloads)],
            "memory": [{"block": i, "backend": p["backend"], "after_timing": p["memory"]["after_timing"]}
                       for i, p in enumerate(payloads)],
            "new_heldout_quality_claim": False, "reconstruction_control_implemented": False, "scope": SCOPE}


def process_snapshot(gpu_uuid, child_pid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    # Reuse the same strict CSV contract as the idle check; own process is allowed here.
    import csv
    pids = []
    for row in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
        require(len(row) == 2 and row[1].strip().isdigit(), "Cannot enumerate in-run GPU processes.")
        if row[0].strip() == gpu_uuid:
            pids.append(int(row[1].strip()))
    return {"utc": now(), "pids": pids, "unexpected_pids": [pid for pid in pids if pid != child_pid]}


def run_process(command, output_dir, block, gpu_uuid):
    index = block["index"]
    log_path = output_dir / f"block_{index}.log"
    telemetry_path = output_dir / f"block_{index}_telemetry.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    observations, stop = [], threading.Event()
    started = now()
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        def watch():
            while not stop.is_set():
                try:
                    observations.append(process_snapshot(gpu_uuid, child.pid))
                except Exception as error:
                    observations.append({"utc": now(), "error": str(error)})
                stop.wait(10)
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = child.wait()
        finally:
            # Only this launcher's own child may be terminated on interruption.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            stop.set()
            thread.join(timeout=20)
            child.stdout.close()
            entry.atomic_json(telemetry_path, {"interval_seconds": 10, "observations": observations,
                                             "scope": "Sampled process checks; not continuous proof of device exclusivity."})
    own_pid_seen = any(child.pid in row.get("pids", []) for row in observations)
    exclusive = own_pid_seen and all(not row.get("error") and not row.get("unexpected_pids") for row in observations)
    return {"pid": child.pid, "started_utc": started, "completed_utc": now(), "return_code": return_code,
            "own_pid_seen": own_pid_seen, "sampled_exclusivity_passed": exclusive, "log_sha256": entry.sha256_file(log_path),
            "telemetry_sha256": entry.sha256_file(telemetry_path)}


def launch(checked, gpu_index):
    required = prerequisites()
    snapshots = entry.idle_preflight(gpu_index)
    gpu_uuid = snapshots[-1]["uuid"]
    import fcntl
    with (Path("/tmp") / f"pagegauge-mlsys-{gpu_uuid}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another PageGauge experiment holds this device.") from error
        output_dir = ROOT / "results/mlsys2027_rtx5090/02_fp16_performance" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Step 02 output: {output_dir}", flush=True)
        manifest = freeze_manifest(checked, required, gpu_uuid)
        entry.atomic_json(output_dir / "manifest.json", manifest)
        payloads, processes = [], []
        try:
            for block in manifest["schedule"]:
                verify_frozen(manifest)
                before = entry.idle_preflight(gpu_index)
                require(before[-1]["uuid"] == gpu_uuid, "GPU identity changed.")
                result_path = output_dir / f"block_{block['index']}_{block['backend']}.json"
                command = worker_command(block, required["model_path"], result_path)
                entry.atomic_json(output_dir / f"block_{block['index']}_invocation.json", {"command": command, "before": before})
                print(f"BLOCK {block['index'] + 1}/8: {block['backend']} seed={block['seed']}", flush=True)
                process = run_process(command, output_dir, block, gpu_uuid)
                processes.append(process)
                require(process["sampled_exclusivity_passed"], "GPU process monitoring failed; timing is ineligible.")
                require(len({p["pid"] for p in processes}) == len(processes), "Fresh-process PID repeated.")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                assessment = validate_worker(payload, block, manifest, process["return_code"])
                verify_frozen(manifest)
                process.update({"block": block, "result_sha256": entry.sha256_file(result_path), "assessment": assessment})
                entry.atomic_json(output_dir / f"block_{block['index']}_completion.json", process)
                payloads.append(payload)
                if len(payloads) % 2 == 0:
                    for key in MATCHED_FIELDS:
                        require(payloads[-1]["pairing"]["configuration"][key] == payloads[-2]["pairing"]["configuration"][key],
                                f"Matched-input gate failed: {key}")
                print(f"Block execution PASS; HF quality diagnostic: {assessment['hf_quality_passed']}", flush=True)
            verify_frozen(manifest, full_inputs=True)
            result = reduce_results(payloads, manifest)
            result["processes"] = processes
            result["completed_utc"] = now()
            entry.atomic_json(output_dir / "analysis.json", result)
            primary = result["endpoints"]["cache_neutral.wall_ms"]
            print(f"STEP 02 COMPLETE: speedup={primary['point_speedup']:.4f}x CI={primary['speedup_95_ci']} "
                  f"speed gate={'PASS' if result['performance_gate_passed'] else 'FAIL'}", flush=True)
            print(f"Analysis: {output_dir / 'analysis.json'}", flush=True)
            return 0 if result["performance_gate_passed"] else 3
        except Exception as error:
            entry.atomic_json(output_dir / "failure.json", {"experiment": EXPERIMENT, "status": "failed",
                "manifest_sha256": manifest["manifest_sha256"], "completed_blocks": len(payloads), "error": str(error),
                "processes": processes, "no_partial_speed_claim": True})
            raise RuntimeError(f"Step 02 stopped: {error}. Evidence: {output_dir}") from error
