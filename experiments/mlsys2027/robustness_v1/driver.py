"""Isolated next-experiment launcher. Never edits the existing 01/02 source closure."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "diagnostics"))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as performance


def sources():
    return [str(path.relative_to(ROOT)) for path in sorted(HERE.glob("*.py"))] + [str((HERE / "plan.json").relative_to(ROOT)),
             str((HERE / "run.sh").relative_to(ROOT)), *performance.EXTRA_SOURCES]


def check():
    result = base.check_environment(sources())
    if not result["passed"]:
        raise RuntimeError(str(result["failures"]))
    plan = json.loads((HERE / "plan.json").read_text(encoding="utf-8"))
    if plan["wikitext_member"] != "wikitext-2-raw/wiki.train.raw" or plan["batch_size"] != 1:
        raise RuntimeError("This development diagnostic is TRAIN-only and B1")
    if importlib.metadata.version("transformers") != plan["transformers_version"]:
        raise RuntimeError("HF capture interface version drift")
    inputs = performance.prerequisites()
    return result, plan, inputs


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if base.canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise RuntimeError("Manifest hash mismatch")
    for name, digest in manifest["source_sha256"].items():
        if base.sha256_file(ROOT / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, evidence in manifest["input_file_evidence"].items():
        if base.sha256_file(Path(name)) != evidence["sha256"]:
            raise RuntimeError(f"Frozen input changed: {name}")
    return manifest


def verify_captures(directory, manifest):
    summary = json.loads((directory / "capture_summary.json").read_text(encoding="utf-8"))
    if summary["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("Capture/manifest mismatch")
    expected = {(layer, generated) for layer in manifest["plan"]["layers"]
                for generated in manifest["plan"]["generated_token_snapshots"]}
    if len(summary["records"]) != len(expected) or {(r["layer"], r["generated_tokens"]) for r in summary["records"]} != expected:
        raise RuntimeError("Missing or duplicate predeclared captures")
    for record in summary["records"]:
        name = record["path"]
        if Path(name).name != name or base.sha256_file(directory / name) != record["sha256"]:
            raise RuntimeError("Capture file identity mismatch")
    return summary


def child(command, directory, gpu_uuid):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    with (directory / (command + ".log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "_" + command, "--run", str(directory)],
                                   cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f"{command} failed; preserve {directory}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "capture", "analyze", "replay", "_capture", "_replay"), nargs="?", default="check")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "check":
        checked, plan, inputs = check()
        print(json.dumps({"passed": True, "gpu_queried": False, "gpu_used": False,
                          "planned_captures": len(plan["layers"]) * len(plan["generated_token_snapshots"]),
                          "model_path": inputs["model_path"], "status": "ready_for_validation_after_current_run"}, indent=2))
        return
    if args.command == "capture":
        checked, plan, inputs = check()
        snapshots = base.idle_preflight(args.gpu_index)
        gpu_uuid = snapshots[-1]["uuid"]
        import fcntl
        with (Path("/tmp") / f"pagegauge-mlsys-{gpu_uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            directory = ROOT / "results/mlsys2027_robustness_v1" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
            directory.mkdir(parents=True)
            print(f"Attribution directory: {directory}", flush=True)
            files = inputs["model_files"] + [inputs["archive"]]
            # Include the HF implementation files whose call interface is intercepted.
            package = Path(importlib.metadata.distribution("transformers").locate_file("transformers"))
            files += [str(package / p) for p in ("models/mistral/modeling_mistral.py", "integrations/sdpa_attention.py", "modeling_utils.py")]
            manifest = {"schema_version": 1, "plan": plan, "model_path": inputs["model_path"], "archive": inputs["archive"],
                        "source_sha256": checked["source_sha256"], "input_file_evidence": {p: performance.file_evidence(p) for p in files},
                        "created_utc": datetime.now(timezone.utc).isoformat(), "idle_check": snapshots}
            manifest["manifest_sha256"] = base.canonical_hash(manifest)
            base.atomic_json(directory / "manifest.json", manifest)
            base.idle_preflight(args.gpu_index)
            child("capture", directory, gpu_uuid)
        return
    if args.run is None:
        parser.error("--run is required")
    directory = args.run.resolve()
    manifest = verify(directory)
    if args.command == "_capture":
        if (directory / "capture_summary.json").exists() or list(directory.glob("*.npz")):
            raise RuntimeError("Refusing to overwrite capture evidence")
        from capture import run
        result = run(manifest, directory)
        for record in result["records"]:
            record["sha256"] = base.sha256_file(directory / record["path"])
        result["manifest_sha256"] = manifest["manifest_sha256"]
        verify(directory)
        base.atomic_json(directory / "capture_summary.json", result)
        return
    summary = verify_captures(directory, manifest)
    if args.command == "replay":
        snapshots = base.idle_preflight(args.gpu_index)
        gpu_uuid = snapshots[-1]["uuid"]
        import fcntl
        with (Path("/tmp") / f"pagegauge-mlsys-{gpu_uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child("replay", directory, gpu_uuid)
        return
    if args.command == "_replay":
        import numpy as np
        import torch
        from replay import run_one
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        if (directory / "replay_summary.json").exists():
            raise RuntimeError("Refusing to overwrite replay")
        records = []
        for record in summary["records"]:
            target = directory / (Path(record["path"]).stem + "_replay.npz")
            if target.exists():
                raise RuntimeError("Partial replay already exists; diagnose, do not overwrite")
            factorized, explicit = run_one(directory / record["path"], manifest["plan"], ROOT)
            np.savez(target, factorized=factorized, explicit=explicit)
            records.append({"capture": record["path"], "path": target.name, "sha256": base.sha256_file(target)})
            print(f"Replayed {record['path']}", flush=True)
        verify(directory)
        base.atomic_json(directory / "replay_summary.json", {"manifest_sha256": manifest["manifest_sha256"], "records": records,
                        "scope": "Untimed static production attention; not append recurrence or model quality"})
        return
    from analyze import analyze_capture
    output = directory / "attribution.json"
    if output.exists():
        raise RuntimeError("Attribution output exists; preserve it")
    replay_records = {}
    if (directory / "replay_summary.json").exists():
        replay_summary = json.loads((directory / "replay_summary.json").read_text())
        if replay_summary["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError("Replay/manifest mismatch")
        expected_captures = {record["path"] for record in summary["records"]}
        if (len(replay_summary["records"]) != len(expected_captures)
                or {row["capture"] for row in replay_summary["records"]} != expected_captures
                or len({row["path"] for row in replay_summary["records"]}) != len(expected_captures)):
            raise RuntimeError("Replay must cover every capture exactly once with distinct outputs")
        for row in replay_summary["records"]:
            if Path(row["path"]).name != row["path"] or base.sha256_file(directory / row["path"]) != row["sha256"]:
                raise RuntimeError("Replay file hash mismatch")
            replay_records[row["capture"]] = directory / row["path"]
    rows = [analyze_capture(directory / record["path"], manifest["plan"], replay_records.get(record["path"])) for record in summary["records"]]
    base.atomic_json(output, {"manifest_sha256": manifest["manifest_sha256"], "rows": rows, "speed_claim": False,
                             "quality_claim": False, "production_replay_available": bool(replay_records),
                             "scope": "Component attribution on fixed HF trajectories; errors are vectors, norms are not additive"})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
