#!/usr/bin/env python3
"""Run a single comparative speed test: FlashInfer FP16 vs PageGauge INT8 with CUDA Graphs
Batch size 4, Context 20,480, 128 decode steps (512 output tokens) on NVIDIA GeForce RTX 5090.
"""
import sys
import os
import json
import subprocess
from pathlib import Path

CUDA_HOME = "/home/anonymous/page_gauge_env_protocol_v2/lib/python3.12/site-packages/nvidia/cu13"
os.environ["CUDA_HOME"] = CUDA_HOME
VENV_BIN = "/home/anonymous/pagegauge_baselines/vllm-7a100bb617471801ee1d5525bfbb8fb238a345ea/.venv/bin"
os.environ["PATH"] = f"{VENV_BIN}:{CUDA_HOME}/bin:" + os.environ.get("PATH", "")
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
os.environ["MAX_JOBS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"

ROOT = Path("/mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090")
OUT_DIR = ROOT / "new_experiments_eval/speed_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = "/home/anonymous/page_gauge_env_protocol_v2/bin/python"
SCRIPT = str(ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py")

def run_backend(backend: str):
    prefix = "4" if backend == "page_gauge" else "0"
    suffix = "128" if backend == "page_gauge" else "0"
    split = "128" if backend == "page_gauge" else "256"
    out_file = OUT_DIR / f"{backend}_b4_20k.json"
    
    cmd = [
        PYTHON, SCRIPT,
        "--backend", backend,
        "--model", "mistralai/Mistral-7B-v0.3",
        "--batch-size", "4",
        "--context", "20480",
        "--decode-steps", "512",
        "--exact-tail", "768",
        "--exact-sink-pages", prefix,
        "--exact-static-suffix-pages", suffix,
        "--prefill-chunk-tokens", "1024",
        "--baseline-split-pages", "256",
        "--candidate-split-pages", split,
        "--tail-attention", "flashinfer_merge",
        "--old-value-scale-placement", "probability",
        "--trajectory-mode", "frozen_hf_teacher_forced",
        "--seed", "20260861",
        "--token-source", "wikitext2",
        "--wikitext-zip", str(ROOT / "data/wikitext-2-raw-v1.zip"),
        "--wikitext-member", "wikitext-2-raw/wiki.test.raw",
        "--token-offset", "0",
        "--token-stride", "23600",
        "--capture-warmups", "1",
        "--maximum-graph-banks", "16",
        "--warmups", "1",
        "--repeats", "1",
        "--cache-scrub-mib", "256",
        "--min-logits-cosine", "0.995",
        "--min-top1-agreement", "0.99",
        "--quality-diagnostics-top-k", "0",
        "--output", str(out_file)
    ]
    
    print(f"\n========================================================")
    print(f"Executing {backend} with CUDA Graphs (B=4, C=20,480)...")
    print(f"========================================================")
    
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError(f"Command failed with return code {res.returncode} and no output file")
        
    data = json.load(open(out_file))
    timing = data["timing_modes"]["cache_hot"]["cuda"]
    return {
        "backend": backend,
        "step_ms": timing["mean_ms_per_decode_step"],
        "token_ms": timing["mean_ms_per_output_token"],
        "tok_per_sec": timing["output_tokens_per_second"],
        "memory_gb": data.get("memory", {}).get("resident_total_bytes", 0) / (1024**3)
    }

def parse_result(out_file: Path, backend: str):
    data = json.load(open(out_file))
    timing = data["timing_modes"]["cache_hot"]["cuda"]
    return {
        "backend": backend,
        "step_ms": timing["mean_ms_per_decode_step"],
        "token_ms": timing["mean_ms_per_output_token"],
        "tok_per_sec": timing["output_tokens_per_second"],
        "memory_gb": data.get("memory", {}).get("resident_total_bytes", 0) / (1024**3)
    }

def main():
    print("Comparative Speed Test on NVIDIA GeForce RTX 5090")
    print("Setup: Batch Size = 4, Context = 20,480 tokens, 512 Decode Steps (2,048 output tokens)")
    
    fi_file = OUT_DIR / "flashinfer_fp16_b4_20k.json"
    pg_file = OUT_DIR / "page_gauge_b4_20k.json"
    
    if fi_file.exists() and fi_file.stat().st_size > 0:
        print("Loading completed FlashInfer FP16 result...")
        fi_res = parse_result(fi_file, "flashinfer_fp16")
    else:
        fi_res = run_backend("flashinfer_fp16")
        
    if pg_file.exists() and pg_file.stat().st_size > 0:
        print("Loading completed PageGauge INT8 result...")
        pg_res = parse_result(pg_file, "page_gauge")
    else:
        pg_res = run_backend("page_gauge")
    
    speedup = fi_res["step_ms"] / pg_res["step_ms"]
    
    print("\n" + "="*70)
    print("FINAL COMPARATIVE SPEED RESULTS (NVIDIA RTX 5090, B=4, Context=20,480):")
    print("="*70)
    print(f"FlashInfer FP16 Baseline:")
    print(f"  - Decode Step Latency: {fi_res['step_ms']:.3f} ms / step")
    print(f"  - Per-Token Latency:   {fi_res['token_ms']:.3f} ms / token")
    print(f"  - Token Throughput:    {fi_res['tok_per_sec']:.1f} tokens/s")
    print(f"PageGauge INT8 (S4/A128/T768):")
    print(f"  - Decode Step Latency: {pg_res['step_ms']:.3f} ms / step")
    print(f"  - Per-Token Latency:   {pg_res['token_ms']:.3f} ms / token")
    print(f"  - Token Throughput:    {pg_res['tok_per_sec']:.1f} tokens/s")
    print(f"SPEEDUP:")
    print(f"  --> PageGauge is {speedup:.3f}x faster ({((speedup - 1.0)*100):.1f}% speedup)!")
    print("="*70)

if __name__ == "__main__":
    main()
