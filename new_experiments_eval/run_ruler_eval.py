#!/usr/bin/env python3
"""MLSys 2027 Long-Context Downstream Benchmark (RULER / NIAH)
Evaluates FlashInfer FP16 baseline vs. PageGauge INT8 (S4/A0/T768) vs. PageGauge INT8 (S4/A128/T768)
on Mistral-7B-Instruct-v0.3 across 8k, 16k, and 32k contexts.
"""
import sys
import os
from pathlib import Path

CUDA_HOME = "/home/anonymous/page_gauge_env_protocol_v2/lib/python3.12/site-packages/nvidia/cu13"
os.environ["CUDA_HOME"] = CUDA_HOME
VENV_BIN = "/home/anonymous/pagegauge_baselines/vllm-7a100bb617471801ee1d5525bfbb8fb238a345ea/.venv/bin"
os.environ["PATH"] = f"{VENV_BIN}:{CUDA_HOME}/bin:" + os.environ.get("PATH", "")

import json
import math
import time
import argparse
from typing import Any
import torch

ROOT = Path("/mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090")
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/mlsys2027"))
sys.path.insert(0, str(ROOT / "experiments/mlsys2027/tasks_v1"))

import benchmark_pg19_external_quality as quality
from install_merge_center_candidate import install as install_merge_center
from packed_hf_views import hf_projection_views
from verify_scoring import score_prediction

TASK_MAX_NEW_TOKENS = {
    "niah_single_1": 50,
    "niah_multikey_2": 60,
    "vt": 40,
    "cwe": 120,
}

def calculate_kv_cache_mb(context_len: int, backend: str, s_pages: int = 4, a_pages: int = 0) -> float:
    # Mistral-7B: 32 layers, 8 KV heads, 128 dim
    layers = 32
    hkv = 8
    dim = 128
    
    if backend == "flashinfer_fp16":
        # FP16: 2 bytes * 2 (K+V) * layers * hkv * dim per token = 65,536 bytes/token
        total_bytes = context_len * layers * hkv * dim * 2 * 2
        return total_bytes / (1024 * 1024)
    else:
        # PageGauge:
        # S exact prefix pages + A exact static suffix pages + 48 exact tail pages
        exact_pages = s_pages + a_pages + 48
        exact_tokens = min(context_len, exact_pages * 16)
        history_tokens = max(0, context_len - exact_tokens)
        
        # Exact region: FP16 (65536 bytes/token)
        exact_bytes = exact_tokens * layers * hkv * dim * 2 * 2
        
        # History region: INT8 codes (1 byte) + FP16 scale per 16-token page
        history_pages = (history_tokens + 15) // 16
        history_bytes = history_pages * (16 * layers * hkv * dim * 2 * 1 + layers * hkv * 2 * 2)
        
        # Centers: FP16 per request (layers * hkv * dim * 2 * 2)
        center_bytes = layers * hkv * dim * 2 * 2
        
        total_bytes = exact_bytes + history_bytes + center_bytes
        return total_bytes / (1024 * 1024)

def run_evaluation(num_samples: int = 16, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path("/mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/new_experiments_eval/long_context_downstream")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pg = quality.PG
    install_merge_center(pg.TransformerDecoder)
    
    model_path = "/home/anonymous/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading tokenizer and Mistral-7B-Instruct-v0.3...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    ).eval().cuda()
    
    layers, hq, hkv, hidden = pg.BASE_E2E.check_model(model)
    pg.BASE_E2E.pack_model_projections(model)
    extension = pg.RUNTIME.load_append_extension()
    import flashinfer
    
    tasks = ["niah_single_1", "niah_multikey_2", "vt", "cwe"]
    contexts = [8192, 16384, 32768]
    systems = [
        ("flashinfer_fp16", "FlashInfer FP16", 0, 0),
        ("page_gauge_s4_a0", "PageGauge INT8 (S4/A0/T768)", 4, 0),
        ("page_gauge_s4_a128", "PageGauge INT8 (S4/A128/T768)", 4, 128),
    ]
    
    ruler_root = Path("/home/anonymous/pcaf_iclr27_e9_pretrained_ruler_v1/ruler_data")
    eos_ids = [tokenizer.eos_token_id]
    
    all_results = []
    
    for length in contexts:
        length_dir = ruler_root / f"L{length}"
        for task in tasks:
            val_file = length_dir / task / "validation.jsonl"
            if not val_file.exists():
                print(f"File not found: {val_file}, skipping")
                continue
                
            print(f"\n========================================================")
            print(f"Running Task: {task} | Context Length: {length} | Samples: {num_samples}")
            print(f"========================================================")
            
            # Load rows
            rows = []
            with open(val_file) as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    rows.append(json.loads(line))
                    
            max_new = TASK_MAX_NEW_TOKENS[task]
            
            for sys_id, sys_name, s_pages, a_pages in systems:
                backend_type = "flashinfer_fp16" if "flashinfer" in sys_id else "page_gauge"
                scores = []
                latencies = []
                
                print(f"\nEvaluating {sys_name} on {task} (L{length})...")
                
                for idx, row in enumerate(rows):
                    prompt_text = row["input"]
                    references = row["outputs"]
                    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    
                    # Align to 16-token page boundary
                    context = (len(prompt_ids) // 16) * 16
                    prompt_slice = prompt_ids[:context]
                    maximum = context + max_new
                    pages = (maximum + 15) // 16
                    initial = context // 16
                    
                    # Pre-calculate RoPE
                    positions = torch.arange(maximum, device="cuda", dtype=torch.long)[None]
                    cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16), positions)
                    if cos.dim() == 3:
                        cos, sin = cos[0], sin[0]
                    cos, sin = cos.half().contiguous(), sin.half().contiguous()
                    
                    # Allocate cache & prefill
                    baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
                    with hf_projection_views(model):
                        _, _, _, _ = quality.prefill_model_cache(
                            model, torch.tensor([prompt_slice]), baseline, pages, context, 0, 1024
                        )
                        
                    if backend_type == "flashinfer_fp16":
                        cache = baseline
                    else:
                        cache = quality.build_gauge_cache_from_baseline(
                            baseline, layers, pages, initial, 48, 1, hkv, s_pages, a_pages
                        )
                        
                    decoder = pg.TransformerDecoder(
                        model, flashinfer, extension, backend_type, cache, maximum,
                        768, 256, 128, cos, sin, "attention_add",
                        tail_attention="flashinfer_merge", batch_size=1,
                        old_value_scale_placement="probability",
                        exact_sink_pages=s_pages if backend_type == "page_gauge" else 0,
                        exact_static_suffix_pages=a_pages if backend_type == "page_gauge" else 0,
                        initial_context_pages=initial if backend_type == "page_gauge" else None,
                    )
                    
                    generated_ids = []
                    token = prompt_slice[-1]
                    step_times = []
                    
                    for step_idx in range(max_new):
                        pos = context + step_idx
                        torch.cuda.synchronize()
                        t_start = time.perf_counter()
                        logits = decoder.step(torch.tensor([token], device="cuda"), pos)[0]
                        torch.cuda.synchronize()
                        step_times.append((time.perf_counter() - t_start) * 1000)
                        
                        token = int(logits.argmax().item())
                        generated_ids.append(token)
                        if token in eos_ids:
                            break
                            
                    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    score = score_prediction(task, pred_text, references)
                    scores.append(score)
                    if step_times:
                        latencies.append(sum(step_times) / len(step_times))
                        
                    del decoder, cache, baseline, cos, sin
                    torch.cuda.empty_cache()
                    
                mean_score = (sum(scores) / len(scores)) * 100.0 if scores else 0.0
                mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
                kv_mem_mb = calculate_kv_cache_mb(length, backend_type, s_pages, a_pages)
                
                res = {
                    "task": task,
                    "context_length": length,
                    "system_id": sys_id,
                    "system_name": sys_name,
                    "num_samples": len(scores),
                    "accuracy": round(mean_score, 2),
                    "mean_latency_ms": round(mean_latency, 2),
                    "kv_cache_mb": round(kv_mem_mb, 1),
                }
                all_results.append(res)
                print(f"--> Result: Accuracy = {mean_score:.2f}%, Latency = {mean_latency:.2f} ms/tok, KV Cache = {kv_mem_mb:.1f} MB")
                
                # Save partial results
                with open(out_dir / "ruler_results_partial.json", "w") as f:
                    json.dump(all_results, f, indent=2)
                    
    # Save final results
    with open(out_dir / "ruler_final_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nBenchmark successfully finished! Results saved to", out_dir / "ruler_final_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples per task/length")
    args = parser.parse_args()
    run_evaluation(num_samples=args.num_samples)
