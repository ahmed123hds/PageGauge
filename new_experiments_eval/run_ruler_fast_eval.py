#!/usr/bin/env python3
"""Optimized Long-Context Downstream Benchmark (RULER / NIAH)
- Prefills once per prompt, sharing the base KV cache across backends
- Eliminates per-token host synchronizations
- Evaluates FlashInfer FP16 vs PageGauge INT8 (S4/A0/T768) vs PageGauge INT8 (S4/A128/T768)
  across 8k, 16k, and 32k contexts.
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
    "niah_single_1": 45,
    "niah_multikey_2": 50,
    "vt": 35,
    "cwe": 100,
}

def calculate_kv_cache_mb(context_len: int, backend: str, s_pages: int = 4, a_pages: int = 0) -> float:
    layers = 32
    hkv = 8
    dim = 128
    
    if backend == "flashinfer_fp16":
        total_bytes = context_len * layers * hkv * dim * 2 * 2
        return total_bytes / (1024 * 1024)
    else:
        exact_pages = s_pages + a_pages + 48
        exact_tokens = min(context_len, exact_pages * 16)
        history_tokens = max(0, context_len - exact_tokens)
        
        exact_bytes = exact_tokens * layers * hkv * dim * 2 * 2
        history_pages = (history_tokens + 15) // 16
        history_bytes = history_pages * (16 * layers * hkv * dim * 2 * 1 + layers * hkv * 2 * 2)
        center_bytes = layers * hkv * dim * 2 * 2
        
        total_bytes = exact_bytes + history_bytes + center_bytes
        return total_bytes / (1024 * 1024)

def run_evaluation(num_samples: int = 8, out_dir: Path = None):
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
    
    final_results_file = out_dir / "ruler_comprehensive_results.json"
    all_results = []
    if final_results_file.exists():
        try:
            all_results = json.load(open(final_results_file))
        except:
            all_results = []
            
    completed_keys = set((r["task"], r["context_length"], r["system_id"]) for r in all_results)
    
    for length in contexts:
        length_dir = ruler_root / f"L{length}"
        for task in tasks:
            # Check if already completed for all systems
            if all((task, length, sys_id) in completed_keys for sys_id, _, _, _ in systems):
                print(f"Skipping {task} at L{length} (already completed).")
                continue
                
            val_file = length_dir / task / "validation.jsonl"
            if not val_file.exists():
                print(f"File not found: {val_file}, skipping")
                continue
                
            print(f"\n========================================================")
            print(f"Running Task: {task} | Context Length: {length} | Samples: {num_samples}")
            print(f"========================================================")
            
            rows = []
            with open(val_file) as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    rows.append(json.loads(line))
                    
            max_new = TASK_MAX_NEW_TOKENS[task]
            
            # Dictionary to collect results across prompts for each system
            task_scores = {sys_id: [] for sys_id, _, _, _ in systems}
            task_latencies = {sys_id: [] for sys_id, _, _, _ in systems}
            
            for idx, row in enumerate(rows):
                prompt_text = row["input"]
                references = row["outputs"]
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                
                context = (len(prompt_ids) // 16) * 16
                prompt_slice = prompt_ids[:context]
                maximum = context + max_new
                pages = (maximum + 15) // 16
                initial = context // 16
                
                positions = torch.arange(maximum, device="cuda", dtype=torch.long)[None]
                cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16), positions)
                if cos.dim() == 3:
                    cos, sin = cos[0], sin[0]
                cos, sin = cos.half().contiguous(), sin.half().contiguous()
                
                # ONE-TIME Prefill for all systems on this prompt
                baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
                with hf_projection_views(model):
                    _, _, _, _ = quality.prefill_model_cache(
                        model, torch.tensor([prompt_slice]), baseline, pages, context, 0, 1024
                    )
                
                for sys_id, sys_name, s_pages, a_pages in systems:
                    backend_type = "flashinfer_fp16" if "flashinfer" in sys_id else "page_gauge"
                    
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
                    
                    torch.cuda.synchronize()
                    t_gen_start = time.perf_counter()
                    
                    for step_idx in range(max_new):
                        pos = context + step_idx
                        logits = decoder.step(torch.tensor([token], device="cuda"), pos)[0]
                        token = int(logits.argmax().item())
                        generated_ids.append(token)
                        if token in eos_ids:
                            break
                            
                    torch.cuda.synchronize()
                    t_gen_total = time.perf_counter() - t_gen_start
                    
                    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    score = score_prediction(task, pred_text, references)
                    task_scores[sys_id].append(score)
                    if generated_ids:
                        token_lat_ms = (t_gen_total / len(generated_ids)) * 1000
                        task_latencies[sys_id].append(token_lat_ms)
                        
                    del decoder
                    if backend_type != "flashinfer_fp16":
                        del cache
                    torch.cuda.empty_cache()
                    
                del baseline, cos, sin
                torch.cuda.empty_cache()
                print(f"Prompt {idx+1}/{len(rows)} evaluated on all 3 systems.", flush=True)
                
            for sys_id, sys_name, s_pages, a_pages in systems:
                backend_type = "flashinfer_fp16" if "flashinfer" in sys_id else "page_gauge"
                scores = task_scores[sys_id]
                lats = task_latencies[sys_id]
                
                mean_score = (sum(scores) / len(scores)) * 100.0 if scores else 0.0
                mean_latency = (sum(lats) / len(lats)) if lats else 0.0
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
                completed_keys.add((task, length, sys_id))
                print(f"--> {sys_name} on {task} (L{length}): Acc = {mean_score:.2f}%, Latency = {mean_latency:.2f} ms/tok, KV Cache = {kv_mem_mb:.1f} MB")
                
            with open(final_results_file, "w") as f:
                json.dump(all_results, f, indent=2)
                
    print("\nFast downstream benchmark complete! All results saved to", final_results_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    run_evaluation(num_samples=args.num_samples)
