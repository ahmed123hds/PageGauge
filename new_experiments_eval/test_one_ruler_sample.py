import sys
import os
from pathlib import Path

CUDA_HOME = "/home/anonymous/page_gauge_env_protocol_v2/lib/python3.12/site-packages/nvidia/cu13"
os.environ["CUDA_HOME"] = CUDA_HOME
VENV_BIN = "/home/anonymous/pagegauge_baselines/vllm-7a100bb617471801ee1d5525bfbb8fb238a345ea/.venv/bin"
os.environ["PATH"] = f"{VENV_BIN}:{CUDA_HOME}/bin:" + os.environ.get("PATH", "")

import json
import time
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

def test_single():
    print("Testing 1 sample of RULER niah_single_1 at L8192...")
    pg = quality.PG
    install_merge_center(pg.TransformerDecoder)
    
    # Load 1 row from L8192 niah_single_1
    data_path = Path("/home/anonymous/pcaf_iclr27_e9_pretrained_ruler_v1/ruler_data/L8192/niah_single_1/validation.jsonl")
    with open(data_path) as f:
        row = json.loads(f.readline())
        
    prompt_text = row["input"]
    expected_outputs = row["outputs"]
    print(f"Row length: {row['length']}, Expected output: {expected_outputs}")
    
    model_path = "/home/anonymous/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Tokenized prompt length: {len(prompt_ids)}")
    
    # Load model
    print("Loading Mistral-7B-Instruct-v0.3...")
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
    
    # Context setup
    max_new_tokens = 50
    context = (len(prompt_ids) // 16) * 16
    prompt_slice = prompt_ids[:context]
    maximum = context + max_new_tokens
    pages = (maximum + 15) // 16
    initial = context // 16
    
    print(f"Context: {context} tokens ({initial} pages), Maximum: {maximum} tokens ({pages} pages)")
    
    # Allocate baseline cache
    baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
    with hf_projection_views(model):
        _, _, prefill, _ = quality.prefill_model_cache(
            model, torch.tensor([prompt_slice]), baseline, pages, context, 0, 1024
        )
    print("Prefill complete.")
    
    positions = torch.arange(maximum, device="cuda", dtype=torch.long)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device="cuda", dtype=torch.float16), positions)
    if cos.dim() == 3:
        cos, sin = cos[0], sin[0]
    cos, sin = cos.half().contiguous(), sin.half().contiguous()
    
    eos_ids = [tokenizer.eos_token_id]
    
    for backend_name, (backend_type, s_pages, a_pages) in [
        ("FlashInfer FP16", ("flashinfer_fp16", 0, 0)),
        ("PageGauge S4/A0/T768", ("page_gauge", 4, 0)),
        ("PageGauge S4/A128/T768", ("page_gauge", 4, 128)),
    ]:
        print(f"\n--- Testing {backend_name} ---")
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
        latencies = []
        
        # Warmup step
        decoder.reset_runtime_operation_counts()
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for step_idx in range(max_new_tokens):
            pos = context + step_idx
            t_step_start = time.perf_counter()
            logits = decoder.step(torch.tensor([token], device="cuda"), pos)[0]
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t_step_start) * 1000)
            
            token = int(logits.argmax().item())
            generated_ids.append(token)
            if token in eos_ids:
                break
                
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) * 1000
        mean_latency = sum(latencies) / len(latencies)
        
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        score = score_prediction("niah_single_1", output_text, expected_outputs)
        print(f"Generated ({len(generated_ids)} tokens): {output_text.strip()[:150]}")
        print(f"Score: {score}, Mean token decode latency: {mean_latency:.2f} ms, Total: {total_time:.1f} ms")

if __name__ == "__main__":
    test_single()
