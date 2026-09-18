"""One exposed synthetic prompt: qualify zero-copy HF views before task use."""
from datetime import datetime, timezone
import argparse
import gc
import json
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
sys.path.insert(0,str(ROOT/'experiments/mlsys2027'))
sys.path.insert(0,str(ROOT/'experiments/mlsys2027/generalization_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
import wsl_gpu_monitor
from qwen_quality import verify


def worker(out):
    import torch
    from transformers import AutoModelForCausalLM
    import benchmark_pg19_external_quality as quality
    from qwen_adapter import install
    from synthetic_generation import hf_greedy
    from packed_hf_views import hf_projection_views
    m = json.loads((out/'manifest.json').read_text()); verify(m)
    fixture = Path(m['fixture'])
    if base.sha256_file(fixture) != m['fixture_sha256']: raise ValueError('Changed exposed synthetic prompt')
    prompt = json.loads(fixture.read_text())['cases'][0]['prompt_ids']
    torch.set_grad_enabled(False); torch.manual_seed(2026090913)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(m['model'],dtype=torch.float16,
        local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').eval().cuda()
    install(quality.PG)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos,int) else list(eos)
    ids = torch.tensor([prompt],device='cuda')
    def last_logits():
        result = model(ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
        value = result.logits.detach().cpu(); del result
        return value
    original_logits = last_logits()
    original_generation = hf_greedy(model,prompt,eos,32)
    quality.PG.BASE_E2E.pack_model_projections(model)
    gc.collect(); torch.cuda.empty_cache()
    def storages():
        return {p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}
    packed_storage = storages()
    with hf_projection_views(model):
        during_storage = storages()
        if packed_storage != during_storage: raise ValueError('HF views duplicate packed weight storage')
        viewed_logits = last_logits()
        viewed_generation = hf_greedy(model,prompt,eos,32)
    if packed_storage != storages(): raise ValueError('Projection views were not restored')
    if not torch.equal(original_logits,viewed_logits): raise ValueError('Full-vocabulary HF logits changed')
    if original_generation != viewed_generation: raise ValueError('Own greedy generation changed')
    if any(layer.self_attn.q_proj is not None or layer.mlp.gate_proj is not None for layer in model.model.layers):
        raise ValueError('HF modules remain in the decoder configuration')
    verify(m)
    base.atomic_json(out/'analysis.json',{'passed':True,'prompt_tokens':len(prompt),
        'full_vocabulary_last_logits_bitwise_equal':True,'greedy_rollout_equal':True,
        'unique_parameter_storage_bytes':sum(packed_storage.values()),'extra_projection_weight_bytes':0,
        'original_generation':original_generation,'viewed_generation':viewed_generation,
        'scope': 'One already-exposed synthetic Qwen prompt, native HF before/after packed zero-copy projection views. Preparation optimization only; no new task score, timing claim, quantizer or decoder-kernel change.'})
    print('Packed HF zero-copy views PASS: '+str(out),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--worker',type=Path); args = p.parse_args()
    if args.worker:
        try: worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
            raise
        return
    import fcntl
    source = ROOT/'results/mlsys2027_tasks_v1/synthetic_20260909T025247Z_a44bf775'
    m = json.loads((source/'manifest.json').read_text()); verify(m)
    for name in ('validate_packed_hf_views.py','packed_hf_views.py'):
        path = Path(__file__).with_name(name).resolve(); m['source_sha256'][str(path)] = base.sha256_file(path)
    fixture = source/'fixtures.json'
    m.update(fixture=str(fixture),fixture_sha256=base.sha256_file(fixture),scope=__doc__)
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_tasks_v1'/('packed_hf_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True); base.atomic_json(out/'manifest.json',m)
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(out)]
        base.atomic_json(out/'invocation.json',{'command':command})
        print('Packed HF view validation: '+str(out),flush=True)
        c = wsl_gpu_monitor.run_process(previous,command,out,{'index':0},gpu)
        base.atomic_json(out/'completion.json',c); verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed']: raise ValueError('View qualification failed, evidence retained')


if __name__ == '__main__': main()
