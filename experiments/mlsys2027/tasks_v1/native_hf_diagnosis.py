"""Native HF partition validation and saved synthetic generation control."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import uuid
from synthetic_generation import ROOT, base, previous, wsl_gpu_monitor, verify


def worker(out):
    import torch
    from transformers import MistralForCausalLM
    from native_hf_generation import generate
    from task_contract import prepare
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    if base.sha256_file(out/'fixtures.json') != m['fixtures_sha256']:
        raise ValueError('Fixture drift')
    case = json.loads((out/'fixtures.json').read_text())['cases'][0]
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model = MistralForCausalLM.from_pretrained(m['model'], torch_dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, attn_implementation='eager').eval()
    contract = prepare(case['task'], case['prompt_ids'], model.config.max_position_embeddings)
    eos = model.config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    result = generate(model, contract, eos)
    short = prepare(case['task'], case['prompt_ids'][:512], model.config.max_position_embeddings)
    manual = generate(model, short, eos)
    ids = torch.tensor([short.prompt_ids], device='cuda', dtype=torch.long)
    with torch.inference_mode():
        native = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=short.max_new_tokens, eos_token_id=eos,
            pad_token_id=eos[0], use_cache=True)[0, len(short.prompt_ids):].tolist()
    with torch.inference_mode():
        full_logits = model(ids, use_cache=False).logits[:, -1].float()
        past = None
        for start in range(0, len(short.prompt_ids), 256):
            output = model(ids[:, start:start+256], past_key_values=past, use_cache=True)
            past = output.past_key_values
        chunk_logits = output.logits[:, -1].float()
    first_difference = next((i for i, (a, b) in enumerate(zip(manual['generated_ids'], native)) if a != b), None)
    base.atomic_json(out/'diagnosis.json', {
        'long_result': result, 'short_manual': manual, 'short_native_ids': native,
        'first_difference': first_difference,
        'full_first_top5': torch.topk(full_logits, 5).indices.tolist(),
        'chunk_first_top5': torch.topk(chunk_logits, 5).indices.tolist(),
        'full_first_values': torch.topk(full_logits, 5).values.tolist(),
        'chunk_first_values': torch.topk(chunk_logits, 5).values.tolist(),
        'first_logits_max_abs': (full_logits-chunk_logits).abs().max().item(),
        'scope': 'Diagnostic preserved even on mismatch; not acceptance or public accuracy.'})
    if manual['generated_ids'] != native:
        raise ValueError('Native generate and partitioned HF differ on synthetic prefix')
    result['short_native_generate_equal'] = True
    verify(m)
    base.atomic_json(out/'analysis.json', {'execution_complete': True, 'case_id': case['case_id'],
        'result': result, 'scope': 'Native HF synthetic control; short partition checked against native generate; no benchmark score.'})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException:
            base.atomic_json(args.worker/'failure.json', {'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    source = ROOT/'results/mlsys2027_tasks_v1/task_fallback_smoke_20260910T081503Z_3aaa0d41'
    m = json.loads((source/'manifest.json').read_text())
    verify(m)
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_tasks_v1'/('native_hf_diagnosis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir()
        fixtures = json.loads((source/'fixtures.json').read_text())
        base.atomic_json(out/'fixtures.json', {'cases': fixtures['cases'][:1]})
        paths = [Path(__file__), Path(__file__).with_name('native_hf_generation.py'),
                 Path(__file__).with_name('task_contract.py'), Path(__file__).with_name('generation_contract.py')]
        paths.extend((ROOT/'experiments/mlsys2027/baselines_v1'/n) for n in ('kivi_adapter.py', 'kivi_quality.py'))
        native = Path('/home/anonymous/pagegauge_baselines/KIVI')
        paths.extend(native/n for n in ('models/mistral_kivi.py', 'quant/new_pack.py', 'quant/matmul.py'))
        for path in paths:
            m['source_sha256'][str(path.resolve())] = base.sha256_file(path)
        m.update(fixtures_sha256=base.sha256_file(out/'fixtures.json'), idle=idle,
                 scope='Native HF synthetic generation control, no public data')
        base.atomic_json(out/'manifest.json', m)
        command = ['/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/bin/python', '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        print(str(out), flush=True)
        completion = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completion)
        verify(m)
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('Qualification failed; inspect retained evidence')


if __name__ == '__main__':
    main()
