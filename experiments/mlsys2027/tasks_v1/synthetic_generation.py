"""Synthetic development smoke for autonomous HF/FI/PG answer generation.

No public benchmark TEST data. Not a task-suite score or performance benchmark.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/generalization_v1'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/ablation_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
import wsl_gpu_monitor
from qwen_quality import verify
from generation_contract import aligned_prefix, greedy_rollout


def make_case(tokenizer, budget, depth, seed, family):
    rng = random.Random(seed)
    codes = rng.sample(range(100000, 1000000), 3001)
    expected = str(codes[0])
    records = [f'Record {i}: archive BIRCH-{i:04d} has code {codes[i+1]}.' for i in range(3000)]
    needle = f'Record special: archive ORCHID has code {expected}.'
    def tokenized(count):
        lines = records[:count]
        insertion = int(count*depth)
        lines.insert(insertion, needle)
        text = ('Read this archive log.\n'+ '\n'.join(lines)+
            '\nAccording to this log, what is the six-digit code for archive ORCHID? Reply with only those six digits.')
        kwargs = {'enable_thinking': False} if family == 'qwen3' else {}
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
            tokenize=True, add_generation_prompt=True, **kwargs)
        return ids, text, insertion
    low, high = 1, len(records)
    while low < high:
        middle = (low+high+1)//2
        if len(tokenized(middle)[0]) <= budget: low = middle
        else: high = middle-1
    ids, text, insertion = tokenized(low)
    if not 2896 < len(ids) <= budget or text.count('archive ORCHID has code') != 1:
        raise ValueError('Invalid synthetic prompt budget/needle')
    return {'case_id': f'c{budget}_depth{int(depth*100)}', 'prompt_ids': ids, 'prompt_text': text,
        'expected_code': expected, 'record_count': low, 'needle_record_index': insertion,
        'target_depth_fraction': depth, 'prompt_token_count': len(ids), 'prompt_budget': budget, 'seed': seed}


def score(text, expected):
    matches = re.findall(r'(?<!\d)\d{6}(?!\d)', text)
    return {'exact_stripped_match': text.strip() == expected,
        'first_six_digit_code_match': bool(matches) and matches[0] == expected}


def hf_greedy(model, prompt, eos, maximum_new):
    import torch
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    transition = None
    for begin in range(0, len(prompt), 1024):
        end = min(len(prompt), begin+1024)
        position = torch.arange(begin, end, device='cuda', dtype=torch.long)
        outputs = model.model(input_ids=torch.tensor([prompt[begin:end]], device='cuda'),
            position_ids=position[None], cache_position=position, past_key_values=cache,
            use_cache=True, return_dict=True)
        cache = outputs.past_key_values
        transition = model.lm_head(outputs.last_hidden_state[:, -1]).float()
        del outputs
    generated = []
    for index in range(maximum_new):
        token = int(transition.argmax(-1).item()); generated.append(token)
        if token in eos: break
        if index+1 == maximum_new: break
        position = torch.tensor([len(prompt)+index], device='cuda', dtype=torch.long)
        outputs = model.model(input_ids=torch.tensor([[token]], device='cuda'),
            position_ids=position[None], cache_position=position, past_key_values=cache,
            use_cache=True, return_dict=True)
        cache = outputs.past_key_values
        transition = model.lm_head(outputs.last_hidden_state[:, -1]).float()
        del outputs
    del cache, transition
    gc.collect(); torch.cuda.empty_cache()
    return {'generated_ids': generated, 'stop_reason': 'eos' if generated[-1] in eos else 'max_new_tokens'}


def worker(out):
    import torch
    import flashinfer
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import benchmark_pg19_external_quality as quality
    from regional_quality import install_probe
    m = json.loads((out/'manifest.json').read_text()); verify(m)
    if base.sha256_file(out/'fixtures.json') != m['fixtures_sha256']: raise ValueError('Changed synthetic inputs')
    cases = json.loads((out/'fixtures.json').read_text())['cases']
    pg = quality.PG
    if m['model_family'] == 'qwen3':
        from qwen_adapter import install
        install(pg)
    torch.set_grad_enabled(False); torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    tokenizer = AutoTokenizer.from_pretrained(m['model'], local_files_only=True, trust_remote_code=False)
    def load_model():
        return AutoModelForCausalLM.from_pretrained(m['model'], dtype=torch.float16,
            local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=True, attn_implementation='sdpa').eval().cuda()
    model = load_model()
    layers, hq, hkv, hidden = pg.BASE_E2E.check_model(model)
    expected_layers = 36 if m['model_family'] == 'qwen3' else 32
    if (layers, hq, hkv, hidden) != (expected_layers, 32, 8, 4096): raise ValueError('Wrong model contract')
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    if not eos: raise ValueError('Explicit native EOS IDs required')
    if model.generation_config.repetition_penalty != 1.0 or model.generation_config.no_repeat_ngram_size != 0:
        raise ValueError('Native generation has logits processing absent from greedy adapter')
    extension = pg.RUNTIME.load_append_extension()
    results = []
    for index, case in enumerate(cases):
        # Production packing removes the separate HF projection modules.
        # Reload the pinned checkpoint per case rather than accidentally using
        # a packed custom model as the next case's native HF reference.
        if index:
            model = load_model()
            if pg.BASE_E2E.check_model(model) != (layers, hq, hkv, hidden):
                raise ValueError('Reloaded model shape changed')
        print('Synthetic generation '+case['case_id'], flush=True)
        prompt = case['prompt_ids']; context = aligned_prefix(len(prompt)); maximum = len(prompt)+m['max_new_tokens']
        if maximum > model.config.max_position_embeddings: raise ValueError('Model context exceeded')
        result = {'case_id': case['case_id'], 'aligned_prefix_tokens': context,
            'prompt_remainder_tokens': len(prompt)-context, 'arms': {}}
        hf = hf_greedy(model, prompt, eos, m['max_new_tokens'])
        if index == 0:
            official = model.generate(torch.tensor([prompt], device='cuda'), do_sample=False,
                num_beams=1, max_new_tokens=m['max_new_tokens'], eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id or eos[0], use_cache=True, logits_to_keep=1)
            generated = official[0, len(prompt):].tolist()
            base.atomic_json(out/'hf_generate_validation.json', {'manual_ids': hf['generated_ids'],
                'official_generate_ids': generated, 'passed': generated == hf['generated_ids'],
                'scope': 'First synthetic case only; native full-prompt generate versus chunk-prefilled manual greedy loop'})
            if generated != hf['generated_ids']: raise ValueError('HF generation adapter differs from native generate')
            del official; gc.collect(); torch.cuda.empty_cache()
        result['arms']['hf'] = hf
        pages, initial = math.ceil(maximum/16), context//16
        baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
        _, _, prefill, fingerprint = quality.prefill_model_cache(model, torch.tensor([prompt]), baseline,
            pages, context, 0, 1024)
        if not all(r['sampled_boundary_copy_bitwise_identical'] for r in prefill): raise ValueError('Prefill copy mismatch')
        result['initial_kv_sample_sha256'] = fingerprint
        pg.BASE_E2E.pack_model_projections(model)
        positions = torch.arange(maximum, device='cuda', dtype=torch.long)[None]
        cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device='cuda', dtype=torch.float16), positions)
        if cos.dim() == 3: cos, sin = cos[0], sin[0]
        cos, sin = cos.half().contiguous(), sin.half().contiguous()
        for backend in ('flashinfer_fp16', 'page_gauge'):
            cache = baseline if backend == 'flashinfer_fp16' else quality.build_gauge_cache_from_baseline(
                baseline, layers, pages, initial, 48, 1, hkv, 4, 128)
            decoder = pg.TransformerDecoder(model, flashinfer, extension, backend, cache, maximum,
                768, 256, 128, cos, sin, 'attention_add', tail_attention='flashinfer_merge', batch_size=1,
                old_value_scale_placement='probability', exact_sink_pages=4 if backend == 'page_gauge' else 0,
                exact_static_suffix_pages=128 if backend == 'page_gauge' else 0,
                initial_context_pages=initial if backend == 'page_gauge' else None)
            probes = []
            if backend == 'page_gauge':
                original = decoder.exact_wrapper.plan
                def exact(indices, length, last, split, page_table_epoch=None):
                    return original(indices, length, last, 32, page_table_epoch)
                decoder.exact_wrapper.plan = exact
                # First recurrent real-prompt token is always visited, even on
                # immediate EOS. Other selected positions may not be reached.
                install_probe(decoder, context, maximum-context, probes)
            def step(token, position):
                logits = decoder.step(torch.tensor([token], device='cuda'), position)[0]
                if not logits.isfinite().all(): raise ValueError('Nonfinite generated logits')
                return int(logits.argmax().item())
            own = greedy_rollout(prompt, context, m['max_new_tokens'], eos, step)
            own['selected_execution_probes'] = probes
            if backend == 'page_gauge' and len(probes) < 6: raise ValueError('Missing first-step execution probes')
            result['arms'][backend] = own
            del decoder, cache, step
            if backend == 'page_gauge': del original, exact
            gc.collect(); torch.cuda.empty_cache()
        for name, arm in result['arms'].items():
            arm['generated_text'] = tokenizer.decode(arm['generated_ids'], skip_special_tokens=True)
            arm['scores'] = score(arm['generated_text'], case['expected_code'])
            arm['identical_token_sequence_to_hf'] = arm['generated_ids'] == hf['generated_ids']
        results.append(result); base.atomic_json(out/f'case_{index}.json', result)
        print(json.dumps({name: arm['scores'] for name, arm in result['arms'].items()}), flush=True)
        del baseline, cos, sin, model; gc.collect(); torch.cuda.empty_cache()
    verify(m)
    base.atomic_json(out/'analysis.json', {'cases': len(results), 'results': results,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'flashinfer': flashinfer.__version__,
        'scope': m['scope'], 'execution_complete': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--model-family', choices=('qwen3',), default='qwen3')
    args = parser.parse_args()
    if args.worker:
        try: worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    from transformers import AutoTokenizer
    import fcntl
    source = ROOT/'results/mlsys2027_generalization_v1/book_qwen3_20260908T174053Z_6d5fa41d/manifest.json'
    m = json.loads(source.read_text()); verify(m)
    tokenizer = AutoTokenizer.from_pretrained(m['model'], local_files_only=True, trust_remote_code=False)
    cases = [make_case(tokenizer, budget, depth, 2026091000+i*3+j, args.model_family)
        for i, budget in enumerate((8192, 20480)) for j, depth in enumerate((.1, .5, .9))]
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_tasks_v1'/('synthetic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True); base.atomic_json(out/'fixtures.json', {'cases': cases})
        for path in (Path(__file__).resolve(), Path(__file__).with_name('generation_contract.py'),
                     Path(wsl_gpu_monitor.__file__).resolve(),
                     ROOT/'experiments/mlsys2027/ablation_v1/regional_quality.py'):
            m['source_sha256'][str(path)] = base.sha256_file(path)
        # Original book source is provenance for the model, not an input to the
        # synthetic answers. Retain its immutable input evidence without reading TEST.
        for key in ('token_provenance', 'tokens_sha256', 'book_index', 'context', 'decode_steps', 'order'):
            m.pop(key, None)
        m.update(fixtures_sha256=base.sha256_file(out/'fixtures.json'), seed=2026091000, max_new_tokens=32,
            process_monitor=wsl_gpu_monitor.CONTRACT,
            prompt_budgets=[8192, 20480], data_split='synthetic_development', order=['hf', 'flashinfer_fp16', 'page_gauge'],
            scope='Six synthetic development retrieval cases, two prompt budgets and three needle depths. Native nonthinking Qwen chat; independent greedy HF/FI/PG answers, first-case native-generate validation and selected PG same-cache execution probes. Fixed S4/A128/T768, exact32/history128. Not public benchmark TEST, task-suite quality, speed or generated-history aging evidence.')
        base.atomic_json(out/'manifest.json', m)
        print('Synthetic generation suite: '+str(out), flush=True)
        command = [sys.executable, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', c); verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed']: raise ValueError('Synthetic generation failure retained')


if __name__ == '__main__': main()
