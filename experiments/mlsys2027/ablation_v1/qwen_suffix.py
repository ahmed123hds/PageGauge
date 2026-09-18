"""Qwen PG19 development transfer: fixed A128 versus A0, all eight books."""
from datetime import datetime, timezone
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/generalization_v1'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from qwen_quality import verify
from regional_quality import install_probe
from reduce_frontier import summarize
import book_tokens

BOOKS = ROOT/'results/mlsys2027_generalization_v1/book_suite_qwen3_20260908T183459Z_18348a6c/analysis.json'
POLICIES = [('reference_policy', 128), ('without_static_suffix', 0)]


def worker(out):
    import torch
    import flashinfer
    import transformers
    from transformers import AutoModelForCausalLM
    import benchmark_pg19_external_quality as quality
    from qwen_adapter import install
    m = json.loads((out/'manifest.json').read_text()); verify(m)
    if base.sha256_file(out/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed tokens')
    pg = quality.PG; install(pg)
    torch.set_grad_enabled(False); torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], dtype=torch.long)
    if tokens.shape != (1, 22017) or m['model_family'] != 'qwen3':
        raise ValueError('Wrong Qwen full recurrent fixture')
    context, steps, maximum, initial, pages = 20480, 1536, 22016, 1280, 1376
    print('Qwen suffix: loading model', flush=True)
    model = AutoModelForCausalLM.from_pretrained(m['model'], dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').eval().cuda()
    layers, hq, hkv, hidden = pg.BASE_E2E.check_model(model)
    if (layers, hq, hkv, hidden) != (36, 32, 8, 4096) or model.config.model_type != 'qwen3':
        raise ValueError('Qwen model contract changed')
    baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
    hf, _, prefill, fingerprint = quality.prefill_model_cache(model, tokens, baseline, pages, context, steps, 1024)
    if fingerprint != m['reference_initial_kv_sample_sha256'] or not all(r['sampled_boundary_copy_bitwise_identical'] for r in prefill):
        raise ValueError('Retained coherent HF prefill mismatch')
    pg.BASE_E2E.pack_model_projections(model)
    positions = torch.arange(maximum, device='cuda', dtype=torch.long)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device='cuda', dtype=torch.float16), positions)
    if cos.dim() == 3: cos, sin = cos[0], sin[0]
    cos, sin = cos.half().contiguous(), sin.half().contiguous()
    extension = pg.RUNTIME.load_append_extension()
    decode = tokens[:, context:maximum].T.contiguous().cuda()
    labels = tokens[:, context+1:maximum+1].T.contiguous()
    windows = book_tokens.request_windows(m['token_provenance'], context, steps, quality)
    results, probes = {}, {}
    try:
        for name, suffix in POLICIES:
            print('Qwen suffix: '+name, flush=True)
            cache = quality.build_gauge_cache_from_baseline(baseline, layers, pages, initial, 48, 1, hkv, 4, suffix)
            decoder = pg.TransformerDecoder(model, flashinfer, extension, 'page_gauge', cache, maximum,
                768, 256, 128, cos, sin, 'attention_add', tail_attention='flashinfer_merge', batch_size=1,
                old_value_scale_placement='probability', exact_sink_pages=4,
                exact_static_suffix_pages=suffix, initial_context_pages=initial)
            original = decoder.exact_wrapper.plan
            def exact(indices, length, last, split, page_table_epoch=None):
                return original(indices, length, last, 32, page_table_epoch)
            decoder.exact_wrapper.plan = exact
            probes[name] = []; install_probe(decoder, context, steps, probes[name])
            logits, consumed = [], set()
            for offset, token in enumerate(decode):
                row = decoder.step(token, context+offset)[0].detach().float().cpu()
                if not row.isfinite().all(): raise ValueError('Nonfinite Qwen logits')
                logits.append(row); consumed.update(p for p in decoder.old_logical_pages if p >= initial)
            if consumed != set(range(1280, 1328)) or len(probes[name]) != 18:
                raise ValueError('Incomplete Qwen recurrence/probes')
            comparison = quality.compare_logits(hf, logits, labels, reference_name='Qwen3 HF SDPA FP16',
                candidate_name=name, predicted_position_start=context+1, request_windows=windows)
            stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in vars(cache).values() if isinstance(t, torch.Tensor)}
            comparison.update(cache_storage_bytes=sum(stores.values()), consumed_new_history_pages=sorted(consumed),
                policy={'S': 4, 'A': suffix, 'T': 768})
            base.atomic_json(out/(name+'.json'), comparison)
            results[name] = {'sha256': base.sha256_file(out/(name+'.json')), 'cache_bytes': sum(stores.values())}
            del decoder, cache, logits, comparison, original, exact
            gc.collect(); torch.cuda.empty_cache()
    finally:
        base.atomic_json(out/'execution_probes.json', {'policies': probes,
            'scope': '18 selected KV-head-group calls/policy: layers0/15/31, steps0/768/1535, heads0/7. Same-cache FP32 .005relative/.02absolute limits; not all layers or HF attention.'})
    verify(m)
    base.atomic_json(out/'analysis.json', dict(results=results, prefill_records=prefill,
        probes_sha256=base.sha256_file(out/'execution_probes.json'), initial_kv_sample_sha256=fingerprint,
        torch=torch.__version__, transformers=transformers.__version__, flashinfer=flashinfer.__version__))


def reduce_runs(directories):
    import numpy as np
    if len(directories) != 8: raise ValueError('All eight Qwen TRAIN books required')
    cells, memory = ({name: [] for name, _ in POLICIES} for _ in range(2))
    evidence = {}; books = set()
    for index, path in enumerate(directories):
        m = json.loads((path/'manifest.json').read_text()); verify(m)
        c = json.loads((path/'completion.json').read_text())
        r = json.loads((path/'analysis.json').read_text())
        if c['return_code'] or not c['sampled_exclusivity_passed'] or m['book_index'] != index:
            raise ValueError('Missing/reordered Qwen book')
        books.add(m['token_provenance']['object_name'])
        for name in ('manifest.json', 'completion.json', 'analysis.json', 'tokens.json', 'execution_probes.json'):
            evidence[str(path/name)] = base.sha256_file(path/name)
        if evidence[str(path/'tokens.json')] != m['tokens_sha256'] or evidence[str(path/'execution_probes.json')] != r['probes_sha256']:
            raise ValueError('Changed tokens/probes')
        probes = json.loads((path/'execution_probes.json').read_text())['policies']
        shared = None
        for name, suffix in POLICIES:
            rawpath = path/(name+'.json'); evidence[str(rawpath)] = base.sha256_file(rawpath)
            if evidence[str(rawpath)] != r['results'][name]['sha256']: raise ValueError('Changed outcome')
            raw = json.loads(rawpath.read_text()); units = raw['distribution_quality']['cluster_bootstrap_units']
            if len(units) != 1 or units[0]['token_count'] != 1536 or raw['consumed_new_history_pages'] != list(range(1280, 1328)) or raw['policy'] != {'S': 4, 'A': suffix, 'T': 768}:
                raise ValueError('Incomplete/changed Qwen policy trajectory')
            unit = units[0]; actual = (unit['cluster_unit_id'], unit['raw_sufficient_statistics']['reference_nll_sum_nats'])
            if shared is not None and actual != shared: raise ValueError('Unmatched book/HF reference')
            shared = actual
            if len(probes[name]) != 18 or any(p['relative_l2'] > .005 or p['absolute_error'] > .02 for p in probes[name]):
                raise ValueError('Failed/missing execution probes')
            cells[name].append(unit); memory[name].append(raw['cache_storage_bytes'])
    if len(books) != 8: raise ValueError('Repeated book')
    rows = {name: summarize(cells[name], memory[name]) for name, _ in POLICIES}
    delta = np.array([b['raw_sufficient_statistics']['candidate_nll_sum_nats']-a['raw_sufficient_statistics']['candidate_nll_sum_nats']
        for a, b in zip(cells['reference_policy'], cells['without_static_suffix'])])
    samples = np.random.default_rng(2026090914).integers(0, 8, size=(5000, 8))
    ratio = math.exp(delta.sum()/12288)
    interval = np.quantile(np.exp(delta[samples].sum(1)/12288), [.025, .975]).tolist()
    return {'rows': rows, 'no_suffix_over_reference_ppl': ratio, 'descriptive_paired_book_ratio95': interval,
        'books': sorted(books), 'runs': [str(p) for p in directories], 'input_sha256': evidence,
        'scope': 'Qwen3-8B fixed A128/A0 transfer on eight preselected PG19 TRAIN books; full1536labels/book/policy, same S4/T768, quantizer and kernels. Descriptive paired-book quality, not speed, superiority, noninferiority, default promotion or final TEST.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try: worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    fixtures = [Path(p) for p in json.loads(BOOKS.read_text())['runs']]
    if len(fixtures) != 8: raise ValueError('Missing retained cohort')
    hashes = {str(p): base.sha256_file(p) for p in (BOOKS, Path(__file__).resolve(), Path(__file__).with_name('regional_quality.py'), ROOT/'experiments/mlsys2027/baselines_v1/reduce_frontier.py')}
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_ablation_v1'/('qwen_suffix_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'manifest.json', {'policies': POLICIES, 'fixtures': [str(p) for p in fixtures], 'input_sha256': hashes,
            'scope': 'All eight preselected development books regardless outcome; no other policies or final TEST'})
        print('Qwen suffix suite: '+str(out), flush=True); directories = []
        for index, source in enumerate(fixtures):
            for path, digest in hashes.items():
                if base.sha256_file(Path(path)) != digest: raise ValueError('Suite source drift')
            m = json.loads((source/'manifest.json').read_text()); verify(m)
            c = json.loads((source/'completion.json').read_text())
            if c['return_code'] or not c['sampled_exclusivity_passed'] or (m['model_family'], m['context'], m['decode_steps'], m['book_index'], m['token_provenance']['split']) != ('qwen3', 20480, 1536, index, 'train'):
                raise ValueError('Invalid retained fixture')
            child = out/f'book_{index}'; child.mkdir()
            base.atomic_json(child/'tokens.json', json.loads((source/'tokens.json').read_text()))
            if base.sha256_file(child/'tokens.json') != m['tokens_sha256']: raise ValueError('Changed token serialization')
            m['source_sha256'].update(hashes)
            for key in ('S', 'A', 'T'): m.pop(key)
            m.update(policies=POLICIES, source_fixture=str(source),
                reference_initial_kv_sample_sha256=json.loads((source/'analysis.json').read_text())['initial_kv_sample_sha256'],
                source_analysis_sha256=base.sha256_file(source/'analysis.json'),
                scope='Qwen fixed A128/A0 development transfer, same S4/T768 and original kernels/quantizer. Two own recurrent PG trajectories versus shared HF reference; full1536labels, selected execution probes, no speed/final TEST/default promotion.')
            base.atomic_json(child/'manifest.json', m); base.idle_preflight(0)
            command = [sys.executable, '-u', str(Path(__file__)), '--worker', str(child)]
            base.atomic_json(child/'invocation.json', {'command': command})
            print(f'Qwen suffix book {index+1}/8', flush=True)
            completed = previous.run_process(command, child, {'index': 0}, gpu)
            base.atomic_json(child/'completion.json', completed); verify(m)
            if completed['return_code'] or not completed['sampled_exclusivity_passed']:
                raise ValueError('Qwen suffix execution failure retained')
            directories.append(child); base.atomic_json(out/'progress.json', {'completed_books': len(directories)})
        result = reduce_runs(directories); base.atomic_json(out/'analysis.json', result)
        print(json.dumps(result['rows'], indent=2), flush=True)


if __name__ == '__main__': main()
