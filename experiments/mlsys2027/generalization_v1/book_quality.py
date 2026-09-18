"""Recurrent Mistral/Qwen quality on a frozen PG19 TRAIN development book.

Shared coherent HF prefill, then unmodified FI/PG serving math. Reports model-
token PPL, not PG19's official whole-corpus word-normalized benchmark score.
"""
import argparse
from datetime import datetime, timezone
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
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from qwen_quality import verify
import book_tokens


def worker(out):
    import torch
    import flashinfer
    import transformers
    from transformers import AutoModelForCausalLM
    import benchmark_pg19_external_quality as quality
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    if base.sha256_file(out/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Token artifact changed')
    pg = quality.PG
    if m['model_family'] == 'qwen3':
        from qwen_adapter import install
        install(pg)
    elif m['model_family'] != 'mistral':
        raise ValueError('Unsupported model family')
    torch.set_grad_enabled(False)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], dtype=torch.long)
    context, steps = m['context'], m['decode_steps']
    maximum = context+steps
    pages, initial = math.ceil(maximum/16), context//16
    if tokens.shape != (1, maximum+1):
        raise ValueError('Incomplete book input/labels')
    print('Book quality: loading '+m['model_family'], flush=True)
    model = AutoModelForCausalLM.from_pretrained(m['model'], dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').eval().cuda()
    layers, hq, hkv, hidden = pg.BASE_E2E.check_model(model)
    expected = (36 if m['model_family'] == 'qwen3' else 32, 32, 8, 4096)
    if (layers, hq, hkv, hidden) != expected or model.config.model_type != m['model_family']:
        raise ValueError('Checkpoint shape/family mismatch')
    baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
    print('Book quality: coherent HF prefill and reference recurrence', flush=True)
    hf, _, prefill, fingerprint = quality.prefill_model_cache(model, tokens, baseline,
        pages, context, steps, 1024)
    if not all(row['sampled_boundary_copy_bitwise_identical'] for row in prefill):
        raise ValueError('HF cache copy failed')
    pg.BASE_E2E.pack_model_projections(model)
    positions = torch.arange(maximum, device='cuda', dtype=torch.long)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device='cuda', dtype=torch.float16), positions)
    if cos.dim() == 3:
        cos, sin = cos[0], sin[0]
    cos, sin = cos.half().contiguous(), sin.half().contiguous()
    extension = pg.RUNTIME.load_append_extension()
    decode = tokens[:, context:maximum].T.contiguous().cuda()
    labels = tokens[:, context+1:maximum+1].T.contiguous()
    windows = book_tokens.request_windows(m['token_provenance'], context, steps, quality)
    results = {}
    for backend in ('flashinfer_fp16', 'page_gauge'):
        cache = baseline if backend == 'flashinfer_fp16' else quality.build_gauge_cache_from_baseline(
            baseline, layers, pages, initial, 48, 1, hkv, 4, 128)
        decoder = pg.TransformerDecoder(model, flashinfer, extension, backend, cache, maximum,
            768, 256, 128, cos, sin, 'attention_add', tail_attention='flashinfer_merge', batch_size=1,
            old_value_scale_placement='probability', exact_sink_pages=4 if backend == 'page_gauge' else 0,
            exact_static_suffix_pages=128 if backend == 'page_gauge' else 0,
            initial_context_pages=initial if backend == 'page_gauge' else None)
        if backend == 'page_gauge':
            original = decoder.exact_wrapper.plan
            def exact_plan(indices, length, last, split, page_table_epoch=None):
                return original(indices, length, last, m['exact_split_pages'], page_table_epoch)
            decoder.exact_wrapper.plan = exact_plan
        print('Book quality: recurrent '+backend, flush=True)
        logits, consumed = [], set()
        for step, token in enumerate(decode):
            logit = decoder.step(token, context+step)[0].detach().float().cpu()
            if not bool(logit.isfinite().all()):
                raise ValueError('Nonfinite '+backend+' logits')
            logits.append(logit)
            if backend == 'page_gauge':
                consumed.update(p for p in decoder.old_logical_pages if p >= initial)
        expected_pages = set(range(initial, math.ceil(maximum/16)-48))
        if backend == 'page_gauge' and consumed != expected_pages:
            raise ValueError('New recurrent INT8 history not consumed')
        comparison = quality.compare_logits(hf, logits, labels,
            reference_name=m['model_family']+' HF SDPA FP16', candidate_name=backend,
            predicted_position_start=context+1, request_windows=windows)
        stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in vars(cache).values() if isinstance(t, torch.Tensor)}
        comparison.update(cache_storage_bytes=sum(stores.values()),
                          consumed_new_history_pages=sorted(consumed), exact_split_pages=m['exact_split_pages'])
        base.atomic_json(out/(backend+'.json'), comparison)
        overall = comparison['distribution_quality']['overall']
        results[backend] = {'sha256': base.sha256_file(out/(backend+'.json')),
            'ppl': overall['candidate_perplexity'], 'hf_ppl': overall['reference_perplexity'],
            'ppl_ratio_to_hf': overall['candidate_to_reference_perplexity_ratio'],
            'top1': comparison['top1_agreement_fraction'], 'minimum_cosine': comparison['minimum_logits_cosine'],
            'cache_storage_bytes': sum(stores.values()), 'consumed_new_history_pages': sorted(consumed)}
        del decoder, logits, comparison, cache
        # The custom exact-plan closure retains its wrapper until explicitly
        # released; no further backend follows PG in this fixed two-arm order.
        gc.collect()
        torch.cuda.empty_cache()
    verify(m)
    base.atomic_json(out/'analysis.json', {'results': results, 'prefill_records': prefill,
        'initial_kv_sample_sha256': fingerprint, 'model_family': m['model_family'],
        'book_index': m['book_index'], 'labels': steps, 'batch': 1,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'flashinfer': flashinfer.__version__,
        'execution_completed': True, 'predictive_quality_acceptance_applied': False, 'scope': m['scope']})
    print('Book quality complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--model-family', choices=('mistral', 'qwen3'), default='mistral')
    parser.add_argument('--book', type=int, default=0)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    if args.cohort is None or not 0 <= args.book < 8:
        raise ValueError('Verified development cohort and book index required')
    import fcntl
    import benchmark_pg19_external_quality as quality
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.model_family == 'mistral':
            source = ROOT/'results/mlsys2027_baselines_v1/bitdecode_quality_20260908T105048Z_28ae38c3/manifest.json'
        else:
            source = ROOT/'results/mlsys2027_generalization_v1/qwen_quality_20260908T142056Z_35911f58/manifest.json'
        original = json.loads(source.read_text())
        context, steps = (4096, 785) if args.short else (20480, 1536)
        tokens, provenance = book_tokens.build_tokens(args.cohort, args.book, original['model'], context, steps, quality)
        if provenance['model_family'] != args.model_family:
            raise ValueError('Tokenizer/config family mismatch')
        e0 = json.loads((ROOT/'results/mlsys2027_representation_v2/ground_truth_20260908T074422Z_8ab9bf75/manifest.json').read_text())
        paths = {ROOT/p for p in e0['source_sha256']}
        paths.update(Path(__file__).with_name(name) for name in ('book_quality.py', 'book_tokens.py', 'qwen_quality.py', 'qwen_adapter.py'))
        selection = args.cohort.resolve()/'selection.json'
        paths.add(selection)
        paths.add(ROOT/'results/mlsys2027_baselines_v1/rtx_exact_split_20260908T115408Z_e0315ebe/analysis.json')
        inputs = dict(original['input_file_evidence'])
        for path in (selection, args.cohort.resolve()/'analysis.json', Path(provenance['document_path'])):
            stat = path.stat()
            inputs[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': base.sha256_file(path)}
        for name, evidence in inputs.items():
            if base.sha256_file(Path(name)) != evidence['sha256']:
                raise ValueError('Model/corpus input changed: '+name)
        out = ROOT/'results/mlsys2027_generalization_v1'/('book_'+args.model_family+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json', {'ids': tokens.tolist()})
        manifest = {'model': original['model'], 'model_family': args.model_family,
            'book_index': args.book, 'context': context, 'decode_steps': steps, 'seed': 2026090901,
            'token_provenance': provenance, 'tokens_sha256': base.sha256_file(out/'tokens.json'),
            'source_sha256': {str(p): base.sha256_file(p) for p in sorted(paths)}, 'input_file_evidence': inputs,
            'S': 4, 'A': 128, 'T': 768, 'exact_split_pages': 32, 'old_split_pages': 128,
            'order': ['hf', 'flashinfer_fp16', 'page_gauge'], 'idle': idle, 'orchestrator_pid': os.getpid(),
            'scope': 'PG19 TRAIN book development only. Shared HF-prefilled FP16 cache, complete recurrent labels with native model/tokenizer semantics, PG exact32 launch selection. Descriptive model-token PPL/top1/cosine, not official word-level corpus score, speed, final TEST or non-inferiority claim.'}
        base.atomic_json(out/'manifest.json', manifest)
        print('Book quality output: '+str(out), flush=True)
        command = [sys.executable, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        result = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', result)
        verify(manifest)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise ValueError('Book worker/exclusivity failed; evidence preserved')


if __name__ == '__main__':
    main()
