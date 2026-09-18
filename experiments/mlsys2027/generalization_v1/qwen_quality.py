"""Qwen3-8B opt-in recurrent integration/quality pilot; exposed TRAIN only.

HF, FI and PG share the FP16 checkpoint and HF-prefilled cache. No speed claim,
quality-driven stopping, new cosine cutoff, or production-default modification.
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
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from qwen_adapter import install
import qwen_tokens


def verify(manifest):
    for name, digest in manifest['source_sha256'].items():
        if base.sha256_file(Path(name)) != digest:
            raise RuntimeError('Source drift: '+name)
    for name, evidence in manifest['input_file_evidence'].items():
        stat = Path(name).stat()
        if stat.st_size != evidence['size'] or stat.st_mtime_ns != evidence['mtime_ns']:
            raise RuntimeError('Input metadata drift: '+name)


def worker(out):
    import torch
    import flashinfer
    import transformers
    from transformers import AutoModelForCausalLM
    import benchmark_pg19_external_quality as quality
    manifest = json.loads((out/'manifest.json').read_text())
    verify(manifest)
    pg = quality.PG
    install(pg)
    torch.set_grad_enabled(False)
    torch.manual_seed(manifest['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], dtype=torch.long)
    if base.sha256_file(out/'tokens.json') != manifest['tokens_sha256']:
        raise RuntimeError('Token input drift')
    context, steps = manifest['context'], manifest['decode_steps']
    maximum = context+steps
    pages, initial = math.ceil(maximum/16), context//16
    print('Qwen: loading model in FP16...', flush=True)
    if manifest.get('synthetic'):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=64, hidden_size=128, intermediate_size=192,
            num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, max_position_embeddings=8192)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).half().eval().cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], dtype=torch.float16,
            local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').eval().cuda()
    layers, hq, hkv, hidden = pg.BASE_E2E.check_model(model)
    expected_shape = (2, 32, 8, 128) if manifest.get('synthetic') else (36, 32, 8, 4096)
    if (layers, hq, hkv, hidden) != expected_shape:
        raise ValueError('Unexpected Qwen3-8B shape')
    baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
    print('Qwen: shared HF prefill and corpus-label reference...', flush=True)
    hf, _, prefill, fingerprint = quality.prefill_model_cache(model, tokens, baseline,
        pages, context, steps, 1024)
    if not all(r['sampled_boundary_copy_bitwise_identical'] for r in prefill):
        raise RuntimeError('HF cache copy failed')
    # Packing occurs only AFTER HF forward has consumed the original projections.
    pg.BASE_E2E.pack_model_projections(model)
    extension = pg.RUNTIME.load_append_extension()
    positions = torch.arange(maximum, device='cuda', dtype=torch.long)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, 1, hidden, device='cuda', dtype=torch.float16), positions)
    if cos.dim() == 3:
        cos, sin = cos[0], sin[0]
    cos, sin = cos.half().contiguous(), sin.half().contiguous()
    decode = tokens[:, context:maximum].T.contiguous().cuda()
    labels = tokens[:, context+1:maximum+1].T.contiguous()
    windows = qwen_tokens.request_windows(manifest['token_provenance'], context, steps, quality)
    results = {}
    for name in ('flashinfer_fp16', 'page_gauge'):
        cache = baseline if name == 'flashinfer_fp16' else quality.build_gauge_cache_from_baseline(
            baseline, layers, pages, initial, 48, 1, hkv, 4, 128)
        decoder = pg.TransformerDecoder(model, flashinfer, extension, name, cache, maximum,
            768, 256, 128, cos, sin, 'attention_add', tail_attention='flashinfer_merge',
            batch_size=1, old_value_scale_placement='probability',
            exact_sink_pages=4 if name == 'page_gauge' else 0,
            exact_static_suffix_pages=128 if name == 'page_gauge' else 0,
            initial_context_pages=initial if name == 'page_gauge' else None)
        print('Qwen: recurrent '+name+'...', flush=True)
        logits = []
        consumed_new_pages = set()
        for step, ids in enumerate(decode):
            logit = decoder.step(ids, context+step)[0].detach().float().cpu()
            if not bool(logit.isfinite().all()):
                raise RuntimeError('Nonfinite '+name+' logits')
            logits.append(logit)
            if name == 'page_gauge':
                consumed_new_pages.update(p for p in decoder.old_logical_pages if p >= initial)
        expected_new_pages = set(range(initial, (maximum+15)//16-48))
        if name == 'page_gauge' and consumed_new_pages != expected_new_pages:
            raise RuntimeError('New recurrent INT8 pages were not consumed as expected')
        comparison = quality.compare_logits(hf, logits, labels, reference_name='Qwen3-8B HF SDPA FP16',
            candidate_name=name, predicted_position_start=context+1, request_windows=windows)
        stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                  for t in vars(cache).values() if isinstance(t, torch.Tensor)}
        comparison['cache_storage_bytes'] = sum(stores.values())
        comparison['consumed_new_history_pages'] = sorted(consumed_new_pages)
        base.atomic_json(out/(name+'.json'), comparison)
        if manifest.get('synthetic') and name == 'flashinfer_fp16':
            limits = manifest['synthetic_fp16_execution_tolerances']
            if (comparison['maximum_logits_absolute_error'] > limits['max_abs']
                    or comparison['maximum_relative_logits_l2'] > limits['max_relative_l2']):
                raise RuntimeError('Synthetic unquantized Qwen execution differs from HF beyond frozen numerical tolerances')
        results[name] = {'sha256': base.sha256_file(out/(name+'.json')),
            'top1': comparison['top1_agreement_fraction'],
            'minimum_logits_cosine': comparison['minimum_logits_cosine'],
            'maximum_relative_logits_l2': comparison['maximum_relative_logits_l2'],
            'maximum_logits_absolute_error': comparison['maximum_logits_absolute_error'],
            'distribution_quality': {k: v for k, v in comparison['distribution_quality'].items()
                                     if k != 'per_token_metrics_step_major'},
            'cache_storage_bytes': comparison['cache_storage_bytes'],
            'consumed_new_history_pages': sorted(consumed_new_pages)}
        del decoder, logits, comparison, cache
        gc.collect()
        torch.cuda.empty_cache()
    verify(manifest)
    base.atomic_json(out/'analysis.json', {'results': results, 'prefill_records': prefill,
        'initial_kv_sample_sha256': fingerprint, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'flashinfer': flashinfer.__version__,
        'labels': steps, 'batch': 1, 'production_default_changed': False,
        'execution_completed': True, 'predictive_quality_acceptance_applied': False,
        'scope': manifest['scope'], 'checkpoint_conversion': ('Random initialized model cast to FP16' if manifest.get('synthetic')
            else 'BF16 checkpoint converted to FP16 for all three arms; not native BF16 quality')})
    print('Qwen quality complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--synthetic', action='store_true', help='Two-layer random-model execution smoke; no pretrained quality claim')
    parser.add_argument('--offset', type=int, default=472000)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as exc:
            base.atomic_json(args.worker/'failure.json', {'type': type(exc).__name__,
                'error': str(exc), 'traceback': traceback.format_exc()})
            raise
        return
    if (args.download is None and not args.synthetic) or args.offset < 472000 or (args.synthetic and args.full):
        raise ValueError('Completed pinned download and exposed TRAIN offset required')
    import fcntl
    import benchmark_pg19_external_quality as quality
    download = ({'snapshot': 'synthetic-random-Qwen3', 'file_evidence': {}} if args.synthetic
                else json.loads((args.download/'analysis.json').read_text()))
    if not args.synthetic and download['revision'] != 'b968826d9c46dd6066d109eabc6255188de91218':
        raise ValueError('Wrong checkpoint revision')
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context, steps = (20480, 1536) if args.full else (4096, 785)
        token_args = SimpleNamespace(context=context, decode_steps=steps, batch_size=1,
            token_source='random' if args.synthetic else 'wikitext2', wikitext_member='wikitext-2-raw/wiki.train.raw',
            wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip', seed=2026090822,
            token_offset=args.offset, token_stride=23600, model=download['snapshot'])
        tokens, provenance = (quality.build_token_matrix(token_args, 64) if args.synthetic
                              else qwen_tokens.build_tokens(token_args, quality, 151936))
        e0 = json.loads((ROOT/'results/mlsys2027_representation_v2/ground_truth_20260908T074422Z_8ab9bf75/manifest.json').read_text())
        paths = {ROOT/name for name in e0['source_sha256']}
        paths.update((Path(__file__), Path(__file__).with_name('qwen_adapter.py'), Path(__file__).with_name('qwen_tokens.py')))
        inputs = {}
        for name, evidence in download['file_evidence'].items():
            path = Path(download['snapshot'])/name
            stat = path.stat()
            if stat.st_size != evidence['size']:
                raise RuntimeError('Checkpoint file size changed: '+name)
            inputs[str(path)] = dict(evidence, mtime_ns=stat.st_mtime_ns)
        corpus = token_args.wikitext_zip
        inputs[str(corpus)] = {'size': corpus.stat().st_size, 'mtime_ns': corpus.stat().st_mtime_ns,
                               'sha256': base.sha256_file(corpus)}
        out = ROOT/'results/mlsys2027_generalization_v1'/('qwen_quality_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json', {'ids': tokens.tolist()})
        manifest = {'model': download['snapshot'], 'synthetic': args.synthetic, 'context': context, 'decode_steps': steps,
            'seed': token_args.seed, 'offset': args.offset, 'source_sha256': {str(p): base.sha256_file(p) for p in sorted(paths)},
            'input_file_evidence': inputs, 'download_analysis_sha256': None if args.synthetic else base.sha256_file(args.download/'analysis.json'),
            'tokens_sha256': base.sha256_file(out/'tokens.json'), 'token_provenance': provenance, 'idle': idle,
            'order': ['hf', 'flashinfer_fp16', 'page_gauge'], 'S': 4, 'A': 128, 'T': 768,
            'exact_split_pages': 128, 'old_split_pages': 128,
            'scope': 'Qwen exposed TRAIN integration/quality pilot; matched FP16 HF reference, descriptive PPL/top1/cosine. No speed, final TEST or non-inferiority claim.'}
        if args.synthetic:
            manifest['scope'] = 'Two-layer random Qwen3 execution smoke, Hq32/Hkv8/D128 with small hidden/vocabulary sizes. No pretrained quality, speed or final TEST claim.'
            manifest['synthetic_fp16_execution_tolerances'] = {'max_abs': .02, 'max_relative_l2': .005,
                'scope': 'Unquantized FI implementation versus same random HF model only; not a predictive quality threshold'}
        base.atomic_json(out/'manifest.json', manifest)
        print('Qwen quality output: '+str(out), flush=True)
        command = [sys.executable, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        completion = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completion)
        verify(manifest)
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('Qwen worker/exclusivity failed; preserve evidence')


if __name__ == '__main__':
    main()
