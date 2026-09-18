"""Fixed one-factor-at-a-time exact-region development ablation, no search."""
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
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/generalization_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from qwen_quality import verify
import book_tokens

POLICIES = [('reference_policy', 4, 128, 768), ('without_prefix', 0, 128, 768),
            ('without_static_suffix', 4, 0, 768), ('minimum_page_tail', 4, 128, 16)]


def row_regions(initial_pages, length, prefix, suffix, old_pages, exact_pages):
    """Disjoint attribution; prefix then static suffix take overlap precedence."""
    if set(old_pages) & set(exact_pages) or sorted(old_pages+exact_pages) != list(range(math.ceil(length/16))):
        raise ValueError('Region pages do not cover the serving domain exactly once')
    regions = []
    for page in old_pages+exact_pages:
        if page in old_pages:
            region = 'history'
        elif page < prefix:
            region = 'prefix'
        elif suffix and initial_pages-suffix <= page < initial_pages:
            region = 'static_suffix'
        else:
            region = 'tail_only'
        regions.extend([region]*min(16, length-page*16))
    if len(regions) != length:
        raise ValueError('Incorrect partial-page coverage')
    return regions


def install_probe(driver, context, steps, stats):
    import torch
    original = driver.eager_attention
    original_step = driver.step
    current_position = None
    selected_steps, selected_layers = {0, 768, steps-1}, {0, 15, 31}
    def step(input_ids, position, *args, **kwargs):
        nonlocal current_position
        current_position = position
        try:
            return original_step(input_ids, position, *args, **kwargs)
        finally:
            current_position = None
    def attention(layer):
        actual = original(layer)
        if current_position is None:
            raise ValueError('Regional attention probe was called outside decoder.step')
        step = current_position-context
        if layer not in selected_layers or step not in selected_steps:
            return actual
        length = current_position+1
        old_pages = list(driver.old_logical_pages)
        exact_pages = [p for p in range(math.ceil(length/16)) if p not in old_pages]
        regions = row_regions(context//16, length, driver.exact_sink_pages,
            driver.exact_static_suffix_pages, old_pages, exact_pages)
        old_ids = driver.old_wrapper.indices[:len(old_pages)].long()
        exact_ids = driver.exact_wrapper.indices[:len(exact_pages)].long()
        cache = driver.cache
        for head in (0, 7):
            old_k = (cache.key_codes[layer, old_ids, :, head].float()*
                cache.key_scales[layer, old_ids, head].float()[:, None, None]).reshape(-1, 128)
            old_v = (cache.value_codes[layer, old_ids, :, head].float()*
                cache.value_scales[layer, old_ids, head].float()[:, None, None]).reshape(-1, 128)
            exact_count = length-len(old_pages)*16
            exact_k = cache.exact_key[layer, exact_ids, :, head].reshape(-1, 128)[:exact_count].float()
            exact_v = cache.exact_value[layer, exact_ids, :, head].reshape(-1, 128)[:exact_count].float()
            keys, values = torch.cat((old_k, exact_k)), torch.cat((old_v, exact_v))
            query = driver.rotated_query[0, head*4:(head+1)*4].float()
            weights = (query@keys.T/math.sqrt(128)).softmax(-1)
            expected = weights@values+cache.output_center[layer, 0, head*4:(head+1)*4].float()
            observed = actual[0, head*4:(head+1)*4].float()
            delta = observed-expected
            relative = float((delta.norm()/expected.norm().clamp_min(1e-30)).item())
            absolute = float(delta.abs().max().item())
            masses = {name: float(weights[:, torch.tensor([r == name for r in regions], device='cuda')].sum(-1).mean().item())
                      for name in ('history', 'prefix', 'static_suffix', 'tail_only')}
            row = dict(step=step, layer=layer, kv_head=head, relative_l2=relative,
                absolute_error=absolute, mean_query_group_attention_mass=masses)
            stats.append(row)
            if not torch.isfinite(expected).all() or not torch.isfinite(observed).all() or relative > .005 or absolute > .02:
                raise ValueError('Selected reconstructed-cache execution check failed: '+str(row))
            if abs(sum(masses.values())-1) > 1e-5:
                raise ValueError('Attention region masses do not sum to one')
        return actual
    driver.eager_attention = attention
    driver.step = step


def worker(out):
    import torch
    import flashinfer
    import transformers
    from transformers import AutoModelForCausalLM
    import benchmark_pg19_external_quality as quality
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    if base.sha256_file(out/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed book tokens')
    pg = quality.PG
    torch.set_grad_enabled(False)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], dtype=torch.long)
    context, steps = m['context'], m['decode_steps']
    maximum, initial = context+steps, context//16
    pages = math.ceil(maximum/16)
    if tokens.shape != (1, maximum+1) or m['model_family'] != 'mistral':
        raise ValueError('Only retained full Mistral B1 fixture supported')
    print('Regional ablation: loading Mistral', flush=True)
    model = AutoModelForCausalLM.from_pretrained(m['model'], dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').eval().cuda()
    layers, hq, hkv, hidden = pg.BASE_E2E.check_model(model)
    if (layers, hq, hkv, hidden) != (32, 32, 8, 4096):
        raise ValueError('Wrong model shape')
    baseline = quality.allocate_baseline_cache(layers, pages, 1, hkv)
    print('Regional ablation: common HF prefill and full reference', flush=True)
    hf, _, prefill, fingerprint = quality.prefill_model_cache(model, tokens, baseline, pages, context, steps, 1024)
    if not all(v['sampled_boundary_copy_bitwise_identical'] for v in prefill):
        raise ValueError('HF initial cache copy mismatch')
    if fingerprint != m['reference_initial_kv_sample_sha256']:
        raise ValueError('HF initial cache sample differs from retained fixture')
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
    results, probes = {}, {}
    try:
        for name, prefix, suffix, tail in POLICIES:
            print('Regional ablation: '+name, flush=True)
            cache = quality.build_gauge_cache_from_baseline(baseline, layers, pages, initial, tail//16, 1, hkv, prefix, suffix)
            decoder = pg.TransformerDecoder(model, flashinfer, extension, 'page_gauge', cache, maximum,
                tail, 256, 128, cos, sin, 'attention_add', tail_attention='flashinfer_merge', batch_size=1,
                old_value_scale_placement='probability', exact_sink_pages=prefix,
                exact_static_suffix_pages=suffix, initial_context_pages=initial)
            original_plan = decoder.exact_wrapper.plan
            def exact_plan(indices, length, last, split, page_table_epoch=None):
                return original_plan(indices, length, last, 32, page_table_epoch)
            decoder.exact_wrapper.plan = exact_plan
            probes[name] = []
            install_probe(decoder, context, steps, probes[name])
            logits, consumed = [], set()
            for step, token in enumerate(decode):
                logit = decoder.step(token, context+step)[0].detach().float().cpu()
                if not logit.isfinite().all():
                    raise ValueError('Nonfinite ablation logits')
                logits.append(logit)
                consumed.update(p for p in decoder.old_logical_pages if p >= initial)
            if consumed != set(range(initial, pages-tail//16)) or len(probes[name]) != 18:
                raise ValueError('Incomplete new-history/probe trajectory')
            comparison = quality.compare_logits(hf, logits, labels,
                reference_name='Mistral HF SDPA FP16', candidate_name=name,
                predicted_position_start=context+1, request_windows=windows)
            stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in vars(cache).values() if isinstance(t, torch.Tensor)}
            comparison.update(cache_storage_bytes=sum(stores.values()),
                consumed_new_history_pages=sorted(consumed), policy={'S': prefix, 'A': suffix, 'T': tail})
            base.atomic_json(out/(name+'.json'), comparison)
            overall = comparison['distribution_quality']['overall']
            results[name] = dict(sha256=base.sha256_file(out/(name+'.json')),
                ppl=overall['candidate_perplexity'], hf_ppl=overall['reference_perplexity'],
                ppl_ratio_to_hf=overall['candidate_to_reference_perplexity_ratio'],
                top1=comparison['top1_agreement_fraction'], cache_storage_bytes=sum(stores.values()),
                consumed_new_history_pages=sorted(consumed))
            base.atomic_json(out/'progress.json', {'results': results})
            del decoder, logits, cache, comparison, original_plan, exact_plan
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        base.atomic_json(out/'regional_execution_probes.json', {'policies': probes,
            'scope': '18 selected KV-head groups/policy, three layers and steps, four queries/group. Same-cache FP32 oracle with unchanged .005 relative/.02 absolute execution limits. Region masses are selected post-quantization attention, not whole-model or original-HF attention. Overlap precedence: prefix, static suffix, remaining tail. Not speed evidence.'})
    verify(m)
    base.atomic_json(out/'analysis.json', dict(results=results, labels=steps, prefill_records=prefill,
        initial_kv_sample_sha256=fingerprint, probes_sha256=base.sha256_file(out/'regional_execution_probes.json'),
        torch=torch.__version__, transformers=transformers.__version__, flashinfer=flashinfer.__version__, scope=m['scope']))
    print('Regional ablation complete: '+str(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--fixture', type=Path)
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            raise
        return
    if args.fixture is None:
        raise ValueError('Completed Mistral PG19 TRAIN fixture required')
    source = args.fixture.resolve()
    m = json.loads((source/'manifest.json').read_text())
    c = json.loads((source/'completion.json').read_text())
    if (m['model_family'], m['context'], m['decode_steps'], m['token_provenance']['split']) != ('mistral', 20480, 1536, 'train') or c['return_code'] or not c['sampled_exclusivity_passed']:
        raise ValueError('Invalid completed development fixture')
    verify(m)
    if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed original tokens')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_ablation_v1'/('regional_quality_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json', json.loads((source/'tokens.json').read_text()))
        m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
        for key in ('S', 'A', 'T'):
            m.pop(key)
        m.update(policies=POLICIES, source_fixture=str(source),
            order=['hf']+[p[0] for p in POLICIES],
            reference_initial_kv_sample_sha256=json.loads((source/'analysis.json').read_text())['initial_kv_sample_sha256'],
            source_analysis_sha256=base.sha256_file(source/'analysis.json'), idle=idle, orchestrator_pid=os.getpid(),
            scope='Fixed one-factor exact-region ablation on an exposed Mistral PG19 TRAIN book. S4/A128/T768, S0/A128/T768, S4/A0/T768, S4/A128/T16, in that order. T16 is the minimum supported nonempty exact page, not zero residual. Shared HF prefill, unchanged INT8 quantizer/kernel/scales/centers and launch splits. Full recurrent corpus-label quality and served KV bytes; selected reconstructed-cache execution/region-mass checks. No timing, policy promotion, universal predictive gate or final TEST claim.')
        base.atomic_json(out/'manifest.json', m)
        print('Regional ablation output: '+str(out), flush=True)
        command = [sys.executable, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        completed = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completed)
        verify(m)
        if completed['return_code'] or not completed['sampled_exclusivity_passed']:
            raise RuntimeError('Regional ablation failed; preserve exact evidence')


if __name__ == '__main__':
    main()
