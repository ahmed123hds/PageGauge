"""Corrected native Kitty Qwen3-8B quality with own HF4.53.2 reference.

Reuse exact tokens from a completed PageGauge development fixture. Full-prefix
prefill in both arms; no timing, generated-task or final TEST claim.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/Kitty')
PYTHON = '/home/anonymous/pagegauge_baselines/kitty_sm120_env/bin/python'
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/generalization_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from kivi_quality import verify
import qwen_tokens


def cache_bytes(cache):
    import torch
    if hasattr(cache, 'kv_cache'):
        tensors = [v for layer in cache.kv_cache for v in vars(layer).values() if isinstance(v, torch.Tensor)]
    else:
        tensors = [v for pair in cache.to_legacy_cache() for v in pair]
    stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return {'unique_storage_bytes': sum(stores.values()),
        'logical_tensor_bytes': sum(t.numel()*t.element_size() for t in tensors),
        'scope': 'Allocated native served KV including code/index/affine metadata, page tables, sink and residual buffers; excludes weights, activations and workspace'}


def worker(out):
    import torch
    import transformers
    from transformers import Qwen3ForCausalLM
    from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
    from kitty.kvcache import get_kvcache_kitty
    import benchmark_pg19_external_quality as quality
    manifest = json.loads((out/'manifest.json').read_text())
    verify(manifest)
    if base.sha256_file(out/'tokens.json') != manifest['tokens_sha256']:
        raise ValueError('Tokens changed')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026090825)
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], device='cuda', dtype=torch.long)
    context, steps = manifest['context'], manifest['decode_steps']
    if tokens.shape != (1, context+steps+1):
        raise ValueError('Wrong token shape')
    labels = tokens[:, context+1:context+steps+1].T.cpu()
    windows = qwen_tokens.request_windows(manifest['token_provenance'], context, steps, quality)
    reference, results = None, {}
    with torch.inference_mode():
        for arm, cls in [('hf', Qwen3ForCausalLM), ('kitty_pro', Qwen3ForCausalLM_Kitty)]:
            print('Kitty quality: loading '+arm+'...', flush=True)
            model, loading = cls.from_pretrained(manifest['model'], torch_dtype=torch.float16,
                local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda:0'},
                attn_implementation='sdpa', output_loading_info=True)
            if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError('Native checkpoint parameter mismatch: '+str(loading))
            model.eval()
            if (model.config.model_type, model.config.num_hidden_layers) != ('qwen3', 36):
                raise ValueError('Wrong model')
            past = None if arm == 'hf' else get_kvcache_kitty(model.config, 1, context+steps)
            print('Kitty quality: full-prefix '+arm+' prefill...', flush=True)
            output = model.model(tokens[:, :context], past_key_values=past, use_cache=True)
            past = output.past_key_values
            del output
            initial_memory = cache_bytes(past)
            observed, consumed = [], [set() for _ in range(36)]
            print('Kitty quality: recurrent '+arm+'...', flush=True)
            for step in range(steps):
                position = context+step
                if arm == 'kitty_pro':
                    for layer, cache in enumerate(past.kv_cache):
                        consumed[layer].add((cache.PageCount_K, cache.PageCount_V))
                output = model(tokens[:, position:position+1], past_key_values=past, use_cache=True)
                past = output.past_key_values
                observed.append(output.logits[:, -1].float().cpu())
                if past.get_seq_length() != position+1:
                    raise RuntimeError('Wrong native recurrent length')
            if not all(bool(row.isfinite().all()) for row in observed):
                raise RuntimeError('Nonfinite logits')
            if arm == 'hf':
                reference = observed
            else:
                expected = {((context+step-32)//128, (context+step-32-128)//128) for step in range(steps)}
                if any(seen != expected for seen in consumed):
                    raise RuntimeError('New native K/V pages not consumed in every layer')
            comparison = quality.compare_logits(reference, observed, labels,
                reference_name='Qwen3-8B HF4.53.2 SDPA FP16 full-prefix reference', candidate_name=arm,
                predicted_position_start=context+1, request_windows=windows)
            comparison['cache_accounting'] = {'initial': initial_memory, 'final': cache_bytes(past)}
            comparison['consumed_page_counts_per_layer'] = [sorted(seen) for seen in consumed] if arm != 'hf' else None
            comparison['final_lengths'] = [past.get_seq_length(layer) for layer in range(36)]
            base.atomic_json(out/(arm+'.json'), comparison)
            results[arm] = {'sha256': base.sha256_file(out/(arm+'.json')),
                'cache_accounting': comparison['cache_accounting'],
                'top1': comparison['top1_agreement_fraction'],
                'distribution_quality': {k: v for k, v in comparison['distribution_quality'].items() if k != 'per_token_metrics_step_major'}}
            del model, past, output, observed, comparison
            gc.collect()
            torch.cuda.empty_cache()
    verify(manifest)
    base.atomic_json(out/'analysis.json', {'results': results, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'scope': manifest['scope'],
        'checkpoint_dtype': 'Common FP16 conversion of BF16 checkpoint',
        'native_port_correction': 'Two uint8-to-int32 address-width casts; original failure preserved'})
    print('Kitty quality complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as exc:
            base.atomic_json(args.worker/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            raise
        return
    if args.fixture is None:
        raise ValueError('Completed Qwen development fixture required')
    original = json.loads((args.fixture/'manifest.json').read_text())
    completed = json.loads((args.fixture/'completion.json').read_text())
    if original['synthetic'] or original['token_provenance']['split'] != 'train' or completed['return_code'] or not completed['sampled_exclusivity_passed']:
        raise ValueError('Invalid Qwen development fixture')
    if base.sha256_file(args.fixture/'tokens.json') != original['tokens_sha256']:
        raise ValueError('Original fixture tokens changed')
    smoke = ROOT/'results/mlsys2027_baselines_v1/kitty_smoke_20260908T150002Z_47d92e07'
    if not json.loads((smoke/'assessment.json').read_text())['execution_passed']:
        raise ValueError('Native SM120 execution not validated')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = {Path(p) for p in original['source_sha256']}
        paths.update((SOURCE/'src/kitty').rglob('*.py'))
        paths.update((Path(__file__), Path(__file__).with_name('kivi_quality.py')))
        fork_package = Path(PYTHON).parent.parent/'lib/python3.12/site-packages/transformers'
        paths.update(fork_package/name for name in ('cache_utils.py', 'masking_utils.py', 'models/qwen3/modeling_qwen3.py'))
        out = ROOT/'results/mlsys2027_baselines_v1'/('kitty_quality_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        tokens = json.loads((args.fixture/'tokens.json').read_text())
        base.atomic_json(out/'tokens.json', tokens)
        manifest = {'model': original['model'], 'context': original['context'], 'decode_steps': original['decode_steps'],
            'token_provenance': original['token_provenance'], 'input_file_evidence': original['input_file_evidence'],
            'tokens_sha256': base.sha256_file(out/'tokens.json'),
            'source_sha256': {str(p): base.sha256_file(p) for p in sorted(paths)}, 'idle': idle,
            'pg_fixture': str(args.fixture), 'pg_fixture_manifest_sha256': base.sha256_file(args.fixture/'manifest.json'),
            'native_source_diff': subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--', 'src/kitty'], text=True),
            'scope': 'Exposed TRAIN corrected native Kitty-Pro Qwen quality; own HF4.53.2 full-prefix reference. No speed, generated-task or final TEST claim.'}
        base.atomic_json(out/'manifest.json', manifest)
        print('Kitty quality output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        completion = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completion)
        verify(manifest)
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('Kitty quality/exclusivity failed; inspect preserved evidence')


if __name__ == '__main__':
    main()
