"""Native NSN INT2 pretrained quality with own HF and V/O-rotation controls.

TRAIN only. Single full-precision prefill per arm avoids quantizing intermediate
prefill chunks. No latency claim or retrospective predictive-quality gate.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/NSNQuant')
PYTHON = '/home/anonymous/pagegauge_baselines/nsn_sm120_env/bin/python'
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from kivi_quality import verify


def cache_bytes(model, cache):
    import torch
    if hasattr(cache, 'caches'):
        tensors = [t for values in cache.caches.values() for t in values if isinstance(t, torch.Tensor)]
    else:
        tensors = [t for pair in cache.to_legacy_cache() for t in pair]
    codebooks = [t for layer in model.model.layers for t in
                 (list(layer.self_attn.quantizer.buffers()) if hasattr(layer.self_attn, 'quantizer') else [])]
    stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors+codebooks}
    return {'unique_storage_bytes': sum(stores.values()),
            'logical_tensor_bytes': sum(t.numel()*t.element_size() for t in tensors+codebooks),
            'quantizer_buffer_bytes': sum(t.numel()*t.element_size() for t in codebooks),
            'scope': 'Native served KV, RoPE cache metadata, and resident quantizer/codebook buffers; excludes weights and workspace'}


def worker(out):
    import torch
    import transformers
    from transformers import MistralForCausalLM
    sys.path.insert(0, str(SOURCE))
    from src.models.mistral import QuantizedMistralForCausalLM
    from src.utils import rotate_v_proj, rotate_o_proj
    import benchmark_pg19_external_quality as quality
    manifest = json.loads((out/'manifest.json').read_text())
    verify(manifest)
    if base.sha256_file(out/'tokens.json') != manifest['tokens_sha256']:
        raise RuntimeError('Tokens changed')
    torch.set_grad_enabled(False)
    torch.manual_seed(manifest['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], device='cuda', dtype=torch.long)
    context, steps = manifest['context'], manifest['decode_steps']
    if tokens.shape != (1, context+steps+1):
        raise ValueError('Wrong token shape')
    labels = tokens[:, context+1:context+steps+1].T.cpu()
    windows = quality.request_window_metadata(manifest['token_provenance'], context, steps, 1)
    reference = None
    results = {}
    with torch.inference_mode():
        for arm in ('hf', 'rotated_identity', 'int2'):
            print('Loading '+arm+'...', flush=True)
            kwargs = dict(torch_dtype=torch.float16, local_files_only=True,
                          low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
            if arm == 'hf':
                model = MistralForCausalLM.from_pretrained(manifest['model'], **kwargs).eval()
            else:
                quant = {'name': 'Identity', 'kwargs': None} if arm == 'rotated_identity' else {
                    'name': 'NSNQuantizer', 'kwargs': {'n_bits': 2,
                    'codebook_path': str(SOURCE/'codebooks/2bit_codebook.pt'),
                    'window_size': 64, 'residual_size': 64, 'hadamard': True}}
                model = QuantizedMistralForCausalLM.from_pretrained(manifest['model'],
                    quant_config=quant, forward_quant=False, **kwargs).half().eval()
                # Upstream get_mistral_model also calls .half() after loading:
                # torch_dtype alone does not cast freshly registered codebooks.
                if arm == 'int2':
                    from src.quantizers.nsn_quantizer import NSNQuantizer
                    released = NSNQuantizer(**quant['kwargs']).half().cuda()
                    expected_buffers = dict(released.named_buffers())
                    for layer in model.model.layers:
                        for name, value in layer.self_attn.quantizer.named_buffers():
                            if not torch.equal(value, expected_buffers[name]):
                                raise RuntimeError('Loaded NSN codebook differs from released artifact: '+name)
                    del released, expected_buffers
                for layer in model.model.layers:
                    rotate_v_proj(layer.self_attn.v_proj, 128)
                    rotate_o_proj(layer.self_attn.o_proj, 128)
            if model.config.sliding_window is not None or len(model.model.layers) != 32:
                raise ValueError('Wrong full-context model')
            print(arm+': full-prefix SDPA prefill...', flush=True)
            # Bypass only the unused prefill LM head, not any decoder layer.
            output = model.model(tokens[:, :context], use_cache=True)
            past = output.past_key_values
            del output
            initial_memory = cache_bytes(model, past)
            observed_packed = [set() for _ in model.model.layers]
            observed = []
            print(arm+': recurrent corpus-label decode...', flush=True)
            for position in range(context, context+steps):
                if arm == 'int2':
                    for i in range(32):
                        observed_packed[i].add(int(past[i]['quantized_key_cache'].shape[-2]))
                output = model(tokens[:, position:position+1], past_key_values=past, use_cache=True)
                past = output.past_key_values
                observed.append(output.logits[:, -1].float().cpu())
                if past.get_seq_length() != position+1:
                    raise RuntimeError('Wrong recurrent length')
            if not all(bool(v.isfinite().all()) for v in observed):
                raise RuntimeError('Nonfinite logits')
            if arm == 'int2':
                expected = set(range(context, context+steps, 64))
                if any(lengths != expected for lengths in observed_packed):
                    raise RuntimeError('New packed history not consumed in every layer')
            if arm == 'hf':
                reference = observed
            comparison = quality.compare_logits(reference, observed, labels,
                reference_name='HF SDPA FP16 transformers 4.48.1, full-prefix prefill',
                candidate_name=arm, predicted_position_start=context+1, request_windows=windows)
            comparison['cache_accounting'] = {'initial': initial_memory, 'final': cache_bytes(model, past)}
            comparison['recurrence_steps'] = steps
            comparison['consumed_packed_lengths_per_layer'] = [sorted(v) for v in observed_packed] if arm == 'int2' else None
            comparison['final_packed_lengths'] = [int(past[i]['quantized_key_cache'].shape[-2]) for i in range(32)] if arm == 'int2' else None
            base.atomic_json(out/(arm+'.json'), comparison)
            results[arm] = {'sha256': base.sha256_file(out/(arm+'.json')),
                'distribution_quality': {k: v for k, v in comparison['distribution_quality'].items() if k != 'per_token_metrics_step_major'},
                'cache_accounting': comparison['cache_accounting']}
            del model, output, past, observed, comparison
            gc.collect()
            torch.cuda.empty_cache()
    verify(manifest)
    base.atomic_json(out/'analysis.json', {'results': results, 'torch': torch.__version__,
        'transformers': transformers.__version__,
        'scope': 'Exposed TRAIN native NSN quality; own HF and V/O-rotation controls, full-prefix prefill. No speed or final TEST claim.'})
    print('NSN quality complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--offset', type=int, default=472000)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as exc:
            base.atomic_json(args.worker/'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    import benchmark_pg19_external_quality as quality
    original = json.loads((ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c/manifest.json').read_text())
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context, steps = (20480, 1536) if args.full else (1024, 129)
        token_args = SimpleNamespace(context=context, decode_steps=steps, batch_size=1,
            token_source='wikitext2', wikitext_member='wikitext-2-raw/wiki.train.raw',
            wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip', seed=2026090815,
            token_offset=args.offset, token_stride=23600, model=original['inputs']['model_path'])
        tokens, provenance = quality.build_token_matrix(token_args, 32768)
        e0 = json.loads((ROOT/'results/mlsys2027_representation_v2/ground_truth_20260908T074422Z_8ab9bf75/manifest.json').read_text())
        paths = {ROOT/name for name in e0['source_sha256']}
        paths.update((Path(__file__), Path(__file__).with_name('kivi_quality.py')))
        paths.update(SOURCE.rglob('*.py'))
        paths.update(SOURCE.rglob('*.cu'))
        paths.update((SOURCE/'codebooks').glob('*bit_codebook.pt'))
        packages = Path('/home/anonymous/pagegauge_baselines/nsn_sm120_env/lib/python3.12/site-packages')
        paths.update(packages.glob('*nsn_tools*.so'))
        paths.update(packages.glob('fast_hadamard_transform_cuda*.so'))
        paths.add(packages/'transformers/models/mistral/modeling_mistral.py')
        out = ROOT/'results/mlsys2027_baselines_v1'/('nsn_quality_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json', {'ids': tokens.tolist()})
        manifest = {'context': context, 'decode_steps': steps, 'seed': token_args.seed,
            'model': token_args.model, 'input_file_evidence': original['input_file_evidence'],
            'source_sha256': {str(p): base.sha256_file(p) for p in sorted(paths)},
            'tokens_sha256': base.sha256_file(out/'tokens.json'), 'token_provenance': provenance,
            'idle': idle, 'scope': 'TRAIN-only baseline pilot/cohort; no predictive-quality acceptance threshold',
            'prefill': 'One full-prefix SDPA invocation per arm; no LM head for unused prefix logits',
            'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()}
        base.atomic_json(out/'manifest.json', manifest)
        print('NSN pretrained quality: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        completion = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completion)
        verify(manifest)
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('NSN worker/exclusivity failed; inspect preserved evidence')


if __name__ == '__main__':
    main()
