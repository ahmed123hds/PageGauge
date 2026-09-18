"""CPU-backed initial state, common-engine decode validation and capacity pilot.

This is not a production/native-best serving benchmark. Prefill/packing/upload
are reported preparation, outside decode timing. No GPU reference cache remains.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from kivi_quality import verify, PYTHON
from common_engine import BACKENDS, check_cache, memory, DEFAULT
from frontier_cache import cpu_state, restore, cpu_snapshot_bytes, verify_restored


def device_memory():
    import torch
    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device_used_bytes_at_boundary': total-free}


def worker(out):
    import torch
    import transformers
    from transformers import MistralForCausalLM
    import benchmark_pg19_external_quality as quality
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    if base.sha256_file(out/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Token artifact changed')
    torch.set_grad_enabled(False)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(m['allocator_budget_bytes']/total)
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    ids = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], device='cuda', dtype=torch.long)
    batch, context, steps, backend = m['batch'], m['context'], m['decode_steps'], m['backend']
    if ids.shape != (batch, context+steps+1):
        raise ValueError('Invalid unpadded request matrix')
    print(f'Frontier loading {backend}, B{batch}/C{context}/D{steps}', flush=True)
    model = MistralForCausalLM.from_pretrained(m['model'], torch_dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, attn_implementation='eager').eval()
    if model.config.sliding_window is not None or len(model.model.layers) != 32:
        raise ValueError('Only pinned full-context Mistral is supported')
    snapshots, reference, preparation = [], [], []
    with torch.inference_mode():
        # Serial coherent prefill avoids an FP16 B-way cache capacity bottleneck.
        # Every request has distinct corpus tokens; it is never a replicated cache.
        for request in range(batch):
            print(f'Preparing independent request {request+1}/{batch}', flush=True)
            past = None
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for start in range(0, context, 256):
                outputs = model.model(ids[request:request+1, start:min(start+256, context)],
                    past_key_values=past, use_cache=True)
                past = outputs.past_key_values
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter()-begin
            legacy = past.to_legacy_cache() if hasattr(past, 'to_legacy_cache') else past
            if any(k.shape != (1, 8, context, 128) for k, v in legacy):
                raise ValueError('Incomplete serial prefill')
            begin = time.perf_counter()
            snapshots.append(cpu_state(legacy, backend, context+steps))
            torch.cuda.synchronize()
            packing_seconds = time.perf_counter()-begin
            del legacy
            if m['validate']:
                request_reference = []
                for position in range(context, context+steps):
                    outputs = model(ids[request:request+1, position:position+1], past_key_values=past, use_cache=True)
                    past = outputs.past_key_values
                    request_reference.append(outputs.logits[:, -1].float().cpu())
                reference.append(request_reference)
            del outputs, past
            gc.collect()
            torch.cuda.empty_cache()
            preparation.append({'request': request, 'prefill_seconds': prefill_seconds,
                                'packing_and_cpu_snapshot_seconds': packing_seconds})
        snapshot_bytes = cpu_snapshot_bytes(snapshots)
        post_preparation = device_memory()
        rows, warmup = [], None
        rounds = 1 if m['validate'] else m['repeats']+1
        for repeat in range(rounds):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            past = restore(model, snapshots, backend, context+steps, m['exact_split_pages'])
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter()-begin
            restore_check = verify_restored(past, snapshots, backend) if m['validate'] else None
            # Peak includes persistent restored serving state, excludes one-time
            # packing allocations. Reset does not subtract model/cache storage.
            torch.cuda.reset_peak_memory_stats()
            initial_memory = device_memory()
            observed = []
            print(f'Frontier {backend} '+('validation' if m['validate'] else 'warmup' if repeat == 0 else f'repeat {repeat}'), flush=True)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start_event.record()
            for position in range(context, context+steps):
                outputs = model(ids[:, position:position+1], past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                if m['validate']:
                    observed.append(outputs.logits[:, -1].float().cpu())
            end_event.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter()-begin)*1000
            cuda_ms = start_event.elapsed_time(end_event)
            final_memory = device_memory()
            check_cache(past, backend, context, steps)
            if not torch.isfinite(outputs.logits).all():
                raise ValueError('Nonfinite final logits')
            cache = memory(past, backend)
            cache['scope'] = 'Served native cache tensors only; no reference cache on GPU. Persistent native staging reported separately where applicable.'
            record = {'repeat': repeat, 'steps': steps, 'request_tokens': batch*steps,
                'state_upload_and_driver_setup_seconds': setup_seconds, 'cache': cache,
                'initial_memory': initial_memory, 'final_memory': final_memory,
                'initial_state_validation': restore_check}
            if m['validate']:
                reference_steps = [torch.cat([reference[r][s] for r in range(batch)], dim=0) for s in range(steps)]
                labels = ids[:, context+1:context+steps+1].T.cpu()
                comparison = quality.compare_logits(reference_steps, observed, labels,
                    reference_name='Serial HF FP16 eager 4.36.2', candidate_name=f'Batched CPU-restored {backend}',
                    predicted_position_start=context+1,
                    request_windows=quality.request_window_metadata(m['token_provenance'], context, steps, batch))
                base.atomic_json(out/'quality.json', comparison)
                rows.append(record)
            else:
                record.update(wall_ms_per_step=wall_ms/steps, cuda_ms_per_step=cuda_ms/steps,
                              aggregate_tokens_per_second=1000*batch*steps/wall_ms)
                if repeat == 0:
                    warmup = record
                else:
                    rows.append(record)
            del past, outputs, observed
            gc.collect()
            torch.cuda.empty_cache()
    verify(m)
    base.atomic_json(out/'analysis.json', {'backend': backend, 'batch': batch, 'context': context,
        'decode_steps': steps, 'validation_only': m['validate'], 'rows': rows, 'warmup': warmup,
        'preparation': preparation, 'cpu_snapshot_bytes': snapshot_bytes,
        'gpu_after_preparation': post_preparation, 'allocator_budget_bytes': m['allocator_budget_bytes'],
        'quality_sha256': base.sha256_file(out/'quality.json') if m['validate'] else None,
        'transformers': transformers.__version__, 'torch': torch.__version__, 'scope': m['scope']})
    print('Frontier complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--backend', choices=BACKENDS, required=False, default='page_gauge')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--context', type=int, choices=(4096, 20480), default=20480)
    parser.add_argument('--steps', type=int, default=1536)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--offset', type=int, default=472000)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'error': str(error), 'type': type(error).__name__,
                'traceback': traceback.format_exc()})
            raise
        return
    if not 1 <= args.batch <= 32 or not 785 <= args.steps <= 1536 or args.repeats < 1 or args.offset < 0:
        raise ValueError('Unsupported capacity/recurrence configuration')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import benchmark_pg19_external_quality as quality
        source = json.loads((DEFAULT/'manifest.json').read_text())
        for name, evidence in source['input_file_evidence'].items():
            if base.sha256_file(Path(name)) != evidence['sha256']:
                raise ValueError('Changed model/corpus input '+name)
        token_args = SimpleNamespace(context=args.context, decode_steps=args.steps, batch_size=args.batch,
            token_source='wikitext2', wikitext_member='wikitext-2-raw/wiki.train.raw',
            wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip', seed=2026090815,
            token_offset=args.offset, token_stride=23600, model=source['model'])
        tokens, provenance = quality.build_token_matrix(token_args, 32768)
        if provenance['split'] != 'train':
            raise ValueError('Development TRAIN only')
        paths = {Path(p) for p in source['source_sha256']}
        paths.update(Path(__file__).with_name(p) for p in
            ('frontier_run.py', 'frontier_cache.py', 'common_engine.py', 'paged_mistral_adapter.py'))
        selection = ROOT/'results/mlsys2027_baselines_v1/rtx_exact_split_20260908T115408Z_e0315ebe/analysis.json'
        paths.add(selection)
        out = ROOT/'results/mlsys2027_baselines_v1'/('frontier_'+args.backend+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json', {'ids': tokens.tolist()})
        manifest = {'backend': args.backend, 'batch': args.batch, 'context': args.context,
            'decode_steps': args.steps, 'repeats': args.repeats, 'validate': args.validate,
            'seed': token_args.seed, 'model': source['model'], 'token_provenance': provenance,
            'tokens_sha256': base.sha256_file(out/'tokens.json'), 'input_file_evidence': source['input_file_evidence'],
            'source_sha256': {str(p): base.sha256_file(p) for p in sorted(paths)},
            'exact_split_pages': 32, 'allocator_budget_bytes': 28*1024**3,
            'idle': idle, 'orchestrator_pid': os.getpid(),
            'scope': 'Exposed TRAIN development common eager Mistral engine. Independent synchronized requests; native cache policies. Same 28 GiB PyTorch allocator budget, not a cap on non-PyTorch CUDA memory. Serial coherent HF prefill, packing, CPU storage and upload/setup excluded from decode timing and reported separately. Only native serving cache on GPU during decode; no retained FP16 reference. No native-prefill, optimized-system, continuous-serving or final TEST claim. Validation timings are not performance evidence.'}
        base.atomic_json(out/'manifest.json', manifest)
        print('Frontier output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        result = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', result)
        verify(manifest)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise RuntimeError('Frontier worker/exclusivity failed; preserved evidence must distinguish capacity OOM from implementation failure')


if __name__ == '__main__':
    main()
