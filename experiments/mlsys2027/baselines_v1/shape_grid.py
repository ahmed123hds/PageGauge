"""Freeze then execute a fixed TRAIN context/batch grid with retained failures.

Uses the historical S4/A128/T768 reference policy. New regional candidates are
separate experiments; this grid does not silently promote A0 or claim native
best serving performance. No original source files or TEST data are changed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
import wsl_gpu_monitor
from kivi_quality import verify, PYTHON
from common_engine import DEFAULT
from frontier_capacity import parameter_bytes
from shape_contract import CONTEXTS, BATCHES, STEPS, STRIDE, BUDGET, schedule, validate_result


def worker(out):
    m = json.loads((out/'manifest.json').read_text())
    if m['backend'] in ('flashinfer_fp16', 'page_gauge'):
        from frontier_eager_plan import worker as execute
    else:
        from frontier_allocator_probe import worker as execute
    execute(out)


def freeze():
    import benchmark_pg19_external_quality as quality
    source = json.loads((DEFAULT/'manifest.json').read_text())
    verify(source)
    model = Path(source['model'])
    config = json.loads((model/'config.json').read_text())
    if config.get('sliding_window') is not None:
        raise ValueError('Only the retained full-context model is supported')
    headers = parameter_bytes(model)
    cells = schedule(headers['fp16_parameter_bytes'], config['max_position_embeddings'])
    out = ROOT/'results/mlsys2027_baselines_v1'/('shape_grid_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    fixtures = {}
    for context in CONTEXTS:
        args = SimpleNamespace(context=context, decode_steps=STEPS, batch_size=max(BATCHES),
            token_source='wikitext2', wikitext_member='wikitext-2-raw/wiki.train.raw',
            wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip', seed=2026090911,
            token_offset=472000, token_stride=STRIDE, model=str(model))
        ids, provenance = quality.build_token_matrix(args, config['vocab_size'])
        if provenance['split'] != 'train':
            raise ValueError('Only TRAIN development data is authorized here')
        path = out/f'context_{context}_tokens.json'
        base.atomic_json(path, {'ids': ids.tolist()})
        fixtures[str(context)] = {'path': str(path), 'sha256': base.sha256_file(path), 'provenance': provenance}
    paths = {Path(p) for p in source['source_sha256']}
    paths.update(Path(__file__).with_name(n) for n in ('shape_grid.py', 'shape_contract.py',
        'frontier_capacity.py', 'frontier_run.py', 'frontier_cache.py', 'common_engine.py',
        'paged_mistral_adapter.py', 'frontier_allocator_probe.py', 'frontier_eager_plan.py',
        'eager_plan_execution.py', 'SHAPE_GRID_PROTOCOL.md'))
    paths.add(Path(wsl_gpu_monitor.__file__).resolve())
    m = {'cells': cells, 'fixtures': fixtures, 'model': str(model),
        'parameter_header_accounting': headers, 'input_file_evidence': source['input_file_evidence'],
        'source_sha256': {str(p.resolve()): base.sha256_file(p) for p in sorted(paths)},
        'seed': 2026090911, 'decode_steps': STEPS, 'exact_split_pages': 32,
        'allocator_budget_bytes': BUDGET, 'allocator_configuration': 'expandable_segments:True',
        'page_gauge_policy': {'prefix_pages': 4, 'static_suffix_pages': 128, 'tail_tokens': 768},
        'process_monitor': wsl_gpu_monitor.CONTRACT,
        'scope': __doc__+' Fixed-order one-process/cell pilots with one warmup and three repeats; medians/ranges, no CI. Full teacher-forced decoder and native cache maintenance in common eager body. Serial prefill/packing/upload/restore excluded and reported separately; no generated-task or online-serving claim.'}
    base.atomic_json(out/'manifest.json', m)
    print('Frozen shape grid: '+str(out), flush=True)
    print(json.dumps({'cells': len(cells), 'analytic_infeasible': sum(c['action'] != 'run' for c in cells)}))


def run(out):
    import fcntl
    m = json.loads((out/'manifest.json').read_text())
    if (out/'progress.json').exists() or (out/'analysis.json').exists() or (out/'cell_0').exists():
        raise ValueError('Grid already started; diagnose/recover retained cells, never overwrite')
    rows = []
    for cell in m['cells']:
        verify(m)
        if cell['action'] != 'run':
            rows.append(dict(cell, outcome='analytically_infeasible'))
            base.atomic_json(out/'progress.json', {'rows': rows})
            continue
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = out/f"cell_{cell['index']}"; target.mkdir()
            fixture = m['fixtures'][str(cell['context'])]
            if base.sha256_file(Path(fixture['path'])) != fixture['sha256']:
                raise ValueError('Changed frozen actual tokens')
            tokens = json.loads(Path(fixture['path']).read_text())['ids'][:cell['batch']]
            base.atomic_json(target/'tokens.json', {'ids': tokens})
            cm = {**m, **cell, 'idle': idle, 'token_provenance': fixture['provenance'],
                'tokens_sha256': base.sha256_file(target/'tokens.json')}
            # The worker's request metadata uses only the first B declared windows.
            base.atomic_json(target/'manifest.json', cm)
            env = os.environ
            env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            env['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(target)]
            base.atomic_json(target/'invocation.json', {'command': command})
            print('Shape cell: '+json.dumps(cell), flush=True)
            c = wsl_gpu_monitor.run_process(previous, command, target, {'index': 0}, gpu)
            base.atomic_json(target/'completion.json', c)
            verify(cm)
            record = dict(cell, run=str(target), completion_sha256=base.sha256_file(target/'completion.json'))
            if not c['sampled_exclusivity_passed']:
                base.atomic_json(out/'failure.json', dict(record, outcome='monitoring_failure'))
                raise ValueError('Unverified GPU ownership, not performance or OOM evidence')
            if c['return_code']:
                failure = json.loads((target/'failure.json').read_text()) if (target/'failure.json').exists() else {}
                record['failure'] = failure
                if cell['validate'] or failure.get('type') != 'OutOfMemoryError' or 'CUDA out of memory' not in failure.get('error', ''):
                    base.atomic_json(out/'failure.json', record)
                    raise ValueError('Validation/non-capacity failure; diagnose before continuing')
                record['outcome'] = 'measured_cuda_oom'
            else:
                r = json.loads((target/'analysis.json').read_text()); validate_result(r, cm)
                record.update(outcome='validated' if cell['validate'] else 'verified_feasible',
                    analysis_sha256=base.sha256_file(target/'analysis.json'))
                if not cell['validate']:
                    times = [s['wall_ms_per_step'] for s in r['rows']]
                    record.update(wall_ms_per_step=statistics.median(times), range_ms_per_step=[min(times), max(times)],
                        aggregate_tokens_per_second=1000*cell['batch']/statistics.median(times),
                        served_cache_bytes=r['rows'][-1]['cache']['unique_storage_bytes'],
                        peak_allocated_bytes=max(s['final_memory']['peak_allocated_bytes'] for s in r['rows']))
            rows.append(record); base.atomic_json(out/'progress.json', {'rows': rows})
    verify(m)
    base.atomic_json(out/'analysis.json', {'rows': rows, 'scope': m['scope'], 'execution_complete': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', type=Path)
    group.add_argument('--worker', type=Path)
    args = p.parse_args()
    if args.worker:
        try: worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
    elif args.freeze: freeze()
    else: run(args.run.resolve())


if __name__ == '__main__': main()
