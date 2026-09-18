"""Isolated allocator-policy diagnosis of a retained capacity fixture."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
import frontier_run
from frontier_cache import verify_restored
from kivi_quality import verify, PYTHON


def worker(out):
    import torch
    m = json.loads((out/'manifest.json').read_text())
    if os.environ.get('PYTORCH_ALLOC_CONF') != m['allocator_configuration'] or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        raise ValueError('Allocator configuration differs from frozen manifest')
    original_restore, original_memory = frontier_run.restore, frontier_run.device_memory
    checks = []
    def restore(model, snapshots, backend, *args, **kwargs):
        past = original_restore(model, snapshots, backend, *args, **kwargs)
        checks.append(verify_restored(past, snapshots, backend))
        return past
    def memory():
        result = original_memory()
        raw = torch.cuda.memory_stats()
        result['allocator_stats'] = {key: raw.get(key) for key in
            ('inactive_split_bytes.all.current', 'inactive_split_bytes.all.peak',
             'active_bytes.all.current', 'num_alloc_retries', 'num_ooms')}
        return result
    frontier_run.restore, frontier_run.device_memory = restore, memory
    try:
        frontier_run.worker(out)
        if len(checks) != m['repeats']+1 or not all(c['bitwise_initial_state_match'] for c in checks):
            raise ValueError('Missing allocator-probe restored-state checks')
    finally:
        diagnostics = {'allocator_configuration': m['allocator_configuration'], 'restored_state_checks': checks,
            'scope': 'Allocator state only; source kernels, cache policies, math and 28 GiB budget unchanged.'}
        if torch.cuda.is_initialized():
            diagnostics['memory'] = memory()
            segments = torch.cuda.memory_snapshot()
            diagnostics['segment_count'] = len(segments)
            diagnostics['expandable_segment_count'] = sum(s.get('is_expandable', False) for s in segments)
            base.atomic_json(out/'allocator_snapshot.json', segments)
            diagnostics['snapshot_sha256'] = base.sha256_file(out/'allocator_snapshot.json')
        base.atomic_json(out/'allocator_diagnostics.json', diagnostics)
        frontier_run.restore, frontier_run.device_memory = original_restore, original_memory
    verify(m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--allocator', choices=('default', 'expandable'), default='default')
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    if args.fixture is None:
        raise ValueError('Retained original capacity fixture required')
    source = args.fixture.resolve()
    m = json.loads((source/'manifest.json').read_text())
    c = json.loads((source/'completion.json').read_text())
    if not c['sampled_exclusivity_passed'] or m['allocator_budget_bytes'] != 28*1024**3 or m['validate']:
        raise ValueError('Invalid original capacity fixture')
    if c['return_code']:
        f = json.loads((source/'failure.json').read_text())
        if f['type'] != 'OutOfMemoryError' or 'CUDA out of memory' not in f['error']:
            raise ValueError('Non-capacity failure must be diagnosed separately')
    verify(m)
    if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed actual tokens')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_baselines_v1'/('allocator_'+args.allocator+'_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
        m.update(allocator_configuration='expandable_segments:'+('True' if args.allocator == 'expandable' else 'False'),
            repeats=1, source_fixture=str(source), source_completion_sha256=base.sha256_file(source/'completion.json'),
            idle=idle, orchestrator_pid=os.getpid(),
            scope='Exposed TRAIN allocator-policy diagnosis. Same original kernels/cache policies/math, full recurrence warmup+one timed repeat, unchanged 28 GiB allocator budget. Bitwise initial state checked after each restore. Not final quality, speed CI or policy promotion.')
        m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
        base.atomic_json(out/'manifest.json', m)
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
        print('Allocator probe output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command, 'allocator_configuration': m['allocator_configuration']})
        result = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', result)
        verify(m)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise RuntimeError('Allocator probe did not complete; preserve and inspect exact failure')


if __name__ == '__main__':
    main()
