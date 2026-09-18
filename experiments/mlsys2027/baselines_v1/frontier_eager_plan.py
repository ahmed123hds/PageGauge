"""Explicit non-graph planning in the common eager FI/PG body; opt-in only."""
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
import frontier_allocator_probe
from eager_plan_execution import install_oracle, verify_oracle
from kivi_quality import verify, PYTHON


def worker(out):
    import flashinfer
    m = json.loads((out/'manifest.json').read_text())
    original_restore = frontier_run.restore
    original_factory = flashinfer.BatchDecodeWithPagedKVCacheWrapper
    records = []
    def restore(model, snapshots, backend, *args, **kwargs):
        stats = {'backend': backend, 'wrapper_count': 0, 'graph_enabled_count': 0}
        def factory(*factory_args, **factory_kwargs):
            factory_kwargs['use_cuda_graph'] = False
            wrapper = original_factory(*factory_args, **factory_kwargs)
            stats['wrapper_count'] += 1
            stats['graph_enabled_count'] += int(wrapper.is_cuda_graph_enabled)
            return wrapper
        flashinfer.BatchDecodeWithPagedKVCacheWrapper = factory
        try:
            past = original_restore(model, snapshots, backend, *args, **kwargs)
        finally:
            flashinfer.BatchDecodeWithPagedKVCacheWrapper = original_factory
        if stats['wrapper_count'] != (1 if backend == 'flashinfer_fp16' else 2) or stats['graph_enabled_count']:
            raise ValueError('Wrong FI/PG non-graph constructor coverage')
        if m['validate']:
            install_oracle(past[0][0], m['context'], m['decode_steps'], stats)
        records.append(stats)
        return past
    frontier_run.restore = restore
    try:
        frontier_allocator_probe.worker(out)
    finally:
        frontier_run.restore = original_restore
        flashinfer.BatchDecodeWithPagedKVCacheWrapper = original_factory
        base.atomic_json(out/'non_graph_diagnostics.json', {'records': records})
    if len(records) != (1 if m['validate'] else m['repeats']+1):
        raise ValueError('Missing non-graph restore rounds')
    for stats in records:
        verify_oracle(stats, m['validate'])
    r = json.loads((out/'analysis.json').read_text())
    r['non_graph_planning'] = records
    base.atomic_json(out/'analysis.json', r)
    verify(m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    if args.fixture is None:
        raise ValueError('Original source-frozen FI/PG fixture required')
    source = args.fixture.resolve()
    m = json.loads((source/'manifest.json').read_text())
    c = json.loads((source/'completion.json').read_text())
    if m['backend'] not in ('flashinfer_fp16', 'page_gauge') or not c['sampled_exclusivity_passed']:
        raise ValueError('Invalid original FI/PG fixture')
    if c['return_code']:
        f = json.loads((source/'failure.json').read_text())
        if f['type'] != 'OutOfMemoryError' or 'CUDA out of memory' not in f['error']:
            raise ValueError('Unexpected original fixture failure')
    verify(m)
    if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed actual tokens')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_baselines_v1'/('eager_plan_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
        m.update(validate=args.validate, repeats=0 if args.validate else 1,
            allocator_configuration='expandable_segments:True', flashinfer_cuda_graph_planning=False,
            source_fixture=str(source), source_completion_sha256=base.sha256_file(source/'completion.json'),
            idle=idle, orchestrator_pid=os.getpid(),
            scope='Opt-in non-graph FI/PG planning for the common eager body with declared expandable allocator, same 28 GiB budget/native kernels/cache policy. Full HF corpus comparison and selected VRAM-sharded reconstructed-cache oracle when validating. No graph replay, production-default promotion or final speed/quality claim.')
        for name in ('frontier_eager_plan.py', 'eager_plan_execution.py', 'frontier_allocator_probe.py'):
            path = Path(__file__).with_name(name).resolve()
            m['source_sha256'][str(path)] = base.sha256_file(path)
        base.atomic_json(out/'manifest.json', m)
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
        print('Eager planning output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        r = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', r)
        verify(m)
        if r['return_code'] or not r['sampled_exclusivity_passed']:
            raise RuntimeError('Non-graph planning run failed; preserve exact evidence')


if __name__ == '__main__':
    main()
