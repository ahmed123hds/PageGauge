"""Full TRAIN validation of combined attention-body and MLP graphs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
import uuid
import frontier_dense_graphs as dense
from frontier_dense_graphs import ROOT, base, previous, wsl_gpu_monitor, verify, PYTHON
import frontier_run
import frontier_eager_plan
import attention_graph_dispatch as attention


def worker(out):
    m = json.loads((out/'manifest.json').read_text())
    original_restore = frontier_run.restore
    original_worker = frontier_eager_plan.worker
    records = []

    def restore(model, *args, **kwargs):
        past = original_restore(model, *args, **kwargs)
        records.append(attention.prepare(past[0][0], m['context'], m['decode_steps'], True))
        return past

    frontier_run.restore = restore
    # Deliberately use normal graph-compatible wrapper construction. The dense
    # validation installs only MLP dispatch; do not install non-graph planning.
    frontier_eager_plan.worker = frontier_run.worker
    try:
        dense.worker(out)
        if len(records) != 1:
            raise ValueError('Missing attention graph round')
        attention.verify_stats(records[0], m['decode_steps'], 32, True)
        r = json.loads((out/'analysis.json').read_text())
        r['attention_graph_dispatch'] = records
        base.atomic_json(out/'analysis.json', r)
    finally:
        frontier_run.restore = original_restore
        frontier_eager_plan.worker = original_worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--fixtures', nargs=2, type=Path)
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as e:
            base.atomic_json(args.worker/'failure.json', {'type': type(e).__name__, 'error': str(e), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    for source in args.fixtures:
        m = json.loads((source/'manifest.json').read_text())
        c = json.loads((source/'completion.json').read_text())
        verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed'] or m['backend'] not in ('flashinfer_fp16', 'page_gauge'):
            raise ValueError('Valid FI/PG source required')
        if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed tokens')
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = ROOT/'results/mlsys2027_baselines_v1'/('combined_graphs_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
            out.mkdir()
            (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            m.update(validate=True, repeats=0, idle=idle, source_fixture=str(source.resolve()),
                     graph_planning=True, scope=__doc__+' Original reference cache policy/math. Same-cache attention oracle, bitwise MLP checks, HF corpus comparison. No timing or final TEST claim.')
            for module in (dense, attention):
                path = Path(module.__file__).resolve()
                m['source_sha256'][str(path)] = base.sha256_file(path)
            m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
            base.atomic_json(out/'manifest.json', m)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(out)]
            base.atomic_json(out/'invocation.json', {'command': command})
            print('Combined graph validation: '+str(out), flush=True)
            c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
            base.atomic_json(out/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Combined validation failed; preserve and stop')


if __name__ == '__main__':
    main()
