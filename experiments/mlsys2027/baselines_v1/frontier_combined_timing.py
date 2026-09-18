"""Combined graph point pilot; correctness precedes timing, no final CI."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import traceback
import uuid
import frontier_dense_timing as dense
from frontier_dense_timing import ROOT, base, previous, wsl_gpu_monitor, verify, PYTHON
import frontier_run
import frontier_eager_plan
import attention_graph_dispatch as attention


def worker(out):
    m = json.loads((out/'manifest.json').read_text())
    original_restore, original_worker = frontier_run.restore, frontier_eager_plan.worker
    records = []
    def restore(model, *args, **kwargs):
        past = original_restore(model, *args, **kwargs)
        records.append(attention.prepare(past[0][0], m['context'], m['decode_steps'], False))
        return past
    frontier_run.restore = restore
    frontier_eager_plan.worker = frontier_run.worker
    try:
        dense.worker(out)
        if len(records) != 4:
            raise ValueError('Missing graph rounds')
        for record in records:
            attention.verify_stats(record, 1536, 32, False)
        r = json.loads((out/'analysis.json').read_text())
        r['attention_graph_dispatch'] = records
        base.atomic_json(out/'analysis.json', r)
    finally:
        frontier_run.restore, frontier_eager_plan.worker = original_restore, original_worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--validations', nargs=2, type=Path)
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as e:
            base.atomic_json(args.worker/'failure.json', {'type': type(e).__name__, 'error': str(e), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    sources = [s.resolve() for s in args.validations]
    manifests = [dense.check(s) for s in sources]
    if [m['backend'] for m in manifests] != ['flashinfer_fp16', 'page_gauge']:
        raise ValueError('FI then PG required')
    for key in ('tokens_sha256', 'model', 'batch', 'context', 'decode_steps', 'allocator_budget_bytes'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Unmatched '+key)
    for s in sources:
        r = json.loads((s/'analysis.json').read_text())
        if len(r['attention_graph_dispatch']) != 1:
            raise ValueError('Missing attention validation')
        attention.verify_stats(r['attention_graph_dispatch'][0], 1536, 32, True)
    out = ROOT/'results/mlsys2027_baselines_v1'/('combined_pair_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir()
    base.atomic_json(out/'manifest.json', {'validation_sha256': {str(s/n): base.sha256_file(s/n) for s in sources for n in ('manifest.json', 'analysis.json', 'completion.json')},
        'scope': __doc__+' Fixed FI/PG order, one process each, warmup plus three repeats. Capture/preparation excluded, copies and dispatch included.'})
    print('Combined timing pair: '+str(out), flush=True)
    rows = {}
    for source, m in zip(sources, manifests):
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = out/m['backend']; target.mkdir()
            (target/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            m.update(validate=False, repeats=3, idle=idle, scope=__doc__, validation_source=str(source))
            for module in (dense, attention):
                path = Path(module.__file__).resolve()
                m['source_sha256'][str(path)] = base.sha256_file(path)
            m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
            base.atomic_json(target/'manifest.json', m)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(target)]
            base.atomic_json(target/'invocation.json', {'command': command})
            c = wsl_gpu_monitor.run_process(previous, command, target, {'index': 0}, gpu)
            base.atomic_json(target/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Combined timing failed; retain and stop')
            r = json.loads((target/'analysis.json').read_text())
            times = [v['wall_ms_per_step'] for v in r['rows']]
            rows[m['backend']] = {'median_ms': statistics.median(times), 'range_ms': [min(times), max(times)], 'analysis_sha256': base.sha256_file(target/'analysis.json')}
            base.atomic_json(out/'progress.json', rows)
    base.atomic_json(out/'analysis.json', {'rows': rows, 'flashinfer_over_page_gauge': rows['flashinfer_fp16']['median_ms']/rows['page_gauge']['median_ms'], 'scope': __doc__})


if __name__ == '__main__':
    main()
