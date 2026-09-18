"""Uninstrumented FI/PG MLP graph pilot after retained recurrence validation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
import uuid
import frontier_dense_graphs as validation
from frontier_dense_graphs import ROOT, base, previous, wsl_gpu_monitor, verify, PYTHON
import frontier_run
import frontier_eager_plan
from dense_graph_dispatch import DenseGraph


def worker(out):
    import torch
    m = json.loads((out/'manifest.json').read_text())
    original = frontier_run.restore
    originals, records = [], []

    def restore(model, *args, **kwargs):
        for module, forward in originals:
            module.forward = forward
        originals.clear()
        past = original(model, *args, **kwargs)
        graphs = []
        for layer in model.model.layers:
            module = layer.mlp
            example = torch.zeros(m['batch'], 1, model.config.hidden_size,
                                  device=next(module.parameters()).device, dtype=torch.float16)
            graph = DenseGraph(module, example)
            originals.append((module, module.forward))
            module.forward = graph
            graphs.append(graph)
        records.append(graphs)
        return past

    frontier_run.restore = restore
    try:
        frontier_eager_plan.worker(out)
        counts = [[g.calls for g in graphs] for graphs in records]
        if len(counts) != m['repeats']+1 or any(len(row) != 32 or any(c != m['decode_steps'] for c in row) for row in counts):
            raise ValueError('Incomplete MLP graph recurrence')
        r = json.loads((out/'analysis.json').read_text())
        r['dense_graph_calls'] = counts
        base.atomic_json(out/'analysis.json', r)
    finally:
        frontier_run.restore = original
        for module, forward in originals:
            module.forward = forward


def check(source):
    m = json.loads((source/'manifest.json').read_text())
    c = json.loads((source/'completion.json').read_text())
    r = json.loads((source/'analysis.json').read_text())
    verify(m)
    if c['return_code'] or not c['sampled_exclusivity_passed'] or not c['own_pid_seen']:
        raise ValueError('Unverified validation worker')
    for name, field in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
        if base.sha256_file(source/name) != c[field]:
            raise ValueError('Changed monitoring evidence')
    if not m['validate'] or not r['validation_only'] or r['dense_graph_validation'] != [{'calls': 49152, 'oracle_calls': 192, 'max_absolute_error': 0.0}]:
        raise ValueError('Full MLP qualification required')
    if base.sha256_file(source/'quality.json') != r['quality_sha256'] or base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed quality or tokens')
    return m


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
    manifests = [check(s) for s in sources]
    if [m['backend'] for m in manifests] != ['flashinfer_fp16', 'page_gauge']:
        raise ValueError('Declared FI then PG pair required')
    for key in ('tokens_sha256', 'model', 'batch', 'context', 'decode_steps', 'allocator_budget_bytes'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Unmatched '+key)
    pair = ROOT/'results/mlsys2027_baselines_v1'/('dense_pair_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    pair.mkdir()
    evidence = {str(s/n): base.sha256_file(s/n) for s in sources for n in ('manifest.json', 'analysis.json', 'completion.json', 'quality.json')}
    base.atomic_json(pair/'manifest.json', {'validation_sha256': evidence, 'order': ['flashinfer_fp16', 'page_gauge'],
        'scope': 'One fresh process per arm, full warmup and three repeats, fixed order, medians/ranges only. MLP graphs with original eager attention and unchanged dense weights. Not serving or CI.'})
    print('Dense timing pair: '+str(pair), flush=True)
    rows = {}
    for source, m in zip(sources, manifests):
        check(source)
        idle = base.idle_preflight(0)
        gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = pair/m['backend']; out.mkdir()
            (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            m.update(validate=False, repeats=3, idle=idle, validation_source=str(source), scope='Dense MLP graph timing pilot; no oracle in timed loop. Prefill/restoration/capture excluded and reported as preparation. No CI or final TEST.')
            m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
            base.atomic_json(out/'manifest.json', m)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(out)]
            base.atomic_json(out/'invocation.json', {'command': command})
            c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
            base.atomic_json(out/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Timing worker failed; retained')
            r = json.loads((out/'analysis.json').read_text())
            times = [v['wall_ms_per_step'] for v in r['rows']]
            rows[m['backend']] = {'median_ms': statistics.median(times), 'range_ms': [min(times), max(times)], 'analysis_sha256': base.sha256_file(out/'analysis.json')}
            base.atomic_json(pair/'progress.json', rows)
    base.atomic_json(pair/'analysis.json', {'rows': rows, 'flashinfer_over_page_gauge': rows['flashinfer_fp16']['median_ms']/rows['page_gauge']['median_ms'], 'scope': 'Development point pilot only, no CI.'})


if __name__ == '__main__':
    main()
