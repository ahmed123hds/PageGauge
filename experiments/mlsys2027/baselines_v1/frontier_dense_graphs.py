"""TRAIN full-recurrence qualification of identical FI/PG MLP graph dispatch."""
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
sys.path.insert(0, str(ROOT/'experiments/mlsys2027'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
import wsl_gpu_monitor
import frontier_run
import frontier_eager_plan
from kivi_quality import verify, PYTHON
from dense_graph_dispatch import DenseGraph


def worker(out):
    import torch
    m = json.loads((out/'manifest.json').read_text())
    original = frontier_run.restore
    records = []
    originals = []

    def restore(model, *args, **kwargs):
        for module, forward in originals:
            module.forward = forward
        originals.clear()
        past = original(model, *args, **kwargs)
        stats = {'calls': 0, 'oracle_calls': 0, 'max_absolute_error': 0.0}
        for layer in model.model.layers:
            module = layer.mlp
            forward = module.forward
            example = torch.zeros(m['batch'], 1, model.config.hidden_size,
                                  device=next(module.parameters()).device, dtype=torch.float16)
            graph = DenseGraph(module, example)
            calls = [0]

            def dispatch(x, graph=graph, forward=forward, calls=calls):
                result = graph(x)
                if calls[0] in (0, 15, 16, 767, 768, m['decode_steps']-1):
                    expected = forward(x)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    stats['oracle_calls'] += 1
                calls[0] += 1
                stats['calls'] += 1
                return result

            originals.append((module, forward))
            module.forward = dispatch
        records.append(stats)
        return past

    frontier_run.restore = restore
    try:
        frontier_eager_plan.worker(out)
        if len(records) != 1 or records[0]['calls'] != 32*m['decode_steps'] or records[0]['oracle_calls'] != 192:
            raise ValueError('Incomplete dense graph validation')
        result = json.loads((out/'analysis.json').read_text())
        result['dense_graph_validation'] = records
        base.atomic_json(out/'analysis.json', result)
    finally:
        frontier_run.restore = original
        for module, forward in originals:
            module.forward = forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--fixtures', nargs='+', type=Path)
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as e:
            base.atomic_json(args.worker/'failure.json', {'type': type(e).__name__, 'error': str(e), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    for source in args.fixtures or []:
        m = json.loads((source/'manifest.json').read_text())
        c = json.loads((source/'completion.json').read_text())
        verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed'] or m['backend'] not in ('flashinfer_fp16', 'page_gauge'):
            raise ValueError('Completed FI/PG fixture required')
        if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed source tokens')
        idle = base.idle_preflight(0)
        gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = ROOT/'results/mlsys2027_baselines_v1'/('dense_graphs_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
            out.mkdir()
            (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            m.update(validate=True, repeats=0, source_fixture=str(source.resolve()),
                     idle=idle, scope=__doc__+' Validation only, unchanged reference cache policy and kernels, full HF comparison and selected same-input MLP bitwise checks. Not speed evidence or final TEST.')
            for name in ('frontier_dense_graphs.py', 'dense_graph_dispatch.py'):
                path = Path(__file__).with_name(name).resolve()
                m['source_sha256'][str(path)] = base.sha256_file(path)
            base.atomic_json(out/'manifest.json', m)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(out)]
            base.atomic_json(out/'invocation.json', {'command': command})
            print('Dense graph validation: '+str(out), flush=True)
            c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
            base.atomic_json(out/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Dense graph qualification failed; preserved, stop pair')


if __name__ == '__main__':
    main()
