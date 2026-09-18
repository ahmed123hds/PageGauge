"""Attention-only graph integration on an unchanged CPU-backed FI/PG fixture."""
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
import attention_graph_dispatch
from kivi_quality import verify, PYTHON


def worker(out):
    m = json.loads((out/'manifest.json').read_text())
    if m['backend'] not in ('flashinfer_fp16', 'page_gauge') or m['attention_graph_mode'] != 'all_prepared_structural_banks':
        raise ValueError('Matched FI/PG attention graph mode required')
    original = frontier_run.restore
    records = []

    def restore(model, *args, **kwargs):
        past = original(model, *args, **kwargs)
        records.append(attention_graph_dispatch.prepare(past[0][0], m['context'], m['decode_steps'], m['validate']))
        return past

    frontier_run.restore = restore
    try:
        frontier_run.worker(out)
    finally:
        frontier_run.restore = original
    expected = 1 if m['validate'] else m['repeats']+1
    if len(records) != expected:
        raise ValueError('Missing restored graph rounds')
    for stats in records:
        attention_graph_dispatch.verify_stats(stats, m['decode_steps'], 32, m['validate'])
    result = json.loads((out/'analysis.json').read_text())
    result['attention_graph_dispatch'] = records
    base.atomic_json(out/'analysis.json', result)
    verify(m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    if args.fixture is None:
        raise ValueError('Completed original-dispatch fixture required')
    source = args.fixture.resolve()
    m = json.loads((source/'manifest.json').read_text())
    c = json.loads((source/'completion.json').read_text())
    if c['return_code'] or not c['sampled_exclusivity_passed'] or m['backend'] not in ('flashinfer_fp16', 'page_gauge'):
        raise ValueError('Invalid FI/PG source fixture')
    if m['batch'] != 4 or m['context'] not in (4096, 20480):
        raise ValueError('Only audited batch-four fixtures supported')
    verify(m)
    if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed actual tokens')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_baselines_v1'/('attention_graphs_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
        m.update(validate=args.validate, repeats=3, attention_graph_mode='all_prepared_structural_banks',
            source_fixture=str(source), source_analysis_sha256=base.sha256_file(source/'analysis.json'),
            idle=idle, orchestrator_pid=os.getpid(),
            scope='Opt-in attention-only CUDA graph dispatch for both FI/PG in the same ungraphed dense Mistral body. Original kernels, quantizers, cache policies, append and planning. All structural banks prepared outside decode timing, setup/memory retained. Same-cache execution oracle plus corpus metrics when validation is enabled. No packed dense projections or complete-layer graphs; no native-best serving/final TEST claim.')
        for path in (Path(__file__), Path(__file__).with_name('attention_graph_dispatch.py')):
            m['source_sha256'][str(path.resolve())] = base.sha256_file(path)
        base.atomic_json(out/'manifest.json', m)
        print('Attention graph output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        result = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', result)
        verify(m)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise ValueError('Attention graph execution/exclusivity failed; preserve evidence')


if __name__ == '__main__':
    main()
