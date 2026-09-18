"""Matched attention-graph timing pilot, only after both full validations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
from kivi_quality import verify
from attention_graph_dispatch import verify_stats


def read_run(path, validation):
    m = json.loads((path/'manifest.json').read_text())
    c = json.loads((path/'completion.json').read_text())
    r = json.loads((path/'analysis.json').read_text())
    if c['return_code'] or not c['sampled_exclusivity_passed'] or r['validation_only'] != validation:
        raise ValueError('Incomplete/wrong-mode graph evidence')
    if (m['batch'], m['context'], m['decode_steps']) != (4, 20480, 1536):
        raise ValueError('Full matched B4 recurrence required')
    verify(m)
    if base.sha256_file(path/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Actual tokens changed')
    rounds = r['rows'] if validation else [r['warmup']]+r['rows']
    if len(rounds) != (1 if validation else 4) or len(r['attention_graph_dispatch']) != len(rounds):
        raise ValueError('Missing graph/warmup/timing rounds')
    for row, stats in zip(rounds, r['attention_graph_dispatch']):
        if row['steps'] != 1536:
            raise ValueError('Incomplete recurrence')
        if validation and not row['initial_state_validation']['bitwise_initial_state_match']:
            raise ValueError('Initial cache mismatch')
        verify_stats(stats, 1536, 32, validation)
    if validation and base.sha256_file(path/'quality.json') != r['quality_sha256']:
        raise ValueError('Changed predictive evidence')
    return m, r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fi-validation', type=Path, required=True)
    p.add_argument('--pg-validation', type=Path, required=True)
    args = p.parse_args()
    validation = {'flashinfer_fp16': args.fi_validation.resolve(), 'page_gauge': args.pg_validation.resolve()}
    manifests, hashes = {}, {str(Path(__file__).resolve()): base.sha256_file(Path(__file__))}
    for backend, path in validation.items():
        m, r = read_run(path, True)
        if m['backend'] != backend:
            raise ValueError('Wrong validation backend')
        manifests[backend] = m
        hashes.update(m['source_sha256'])
        for name in ('manifest.json', 'analysis.json', 'completion.json', 'quality.json'):
            hashes[str(path/name)] = base.sha256_file(path/name)
    fi, pg = manifests.values()
    for key in ('tokens_sha256', 'model', 'allocator_budget_bytes', 'exact_split_pages'):
        if fi[key] != pg[key]:
            raise ValueError('Unmatched '+key)
    out = ROOT/'results/mlsys2027_baselines_v1'/('graph_pair_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    scope = ('Matched TRAIN attention-only graph development pilot: one fresh process per arm, '
             'full recurrence warmup and three repeats. Both arms use original common dense body. '
             'Point estimates/ranges, not final confidence intervals or native-best serving.')
    base.atomic_json(out/'manifest.json', {'order': list(validation), 'validation': {k: str(v) for k, v in validation.items()},
        'input_sha256': hashes, 'tokens_sha256': fi['tokens_sha256'], 'scope': scope})
    print('Graph timing pair: '+str(out), flush=True)
    rows = {}
    for backend, m in manifests.items():
        for name, digest in hashes.items():
            if base.sha256_file(Path(name)) != digest:
                raise ValueError('Evidence/source drift: '+name)
        command = [sys.executable, '-u', str(Path(__file__).with_name('frontier_attention_graphs.py')),
                   '--fixture', m['source_fixture']]
        target = None
        with (out/(backend+'.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Attention graph output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/(backend+'_completion.json'), {'run': str(target), 'return_code': code, 'command': command})
        if code or target is None:
            raise ValueError('Graph pilot failure retained; diagnose before continuing')
        actual, r = read_run(target, False)
        if actual['tokens_sha256'] != fi['tokens_sha256'] or actual['source_sha256'] != m['source_sha256']:
            raise ValueError('Unmatched timing source/tokens')
        times = [v['wall_ms_per_step'] for v in r['rows']]
        rows[backend] = {'run': str(target), 'analysis_sha256': base.sha256_file(target/'analysis.json'),
            'wall_ms_per_step_median': statistics.median(times), 'repeat_range_ms': [min(times), max(times)],
            'aggregate_tokens_per_second_median': statistics.median(v['aggregate_tokens_per_second'] for v in r['rows']),
            'cache_bytes': r['rows'][-1]['cache']['unique_storage_bytes'],
            'decode_peak_allocated_bytes': max(v['final_memory']['peak_allocated_bytes'] for v in r['rows']),
            'graph_setup_seconds': [s['total_graph_setup_seconds'] for s in r['attention_graph_dispatch']],
            'graph_device_memory_delta_bytes': [s['graph_setup_device_used_delta_bytes'] for s in r['attention_graph_dispatch']]}
        base.atomic_json(out/'progress.json', {'rows': rows})
    for name, digest in hashes.items():
        if base.sha256_file(Path(name)) != digest:
            raise ValueError('Evidence/source drift: '+name)
    speedup = rows['flashinfer_fp16']['wall_ms_per_step_median']/rows['page_gauge']['wall_ms_per_step_median']
    base.atomic_json(out/'analysis.json', {'rows': rows, 'flashinfer_over_page_gauge': speedup, 'scope': scope})
    print(json.dumps({'rows': rows, 'flashinfer_over_page_gauge': speedup}, indent=2), flush=True)


if __name__ == '__main__':
    main()
