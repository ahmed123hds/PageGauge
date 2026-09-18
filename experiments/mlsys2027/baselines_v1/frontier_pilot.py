"""Five CPU-backed fixed-batch common-engine timing pilots; no final CI."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    validation = args.validation.resolve()
    result = json.loads((validation/'analysis.json').read_text())
    manifest = json.loads((validation/'manifest.json').read_text())
    order = ['flashinfer_fp16', 'kivi_int4', 'page_gauge', 'bitdecode_int4', 'kivi_int2']
    if not result['initial_state_validation_passed'] or set(result['rows']) != set(order):
        raise ValueError('Complete five-backend restored-state validation required')
    for row in result['rows'].values():
        path = Path(row['run'])
        if base.sha256_file(path/'analysis.json') != row['analysis_sha256'] or base.sha256_file(path/'quality.json') != row['quality_sha256']:
            raise ValueError('Validation evidence changed')
    hashes = dict(manifest['source_sha256'])
    hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    out = ROOT/'results/mlsys2027_baselines_v1'/('frontier_pilot_b'+str(args.batch)+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'order': order, 'batch': args.batch,
        'context': 20480, 'decode_steps': 1536, 'timed_repeats': 3, 'warmup_steps': 1536,
        'validation': str(validation), 'validation_sha256': base.sha256_file(validation/'analysis.json'),
        'source_sha256': hashes, 'allocator_budget_bytes': 28*1024**3, 'orchestrator_pid': os.getpid(),
        'scope': 'Single matched TRAIN workload, five fresh processes, three within-process repeats. Common eager model engine, CPU-backed initial caches, 28 GiB allocator budget. Point estimates and repeat range only, not hierarchical CIs or optimized/prefill-inclusive serving.'})
    print('Frontier timing pilot: '+str(out), flush=True)
    rows, tokens_hash = {}, None
    for index, backend in enumerate(order):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source drift '+path)
        command = [sys.executable, '-u', str(Path(__file__).with_name('frontier_run.py')),
            '--backend', backend, '--batch', str(args.batch), '--context', '20480', '--steps', '1536', '--repeats', '3']
        target = None
        with (out/f'{index}_{backend}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Frontier output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'{index}_{backend}_completion.json', {'run': str(target), 'return_code': code, 'command': command})
        if code or target is None:
            raise ValueError('Timing pilot failed; diagnose preserved evidence')
        completion = json.loads((target/'completion.json').read_text())
        m = json.loads((target/'manifest.json').read_text())
        r = json.loads((target/'analysis.json').read_text())
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Invalid process/exclusivity')
        actual = base.sha256_file(target/'tokens.json')
        tokens_hash = tokens_hash or actual
        if actual != tokens_hash or m['tokens_sha256'] != actual:
            raise ValueError('Unmatched actual token IDs')
        if r['validation_only'] or r['batch'] != args.batch or r['decode_steps'] != 1536 or len(r['rows']) != 3 or r['warmup']['steps'] != 1536:
            raise ValueError('Incomplete fixed-batch timing')
        wall = [v['wall_ms_per_step'] for v in r['rows']]
        rows[backend] = {'run': str(target), 'analysis_sha256': base.sha256_file(target/'analysis.json'),
            'wall_ms_per_step_median': statistics.median(wall), 'wall_repeat_range_ms': [min(wall), max(wall)],
            'cuda_ms_per_step_median': statistics.median(v['cuda_ms_per_step'] for v in r['rows']),
            'aggregate_tokens_per_second_median': statistics.median(v['aggregate_tokens_per_second'] for v in r['rows']),
            'cache': r['rows'][-1]['cache'], 'cpu_snapshot_bytes': r['cpu_snapshot_bytes'],
            'decode_peak_allocated_bytes': max(v['final_memory']['peak_allocated_bytes'] for v in r['rows']),
            'decode_peak_reserved_bytes': max(v['final_memory']['peak_reserved_bytes'] for v in r['rows'])}
        base.atomic_json(out/'progress.json', {'rows': rows})
    for path, digest in hashes.items():
        if base.sha256_file(Path(path)) != digest:
            raise ValueError('Source drift')
    for row in rows.values():
        row['wall_latency_over_page_gauge'] = row['wall_ms_per_step_median']/rows['page_gauge']['wall_ms_per_step_median']
    base.atomic_json(out/'analysis.json', {'rows': rows, 'batch': args.batch, 'tokens_sha256': tokens_hash,
        'scope': 'Common eager fixed-batch development pilot. No retained GPU reference cache. Single matched TRAIN fixture, one fresh process per backend, three timing repeats; no hierarchical CI, native-best engine, or continuous-serving claim.'})
    print(json.dumps({k: {f: r[f] for f in ('wall_ms_per_step_median', 'aggregate_tokens_per_second_median', 'wall_latency_over_page_gauge')} for k, r in rows.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
