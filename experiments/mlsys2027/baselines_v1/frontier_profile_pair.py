"""Matched FI/PG attribution of the completed B4 common-engine timing pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def category(name):
    if 'BatchPrefillWithPagedKVCacheKernel' in name:
        return 'int8_history_attention' if 'PageAffineInt8Attention' in name else 'fp16_attention'
    if 'MergeState' in name:
        return 'merge'
    if 'rope_append' in name.lower():
        return 'rope_append'
    if 'gemv' in name.lower() or 'gemm' in name.lower():
        return 'projection_and_lm_matmul'
    return 'other'


def reduce_run(directory):
    m = json.loads((directory/'manifest.json').read_text())
    a = json.loads((directory/'analysis.json').read_text())
    c = json.loads((directory/'completion.json').read_text())
    if not a['profile_only'] or c['return_code'] or not c['sampled_exclusivity_passed']:
        raise ValueError('Invalid profile worker')
    p = json.loads((directory/'profile.json').read_text())
    if base.sha256_file(directory/'profile.json') != a['profile_sha256'] or base.sha256_file(directory/'trace.json') != a['trace_sha256']:
        raise ValueError('Profile evidence changed')
    if p['counts'] != {'steps': 1536, 'starts': 1, 'stops': 1, 'restores': 1} or p['profiled_steps'] != 129:
        raise ValueError('Unmatched profiler window')
    gpu = [event for event in p['events'] if event['device_type'] == 'DeviceType.CUDA']
    if not gpu:
        raise ValueError('Missing GPU attribution')
    groups, counts = {}, {}
    for event in gpu:
        key = category(event['name'])
        groups[key] = groups.get(key, 0)+event['self_device_us']/129000
        counts[key] = counts.get(key, 0)+event['count']/129
    cpu = [event for event in p['events'] if event['device_type'] == 'DeviceType.CPU']
    launches = [event for event in cpu if event['name'] in ('cudaLaunchKernel', 'cudaLaunchKernelExC', 'cudaGraphLaunch')]
    return {'run': str(directory), 'profile_sha256': a['profile_sha256'], 'trace_sha256': a['trace_sha256'],
        'tokens_sha256': m['tokens_sha256'], 'gpu_self_ms_per_step': groups,
        'gpu_total_self_ms_per_step': sum(groups.values()), 'gpu_events_per_step': counts,
        'cpu_launch_api_calls_per_step': sum(event['count'] for event in launches)/129,
        'instrumented_cpu_launch_self_ms_per_step': sum(event['self_cpu_us'] for event in launches)/129000,
        'largest_cpu_self_events': sorted(cpu, key=lambda event: event['self_cpu_us'], reverse=True)[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    source = json.loads((pilot/'analysis.json').read_text())
    m = json.loads((pilot/'manifest.json').read_text())
    if source['batch'] != 4 or set(source['rows']) != {'flashinfer_fp16', 'page_gauge', 'kivi_int4', 'kivi_int2', 'bitdecode_int4'}:
        raise ValueError('Complete five-method B4 timing cohort required')
    hashes = dict(m['source_sha256'])
    for path in (Path(__file__), Path(__file__).with_name('frontier_profile.py')):
        hashes[str(path.resolve())] = base.sha256_file(path)
    out = ROOT/'results/mlsys2027_baselines_v1'/('frontier_profile_pair_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'pilot': str(pilot), 'pilot_sha256': base.sha256_file(pilot/'analysis.json'),
        'source_sha256': hashes, 'order': ['flashinfer_fp16', 'page_gauge'], 'orchestrator_pid': os.getpid(),
        'scope': 'Instrumented attribution only; not additional speed-acceptance blocks.'})
    print('Frontier profile pair: '+str(out), flush=True)
    rows = {}
    for backend in ('flashinfer_fp16', 'page_gauge'):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source drift')
        fixture = Path(source['rows'][backend]['run'])
        if base.sha256_file(fixture/'analysis.json') != source['rows'][backend]['analysis_sha256']:
            raise ValueError('Changed timed source')
        command = [sys.executable, '-u', str(Path(__file__).with_name('frontier_profile.py')), '--fixture', str(fixture)]
        target = None
        with (out/(backend+'.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Frontier profile output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/(backend+'_completion.json'), {'run': str(target), 'return_code': code, 'command': command})
        if code or target is None:
            raise ValueError('Profile worker failed; preserve evidence')
        rows[backend] = reduce_run(target)
    for path, digest in hashes.items():
        if base.sha256_file(Path(path)) != digest:
            raise ValueError('Source drift')
    if len({row['tokens_sha256'] for row in rows.values()}) != 1:
        raise ValueError('Different actual token matrices')
    base.atomic_json(out/'analysis.json', {'rows': rows,
        'scope': 'Matched B4/C20480/D1536 common-engine profiles, last129steps. Kernel self times and instrumented CPU API rows are separate, overlapping clocks; not wall latency or final CIs. Chrome traces retained for launch gaps/overlap.'})
    print(json.dumps({key: {field: row[field] for field in ('gpu_self_ms_per_step', 'cpu_launch_api_calls_per_step')} for key, row in rows.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
