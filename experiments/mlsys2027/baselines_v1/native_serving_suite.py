"""Run the four predeclared own-stack native serving pilots serially."""
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


def summarize(raw):
    if raw['warmup']['steps'] != 1536 or len(raw['rows']) != 3:
        raise ValueError('Incomplete native request timing pilot')
    rows = raw['rows']
    expected_layers = 32 if raw['family'] == 'nsn' else 36
    if any(row['steps'] != 1536 or row['final_lengths'] != [22016]*expected_layers for row in rows):
        raise ValueError('Incomplete native recurrence')
    cache_bytes = {row['final_cache']['unique_storage_bytes'] for row in rows}
    if len(cache_bytes) != 1:
        raise ValueError('Cache accounting changed across fixed repeated requests')
    result = {'cache_bytes': cache_bytes.pop()}
    for field in ('prefill_wall_ms', 'decode_wall_ms_per_step', 'timed_segment_sum_ms', 'decode_tokens_per_second'):
        values = [row[field] for row in rows]
        if any(not 0 < value < float('inf') for value in values):
            raise ValueError('Invalid timing value')
        result[field] = {'median': statistics.median(values), 'range': [min(values), max(values)]}
    result['peak_allocated_bytes'] = max(row[phase]['peak_allocated_bytes'] for row in rows for phase in ('prefill_memory', 'decode_memory'))
    return result


def main():
    home = ROOT/'results/mlsys2027_baselines_v1'
    jobs = [dict(family=family, backend=backend, fixture=str(home/fixture))
        for family, fixture, backends in (
            ('nsn', 'nsn_quality_20260908T133908Z_a67611e0', ('hf', 'nsn_int2')),
            ('kitty', 'kitty_quality_20260908T150715Z_8cf039f7', ('hf', 'kitty_pro')))
        for backend in backends]
    hashes = {}
    for job in jobs:
        source = Path(job['fixture'])
        m = json.loads((source/'manifest.json').read_text())
        verify(m)
        hashes.update(m['source_sha256'])
        for name in ('manifest.json', 'analysis.json', 'completion.json', 'tokens.json'):
            hashes[str(source/name)] = base.sha256_file(source/name)
    for path in (Path(__file__), Path(__file__).with_name('native_serving_pilot.py'), Path(__file__).with_name('NATIVE_SERVING_PROTOCOL.md')):
        hashes[str(path.resolve())] = base.sha256_file(path)
    out = home/('native_serving_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    scope = 'Four serial fresh-process native-serving pilots, own-stack HF then native for NSN and Kitty. Full-request warmup then three repeats. Repeat ranges are not CIs; no cross-stack or PageGauge-relative speed claim, final TEST, production maximum or universal superiority.'
    base.atomic_json(out/'manifest.json', dict(jobs=jobs, input_sha256=hashes, scope=scope))
    print('Native serving suite: '+str(out), flush=True)
    rows = []
    for index, job in enumerate(jobs):
        for name, digest in hashes.items():
            if base.sha256_file(Path(name)) != digest:
                raise ValueError('Source/evidence drift: '+name)
        command = [sys.executable, '-u', str(Path(__file__).with_name('native_serving_pilot.py')),
            '--family', job['family'], '--backend', job['backend'], '--fixture', job['fixture']]
        target = None
        with (out/f'job_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Native serving output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'job_{index}_completion.json', dict(job, run=str(target), return_code=code))
        if code or target is None:
            raise ValueError('Native serving pilot failed; diagnose retained failure, no fallback')
        completed = json.loads((target/'completion.json').read_text())
        if completed['return_code'] or not completed['sampled_exclusivity_passed']:
            raise ValueError('Invalid native serving process/exclusivity')
        raw = json.loads((target/'analysis.json').read_text())
        rows.append(dict(job, run=str(target), analysis_sha256=base.sha256_file(target/'analysis.json'), summary=summarize(raw)))
        base.atomic_json(out/'progress.json', {'rows': rows})
    for name, digest in hashes.items():
        if base.sha256_file(Path(name)) != digest:
            raise ValueError('Source/evidence drift: '+name)
    base.atomic_json(out/'analysis.json', dict(rows=rows, scope=scope))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
