"""Predeclared default/expandable diagnosis for four failed native grid cells."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
from kivi_quality import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=Path, required=True)
    args = p.parse_args()
    grid = args.grid.resolve()
    r = json.loads((grid/'analysis.json').read_text())
    order = [('kivi_int4', 8), ('kivi_int4', 16), ('bitdecode_int4', 16), ('kivi_int2', 16)]
    sources, hashes = [], {}
    for backend, batch in order:
        cells = [v for v in r['rows'] if (v['backend'], v['batch']) == (backend, batch)]
        if len(cells) != 1 or cells[0]['outcome'] != 'measured_cuda_oom':
            raise ValueError('Expected original measured native OOM cell')
        source = Path(cells[0]['run'])
        m = json.loads((source/'manifest.json').read_text())
        verify(m)
        hashes.update(m['source_sha256'])
        for name in ('manifest.json', 'completion.json', 'failure.json', 'tokens.json'):
            hashes[str(source/name)] = base.sha256_file(source/name)
        sources.append({'backend': backend, 'batch': batch, 'fixture': str(source)})
    for path in (grid/'analysis.json', Path(__file__), Path(__file__).with_name('frontier_allocator_probe.py')):
        hashes[str(path.resolve())] = base.sha256_file(path)
    out = ROOT/'results/mlsys2027_baselines_v1'/('native_allocator_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    jobs = [dict(source, allocator=policy) for source in sources for policy in ('default', 'expandable')]
    scope = ('All four originally failed native capacity cells, explicit default then expandable per cell. '
             'Unchanged 28 GiB budget, full recurrence warmup and one timing repeat on success. '
             'Failures retained; not a best-allocator search, final timing CI, PageGauge speed comparison or maximum capacity.')
    base.atomic_json(out/'manifest.json', {'jobs': jobs, 'input_sha256': hashes, 'scope': scope})
    print('Native allocator suite: '+str(out), flush=True)
    rows = []
    for index, job in enumerate(jobs):
        for name, digest in hashes.items():
            if base.sha256_file(Path(name)) != digest:
                raise ValueError('Source/evidence drift: '+name)
        command = [sys.executable, '-u', str(Path(__file__).with_name('frontier_allocator_probe.py')),
            '--fixture', job['fixture'], '--allocator', job['allocator']]
        target = None
        with (out/f'job_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Allocator probe output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        row = dict(job, run=str(target), return_code=code)
        base.atomic_json(out/f'job_{index}_completion.json', row)
        if target is None or not (target/'completion.json').exists():
            raise ValueError('Host/preflight failure; not a capacity outcome')
        completed = json.loads((target/'completion.json').read_text())
        if not completed['sampled_exclusivity_passed']:
            raise ValueError('Invalid GPU exclusivity')
        d = json.loads((target/'allocator_diagnostics.json').read_text())
        row.update(allocator_diagnostics_sha256=base.sha256_file(target/'allocator_diagnostics.json'),
                   allocator_memory=d.get('memory'), expandable_segment_count=d.get('expandable_segment_count'))
        if code:
            failure = json.loads((target/'failure.json').read_text())
            if failure['type'] != 'OutOfMemoryError' or 'CUDA out of memory' not in failure['error']:
                base.atomic_json(out/'failure.json', dict(row, failure=failure))
                raise ValueError('Non-OOM implementation failure; diagnose before continuing')
            row.update(outcome='measured_cuda_oom', failure_sha256=base.sha256_file(target/'failure.json'))
        else:
            raw = json.loads((target/'analysis.json').read_text())
            if raw['validation_only'] or raw['warmup']['steps'] != 1536 or len(raw['rows']) != 1 or raw['rows'][0]['steps'] != 1536:
                raise ValueError('Incomplete native allocator recurrence')
            if len(d['restored_state_checks']) != 2 or not all(v['bitwise_initial_state_match'] for v in d['restored_state_checks']):
                raise ValueError('Missing native bitwise restored-state checks')
            row.update(outcome='verified_feasible', analysis_sha256=base.sha256_file(target/'analysis.json'),
                wall_ms_per_step=raw['rows'][0]['wall_ms_per_step'],
                aggregate_tokens_per_second=raw['rows'][0]['aggregate_tokens_per_second'],
                peak_allocated_bytes=raw['rows'][0]['final_memory']['peak_allocated_bytes'])
        rows.append(row)
        base.atomic_json(out/'progress.json', {'rows': rows})
    for name, digest in hashes.items():
        if base.sha256_file(Path(name)) != digest:
            raise ValueError('Source/evidence drift: '+name)
    base.atomic_json(out/'analysis.json', {'rows': rows, 'scope': scope})
    print(json.dumps([{k: v for k, v in row.items() if k not in ('allocator_memory', 'fixture')} for row in rows], indent=2), flush=True)


if __name__ == '__main__':
    main()
