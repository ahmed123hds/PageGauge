"""Complete the same eight exposed TRAIN windows, retaining the full NSN pilot."""
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
from reduce_frontier import summarize


def reduce_runs(directories):
    if len(directories) != 8:
        raise ValueError('Eight full windows required')
    pg = json.loads((ROOT/'results/mlsys2027_representation_v2/quality_suite_20260908T075133Z_92588ba6/analysis.json').read_text())
    cells = {name: [] for name in ('hf', 'rotated_identity', 'int2')}
    memory = {name: [] for name in cells}
    evidence = {}
    for index, directory in enumerate(directories):
        manifest = json.loads((directory/'manifest.json').read_text())
        if (manifest['context'], manifest['decode_steps']) != (20480, 1536):
            raise ValueError('Wrong recurrent configuration')
        pg_tokens = json.loads((Path(pg['runs'][index])/'token_provenance.json').read_text())
        if manifest['token_provenance']['token_ids_sha256'] != pg_tokens['token_ids_sha256']:
            raise ValueError('Different actual token IDs from PageGauge cohort')
        process = json.loads((directory/'completion.json').read_text())
        if process['return_code'] or not process['sampled_exclusivity_passed']:
            raise ValueError('Invalid native run')
        result = json.loads((directory/'analysis.json').read_text())
        for name in cells:
            path = directory/(name+'.json')
            digest = base.sha256_file(path)
            if digest != result['results'][name]['sha256']:
                raise ValueError('Changed native evidence')
            evidence[str(path)] = digest
            row = json.loads(path.read_text())
            units = row['distribution_quality']['cluster_bootstrap_units']
            if len(units) != 1 or units[0]['token_count'] != 1536:
                raise ValueError('Wrong label count')
            if name == 'int2':
                if row['final_packed_lengths'] != [22016]*32:
                    raise ValueError('Incomplete native finalization')
                if any(lengths != list(range(20480, 22016, 64)) for lengths in row['consumed_packed_lengths_per_layer']):
                    raise ValueError('Missing recurrent packed-cache consumption')
            cells[name].append(units[0])
            memory[name].append(row['cache_accounting']['final']['unique_storage_bytes'])
    rows = {name: summarize(cells[name], memory[name]) for name in cells}
    for row in rows.values():
        row['kv_reduction_fraction'] = 1-row['cache_bytes']/rows['hf']['cache_bytes']
    return {'rows': rows, 'runs': [str(p) for p in directories], 'input_sha256': evidence,
            'matched_actual_token_ids_to_pagegauge': True,
            'scope': 'Eight exposed TRAIN windows, native NSN INT2 and unquantized rotation control with own HF4.48.1 SDPA full-prefix reference. No speed or final TEST claim.',
            'cache_scope': 'Served native KV including resident quantizer/codebook buffers; not total process memory',
            'uncertainty': 'Descriptive paired-window bootstrap; not independent books or prospective non-inferiority'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    completion = json.loads((pilot/'completion.json').read_text())
    if completion['return_code'] or not completion['sampled_exclusivity_passed']:
        raise RuntimeError('Pilot has not completed validly')
    old = json.loads((pilot/'manifest.json').read_text())
    if (old['context'], old['decode_steps']) != (20480, 1536):
        raise ValueError('Need full-context pilot')
    paths = {Path(p) for p in old['source_sha256']}
    paths.update((Path(__file__), Path(__file__).with_name('reduce_frontier.py')))
    hashes = {str(p): base.sha256_file(p) for p in paths}
    for name, digest in old['source_sha256'].items():
        if hashes[name] != digest:
            raise ValueError('Pilot source changed; do not mix silently')
    offsets = [472000+23600*i for i in range(1, 8)]
    out = ROOT/'results/mlsys2027_baselines_v1'/('nsn_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'pilot': str(pilot), 'pilot_analysis_sha256': base.sha256_file(pilot/'analysis.json'),
        'new_offsets': offsets, 'source_sha256': hashes, 'orchestrator_pid': os.getpid(),
        'scope': 'Remaining seven native baseline windows on the existing eight-window TRAIN cohort; no tuning between windows'})
    print('NSN quality suite: '+str(out), flush=True)
    directories = [pilot]
    for index, offset in enumerate(offsets):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise RuntimeError('Source changed during cohort')
        command = [sys.executable, '-u', str(Path(__file__).with_name('nsn_quality.py')), '--full', '--offset', str(offset)]
        target = None
        with (out/f'window_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    # Retain verbose model-loader warnings in the log, not repeated chat output.
                    if not line.startswith(('Some weights of ', 'You should probably TRAIN')):
                        print(line, end='', flush=True)
                    if line.startswith('NSN pretrained quality: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'window_{index}_completion.json', {'offset': offset, 'return_code': code,
            'run': str(target) if target else None, 'command': command})
        if code or target is None:
            raise RuntimeError('Native window failed; completed evidence preserved')
        directories.append(target)
    result = reduce_runs(directories)
    base.atomic_json(out/'analysis.json', result)
    print(json.dumps(result['rows'], indent=2), flush=True)


if __name__ == '__main__':
    main()
