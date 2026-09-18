"""Matched eight-window Qwen/Kitty cohort; preserve native corrected-port policy."""
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

QWEN = ROOT/'results/mlsys2027_generalization_v1/qwen_suite_20260908T142521Z_58a37e17/analysis.json'


def reduce_runs(directories):
    qwen = json.loads(QWEN.read_text())
    if len(directories) != 8:
        raise ValueError('Eight full-context windows required')
    cells = {name: [] for name in ('hf', 'kitty_pro')}
    memory = {name: [] for name in cells}
    evidence, first_sources = {str(QWEN): base.sha256_file(QWEN)}, None
    for index, directory in enumerate(directories):
        manifest = json.loads((directory/'manifest.json').read_text())
        pg_manifest = json.loads((Path(qwen['runs'][index])/'manifest.json').read_text())
        if (manifest['context'], manifest['decode_steps']) != (20480, 1536):
            raise ValueError('Wrong recurrence configuration')
        if manifest['token_provenance']['token_ids_sha256'] != pg_manifest['token_provenance']['token_ids_sha256']:
            raise ValueError('Actual token IDs differ from Qwen/PageGauge cohort')
        if first_sources is None:
            first_sources = manifest['source_sha256']
        elif first_sources != manifest['source_sha256']:
            raise ValueError('Source change between native windows')
        completion = json.loads((directory/'completion.json').read_text())
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Invalid process/exclusivity')
        result = json.loads((directory/'analysis.json').read_text())
        for name in cells:
            path = directory/(name+'.json')
            digest = base.sha256_file(path)
            if digest != result['results'][name]['sha256']:
                raise ValueError('Changed native evidence')
            evidence[str(path)] = digest
            row = json.loads(path.read_text())
            units = row['distribution_quality']['cluster_bootstrap_units']
            if len(units) != 1 or units[0]['token_count'] != 1536 or row['final_lengths'] != [22016]*36:
                raise ValueError('Incomplete label/cache trajectory')
            if name == 'kitty_pro' and row['consumed_page_counts_per_layer'] != [[[k, k-1] for k in range(159, 172)]]*36:
                raise ValueError('New native packed pages not consumed')
            cells[name].append(units[0])
            memory[name].append(row['cache_accounting']['final']['unique_storage_bytes'])
    rows = {name: summarize(cells[name], memory[name]) for name in cells}
    for row in rows.values():
        row['kv_reduction_fraction'] = 1-row['cache_bytes']/rows['hf']['cache_bytes']
    return {'rows': rows, 'runs': [str(p) for p in directories], 'input_sha256': evidence,
        'matched_actual_token_ids_to_qwen_pagegauge': True,
        'scope': 'Eight exposed TRAIN windows, corrected native Kitty-Pro with own HF4.53.2 full-prefix SDPA reference. Descriptive paired-window bootstrap, not final TEST/non-inferiority or speed.',
        'port_disclosure': 'Both boosted-channel byte-address calculations widened from uint8 to int32; native representation unchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    completion = json.loads((pilot/'completion.json').read_text())
    old = json.loads((pilot/'manifest.json').read_text())
    qwen = json.loads(QWEN.read_text())
    if completion['return_code'] or not completion['sampled_exclusivity_passed']:
        raise ValueError('Native pilot incomplete/invalid')
    if (old['context'], old['decode_steps']) != (20480, 1536) or Path(old['pg_fixture']) != Path(qwen['runs'][0]):
        raise ValueError('Wrong first full pilot fixture')
    hashes = dict(old['source_sha256'])
    for path in (Path(__file__), Path(__file__).with_name('reduce_frontier.py')):
        hashes[str(path.resolve())] = base.sha256_file(path)
    out = ROOT/'results/mlsys2027_baselines_v1'/('kitty_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'pilot': str(pilot), 'pilot_analysis_sha256': base.sha256_file(pilot/'analysis.json'),
        'remaining_pg_fixtures': qwen['runs'][1:], 'qwen_analysis_sha256': base.sha256_file(QWEN),
        'source_sha256': hashes, 'orchestrator_pid': os.getpid(),
        'scope': 'Remaining seven already-exposed Qwen TRAIN windows; no method tuning between windows'})
    print('Kitty quality suite: '+str(out), flush=True)
    directories = [pilot]
    for index, fixture in enumerate(qwen['runs'][1:]):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source drift during cohort')
        command = [sys.executable, '-u', str(Path(__file__).with_name('kitty_quality.py')), '--fixture', fixture]
        target = None
        with (out/f'window_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                    if line.startswith('Kitty quality output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'window_{index}_completion.json', {'run': str(target) if target else None,
            'return_code': code, 'command': command})
        if code or target is None:
            raise RuntimeError('Native window failed; preserve completed evidence')
        directories.append(target)
    result = reduce_runs(directories)
    base.atomic_json(out/'analysis.json', result)
    print(json.dumps(result['rows'], indent=2), flush=True)


if __name__ == '__main__':
    main()
