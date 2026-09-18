"""Eight exposed TRAIN windows for Qwen; retain the completed full pilot."""
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
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import mlsys_rtx5090_entry as base
from reduce_frontier import summarize


def reduce_runs(directories):
    if len(directories) != 8:
        raise ValueError('Eight completed full-context windows required')
    cells = {name: [] for name in ('flashinfer_fp16', 'page_gauge')}
    memory = {name: [] for name in cells}
    evidence, first_sources, token_hashes = {}, None, set()
    for index, directory in enumerate(directories):
        manifest = json.loads((directory/'manifest.json').read_text())
        if (manifest['context'], manifest['decode_steps'], manifest['offset'], manifest['synthetic']) != (20480, 1536, 472000+23600*index, False):
            raise ValueError('Wrong configuration/offset')
        provenance = manifest['token_provenance']
        if provenance['bos_prepended_per_request'] or provenance['split'] != 'train':
            raise ValueError('Wrong native token policy')
        token_hashes.add(provenance['token_ids_sha256'])
        if first_sources is None:
            first_sources = manifest['source_sha256']
        elif first_sources != manifest['source_sha256']:
            raise ValueError('Source drift between windows')
        completion = json.loads((directory/'completion.json').read_text())
        if completion['return_code'] or not completion['sampled_exclusivity_passed'] or not completion['own_pid_seen']:
            raise ValueError('Invalid execution')
        result = json.loads((directory/'analysis.json').read_text())
        for name in cells:
            path = directory/(name+'.json')
            digest = base.sha256_file(path)
            if digest != result['results'][name]['sha256']:
                raise ValueError('Changed raw result')
            evidence[str(path)] = digest
            raw = json.loads(path.read_text())
            units = raw['distribution_quality']['cluster_bootstrap_units']
            if len(units) != 1 or units[0]['token_count'] != 1536:
                raise ValueError('Wrong label count')
            if name == 'page_gauge' and raw['consumed_new_history_pages'] != list(range(1280, 1328)):
                raise ValueError('Missing recurrent new history')
            cells[name].append(units[0])
            memory[name].append(raw['cache_storage_bytes'])
    if len(token_hashes) != 8:
        raise ValueError('Duplicate token inputs')
    rows = {name: summarize(cells[name], memory[name]) for name in cells}
    for row in rows.values():
        row['kv_reduction_fraction'] = 1-row['cache_bytes']/rows['flashinfer_fp16']['cache_bytes']
    return {'rows': rows, 'runs': [str(p) for p in directories], 'input_sha256': evidence,
        'model': 'Qwen3-8B', 'checkpoint_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
        'dtype': 'FP16 conversion of BF16 checkpoint for all arms',
        'token_policy': 'Raw WikiText TRAIN with native tokenizer, no BOS/EOS insertion or chat template',
        'scope': 'Eight exposed TRAIN windows, descriptive window bootstrap; no speed, generated-task, independent TEST or non-inferiority claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--download', type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    old = json.loads((pilot/'manifest.json').read_text())
    completion = json.loads((pilot/'completion.json').read_text())
    if (old['context'], old['decode_steps'], old['offset'], old['synthetic']) != (20480, 1536, 472000, False):
        raise ValueError('Expected full first Qwen pilot')
    if completion['return_code'] or not completion['sampled_exclusivity_passed']:
        raise ValueError('Pilot execution invalid')
    if base.sha256_file(args.download/'analysis.json') != old['download_analysis_sha256']:
        raise ValueError('Changed checkpoint preparation')
    hashes = dict(old['source_sha256'])
    hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    reducer = ROOT/'experiments/mlsys2027/baselines_v1/reduce_frontier.py'
    hashes[str(reducer)] = base.sha256_file(reducer)
    out = ROOT/'results/mlsys2027_generalization_v1'/('qwen_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    offsets = [472000+23600*i for i in range(1, 8)]
    base.atomic_json(out/'manifest.json', {'pilot': str(pilot), 'new_offsets': offsets,
        'pilot_analysis_sha256': base.sha256_file(pilot/'analysis.json'), 'source_sha256': hashes,
        'orchestrator_pid': os.getpid(), 'scope': 'Remaining seven predeclared exposed TRAIN windows; no tuning between windows'})
    print('Qwen quality suite: '+str(out), flush=True)
    directories = [pilot]
    for index, offset in enumerate(offsets):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source changed during cohort')
        command = [sys.executable, '-u', str(Path(__file__).with_name('qwen_quality.py')),
            '--full', '--offset', str(offset), '--download', str(args.download)]
        target = None
        with (out/f'window_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                    if line.startswith('Qwen quality output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'window_{index}_completion.json', {'offset': offset,
            'return_code': code, 'run': str(target) if target else None, 'command': command})
        if code or target is None:
            raise RuntimeError('Qwen window failed; preserve all completed results')
        directories.append(target)
    result = reduce_runs(directories)
    base.atomic_json(out/'analysis.json', result)
    print(json.dumps(result['rows'], indent=2), flush=True)


if __name__ == '__main__':
    main()
