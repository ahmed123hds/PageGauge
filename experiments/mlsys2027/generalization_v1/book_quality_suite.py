"""Complete eight frozen PG19 TRAIN books after a retained first-book pilot."""
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
        raise ValueError('Eight complete preselected development books required')
    cells = {key: [] for key in ('flashinfer_fp16', 'page_gauge')}
    memory = {key: [] for key in cells}
    evidence, books, first = {}, set(), None
    for index, directory in enumerate(directories):
        m = json.loads((directory/'manifest.json').read_text())
        c = json.loads((directory/'completion.json').read_text())
        r = json.loads((directory/'analysis.json').read_text())
        if c['return_code'] or not c['sampled_exclusivity_passed'] or not r['execution_completed']:
            raise ValueError('Invalid book execution')
        if (m['book_index'], m['context'], m['decode_steps'], m['S'], m['A'], m['T'], m['exact_split_pages']) != (index, 20480, 1536, 4, 128, 768, 32):
            raise ValueError('Wrong book/configuration')
        contract = (m['model'], m['model_family'], m['source_sha256'])
        if first is None:
            first = contract
        elif first != contract:
            raise ValueError('Model/source drift across books')
        p = m['token_provenance']
        if p['kind'] != 'pg19' or p['split'] != 'train' or not p['object_name'].startswith('train/'):
            raise ValueError('Unexpected corpus split')
        if base.sha256_file(directory/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed book tokens')
        books.add(p['object_name'])
        for backend in cells:
            path = directory/(backend+'.json')
            digest = base.sha256_file(path)
            if digest != r['results'][backend]['sha256']:
                raise ValueError('Changed book quality evidence')
            evidence[str(path)] = digest
            raw = json.loads(path.read_text())
            units = raw['distribution_quality']['cluster_bootstrap_units']
            if len(units) != 1 or units[0]['token_count'] != 1536:
                raise ValueError('Incomplete book label trajectory')
            if backend == 'page_gauge' and raw['consumed_new_history_pages'] != list(range(1280, 1328)):
                raise ValueError('New history not consumed')
            cells[backend].append(units[0])
            memory[backend].append(raw['cache_storage_bytes'])
    if len(books) != 8:
        raise ValueError('Repeated book')
    rows = {key: summarize(cells[key], memory[key]) for key in cells}
    for row in rows.values():
        row['kv_reduction_fraction'] = 1-row['cache_bytes']/rows['flashinfer_fp16']['cache_bytes']
    return {'rows': rows, 'runs': [str(path) for path in directories], 'input_sha256': evidence,
        'model_family': first[1], 'book_objects': sorted(books),
        'scope': 'Eight metadata-selected PG19 TRAIN development books, one recurrent1536-label window/book. Descriptive paired-book bootstrap of model-token PPL ratios. Not official whole-corpus word-level PG19, final TEST, speed or non-inferiority evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    m = json.loads((pilot/'manifest.json').read_text())
    c = json.loads((pilot/'completion.json').read_text())
    if (m['book_index'], m['context'], m['decode_steps']) != (0, 20480, 1536) or c['return_code'] or not c['sampled_exclusivity_passed']:
        raise ValueError('Complete first full-book pilot required')
    cohort = Path(m['token_provenance']['selection_path']).parent
    if base.sha256_file(cohort/'selection.json') != m['token_provenance']['selection_sha256']:
        raise ValueError('Changed book selection')
    hashes = dict(m['source_sha256'])
    for path in (Path(__file__), ROOT/'experiments/mlsys2027/baselines_v1/reduce_frontier.py'):
        hashes[str(path.resolve())] = base.sha256_file(path)
    out = ROOT/'results/mlsys2027_generalization_v1'/('book_suite_'+m['model_family']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'pilot': str(pilot), 'pilot_sha256': base.sha256_file(pilot/'analysis.json'),
        'cohort': str(cohort), 'selection_sha256': m['token_provenance']['selection_sha256'],
        'model_family': m['model_family'], 'remaining_books': list(range(1, 8)), 'source_sha256': hashes,
        'orchestrator_pid': os.getpid(), 'scope': 'Complete all preselected TRAIN books; no quality-driven stopping or replacements.'})
    print('Book quality suite: '+str(out), flush=True)
    directories = [pilot]
    for index in range(1, 8):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source drift')
        command = [sys.executable, '-u', str(Path(__file__).with_name('book_quality.py')), '--cohort', str(cohort),
                   '--model-family', m['model_family'], '--book', str(index)]
        target = None
        with (out/f'book_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Book quality output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'book_{index}_completion.json', {'run': str(target), 'return_code': code, 'command': command})
        if code or target is None:
            raise ValueError('Book execution/infeasibility failure retained; do not silently replace the selected book')
        directories.append(target)
        base.atomic_json(out/'progress.json', {'runs': [str(path) for path in directories]})
    for path, digest in hashes.items():
        if base.sha256_file(Path(path)) != digest:
            raise ValueError('Source drift')
    result = reduce_runs(directories)
    base.atomic_json(out/'analysis.json', result)
    print(json.dumps(result['rows'], indent=2), flush=True)


if __name__ == '__main__':
    main()
