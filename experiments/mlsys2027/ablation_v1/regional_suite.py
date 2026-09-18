"""Complete all four fixed residual policies on eight preselected TRAIN books."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import mlsys_rtx5090_entry as base
from kivi_quality import verify
from reduce_frontier import summarize
from regional_quality import POLICIES

BOOKS = ROOT/'results/mlsys2027_generalization_v1/book_suite_mistral_20260908T163058Z_c15822be/analysis.json'


def reduce_runs(directories):
    if len(directories) != 8:
        raise ValueError('All eight predeclared books required')
    cells = {name: [] for name, *_ in POLICIES}
    memory, probes = ({name: [] for name in cells} for _ in range(2))
    evidence, books, first = {}, set(), None
    for index, directory in enumerate(directories):
        m = json.loads((directory/'manifest.json').read_text())
        c = json.loads((directory/'completion.json').read_text())
        r = json.loads((directory/'analysis.json').read_text())
        verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed'] or m['book_index'] != index:
            raise ValueError('Incomplete/reordered ablation book')
        if m['policies'] != [list(p) for p in POLICIES] or m['token_provenance']['split'] != 'train':
            raise ValueError('Changed ablation policy/split')
        contract = (m['model'], m['source_sha256'], m['context'], m['decode_steps'])
        if first is None:
            first = contract
        elif contract != first:
            raise ValueError('Changed source/model/configuration across ablation books')
        if base.sha256_file(directory/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed book tokens')
        books.add(m['token_provenance']['object_name'])
        for name in ('manifest.json', 'completion.json', 'analysis.json', 'tokens.json'):
            evidence[str(directory/name)] = base.sha256_file(directory/name)
        pp = directory/'regional_execution_probes.json'
        if base.sha256_file(pp) != r['probes_sha256']:
            raise ValueError('Changed execution probes')
        evidence[str(pp)] = r['probes_sha256']
        probe = json.loads(pp.read_text())['policies']
        book_unit = None
        for name, _, _, tail in POLICIES:
            path = directory/(name+'.json')
            digest = base.sha256_file(path)
            if digest != r['results'][name]['sha256']:
                raise ValueError('Changed ablation outcome')
            evidence[str(path)] = digest
            raw = json.loads(path.read_text())
            units = raw['distribution_quality']['cluster_bootstrap_units']
            if len(units) != 1 or units[0]['token_count'] != 1536 or raw['consumed_new_history_pages'] != list(range(1280, 1376-tail//16)):
                raise ValueError('Incomplete recurrent ablation labels/history')
            u = units[0]
            shared = (u['cluster_unit_id'], u['raw_sufficient_statistics']['reference_nll_sum_nats'])
            if book_unit is None:
                book_unit = shared
            elif shared != book_unit:
                raise ValueError('Policies do not share actual book/HF reference')
            if len(probe[name]) != 18 or any(p['relative_l2'] > .005 or p['absolute_error'] > .02 for p in probe[name]):
                raise ValueError('Missing/failed selected execution probes')
            cells[name].append(u)
            memory[name].append(raw['cache_storage_bytes'])
            probes[name].extend(probe[name])
    if len(books) != 8 or first[2:] != (20480, 1536):
        raise ValueError('Wrong ablation cohort/configuration')
    rows = {name: summarize(cells[name], memory[name]) for name in cells}
    original = np.array([u['raw_sufficient_statistics']['candidate_nll_sum_nats'] for u in cells['reference_policy']])
    counts = np.array([u['token_count'] for u in cells['reference_policy']])
    samples = np.random.default_rng(2026090913).integers(0, 8, size=(5000, 8))
    for name, row in rows.items():
        candidate = np.array([u['raw_sufficient_statistics']['candidate_nll_sum_nats'] for u in cells[name]])
        delta = candidate-original
        values = np.exp(delta[samples].sum(1)/counts[samples].sum(1))
        row.update(ppl_ratio_to_reference_policy=math.exp(delta.sum()/counts.sum()),
            descriptive_paired_book_ratio95=np.quantile(values, [.025, .975]).tolist(),
            cache_reduction_to_reference_policy=1-row['cache_bytes']/rows['reference_policy']['cache_bytes'],
            selected_execution_probe_calls=len(probes[name]),
            maximum_relative_l2=max(p['relative_l2'] for p in probes[name]),
            maximum_absolute_error=max(p['absolute_error'] for p in probes[name]),
            selected_group_mean_attention_mass={region: float(np.mean([p['mean_query_group_attention_mass'][region] for p in probes[name]]))
                for region in ('history', 'prefix', 'static_suffix', 'tail_only')})
    return dict(rows=rows, runs=[str(p) for p in directories], input_sha256=evidence,
        books=sorted(books), policies=POLICIES,
        scope='Eight preselected Mistral PG19 TRAIN development books, four one-factor exact-region policies, full1536 recurrent labels/book/policy. PPL aggregated from NLL sums; paired-book descriptive bootstrap versus HF and versus reference policy. Top1/KL are versus HF only. Selected post-quantization attention masses, not whole-model/HF mass. No speed, policy promotion, prospective quality gate, zero-residual T0 or independent final TEST claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    args = p.parse_args()
    pilot = args.pilot.resolve()
    m = json.loads((pilot/'manifest.json').read_text())
    c = json.loads((pilot/'completion.json').read_text())
    reference = json.loads(BOOKS.read_text())
    fixtures = [Path(p) for p in reference['runs']]
    if len(fixtures) != 8 or m['source_fixture'] != str(fixtures[0]) or c['return_code'] or not c['sampled_exclusivity_passed']:
        raise ValueError('Completed first-book ablation required')
    verify(m)
    hashes = dict(m['source_sha256'])
    for path in (BOOKS, Path(__file__), ROOT/'experiments/mlsys2027/baselines_v1/reduce_frontier.py', pilot/'analysis.json'):
        hashes[str(path.resolve())] = base.sha256_file(path)
    out = ROOT/'results/mlsys2027_ablation_v1'/('regional_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', dict(pilot=str(pilot), fixtures=[str(p) for p in fixtures],
        policies=POLICIES, input_sha256=hashes, scope='Complete all remaining seven preselected TRAIN books; no quality-driven stopping, replacement, or policy search.'))
    print('Regional ablation suite: '+str(out), flush=True)
    directories = [pilot]
    for index in range(1, 8):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source/evidence drift: '+path)
        command = [sys.executable, '-u', str(Path(__file__).with_name('regional_quality.py')), '--fixture', str(fixtures[index])]
        target = None
        with (out/f'book_{index}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Regional ablation output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'book_{index}_completion.json', dict(run=str(target), return_code=code))
        if code or target is None:
            raise ValueError('Ablation execution failure retained; diagnose before continuing')
        directories.append(target)
        base.atomic_json(out/'progress.json', {'runs': [str(p) for p in directories]})
    for path, digest in hashes.items():
        if base.sha256_file(Path(path)) != digest:
            raise ValueError('Source/evidence drift: '+path)
    result = reduce_runs(directories)
    base.atomic_json(out/'analysis.json', result)
    print(json.dumps(result['rows'], indent=2), flush=True)


if __name__ == '__main__':
    main()
