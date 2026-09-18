"""Score a complete saved run against explicitly supplied reference records."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from cohort_scoring import score_cohort
from result_adapter import normalize

METRIC_SHA = 'e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323'


def reduce_run(analysis, fixtures, references, backends, metrics, seed, draws):
    predictions = normalize(analysis, fixtures, backends)
    expected = {c['case_id']: c['task'] for c in fixtures['cases']}
    if len(references) != len(expected) or {r['id'] for r in references} != set(expected):
        raise ValueError('Reference cohort must exactly match generated cohort')
    if any(r['task'] != expected[r['id']] for r in references):
        raise ValueError('Reference task mismatch')
    tasks = {}
    for task in sorted(set(expected.values())):
        examples = [r for r in references if r['task'] == task]
        ids = {r['id'] for r in examples}
        tasks[task] = score_cohort(task, examples,
            [p for p in predictions if p['id'] in ids], backends, metrics, seed, draws)
    return {'tasks': tasks, 'normalized_predictions': predictions,
            'scope': 'Quality scores only; no speed or quantized-execution claim for native fallback.'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('analysis', 'fixtures', 'references', 'metrics', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--backends', nargs='+', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a previous scoring artifact')
    paths = {k: getattr(args, k) for k in ('analysis', 'fixtures', 'references', 'metrics')}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    if hashes['metrics'] != METRIC_SHA:
        raise ValueError('Official metric source differs from qualified revision')
    spec = importlib.util.spec_from_file_location('pinned_longbench_metrics', args.metrics)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    if metrics.fuzz.SequenceMatcher.__module__ != 'difflib':
        raise ValueError('Code-similarity backend differs from qualification')
    result = reduce_run(*(json.loads(paths[k].read_text()) for k in
                        ('analysis', 'fixtures', 'references')),
                        args.backends, metrics, args.seed, args.draws)
    result['input_sha256'] = hashes
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), *[Path(__file__).with_name(n+'.py') for n in
                  ('cohort_scoring', 'result_adapter', 'scoring_contract')]]}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
