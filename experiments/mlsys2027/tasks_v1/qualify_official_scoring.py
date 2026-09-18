"""CPU equivalence to pinned LongBench scoring on synthetic strings only."""
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys
from scoring_contract import METRICS, example_score, aggregate

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = Path('/home/anonymous/pagegauge_baselines/longbench_scoring_2e00731')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(OFFICIAL))
    import metrics
    spec = importlib.util.spec_from_file_location('longbench_official_eval', OFFICIAL/'eval.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    cases = {
        'qa': (['The blue bird', '\nblue\nextra words'], [['blue bird', 'bird'], ['blue']]),
        'summary': (['A bird flies over a lake.', ''], [['A bird flies over the lake.'], ['Some text.']]),
        'code': (['```python\n# comment\nreturn value', '\nreturn other'], [['return value'], ['return value']]),
    }
    rows = {}
    for task in METRICS:
        group = 'summary' if task in ('gov_report', 'qmsum') else 'code' if task in ('lcc', 'repobench-p') else 'qa'
        predictions, answers = cases[group]
        values = [example_score(task, pred, refs, metrics) for pred, refs in zip(predictions, answers)]
        reduced = aggregate([{'id': str(i), 'score': value, 'fallback': False} for i, value in enumerate(values)], ['0', '1'])
        expected = official.scorer(task, predictions, answers, [])
        assert round(reduced['score_percent'], 2) == expected
        rows[task] = {'per_example': values, 'official_percent': expected, 'adapter_percent': reduced['score_percent']}
    result = {'passed': True, 'synthetic_tasks_checked': len(rows), 'rows': rows,
        'official_revision': '2e00731f8d0bff23dc4325161044d0ed8af94c1e',
        'file_sha256': {str(p): sha(p) for p in (OFFICIAL/'metrics.py', OFFICIAL/'eval.py', Path(__file__), Path(__file__).with_name('scoring_contract.py'))},
        'packages': {name: version(name) for name in ('numpy', 'rouge', 'fuzzywuzzy', 'jieba', 'six')},
        'code_similarity_backend': metrics.fuzz.SequenceMatcher.__module__,
        'scope': 'Synthetic string equivalence only; no benchmark examples or answers, no model scores.'}
    out = ROOT/'results/mlsys2027_tasks_v1/official_scoring_qualification'
    out.mkdir(exist_ok=True)
    target = out/'analysis.json'
    if target.exists():
        assert json.loads(target.read_text()) == result
    else:
        target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
