import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from score_saved_run import reduce_run


class Tests(unittest.TestCase):
    def test_saved_synthetic_cohort(self):
        root = Path(__file__).resolve().parents[3]
        folder = root/'results/mlsys2027_tasks_v1/task_fallback_smoke_20260910T081503Z_3aaa0d41'
        analysis = json.loads((folder/'analysis.json').read_text())
        fixtures = json.loads((folder/'fixtures.json').read_text())
        # Deliberately artificial references: this tests plumbing, not accuracy.
        refs = [{'id': c['case_id'], 'task': c['task'], 'answers': ['synthetic']} for c in fixtures['cases']]
        metrics = SimpleNamespace(qa_f1_score=lambda *a, **kw: 0.0)
        result = reduce_run(analysis, fixtures, refs,
                            ['hf', 'flashinfer_fp16', 'page_gauge'], metrics, 7, 100)
        self.assertEqual(result['tasks']['qasper']['rows']['page_gauge']['fallback_fraction'], .5)
        self.assertEqual(result['tasks']['qasper']['rows']['page_gauge']['paired_example_95_interval'], [0, 0])
        with self.assertRaises(ValueError):
            reduce_run(analysis, fixtures, refs[:-1], ['hf', 'flashinfer_fp16', 'page_gauge'], metrics, 7, 100)


if __name__ == '__main__':
    unittest.main()
