import copy
import json
from pathlib import Path
import unittest
from result_adapter import normalize

ROOT = Path(__file__).resolve().parents[3]


class Tests(unittest.TestCase):
    def test_actual_synthetic_serialization(self):
        d = ROOT/'results/mlsys2027_tasks_v1/task_fallback_smoke_20260910T081503Z_3aaa0d41'
        analysis = json.loads((d/'analysis.json').read_text())
        fixtures = json.loads((d/'fixtures.json').read_text())
        backends = ['hf', 'flashinfer_fp16', 'page_gauge']
        rows = normalize(analysis, fixtures, backends)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r['fallback'] for r in rows), 3)
        bad = copy.deepcopy(analysis)
        del bad['results'][0]['arms']['page_gauge']
        with self.assertRaises(ValueError):
            normalize(bad, fixtures, backends)
        bad = copy.deepcopy(analysis)
        bad['results'][1]['arms']['page_gauge']['executed_backend'] = 'page_gauge'
        with self.assertRaises(ValueError):
            normalize(bad, fixtures, backends)


if __name__ == '__main__':
    unittest.main()
