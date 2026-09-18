import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from full_native_result_adapter import normalize


class Tests(unittest.TestCase):
    def test_native_short_is_not_pg_fallback(self):
        root = Path(__file__).resolve().parents[3]
        directory = root/'results/mlsys2027_tasks_v1/nsn_hf_cohort_20260910T094041Z_7d48c1d6'
        analysis = json.loads((directory/'analysis.json').read_text())
        fixtures = json.loads((directory/'fixtures.json').read_text())
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
        rows = normalize(analysis, fixtures, 'hf', 'nsn', tokenizer, 32768, [2])
        self.assertEqual(len(rows), 2)
        self.assertFalse(any(r['fallback'] for r in rows))
        bad = copy.deepcopy(analysis)
        bad['results'][1]['result']['final_lengths'][0] += 1
        with self.assertRaises(ValueError):
            normalize(bad, fixtures, 'hf', 'nsn', tokenizer, 32768, [2])


if __name__ == '__main__':
    unittest.main()
