import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/baselines_v1'))
from shape_contract import schedule, validate_result, BUDGET, STEPS, STRIDE, CONTEXTS


class ShapeContract(unittest.TestCase):
    def test_fixed_grid_and_conservative_bounds(self):
        rows = schedule(14496047104, 32768)
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(r['validate'] for r in rows[:15]))
        self.assertFalse(any(r['validate'] for r in rows[15:]))
        self.assertTrue(all(c+STEPS+1 <= STRIDE for c in CONTEXTS))
        row = next(r for r in rows if not r['validate'] and r['backend'] == 'flashinfer_fp16' and r['batch'] == 8 and r['context'] == 30720)
        self.assertGreater(row['lower_bound_bytes'], BUDGET)
        self.assertEqual(row['action'], 'analytically_infeasible')
        with self.assertRaises(ValueError): schedule(14496047104, 32000)

    def test_full_recurrence_not_short_timing(self):
        m = {'backend': 'page_gauge', 'batch': 4, 'context': 8192, 'decode_steps': STEPS, 'validate': False}
        row = {'steps': STEPS, 'request_tokens': 4*STEPS}
        r = {**m, 'validation_only': False, 'allocator_budget_bytes': BUDGET,
            'warmup': row, 'rows': [dict(row) for _ in range(3)]}
        validate_result(r, m)
        bad = copy.deepcopy(r); bad['rows'][1]['steps'] -= 1
        with self.assertRaises(ValueError): validate_result(bad, m)
        bad = copy.deepcopy(r); bad['batch'] = 1
        with self.assertRaises(ValueError): validate_result(bad, m)


if __name__ == '__main__': unittest.main()
