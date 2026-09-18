"""Prospective recurrence boundaries and dispatch completeness checks."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('graph_dispatch', ROOT/'experiments/mlsys2027/baselines_v1/attention_graph_dispatch.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class GraphScheduleTests(unittest.TestCase):
    def test_boundary_oracles(self):
        for steps in (785, 1536):
            values = MOD.oracle_steps(steps)
            self.assertEqual(values, sorted(set(values)))
            self.assertIn(768, values)
            self.assertIn(steps-1, values)
            self.assertTrue(all(0 <= x < steps for x in values))

    def test_no_eager_fallback_or_partial_success(self):
        stats = {'served_plan_calls': 785, 'graph_replays': 785*32,
                 'missing_bank_count': 0, 'same_cache_oracle_calls': 32*len(MOD.oracle_steps(785))}
        MOD.verify_stats(stats, 785, 32, True)
        for key in stats:
            bad = dict(stats)
            bad[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                MOD.verify_stats(bad, 785, 32, True)


if __name__ == '__main__':
    unittest.main()
