"""Conservative capacity exclusions; no GPU/OOM claims from these tests."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
from frontier_capacity import lower_bound


class CapacityBounds(unittest.TestCase):
    def test_only_proven_exclusions(self):
        weights, budget = 14496047104, 28*1024**3
        self.assertEqual(lower_bound(weights, 8, 16), 37581496320)
        self.assertGreater(lower_bound(weights, 8, 16), budget)
        self.assertGreater(lower_bound(weights, 16, 8), budget)
        self.assertLess(lower_bound(weights, 8, 8), budget)
        self.assertLess(lower_bound(weights, 16, 4), budget)
        # A lower bound below budget does NOT establish feasibility.
        self.assertLess(lower_bound(weights, 16, 2), budget)


if __name__ == '__main__':
    unittest.main()
