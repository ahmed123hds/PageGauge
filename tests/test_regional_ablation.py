import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/ablation_v1'))
from regional_quality import POLICIES, row_regions, install_probe


class RegionalAblation(unittest.TestCase):
    def test_one_factor_at_a_time_and_nonempty_tail(self):
        original = POLICIES[0][1:]
        for _, *policy in POLICIES[1:]:
            self.assertEqual(sum(x != y for x, y in zip(original, policy)), 1)
            self.assertGreaterEqual(policy[2], 16)

    def test_disjoint_overlap_and_partial_last_page(self):
        labels = row_regions(5, 83, 1, 2, [1, 2], [0, 3, 4, 5])
        self.assertEqual(len(labels), 83)
        self.assertEqual(labels.count('history'), 32)
        self.assertEqual(labels.count('prefix'), 16)
        self.assertEqual(labels.count('static_suffix'), 32)
        self.assertEqual(labels.count('tail_only'), 3)
        with self.assertRaises(ValueError):
            row_regions(5, 83, 1, 2, [1, 2], [0, 2, 3, 4, 5])

    def test_tracks_actual_step_without_other_harness_fields(self):
        class MinimalProductionDriver:
            def eager_attention(self, layer):
                return ('actual', layer)
            def step(self, input_ids, position):
                return self.eager_attention(0)
        driver = MinimalProductionDriver()
        stats = []
        install_probe(driver, 20480, 1536, stats)
        # Unselected step: verifies production call tracking without requiring
        # the other harness's nonexistent logical_lengths or a GPU fixture.
        self.assertEqual(driver.step(None, 20481), ('actual', 0))
        self.assertEqual(stats, [])
        with self.assertRaises(ValueError):
            driver.eager_attention(0)


if __name__ == '__main__':
    unittest.main()
