import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/mlsys2027/representation_v2'))
from fixed_gain_pilot import calibrated_gain, value_cache


class FixedGainPilotTests(unittest.TestCase):
    def test_gain_ignores_future_values(self):
        x = np.random.default_rng(11).normal(size=(32, 2, 4)).astype(np.float16)
        x[:, :, 0] *= 16
        gain = calibrated_gain(x, 16)
        x[16:] *= 100
        np.testing.assert_array_equal(gain, calibrated_gain(x, 16))
        self.assertTrue(np.isfinite(gain).all())
        np.testing.assert_array_equal(np.log2(gain), np.rint(np.log2(gain)))

    def test_rejects_short_calibration(self):
        with self.assertRaises(ValueError):
            calibrated_gain(np.ones((3, 2, 4), dtype=np.float16), 4)

    def test_fixed_unit_gain_matches_zero_center_baseline(self):
        x = np.tile(np.array([-4., -2., 2., 4.], dtype=np.float16)[:, None, None], (8, 2, 4))
        policy = {'page_size': 16, 'exact_prefix_pages': 0,
                  'exact_static_suffix_pages': 0, 'exact_tail_tokens': 16}
        center = np.zeros((2, 4), dtype=np.float16)
        a = value_cache(x, center, 16, policy, None)
        b = value_cache(x, center, 16, policy, np.ones_like(center))
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
