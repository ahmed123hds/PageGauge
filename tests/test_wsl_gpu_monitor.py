import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027'))
from wsl_gpu_monitor import assess, parse_host


class Ownership(unittest.TestCase):
    def test_owned_and_foreign_contexts(self):
        self.assertEqual(assess([120], [4], 120)['unexpected_pids'], [])
        self.assertEqual(assess([120, 121], [4], 120)['unexpected_pids'], [121])
        self.assertIn('error', assess([120], [4, 900], 120))

    def test_absence_is_not_own_presence(self):
        self.assertNotIn(120, assess([], [4], 120)['pids'])
        self.assertNotIn(120, assess([], [], 120)['pids'])

    def test_host_parse_fails_closed(self):
        self.assertEqual(parse_host('GPU-a, 4\nGPU-b, 7', 'GPU-a'), [4])
        with self.assertRaises(ValueError):
            parse_host('GPU-a, [N/A]', 'GPU-a')


if __name__ == '__main__':
    unittest.main()
