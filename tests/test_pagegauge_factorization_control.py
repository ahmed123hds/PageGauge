import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('factorization_control', ROOT/'experiments/mlsys2027/factorization_v1/control.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class FactorizationControlTests(unittest.TestCase):
    def test_transformation_is_isolated_and_deterministic(self):
        before = c.SOURCE.read_bytes()
        first = c.transformed_header(before)
        self.assertEqual(first, c.transformed_header(before))
        self.assertEqual(first.count('if constexpr (false && PAGE_GAUGE) {'), 2)
        self.assertEqual(first.count('__hmul2(*reinterpret_cast<half2*>(&b_frag[packed]), scale2);'), 2)
        self.assertEqual(c.SOURCE.read_bytes(), before)
        self.assertEqual(hashlib.sha256(before).hexdigest(), c.EXPECTED)

    def test_rejects_unknown_upstream(self):
        with self.assertRaises(ValueError):
            c.transformed_header(b'changed')


if __name__ == '__main__':
    unittest.main()
