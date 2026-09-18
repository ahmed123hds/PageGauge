"""Metadata selection must be stable and cannot select TEST/validation books."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pg19_development', ROOT/'experiments/mlsys2027/generalization_v1/download_pg19_development.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.items = [{'name': f'train/{i}.txt', 'size': '250000', 'generation': '123', 'md5Hash': 'fake'} for i in range(20)]

    def test_order_invariance_and_split(self):
        first, count = MODULE.select(self.items)
        second, _ = MODULE.select(list(reversed(self.items))+[
            {'name': 'test/1.txt'}, {'name': 'validation/2.txt'}, {'name': 'train/../test/1.txt'}])
        self.assertEqual(first, second)
        self.assertEqual(count, 20)
        self.assertEqual(len(first), 8)
        self.assertTrue(all(x['name'].startswith('train/') for x in first))

    def test_metadata_eligibility_only(self):
        items = self.items+[{'name': 'train/999.txt', 'size': '199999'}]
        self.assertEqual(MODULE.select(items), MODULE.select(self.items))
        with self.assertRaises(ValueError):
            MODULE.select(self.items[:7])
        with self.assertRaises(ValueError):
            MODULE.select(self.items+[self.items[0]])


if __name__ == '__main__':
    unittest.main()
