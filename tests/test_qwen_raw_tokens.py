"""CPU label-alignment checks for Qwen's no-BOS corpus preparation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('qwen_tokens', ROOT/'experiments/mlsys2027/generalization_v1/qwen_tokens.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class RawTokenTests(unittest.TestCase):
    def test_native_slice_and_labels(self):
        ids, starts, ends = MOD.raw_windows(list(range(100)), 3, 2, 10, 2, 8)
        self.assertEqual(ids, [list(range(10, 16)), list(range(18, 24))])
        self.assertEqual([row[4:] for row in ids], [[14, 15], [22, 23]])
        self.assertEqual(starts, [10, 18])
        self.assertEqual(ends, [16, 24])

    def test_overlap_and_short_corpus_rejected(self):
        for stride in (1, 5):
            with self.assertRaises(ValueError):
                MOD.raw_windows(list(range(100)), 3, 2, 10, 2, stride)
        with self.assertRaises(ValueError):
            MOD.raw_windows(list(range(15)), 3, 2, 10)

    def test_no_bos_cluster_offset(self):
        quality = SimpleNamespace(request_window_metadata=lambda *args: [{'corpus_window_start_offset': 10}])
        provenance = {'bos_prepended_per_request': False, 'archive_member_sha256': 'a'*64, 'shape': [1, 6]}
        row = MOD.request_windows(provenance, 3, 2, quality)[0]
        self.assertEqual(row['corpus_label_start_offset'], 14)
        self.assertEqual(row['corpus_label_end_offset_exclusive'], 16)
        self.assertTrue(row['cluster_unit_id'].endswith('-14-16'))


if __name__ == '__main__':
    unittest.main()
