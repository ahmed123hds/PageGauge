"""CPU initial-state merge/accounting tests; not a GPU capacity result."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('frontier_cache', ROOT/'experiments/mlsys2027/baselines_v1/frontier_cache.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FrontierCacheTests(unittest.TestCase):
    def test_native_and_request_major_layouts(self):
        a = torch.ones(1, 2, 3)
        b = a*2
        merged = MOD.merge_field([a, b], 0, 'cpu')
        self.assertEqual(merged.shape, (2, 2, 3))
        torch.testing.assert_close(merged[1], b[0])
        paged = MOD.merge_field([a.transpose(0, 1), b.transpose(0, 1)], 1, 'cpu')
        self.assertEqual(paged.shape, (2, 2, 3))
        torch.testing.assert_close(paged[:, 1], b.transpose(0, 1)[:, 0])

    def test_native_scalars_and_dtype(self):
        self.assertIsNone(MOD.merge_field([None, None], 0, 'cpu'))
        self.assertEqual(MOD.merge_field([20480, 20480], 0, 'cpu'), 20480)
        with self.assertRaises(ValueError):
            MOD.merge_field([20480, 20481], 0, 'cpu')
        with self.assertRaises(ValueError):
            MOD.merge_field([torch.ones(1), torch.ones(1).half()], 0, 'cpu')

    def test_storage_alias_counted_once(self):
        tensor = torch.zeros(16, dtype=torch.float16)
        self.assertEqual(MOD.cpu_snapshot_bytes([{'base': tensor, 'view': tensor[:8]}]), 32)

    def test_restore_bitwise_and_request_order(self):
        a = torch.tensor([[float('nan'), 1.0]], dtype=torch.float16)
        b = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
        snapshots = [{'layers': [(v, None, 20480)]} for v in (a, b)]
        combined = torch.cat((a, b), dim=0)
        past = ((combined, None, 20480),)
        result = MOD.verify_restored(past, snapshots, 'kivi_int4')
        self.assertTrue(result['bitwise_initial_state_match'])
        self.assertEqual(result['fields_checked'], 3)
        with self.assertRaises(ValueError):
            MOD.verify_restored(((combined.flip(0), None, 20480),), snapshots, 'kivi_int4')


if __name__ == '__main__':
    unittest.main()
