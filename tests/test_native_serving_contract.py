import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
from native_serving_pilot import contract
from native_serving_suite import summarize


class NativeServingContract(unittest.TestCase):
    def test_native_family_and_backend(self):
        m = {'context': 20480, 'decode_steps': 1536, 'token_provenance': {'split': 'train'}}
        for family, backend in (('nsn', 'hf'), ('nsn', 'nsn_int2'), ('kitty', 'hf'), ('kitty', 'kitty_pro')):
            contract(m, family, backend)
        for family, backend in (('nsn', 'kitty_pro'), ('kitty', 'nsn_int2'), (None, 'hf')):
            with self.assertRaises(ValueError):
                contract(m, family, backend)

    def test_reject_nonfull_or_test_fixture(self):
        for context, steps, split in ((20480, 16, 'train'), (1024, 1536, 'train'), (20480, 1536, 'test')):
            with self.assertRaises(ValueError):
                contract({'context': context, 'decode_steps': steps, 'token_provenance': {'split': split}}, 'nsn', 'hf')

    def test_summary_requires_full_recurrence(self):
        rows = [dict(steps=1536, final_lengths=[22016]*32,
            final_cache={'unique_storage_bytes': 100}, prefill_wall_ms=20,
            decode_wall_ms_per_step=10+i, timed_segment_sum_ms=20+(10+i)*1536,
            decode_tokens_per_second=1000/(10+i),
            prefill_memory={'peak_allocated_bytes': 500},
            decode_memory={'peak_allocated_bytes': 400}) for i in range(3)]
        raw = {'family': 'nsn', 'warmup': {'steps': 1536}, 'rows': rows}
        result = summarize(raw)
        self.assertEqual(result['decode_wall_ms_per_step']['median'], 11)
        self.assertEqual(result['peak_allocated_bytes'], 500)
        rows[0]['final_lengths'][31] -= 1
        with self.assertRaises(ValueError):
            summarize(raw)


if __name__ == '__main__':
    unittest.main()
