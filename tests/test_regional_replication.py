from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/ablation_v1'))
from regional_replication import schedule, reduce_results
from regional_replication import previous


class Replication(unittest.TestCase):
    def test_roles_are_balanced_without_relabeling_backends(self):
        for contrast in ('reference_a0', 'fi_a0'):
            rows = schedule(contrast)
            self.assertEqual([r['role'] for r in rows], ['reference','a0','a0','reference','a0','reference','reference','a0'])
            self.assertEqual(len({r['seed'] for r in rows}), 2)
            for r in rows:
                if r['role'] == 'a0':
                    self.assertEqual((r['backend'], r['policy']), ('page_gauge','without_static_suffix'))
                else:
                    self.assertEqual(r['backend'], 'flashinfer_fp16' if contrast == 'fi_a0' else 'page_gauge')

    def test_partial_comparisons_are_not_reduced(self):
        with self.assertRaises(ValueError): reduce_results([], schedule('fi_a0'))

    def test_ratio_direction_and_pair_matching(self):
        blocks = schedule('fi_a0')
        payloads = [{'pairing': {'configuration': {k: block['seed'] for k in previous.MATCHED_FIELDS}},
            'timed_work': {'decoder': 'full'}, 'timing_modes': {mode: {'raw_samples': [
                {metric: (20.0 if block['role'] == 'reference' else 10.0) for metric in previous.METRICS}
                for _ in range(3)]} for mode in previous.MODES}} for block in blocks]
        with patch.object(previous, '_hierarchical_bootstrap_ci', return_value={'speedup_95_ci': [2.0,2.0]}):
            result = reduce_results(payloads, blocks)
            self.assertAlmostEqual(result['endpoints']['cache_neutral.wall_ms']['reference_over_a0'], 2.0)
            self.assertEqual(len(result['endpoints']['cache_neutral.wall_ms']['pairs']), 4)
            payloads[1]['pairing']['configuration'][previous.MATCHED_FIELDS[0]] = 'different'
            with self.assertRaises(RuntimeError): reduce_results(payloads, blocks)


if __name__ == '__main__': unittest.main()
