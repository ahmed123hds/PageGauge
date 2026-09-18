import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/factorization_v1'))
spec = importlib.util.spec_from_file_location('e1_full_model', ROOT/'experiments/mlsys2027/factorization_v1/full_model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def payloads():
    result = []
    for block in m.schedule():
        latency = 12 if block['arm'] == 'register_control' else 10
        result.append({'pairing':{'configuration':{key:str(block['seed']) for key in m.previous.MATCHED_FIELDS}},
            'timed_work':{'steps':1536},
            'timing_modes':{mode:{'raw_samples':[{'wall_ms':latency*1536, 'cuda_ms':latency*1536} for _ in range(3)]}
                           for mode in m.previous.MODES},
            'correctness':{'backend_vs_hf_sdpa_fp16':{'passed':True}}})
    return result


class E1FullModelTests(unittest.TestCase):
    def test_balanced_fresh_process_schedule(self):
        blocks = m.schedule()
        self.assertEqual([b['arm'] for b in blocks],
            ['register_control','factorized','factorized','register_control',
             'factorized','register_control','register_control','factorized'])
        self.assertTrue(all(b['backend'] == 'page_gauge' for b in blocks))
        self.assertEqual(len({b['seed'] for b in blocks}), 2)

    def test_ratio_direction_and_cluster_reduction(self):
        r = m.reduce_results(payloads(), m.schedule(), bootstrap_samples=200)
        for endpoint in r['endpoints'].values():
            self.assertAlmostEqual(endpoint['register_over_factorized'], 1.2)
            for bound in endpoint['speedup_95_ci']:
                self.assertAlmostEqual(bound, 1.2)
        self.assertEqual(r['fixture_clusters'], 2)

    def test_rejects_partial_or_unmatched(self):
        p = payloads()
        with self.assertRaises(ValueError):
            m.reduce_results(p[:-1], m.schedule(), 200)
        p[1]['pairing']['configuration']['teacher_inputs_sha256'] = 'different'
        with self.assertRaises(RuntimeError):
            m.reduce_results(p, m.schedule(), 200)

    def test_does_not_require_positive_result(self):
        p = payloads()
        for row in p:
            row['timing_modes'] = copy.deepcopy(p[0]['timing_modes'])
        r = m.reduce_results(p, m.schedule(), 200)
        self.assertAlmostEqual(r['endpoints']['cache_neutral.wall_ms']['register_over_factorized'], 1)


if __name__ == '__main__':
    unittest.main()
