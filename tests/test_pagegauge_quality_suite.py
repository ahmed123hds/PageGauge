import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/representation_v2'))
spec = importlib.util.spec_from_file_location('e0_quality_suite', ROOT/'experiments/mlsys2027/representation_v2/quality_suite.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixtures():
    files = {}
    dirs = [Path('/virtual/window0'), Path('/virtual/window1')]
    for i, directory in enumerate(dirs):
        count = (1,3)[i]
        nll = (1,6)[i]
        files[directory/'completion.json'] = {'return_code':0, 'sampled_exclusivity_passed':True}
        files[directory/'analysis.json'] = {'results':{name:{'sha256':'valid'} for name in m.VARIANTS}}
        for name in m.VARIANTS:
            candidate = nll + (0.1*count if name == 'fixed_pg' else 0)
            unit = {'cluster_unit_id':str(i), 'token_count':count, 'raw_sufficient_statistics':{
                'reference_nll_sum_nats':nll, 'candidate_nll_sum_nats':candidate,
                'nll_delta_sum_nats':candidate-nll, 'forward_kl_sum_nats':0,
                'top1_agreement_count':count, 'candidate_true_token_top1_count':count}}
            files[directory/(name+'.json')] = {'distribution_quality':{'cluster_bootstrap_units':[unit]}}
    return dirs, files


class QualitySuiteTests(unittest.TestCase):
    def test_pools_nll_before_exponentiating_and_keeps_negative_result(self):
        dirs, files = fixtures()
        with patch.object(Path, 'read_text', lambda path:json.dumps(files[path])), patch.object(m,'digest',lambda path:'valid'):
            r = m.reduce_runs(dirs)
        self.assertAlmostEqual(r['summary']['original_pg']['ppl'], math.exp(7/4))
        self.assertAlmostEqual(r['fixed_over_original_pg_ppl']['point'], math.exp(.1))
        self.assertEqual(r['fixed_over_original_pg_ppl']['windows_with_higher_nll'], 2)
        self.assertFalse(r['production_default_changed'])

    def test_rejects_changed_evidence(self):
        dirs, files = fixtures()
        with patch.object(Path, 'read_text', lambda path:json.dumps(files[path])), patch.object(m,'digest',lambda path:'changed'):
            with self.assertRaises(RuntimeError):m.reduce_runs(dirs)

    def test_rejects_unpaired_windows(self):
        dirs, files = fixtures()
        files[dirs[0]/'fixed_pg.json']['distribution_quality']['cluster_bootstrap_units'][0]['cluster_unit_id'] = 'wrong'
        with patch.object(Path, 'read_text', lambda path:json.dumps(files[path])), patch.object(m,'digest',lambda path:'valid'):
            with self.assertRaises(ValueError):m.reduce_runs(dirs)


if __name__ == '__main__':unittest.main()
