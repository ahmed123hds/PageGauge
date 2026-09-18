import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kivi_cohort',ROOT/'experiments/mlsys2027/baselines_v1/kivi_quality_suite.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@unittest.skipUnless(m.PILOT.is_dir(),'Local frozen native-KIVI pilot required')
class KiviCohortTests(unittest.TestCase):
    def test_matches_frozen_single_window_sufficient_statistics(self):
        result=m.reduce_runs([m.PILOT])
        for name in ('int2','int4'):
            p=json.loads((m.PILOT/(name+'.json')).read_text())
            u=p['distribution_quality']['cluster_bootstrap_units'][0]
            self.assertAlmostEqual(result['summary'][name]['ppl'],u['candidate_perplexity'])
            self.assertEqual(result['summary'][name]['tokens'],1536)

    def test_rejects_duplicate_pilot_as_independent_evidence(self):
        with self.assertRaises(ValueError):m.reduce_runs([m.PILOT,m.PILOT])


if __name__=='__main__':unittest.main()
