import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bitdecode_cohort',ROOT/'experiments/mlsys2027/baselines_v1/bitdecode_quality_suite.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PILOT=ROOT/'results/mlsys2027_baselines_v1/bitdecode_quality_20260908T105048Z_28ae38c3'


@unittest.skipUnless((PILOT/'analysis.json').is_file(),'Local frozen BitDecoding pilot required')
class BitDecodeCohortTests(unittest.TestCase):
    def test_single_window_statistics(self):
        result=m.reduce_runs([PILOT])['summary']['int4']
        p=json.loads((PILOT/'int4.json').read_text())['distribution_quality']['overall']
        self.assertAlmostEqual(result['ppl'],p['candidate_perplexity'])
        self.assertEqual(result['tokens'],1536)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):m.reduce_runs([PILOT,PILOT])


if __name__=='__main__':unittest.main()
