import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/ablation_v1'))
from recover_fi_a0 import match_work, COUNTERS

class Recovery(unittest.TestCase):
    def pair(self):
        return [{'backend':backend, 'timed_work':{'output_tokens':6144,
            **{k:v*mult for k,v in COUNTERS.items()}}}
            for backend,mult in [('flashinfer_fp16',1),('page_gauge',2)]]
    def test_expected_backend_costs_allowed(self): match_work(*self.pair())
    def test_each_wrong_counter_rejected(self):
        for key in COUNTERS:
            pair=self.pair(); pair[1]['timed_work'][key]-=1
            with self.assertRaises(RuntimeError): match_work(*pair)
    def test_changed_work_rejected(self):
        pair=self.pair(); pair[1]['timed_work']['output_tokens']=6143
        with self.assertRaises(RuntimeError): match_work(*pair)
    def test_unknown_extra_field_rejected(self):
        pair=self.pair(); pair[1]['timed_work']['new_cost']=1
        with self.assertRaises(RuntimeError): match_work(*pair)

if __name__ == '__main__': unittest.main()
