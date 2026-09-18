from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/serving_v1'))
from metrics import summarize


class ServingMetrics(unittest.TestCase):
    def test_queue_time_denominators_failures_and_one_token(self):
        rows = [dict(request_id='a',arrival_s=0.,completed_s=8.,output_token_times_s=[5.,6.,7.],output_tokens=3,success=True),
            dict(request_id='b',arrival_s=1.,completed_s=3.,output_token_times_s=[2.],output_tokens=1,success=True),
            dict(request_id='failed',arrival_s=2.,completed_s=10.,output_token_times_s=[4.],output_tokens=1,success=False)]
        result = summarize(rows,measurement_end_s=10.,ttft_slo_s=2.,tpot_slo_s=1.)
        self.assertEqual(result['request_rows'][0]['ttft_s'],5.)
        self.assertEqual(result['request_rows'][0]['mean_tpot_s'],1.)
        self.assertIsNone(result['request_rows'][1]['mean_tpot_s'])
        self.assertEqual(result['output_tokens_per_second'],.4)
        self.assertEqual(result['slo_goodput_requests_per_second'],.1)
        self.assertEqual(result['slo_pass_fraction_all_offered_requests'],1/3)
        self.assertEqual(result['failed_requests'],1)

    def test_missing_timestamps_are_not_silently_averaged(self):
        row = dict(request_id='a',arrival_s=0.,completed_s=3.,output_token_times_s=[1.],output_tokens=2,success=True)
        with self.assertRaises(ValueError): summarize([row],measurement_end_s=3.,ttft_slo_s=2.,tpot_slo_s=1.)
        row['output_tokens'] = 1; row['output_token_times_s'] = [-1.]
        with self.assertRaises(ValueError): summarize([row],measurement_end_s=3.,ttft_slo_s=2.,tpot_slo_s=1.)


if __name__ == '__main__': unittest.main()
