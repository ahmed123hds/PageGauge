import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'diagnostics'))
from benchmark_pg19_external_quality import compare_logits


class GroundTruthMetricsTests(unittest.TestCase):
    def test_true_labels_not_argmax_are_scored(self):
        reference=[torch.tensor([[3.,1.,-2.]]),torch.tensor([[0.,2.,1.]])]
        candidate=[x.clone() for x in reference]
        labels=torch.tensor([[2],[0]])
        report=compare_logits(reference,candidate,labels,predicted_position_start=20481)
        rows=report['distribution_quality']['per_token_metrics_step_major']
        self.assertEqual(len(rows),2)
        for i,row in enumerate(rows):
            expected=-torch.log_softmax(reference[i].double(),dim=-1)[0,labels[i,0]].item()
            self.assertAlmostEqual(row['reference_nll_nats'],expected,places=10)
            self.assertAlmostEqual(row['nll_delta_nats'],0.,places=10)
            self.assertEqual(row['true_token_id'],labels[i,0].item())

    def test_rejects_missing_last_label(self):
        with self.assertRaises(ValueError):
            compare_logits([torch.zeros(1,3)]*2,[torch.zeros(1,3)]*2,torch.zeros(1,1,dtype=torch.long))


if __name__=='__main__':unittest.main()
