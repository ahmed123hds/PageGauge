"""FP64 sanity checks supplement the analytic proof; not empirical quality."""
import math
import unittest
import torch


class ShiftInvariantBounds(unittest.TestCase):
    def test_attainable_two_position_bound(self):
        for width in (0.0, .01, .5, 2.0, 12.0):
            x = torch.tensor([width/2, 0], dtype=torch.float64)
            delta = torch.tensor([-width/2, width/2], dtype=torch.float64)
            p, q = x.softmax(0), (x+delta).softmax(0)
            tv = (p-q).abs().sum().item()/2
            self.assertAlmostEqual(tv, math.tanh(width/4), places=13)

    def test_arbitrary_values_and_common_offsets(self):
        gen = torch.Generator().manual_seed(20260909)
        for _ in range(64):
            x = torch.randn(17, generator=gen, dtype=torch.float64)
            delta = torch.randn(17, generator=gen, dtype=torch.float64)*.4
            v = torch.randn(17, 8, generator=gen, dtype=torch.float64)
            ev = torch.randn(17, 8, generator=gen, dtype=torch.float64)*.03
            p, q = x.softmax(0), (x+delta).softmax(0)
            width = (delta.max()-delta.min()).item()
            diameter = torch.cdist(v, v).max().item()
            bound = (q*ev.norm(dim=1)).sum().item()+diameter*math.tanh(width/4)
            self.assertLessEqual((q@(v+ev)-p@v).norm().item(), bound+1e-12)
            torch.testing.assert_close(q, (x+delta+100).softmax(0), rtol=1e-12, atol=1e-14)
            nll_change = (x+delta).log_softmax(0)-x.log_softmax(0)
            self.assertLessEqual(nll_change.abs().max().item(), width+1e-12)


if __name__ == '__main__':
    unittest.main()
