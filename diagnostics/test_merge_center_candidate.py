"""Synthetic GPU tests; no public-task inputs or model tuning."""
import unittest
import torch
from merge_center_candidate import merge_center


class MergeTests(unittest.TestCase):
    def test_reference_and_aliasing(self):
        torch.manual_seed(20260910)
        for batch in (1, 4, 16):
            for empty in ('none', 'history', 'exact', 'both'):
                with self.subTest(batch=batch, empty=empty):
                    shape = (batch, 32, 128)
                    c = (torch.randn(shape, device='cuda')*3).half()
                    h = (-c.float()+torch.randn(shape, device='cuda')*.1).half()
                    e = (-c.float()+torch.randn(shape, device='cuda')*.1).half()
                    a = torch.randn(shape[:-1], device='cuda')*30
                    b = torch.randn_like(a)*30
                    if empty in ('history', 'both'):
                        a.fill_(-torch.inf)
                    if empty in ('exact', 'both'):
                        b.fill_(-torch.inf)
                    logits = torch.stack((a.double(), b.double()))
                    # LSE inputs are base 2, whereas torch softmax takes ln.
                    logits = logits*torch.log(torch.tensor(2., dtype=torch.float64, device='cuda'))
                    w = logits.softmax(0).nan_to_num(0.)
                    ref = (w[0, ..., None]*h.double()+w[1, ..., None]*e.double()+c.double()).half()
                    out = torch.empty_like(h)
                    merge_center(h, e, a, b, c, out)
                    torch.testing.assert_close(out, ref, rtol=.002, atol=2e-5)
                    merge_center(h, e, a, b, c, h)
                    torch.testing.assert_close(h, out, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
