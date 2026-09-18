"""Small GPU execution tests, not model quality or performance evidence."""
import unittest
import torch
from segment_graph_dispatch import SegmentGraph


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(128, 128, bias=False)

    def forward(self, attended, residual):
        combined = residual+self.projection(attended)
        return combined, combined.square()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Tests(unittest.TestCase):
    def test_residual_and_lifetimes(self):
        torch.manual_seed(2026090914)
        model = Residual().cuda().half().eval()
        with torch.no_grad():
            x = torch.randn(4, 1, 128, device='cuda', dtype=torch.float16)
            residual = torch.randn_like(x)
            graph = SegmentGraph(model, (x, residual))
            for _ in range(5):
                x.normal_(); residual.normal_()
                actual = graph(x, residual)
                expected = model(x, residual)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                graph(x, actual[0])
            with self.assertRaises(ValueError):
                graph(x)
            with self.assertRaises(ValueError):
                graph(x[:1], residual[:1])
            self.assertEqual(graph.calls, 5)


if __name__ == '__main__':
    unittest.main()
