"""GPU execution qualification only; no performance or model-quality claim."""
import unittest
import torch
from dense_graph_dispatch import DenseGraph


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DenseGraphTests(unittest.TestCase):
    def test_replay_and_guards(self):
        torch.manual_seed(2026090913)
        module = torch.nn.Sequential(torch.nn.Linear(128, 256, bias=False),
                                     torch.nn.SiLU(), torch.nn.Linear(256, 128, bias=False)).cuda().half().eval()
        with torch.no_grad():
            x = torch.randn(4, 1, 128, device='cuda', dtype=torch.float16)
            graph = DenseGraph(module, x)
            for _ in range(5):
                x.normal_()
                expected = module(x)
                actual = graph(x).clone()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(graph.calls, 5)
            with self.assertRaises(ValueError):
                graph(x[:1])
            with torch.cuda.stream(torch.cuda.Stream()):
                with self.assertRaises(ValueError):
                    graph(x)
            next(module.parameters()).add_(1)
            with self.assertRaises(ValueError):
                graph(x)


if __name__ == '__main__':
    unittest.main()
