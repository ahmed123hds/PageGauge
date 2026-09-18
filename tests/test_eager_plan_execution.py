"""CPU validation of sharded reconstruction and non-graph coverage checks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('eager_execution', ROOT/'experiments/mlsys2027/baselines_v1/eager_plan_execution.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class EagerPlanChecks(unittest.TestCase):
    def test_sharded_mixed_cache_and_last_page(self):
        gen = torch.Generator().manual_seed(814)
        codes = lambda: torch.randint(-127, 128, (1, 2, 16, 8, 128), generator=gen, dtype=torch.int8)
        exact = lambda: torch.randn(1, 2, 16, 8, 128, generator=gen).half()*.1
        c = NS(key_codes=codes(), value_codes=codes(),
            key_scales=torch.rand(1, 2, 8, generator=gen)*.01,
            value_scales=torch.rand(1, 2, 8, generator=gen)*.02,
            exact_key=exact(), exact_value=exact(),
            value_center=torch.randn(1, 2, 8, 128, generator=gen).half()*.1)
        c.output_center = c.value_center.repeat_interleave(4, dim=2).contiguous()
        d = NS(cache=c, backend='page_gauge', logical_lengths=[20],
            rotated_query=torch.randn(2, 32, 128, generator=gen).half(),
            old_wrapper=NS(indptr=torch.tensor([0, 1, 2]), indices=torch.tensor([0, 1]), last_len=torch.tensor([16, 16])),
            exact_wrapper=NS(indptr=torch.tensor([0, 1, 2]), indices=torch.tensor([1, 0]), last_len=torch.tensor([5, 5])))
        for request in (0, 1):
            for head in (0, 7):
                k = torch.cat((c.key_codes[0, request, :, head].double()*c.key_scales[0, request, head].double(),
                               c.exact_key[0, 1-request, :5, head].double()))
                v = torch.cat((c.value_codes[0, request, :, head].double()*c.value_scales[0, request, head].double(),
                               c.exact_value[0, 1-request, :5, head].double()))
                q = d.rotated_query[request, head*4:head*4+4].double()
                expected = torch.nn.functional.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None])[0, 0]
                expected += c.value_center[0, request, head].double()
                actual = MOD.reference_head(d, 0, request, head)
                torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
        d.logical_lengths = [21]
        with self.assertRaises(ValueError):
            MOD.reference_head(d, 0, 0, 0)

    def test_coverage_is_not_optional(self):
        stats = {'wrapper_count': 2, 'graph_enabled_count': 0,
                 'oracle_steps': [0, 768, 1535], 'oracle_layers': [0, 15, 31],
                 'oracle_requests': [0, 7], 'oracle_kv_heads': [0, 7], 'oracle_calls': 36}
        MOD.verify_oracle(stats, True)
        stats['oracle_calls'] = 35
        with self.assertRaises(ValueError):
            MOD.verify_oracle(stats, True)
        stats['graph_enabled_count'] = 1
        with self.assertRaises(ValueError):
            MOD.verify_oracle(stats, False)


if __name__ == '__main__':
    unittest.main()
