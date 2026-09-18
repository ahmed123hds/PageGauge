from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/tasks_v1'))
from packed_hf_views import hf_projection_views


class PackedViews(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(2026090913)
        attention,mlp = torch.nn.Module(),torch.nn.Module()
        for owner,names,widths,label in (
            (attention,('q_proj','k_proj','v_proj'),(8,2,2),'qkv_weight'),
            (mlp,('gate_proj','up_proj'),(12,12),'gate_up_weight')):
            tensors = [torch.randn(width,8,dtype=torch.float16) for width in widths]
            owner.register_parameter(label,torch.nn.Parameter(torch.cat(tensors),requires_grad=False))
            for name in names: setattr(owner,name,None)
        attention._pkv_q_width,attention._pkv_kv_width = 8,2
        mlp._pkv_intermediate = 12
        return SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention,mlp=mlp)]))

    def test_exact_linear_outputs_without_new_weight_storage(self):
        model = self.fixture(); layer = model.model.layers[0]
        qkv = layer.self_attn.qkv_weight; gu = layer.mlp.gate_up_weight
        x = torch.randn(3,8,dtype=torch.float16)
        with hf_projection_views(model):
            for module,weight in ((layer.self_attn.q_proj,qkv[:8]),(layer.self_attn.k_proj,qkv[8:10]),
                (layer.self_attn.v_proj,qkv[10:]),(layer.mlp.gate_proj,gu[:12]),(layer.mlp.up_proj,gu[12:])):
                self.assertEqual(module.weight.untyped_storage().data_ptr(),weight.untyped_storage().data_ptr())
                self.assertTrue(torch.equal(module(x),torch.nn.functional.linear(x,weight)))
        self.assertIsNone(layer.self_attn.q_proj); self.assertIsNone(layer.mlp.up_proj)
        self.assertIs(layer.self_attn.qkv_weight,qkv)

    def test_failure_unwinds_partial_projection_views(self):
        model = self.fixture(); layer = model.model.layers[0]
        layer.mlp.gate_up_bias = torch.zeros(24,dtype=torch.float16)
        with self.assertRaises(ValueError):
            with hf_projection_views(model): pass
        self.assertIsNone(layer.self_attn.q_proj)
        self.assertIsNone(layer.self_attn.k_proj)
        self.assertIsNone(layer.mlp.gate_proj)


if __name__ == '__main__': unittest.main()
