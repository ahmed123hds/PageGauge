"""CPU semantic checks; no claim of GPU or full-model integration correctness."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('qwen_adapter', ROOT/'experiments/mlsys2027/generalization_v1/qwen_adapter.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class RecordingDecoder:
    def append(self, layer, query, key, value, position):
        self.record = (layer, query, key, value, position)


class QwenAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026090821)
        self.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=128,
            intermediate_size=192, num_hidden_layers=1, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128, max_position_embeddings=512)).eval()
        self.production = SimpleNamespace(BASE_E2E=SimpleNamespace(check_model=lambda model: 'original-check'),
                                          TransformerDecoder=RecordingDecoder)

    def test_native_norm_before_append(self):
        cls = MOD.install(self.production)
        self.assertIs(MOD.install(self.production), cls)
        decoder = cls()
        decoder.model = self.model
        q, k, v = torch.randn(2, 32, 128), torch.randn(2, 8, 128), torch.randn(2, 8, 128)
        attn = self.model.model.layers[0].self_attn
        with torch.no_grad():
            attn.q_norm.weight.uniform_(.7, 1.3)
            attn.k_norm.weight.uniform_(.7, 1.3)
            decoder.append(0, q, k, v, 16)
        record = decoder.record
        torch.testing.assert_close(record[1], attn.q_norm(q), rtol=0, atol=0)
        torch.testing.assert_close(record[2], attn.k_norm(k), rtol=0, atol=0)
        self.assertIs(record[3], v)
        self.assertEqual(record[4], 16)
        self.assertEqual(self.production.BASE_E2E.check_model(self.model), (1, 32, 8, 128))

    def test_other_model_unchanged(self):
        decoder = MOD.install(self.production)()
        decoder.model = SimpleNamespace(config=SimpleNamespace(model_type='mistral'))
        q, k, v = object(), object(), object()
        decoder.append(0, q, k, v, 12)
        self.assertIs(decoder.record[1], q)
        self.assertIs(decoder.record[2], k)
        self.assertEqual(self.production.BASE_E2E.check_model(decoder.model), 'original-check')

    def test_missing_norm_and_sliding_window_rejected(self):
        self.model.config.num_attention_heads = 16
        with self.assertRaises(ValueError):
            MOD.validate_qwen_model(self.model)
        self.model.config.num_attention_heads = 32
        self.model.config.use_sliding_window = True
        with self.assertRaises(ValueError):
            MOD.validate_qwen_model(self.model)
        self.model.config.use_sliding_window = False
        self.model.model.layers[0].self_attn.q_norm = None
        with self.assertRaises(ValueError):
            MOD.validate_qwen_model(self.model)


if __name__ == '__main__':
    unittest.main()
