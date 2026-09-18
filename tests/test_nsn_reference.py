"""CPU tests for the independent NSN smoke reference, no native NSN import."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('nsn_smoke', ROOT/'experiments/mlsys2027/baselines_v1/nsn_smoke.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class NSNReferenceTests(unittest.TestCase):
    def test_hadamard_involution(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 128)
        torch.testing.assert_close(MOD.hadamard_reference(MOD.hadamard_reference(x)), x)
        torch.testing.assert_close(MOD.hadamard_reference(x).norm(), x.norm())

    def test_vector_index_and_sign_unpack(self):
        codebook = torch.arange(256*8).reshape(256, 8).float()
        # Two 8D codewords, first positive and second negative.
        packed = torch.tensor([[[[3 | (5 << 8) | (255 << 16)]]]], dtype=torch.int32)
        expected = torch.cat((codebook[3], -codebook[5])).reshape(1, 1, 1, 16)
        torch.testing.assert_close(MOD.code_vectors(packed, codebook), expected)

    def test_metadata_unpack(self):
        packed = torch.tensor([[0x76543210]], dtype=torch.int32)
        scale = torch.tensor([[.5]], dtype=torch.float16)
        offset = torch.tensor([[-1.]], dtype=torch.float16)
        torch.testing.assert_close(MOD.unpack4(packed, scale, offset),
                                   (torch.arange(8).float()*.5-1).reshape(1, 8))


if __name__ == '__main__':
    unittest.main()
