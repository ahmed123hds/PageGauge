"""CPU packed-field checks, independent of native Kitty/Triton imports."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('kitty_reference', ROOT/'experiments/mlsys2027/baselines_v1/kitty_reference.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class KittyReferenceTests(unittest.TestCase):
    def test_byte_offsets_require_widening(self):
        narrow = torch.arange(32, dtype=torch.uint8)*32
        wide = torch.arange(32, dtype=torch.uint8).int()*32
        self.assertEqual(narrow.unique().numel(), 8)
        self.assertEqual(wide.unique().numel(), 32)
        self.assertEqual(wide[-1].item(), 992)

    def test_key_boosted_bits_and_time_order(self):
        dim, page, boosted = 8, 4, 2
        # All low fields encode chronological values 0,1,2,3; boosted high
        # fields add 4 and 8 respectively. Channel6 and channel2 are boosted.
        index = torch.tensor([2, 3, 1, 4, 5, 6, 0, 7], dtype=torch.uint8)
        raw = torch.cat((torch.full((dim,), 228, dtype=torch.uint8),
                         torch.tensor([85, 170], dtype=torch.uint8), index))[None]
        meta = torch.zeros(1, 1, dim, 2, dtype=torch.float16)
        meta[..., 0] = .5
        meta[..., 1] = -1
        out = MOD.unpack_keys(raw, meta, 1, dim, page, boosted)[0, 0]
        expected = torch.arange(4)[:, None].expand(4, dim).clone()
        expected[:, 6] += 4
        expected[:, 2] += 8
        torch.testing.assert_close(out, (expected*.5-1).half(), rtol=0, atol=0)

    def test_value_channel_bits_and_token_metadata(self):
        raw = torch.full((1, 8), 228, dtype=torch.uint8)
        meta = torch.zeros(1, 1, 4, 2, dtype=torch.float16)
        meta[0, 0, :, 0] = torch.arange(1, 5)
        meta[0, 0, :, 1] = -1
        out = MOD.unpack_values(raw, meta, 1, 8, 4)[0, 0]
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])[None]*torch.arange(1, 5)[:, None]-1
        torch.testing.assert_close(out, expected.half(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
