"""CPU-only tests of the separate representation prototype."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/representation_v2'))
import prototype as p
from affine_reference import mixed_reconstructions, attention


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(82)
        self.x = (rng.normal(size=(96, 2, 8))*np.exp2(np.arange(8)-4)).astype(np.float16)
        self.c = self.x[:64].astype(np.float32).mean(0).astype(np.float16)
        self.policy = {'page_size':16, 'exact_prefix_pages':1, 'exact_static_suffix_pages':1, 'exact_tail_tokens':16}

    def test_baseline_reproduces_existing_reference(self):
        actual = p.encode(self.x,self.c,64,self.policy,False)
        expected = mixed_reconstructions(self.x,self.c,64,self.policy)
        np.testing.assert_array_equal(actual['real'],expected['real'])

    def test_exact_regions_unchanged(self):
        a,b = [p.encode(self.x,self.c,64,self.policy,flag) for flag in (False,True)]
        np.testing.assert_array_equal(a['real'][a['mask']],b['real'][b['mask']])

    def test_decode_cannot_change_prefill_gain(self):
        a = p.encode(self.x,self.c,64,self.policy,True)
        changed = self.x.copy()
        changed[64:] *= 2
        b = p.encode(changed,self.c,64,self.policy,True)
        np.testing.assert_array_equal(a['gain'],b['gain'])

    def test_power_two_and_zero_channels(self):
        gain = p.fit_gain(np.zeros((64,2,8)))
        np.testing.assert_array_equal(gain,np.ones((2,8)))
        gain = p.fit_gain(self.x)
        np.testing.assert_array_equal(np.log2(gain),np.rint(np.log2(gain)))

    def test_mixed_factorization(self):
        cache = p.encode(self.x,self.c,64,self.policy,True)
        q = np.random.default_rng(2).normal(size=(4,8))
        for h in range(2):
            np.testing.assert_allclose(p.factorized(q,cache,cache,h),attention(q,cache['real'][:,h],cache['real'][:,h]),atol=1e-11,rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
