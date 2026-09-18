import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
import hashlib
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import page_gauge_value_conditioning as c


class FoldedValueTests(unittest.TestCase):
    def test_fixed_gain_does_not_depend_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'gain.pt'
            gain = torch.tensor([[[2., .5], [4., 1.]]]).half()
            torch.save({'gain':gain}, path)
            env = {'PAGEGAUGE_VALUE_CONDITIONING':c.FIXED_MODE,
                   'PAGEGAUGE_FIXED_GAIN_PATH':str(path),
                   'PAGEGAUGE_FIXED_GAIN_SHA256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.dict(os.environ, env):
                cache = NS(value_center=torch.zeros(1,2,2,2).half())
                x = torch.ones(16,2,2).half()
                result = c.condition_prefill(cache,0,3,x)
                torch.testing.assert_close(result, x/gain[0], rtol=0, atol=0)
                c.condition_prefill(cache,0,0,x*16)
                self.assertTrue(torch.equal(cache.value_channel_gain,gain))
            with patch.dict(os.environ, dict(env, PAGEGAUGE_FIXED_GAIN_SHA256='bad')):
                with self.assertRaises(ValueError):
                    c.condition_prefill(NS(value_center=torch.zeros(1,2,2,2).half()),0,0,x)

    def test_disabled_is_identity(self):
        with patch.dict(os.environ,{'PAGEGAUGE_VALUE_CONDITIONING':'none'}):
            x=torch.ones(16,2,2,dtype=torch.float16)
            self.assertIs(c.condition_prefill(NS(),0,0,x),x)

    def test_shared_gain_does_not_refit_later_requests(self):
        with patch.dict(os.environ,{'PAGEGAUGE_VALUE_CONDITIONING':c.MODE}):
            cache=NS(value_center=torch.zeros(1,2,2,2,dtype=torch.float16))
            x=torch.arange(64,dtype=torch.float32).reshape(16,2,2).half()
            y=c.condition_prefill(cache,0,0,x)
            gain=cache.value_channel_gain.clone()
            z=c.condition_prefill(cache,0,1,x*2)
            self.assertTrue(torch.equal(gain,cache.value_channel_gain))
            self.assertTrue(torch.equal(y*gain[0],x))
            self.assertTrue(torch.equal(z*gain[0],x*2))

    def test_folded_gqa_value_output_identity(self):
        torch.manual_seed(7)
        with patch.dict(os.environ,{'PAGEGAUGE_VALUE_CONDITIONING':c.MODE}):
            attn=NS(_pkv_q_width=8,_pkv_kv_width=4,qkv_weight=torch.randn(16,8).half(),o_proj=NS(weight=torch.randn(8,8).half()))
            model=NS(model=NS(layers=[NS(self_attn=attn)]))
            cache=NS(value_channel_gain=torch.tensor([[[2.,.5],[4.,1.]]]).half(),value_conditioning_fitted={0},value_conditioning_setup_seconds=0.)
            original_v=attn.qkv_weight[12:].double().clone()
            original_o=attn.o_proj.weight.double().clone()
            original_qk=attn.qkv_weight[:12].clone()
            metadata=c.fold_packed_weights(model,cache)
            x=torch.randn(5,8,dtype=torch.float64)
            weights=torch.softmax(torch.randn(4,5,dtype=torch.float64),-1)
            def out(wv,wo):
                v=(x@wv.T).reshape(5,2,2).repeat_interleave(2,dim=1)
                a=torch.einsum('ht,thd->hd',weights,v).reshape(8)
                return a@wo.T
            torch.testing.assert_close(out(original_v,original_o),out(attn.qkv_weight[12:].double(),attn.o_proj.weight.double()),rtol=1e-10,atol=1e-10)
            self.assertTrue(torch.equal(original_qk,attn.qkv_weight[:12]))
            self.assertEqual(metadata['decode_extra_operations'],0)
            with self.assertRaises(ValueError):c.fold_packed_weights(model,cache)


if __name__=='__main__':unittest.main()
