"""Native BitDecoding INT4 cache adapter, retaining a 128-token FP16 append block.

This integrates released kernels; it is not an upstream Mistral serving engine.
"""
import math
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=1)
def kernel_interface():
    # The package __init__ imports unrelated new-Transformers model classes.
    # Load the unchanged standalone kernel interface in the older Mistral stack.
    path=Path('/home/anonymous/pagegauge_baselines/bitdecode_sm120_env/lib/python3.12/site-packages/bit_decode/bit_decode_interface.py')
    spec=importlib.util.spec_from_file_location('pagegauge_bitdecode_kernel_interface',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BitDecodeCache:
    def __init__(self,key,value):
        import torch
        api=kernel_interface()
        if key.shape!=value.shape or key.ndim!=4 or key.shape[-1]!=128:
            raise ValueError('Expected matching B/T/Hkv/128 tensors')
        if key.dtype!=torch.float16 or value.dtype!=torch.float16:
            raise ValueError('FP16 inputs required')
        batch,length,heads,dim=key.shape
        count=length//128*128
        if count<128:raise ValueError('At least one packed block required')
        self.length=length;self.residual=length-count;self.packed_length=count
        self.finalized_blocks=0
        self.consumed_finalized_blocks=0
        options={'device':key.device}
        self.kp=torch.zeros(batch,count//4,heads,dim,dtype=torch.uint16,**options)
        self.ks=torch.zeros(batch,count//32,heads,dim,dtype=torch.float32,**options)
        self.vp=torch.zeros(batch,count,heads,dim//4,dtype=torch.uint16,**options)
        self.vs=torch.zeros(batch,dim//32,heads,count,dtype=torch.float32,**options)
        indptr=torch.arange(0,(batch+1)*count,count,dtype=torch.int32,**options)
        api.kvcache_pack_int(key[:,:count].contiguous(),self.kp,self.ks,
            value[:,:count].contiguous(),self.vp,self.vs,None,indptr,count,'k-channel',32,4)
        self.kr=torch.zeros(batch,128,heads,dim,dtype=torch.float16,**options)
        self.vr=torch.zeros_like(self.kr)
        if self.residual:
            self.kr[:,:self.residual].copy_(key[:,count:])
            self.vr[:,:self.residual].copy_(value[:,count:])
        self.kn=torch.empty(batch,32,heads,dim,dtype=torch.uint16,**options)
        self.ksn=torch.empty(batch,4,heads,dim,dtype=torch.float32,**options)
        self.vn=torch.empty(batch,128,heads,32,dtype=torch.uint16,**options)
        self.vsn=torch.empty(batch,4,heads,128,dtype=torch.float32,**options)
        self.lengths=torch.full((batch,),count,dtype=torch.int32,**options)

    def step(self,q,key,value):
        import torch
        api=kernel_interface()
        if key.shape!=self.kr[:,:1].shape or value.shape!=key.shape or q.shape[1]!=1:
            raise ValueError('Exactly one new KV token per request required')
        slot=self.residual
        self.kr[:,slot:slot+1].copy_(key);self.vr[:,slot:slot+1].copy_(value)
        self.residual+=1
        self.consumed_finalized_blocks=max(self.consumed_finalized_blocks,self.finalized_blocks)
        out,self.kn,self.ksn,self.vn,self.vsn=api.fwd_kvcache_int(q.contiguous(),self.kp,self.ks,self.vp,self.vs,
            self.kr,self.vr,self.lengths,self.kn,self.ksn,self.vn,self.vsn,None,
            1/math.sqrt(128),'k-channel',32,128,self.residual,4)
        self.length+=1
        if self.residual==128:
            # Mirror upstream DynamicCache's chronological append dimensions.
            self.kp=torch.cat((self.kp,self.kn),dim=1)
            self.ks=torch.cat((self.ks,self.ksn),dim=1)
            self.vp=torch.cat((self.vp,self.vn),dim=1)
            self.vs=torch.cat((self.vs,self.vsn),dim=3)
            self.packed_length+=128;self.finalized_blocks+=1
            self.lengths.fill_(self.packed_length)
            self.residual=0;self.kr.zero_();self.vr.zero_()
        if self.packed_length+self.residual!=self.length:raise RuntimeError('Cache partition drift')
        return out

    def memory(self):
        packed=sum(t.numel()*t.element_size() for t in (self.kp,self.ks,self.vp,self.vs))
        tail=sum(t.numel()*t.element_size() for t in (self.kr,self.vr))
        staging=sum(t.numel()*t.element_size() for t in (self.kn,self.ksn,self.vn,self.vsn,self.lengths))
        return {'packed_bytes':packed,'fp16_append_capacity_bytes':tail,'persistent_staging_bytes':staging,
            'total_persistent_bytes':packed+tail+staging,'packed_tokens':self.packed_length,
            'fp16_valid_tokens':self.residual,'logical_tokens':self.length}
