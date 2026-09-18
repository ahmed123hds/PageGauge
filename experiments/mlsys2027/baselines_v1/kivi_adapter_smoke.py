"""Check shared-prefill adapter against native KIVI, same small random model."""
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
from kivi_adapter import imports,configure_decode,pack_initial_cache


def main():
    import fcntl
    idle=base.idle_preflight(0)
    gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES']=gpu
        import torch
        from transformers import MistralConfig,MistralForCausalLM
        native,_=imports()
        torch.set_grad_enabled(False)
        torch.manual_seed(2026090814)
        out=ROOT/'results/mlsys2027_baselines_v1'/('kivi_adapter_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'manifest.json',{'idle':idle,'source_sha256':{
            str(p):base.sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('kivi_adapter.py'))},
            'scope':'Random-model adapter equality; no pretrained quality or performance claim'})
        rows=[]
        for bits in (2,4):
            c=MistralConfig(vocab_size=256,hidden_size=512,intermediate_size=1024,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,sliding_window=None)
            c.k_bits=c.v_bits=bits;c.group_size=c.residual_length=32;c.use_flash=False;c._attn_implementation='eager'
            a=native.MistralForCausalLM_KIVI(c).eval().half().cuda()
            b=MistralForCausalLM(c).eval().half().cuda()
            b.load_state_dict(a.state_dict(),strict=True)
            tokens=torch.randint(3,256,(1,161),device='cuda')
            with torch.inference_mode():
                x=a(tokens[:,:128],use_cache=True)
                y=b(tokens[:,:128],use_cache=True)
                legacy=y.past_key_values.to_legacy_cache() if hasattr(y.past_key_values,'to_legacy_cache') else y.past_key_values
                packed=pack_initial_cache(legacy,bits)
                for left,right in zip(x.past_key_values,packed):
                    for l,r in zip(left,right):
                        if isinstance(l,torch.Tensor):
                            if not torch.equal(l,r):raise RuntimeError('Adapter/native initial cache differs')
                        elif l!=r:raise RuntimeError('Adapter/native metadata differs')
                configure_decode(b,bits)
                p=x.past_key_values
                for position in range(128,161):
                    x=a(tokens[:,position:position+1],past_key_values=p,use_cache=True)
                    y=b(tokens[:,position:position+1],past_key_values=packed,use_cache=True)
                    if not torch.equal(x.logits,y.logits):raise RuntimeError('Native/adapter decode differs')
                    p,packed=x.past_key_values,y.past_key_values
                rows.append({'bits':bits,'initial_cache_bitwise_equal':True,'decode_logits_bitwise_equal':True,'steps':33})
            del a,b,x,y,p,packed,legacy
            torch.cuda.empty_cache()
        base.atomic_json(out/'analysis.json',{'passed':True,'rows':rows,'scope':'Adapter integration only'})
        print('PASS '+str(out),flush=True)


if __name__=='__main__':main()
