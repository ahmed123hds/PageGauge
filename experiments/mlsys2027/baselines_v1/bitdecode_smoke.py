"""Disclosed SM120 port: exactly representable INT2/INT4 history + FP16 tail.

Not a general quantization-quality, model, recurrence or timing result.
"""
from datetime import datetime,timezone
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path('/home/anonymous/pagegauge_baselines/BitDecoding')
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits',type=int,nargs='+',choices=(2,4),default=[2,4])
    args=parser.parse_args()
    import fcntl
    idle=base.idle_preflight(0);gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES']=gpu
        import torch
        import torch.nn.functional as F
        import bit_decode_cuda
        import bit_decode.bit_decode_interface as api
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(2026090816)
        out=ROOT/'results/mlsys2027_baselines_v1'/('bitdecode_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        paths=[Path(__file__),Path(bit_decode_cuda.__file__),Path(api.__file__),SOURCE/'setup.py',SOURCE/'csrc/bit_decode/decode_api.cpp']
        manifest={'idle':idle,'source_sha256':{str(p):base.sha256_file(p) for p in paths},
            'commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
            'port_patch':subprocess.check_output(['git','-C',str(SOURCE),'diff','--','setup.py','csrc/bit_decode/decode_api.cpp'],text=True),
            'cases':[[b,h,bits] for b,h in ((1,8),(1,32),(4,8)) for bits in args.bits],
            'context':256,'hq':32,'head_dim':128,'group':32,'residual_capacity':128,'tail_tokens':1,
            'maximum_relative_l2':.003,'maximum_absolute_error':.03,
            'scope':'SM120 portability numerical smoke. Binary endpoint data makes page/channel min-max scale exactly 1. No timing claim.'}
        base.atomic_json(out/'manifest.json',manifest)
        print('BitDecoding SM120 smoke: '+str(out),flush=True)
        rows=[]
        try:
            for batch,heads,bits in manifest['cases']:
                levels=2**bits-1;pack=16//bits;context=256;dim=128
                k=(torch.randint(0,2,(batch,context,heads,dim),device='cuda')*levels).half()
                v=(torch.randint(0,2,k.shape,device='cuda')*levels).half()
                k[:,0::32]=0;k[:,1::32]=levels
                v[:,:,:,0::32]=0;v[:,:,:,1::32]=levels
                q=torch.randn(batch,1,32,dim,device='cuda',dtype=torch.float16)*.03
                kp=torch.zeros(batch,context//pack,heads,dim,device='cuda',dtype=torch.uint16)
                ks=torch.zeros(batch,context//32,heads,dim,device='cuda',dtype=torch.float32)
                vp=torch.zeros(batch,context,heads,dim//pack,device='cuda',dtype=torch.uint16)
                vs=torch.zeros(batch,dim//32,heads,context,device='cuda',dtype=torch.float32)
                indptr=torch.arange(0,(batch+1)*context,context,device='cuda',dtype=torch.int32)
                api.kvcache_pack_int(k,kp,ks,v,vp,vs,None,indptr,context,'k-channel',32,bits)
                torch.cuda.synchronize()
                kr=torch.zeros(batch,128,heads,dim,device='cuda',dtype=torch.float16)
                vr=torch.zeros_like(kr)
                kr[:,:1]=torch.randn_like(kr[:,:1])*.1
                vr[:,:1]=torch.randn_like(vr[:,:1])*.1
                knew=torch.empty(batch,128//pack,heads,dim,device='cuda',dtype=torch.uint16)
                ksnew=torch.empty(batch,128//32,heads,dim,device='cuda',dtype=torch.float32)
                vnew=torch.empty(batch,128,heads,dim//pack,device='cuda',dtype=torch.uint16)
                vsnew=torch.empty(batch,dim//32,heads,128,device='cuda',dtype=torch.float32)
                lengths=torch.full((batch,),context,device='cuda',dtype=torch.int32)
                actual,*_=api.fwd_kvcache_int(q,kp,ks,vp,vs,kr,vr,lengths,knew,ksnew,vnew,vsnew,
                    None,1/math.sqrt(dim),'k-channel',32,128,1,bits)
                torch.cuda.synchronize()
                key=torch.cat((k,kr[:,:1]),dim=1).transpose(1,2).float().repeat_interleave(32//heads,dim=1)
                value=torch.cat((v,vr[:,:1]),dim=1).transpose(1,2).float().repeat_interleave(32//heads,dim=1)
                scores=q.transpose(1,2).float()@key.transpose(-2,-1)/math.sqrt(dim)
                expected=(scores.softmax(-1)@value).transpose(1,2)
                delta=actual.float()-expected
                relative=float(delta.norm()/expected.norm().clamp_min(1e-12));maximum=float(delta.abs().max())
                finite=bool(torch.isfinite(actual).all())
                row={'batch':batch,'hkv':heads,'bits':bits,'relative_l2':relative,'max_abs':maximum,
                    'passed':finite and relative<=.003 and maximum<=.03}
                rows.append(row);print(json.dumps(row),flush=True)
            for name,sha in manifest['source_sha256'].items():
                if base.sha256_file(Path(name))!=sha:raise RuntimeError('Source drift '+name)
            base.atomic_json(out/'analysis.json',{'passed':all(r['passed'] for r in rows),'rows':rows,
                'scope':manifest['scope']})
            if not all(r['passed'] for r in rows):raise SystemExit(2)
        except BaseException as error:
            base.atomic_json(out/'failure.json',{'error_type':type(error).__name__,'error':str(error),
                'traceback':traceback.format_exc()})
            raise


if __name__=='__main__':main()
