"""INT4 cache close/consume check: 129 appends cross the 128-token boundary."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    import fcntl
    idle=base.idle_preflight(0);gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES']=gpu
        import torch
        import bit_decode_cuda
        from bitdecode_cache import BitDecodeCache
        torch.set_grad_enabled(False);torch.manual_seed(2026090818)
        torch.backends.cuda.matmul.allow_tf32=False
        out=ROOT/'results/mlsys2027_baselines_v1'/('bitdecode_recurrence_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        paths=(Path(__file__),Path(__file__).with_name('bitdecode_cache.py'),Path(bit_decode_cuda.__file__))
        base.atomic_json(out/'manifest.json',{'source_sha256':{str(p):base.sha256_file(p) for p in paths},
            'idle':idle,'batch_sizes':[1,4],'context':256,'steps':129,'hkv':8,'hq':32,'head_dim':128,
            'atol':.03,'relative_l2_limit':.003,'scope':'Exactly representable INT4 history and appended blocks; no model-quality or speed claim'})
        print('BitDecoding recurrence: '+str(out),flush=True)
        rows=[]
        try:
            for batch in (1,4):
                k=(torch.randint(0,2,(batch,385,8,128),device='cuda')*15).half()
                v=(torch.randint(0,2,k.shape,device='cuda')*15).half()
                k[:,0::32]=0;k[:,1::32]=15
                v[:,:,:,0::32]=0;v[:,:,:,1::32]=15
                cache=BitDecodeCache(k[:,:256],v[:,:256])
                metrics=[]
                for position in range(256,385):
                    q=torch.randn(batch,1,32,128,device='cuda',dtype=torch.float16)*.03
                    actual=cache.step(q,k[:,position:position+1],v[:,position:position+1])
                    torch.cuda.synchronize()
                    key=k[:,:position+1].transpose(1,2).float().repeat_interleave(4,dim=1)
                    value=v[:,:position+1].transpose(1,2).float().repeat_interleave(4,dim=1)
                    expected=((q.transpose(1,2).float()@key.transpose(-2,-1)/math.sqrt(128)).softmax(-1)@value).transpose(1,2)
                    delta=actual.float()-expected
                    maximum=float(delta.abs().max());relative=float(delta.norm()/expected.norm().clamp_min(1e-12))
                    metrics.append({'step':position-255,'max_abs':maximum,'relative_l2':relative,
                        'passed':bool(torch.isfinite(actual).all()) and maximum<=.03 and relative<=.003})
                if cache.finalized_blocks!=1 or cache.packed_length!=384 or cache.residual!=1:
                    raise RuntimeError('Did not close and consume new packed block')
                row={'batch':batch,'passed':all(p['passed'] for p in metrics),'steps':metrics,
                    'memory':cache.memory(),'finalized_blocks':cache.finalized_blocks}
                rows.append(row)
                print(json.dumps({'batch':batch,'passed':row['passed'],'maximum_error':max(p['max_abs'] for p in metrics),
                    'boundary_steps':metrics[126:129]}),flush=True)
            base.atomic_json(out/'analysis.json',{'passed':all(r['passed'] for r in rows),'rows':rows,
                'scope':'SM120 port recurrence on exact-representable INT4 operands; not general quantization or end-to-end performance'})
            if not all(r['passed'] for r in rows):raise SystemExit(2)
        except BaseException as error:
            base.atomic_json(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
            raise


if __name__=='__main__':main()
