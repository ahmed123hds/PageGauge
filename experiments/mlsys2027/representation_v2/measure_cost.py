"""Short launch-overhead experiment; not an end-to-end speed claim."""
import json
import os
from pathlib import Path
import sys
import statistics
from datetime import datetime, timezone
import uuid
from prototype import ROOT, write_json, digest


def main():
    sys.path.insert(0,str(ROOT/'diagnostics'))
    import mlsys_rtx5090_entry as base
    idle=base.idle_preflight(0)
    gpu=idle[-1]['uuid']
    import fcntl
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES']=gpu
        import torch
        import flashinfer
        torch.manual_seed(20260861)
        torch.set_grad_enabled(False)
        a=torch.randn((4,32,128),device='cuda',dtype=torch.float16)
        b=torch.randn_like(a)
        sa=torch.randn((4,32),device='cuda')
        sb=torch.randn_like(sa)
        center=torch.randn_like(a)
        gain=torch.exp2(torch.randint(-4,5,a.shape,device='cuda')).half()
        out=torch.empty_like(a)
        lse=torch.empty_like(sa)
        def call(conditioned):
            out.copy_(a)
            lse.copy_(sa)
            if conditioned:
                out.mul_(gain)
            flashinfer.merge_state_in_place(out,lse,b,sb)
            out.add_(center)
        graphs={}
        for flag in (False,True):
            for _ in range(5):call(flag)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(128):call(flag)
            graphs[flag]=graph
        rows=[]
        for order in (False,True,True,False)*5:
            graphs[order].replay()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):graphs[order].replay()
            end.record();end.synchronize()
            rows.append({'conditioned':order,'us_per_attention_postprocess':start.elapsed_time(end)*1000/(128*10)})
        med={str(flag):statistics.median(r['us_per_attention_postprocess'] for r in rows if r['conditioned']==flag) for flag in (False,True)}
        extra=med['True']-med['False']
        result={'seed':20260861,'rows':rows,'median_us':med,'extra_us_per_layer':extra,
                'estimated_32_layer_extra_ms':extra*32/1000,
                'scope':'B4 captured postprocessing microbenchmark with identical reset copies; no cache read/append or full model',
                'alternative':'Fold shared positive diagonal into V-projection rows and output-projection columns before timing; no added decode operations',
                'idle':idle,'source_sha256':digest(__file__)}
        path=ROOT/'results/mlsys2027_representation_v2'/('cost_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        path.mkdir(parents=True)
        write_json(path/'analysis.json',result)
        print(path)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','idle')},indent=2))


if __name__=='__main__':main()
