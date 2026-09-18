"""E1 synthetic GPU validation/microtiming, only after the device is idle."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid
from control import ROOT, SOURCE, EXPECTED, make_wrapper, prepare

sys.path.insert(0, str(ROOT/'experiments/mlsys2027/representation_v2'))
from prototype import digest, write_json


def main():
    sys.path.insert(0, str(ROOT/'diagnostics'))
    import mlsys_rtx5090_entry as base
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        import torch
        import flashinfer
        sys.path.insert(0, str(ROOT/'scripts'))
        import benchmark_flashinfer_page_affine_int8 as pg
        torch.set_grad_enabled(False)
        torch.manual_seed(20260908)
        torch.backends.cuda.matmul.allow_tf32 = False
        shapes = [(1,33),(4,20480)]
        out = ROOT/'results/mlsys2027_factorization_v1'/('smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        include, control_sha = prepare()
        files = [Path(__file__), Path(__file__).with_name('control.py'), Path(__file__).with_name('run.sh'),
                 ROOT/'scripts/benchmark_flashinfer_page_affine_int8.py']
        hashes = {str(p.relative_to(ROOT)):digest(p) for p in files}
        write_json(out/'manifest.json', {'stage':'E1_synthetic_control_validation', 'shapes':shapes,
            'seed':20260908, 'source_sha256':hashes, 'production_header_sha256':EXPECTED,
            'control_header_sha256':control_sha, 'idle_check':idle,
            'scope':'Centered historical attention only; no mixed-cache/full-model or final speed claim',
            'execution_tolerance':{'atol':0.0005,'rtol':0.02}, 'fixed_split_pages':128})
        print('E1 smoke output: '+str(out), flush=True)
        work = [torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda') for _ in range(3)]
        factorized = pg.make_page_gauge_wrapper(flashinfer,work[0])
        control = make_wrapper(flashinfer,work[1])
        reference = flashinfer.BatchDecodeWithPagedKVCacheWrapper(work[2],'NHD',use_tensor_cores=True,backend='fa2')
        scrub = torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda')
        rows = []
        for batch,length in shapes:
            pages = (length+15)//16
            total = batch*pages
            codes = tuple(torch.randint(-127,128,(total,16,8,128),dtype=torch.int8,device='cuda') for _ in range(2))
            scales = tuple((torch.rand(total,8,device='cuda')*.007+.001).half() for _ in range(2))
            indices = torch.randperm(total,device='cuda').int()
            indptr = torch.arange(0,total+1,pages,device='cuda',dtype=torch.int32)
            last = torch.full((batch,), (length-1)%16+1,device='cuda',dtype=torch.int32)
            query = torch.randn(batch,32,128,device='cuda').half()
            reconstructed = tuple((code.float()*scale[:,None,:,None].float()).half() for code,scale in zip(codes,scales))
            for wrapper,dtype in ((factorized,torch.int8),(control,torch.int8),(reference,torch.float16)):
                pg.plan(wrapper,indptr,indices,last,dtype,fixed_split_pages=128)
            outputs = [torch.empty_like(query) for _ in range(3)]
            def factorized_call():
                factorized.run(query,codes,*scales,1/(128**.5),out=outputs[0])
            def control_call():
                control.run(query,codes,*scales,1/(128**.5),out=outputs[1])
            factorized_call(); control_call()
            reference.run(query,reconstructed,out=outputs[2])
            row = {'batch':batch,'tokens':length,'checks':{}}
            for label,value in zip(('factorized','register_control'),outputs[:2]):
                difference = value.float()-outputs[2].float()
                row['checks'][label] = {'max_abs_vs_explicit_fp16':float(difference.abs().max()),
                    'rmse_vs_explicit_fp16':float(difference.square().mean().sqrt()),
                    'finite':bool(torch.isfinite(value).all()),
                    'close':bool(torch.allclose(value,outputs[2],atol=.0005,rtol=.02))}
            if all(m['finite'] and m['close'] for m in row['checks'].values()):
                graphs = {}
                for name,call in (('factorized',factorized_call),('register_control',control_call)):
                    for _ in range(3):call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):call()
                    graphs[name] = graph
                row['microtiming'] = pg.controlled_event_times({name:g.replay for name,g in graphs.items()},
                    warmup=5,repeats=30,cache_scrub=lambda:scrub.zero_())
            rows.append(row)
            print(json.dumps({'batch':batch,'tokens':length,'checks':row['checks']}),flush=True)
        for name,sha in hashes.items():
            if digest(ROOT/name) != sha:raise RuntimeError('Source drift')
        if digest(SOURCE) != EXPECTED:raise RuntimeError('Production header changed')
        passed = all(m['finite'] and m['close'] for row in rows for m in row['checks'].values())
        write_json(out/'analysis.json',{'passed':passed,'rows':rows,
            'scope':'Synthetic single-process graph microbenchmark; not full E1 or publication speed confirmation'})
        if not passed:raise RuntimeError('Control correctness failed; inspect saved analysis')


if __name__ == '__main__':main()
