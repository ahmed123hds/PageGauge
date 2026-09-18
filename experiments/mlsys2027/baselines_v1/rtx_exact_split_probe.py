"""Bounded SM120 exact-region split pilot prompted by measured B1 overhead.

Only launch partitioning changes. Same quantizer, INT8 kernel, exact token count,
and centers. Synthetic attention microbenchmark, not model-quality/e2e evidence.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import contextlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
TARGET=ROOT/'scripts/benchmark_flashinfer_page_affine_int8.py'


def worker(out,batch,split):
    import torch
    if torch.cuda.get_device_capability()!=(12,0):raise RuntimeError('SM120 required')
    spec=importlib.util.spec_from_file_location('rtx_split_micro',TARGET)
    target=importlib.util.module_from_spec(spec);spec.loader.exec_module(target)
    original=target.plan;fp16_calls=0
    def plan(*args,**kwargs):
        nonlocal fp16_calls
        if args[4]==torch.float16:
            fp16_calls+=1
            if fp16_calls==2:args=(*args[:5],split,*args[6:])
        return original(*args,**kwargs)
    target.plan=plan
    timing=target.controlled_event_times;observations=[];captured={}
    def measure(operations,*args,**kwargs):
        observations.append(previous.process_snapshot(os.environ['CUDA_VISIBLE_DEVICES'],os.getpid()))
        result=timing(operations,*args,**kwargs)
        observations.append(previous.process_snapshot(os.environ['CUDA_VISIBLE_DEVICES'],os.getpid()))
        candidate=next(op for name,op in operations.items() if name!='flashinfer_fp16')
        captured['output']=inspect.getclosurevars(candidate).nonlocals['affine_output'].detach().cpu().clone()
        return result
    target.controlled_event_times=measure
    sys.argv=[str(TARGET),'--lengths',','.join(['22016']*batch),'--representation','page_gauge',
        '--exact-tail','768','--exact-sink','2112','--baseline-fixed-split-pages','256',
        '--candidate-fixed-split-pages','128','--warmup','20','--repeats','100',
        '--cache-scrub-mib','256','--seed','2026090818','--output',str(out)]
    with contextlib.redirect_stdout(io.StringIO()):target.main()
    if fp16_calls!=2:raise RuntimeError('Unexpected plan order')
    result=json.loads(out.read_text())
    result['exact_split_override']={'pages':split,'fp16_plan_calls':fp16_calls,
        'scope':'Only second FP16 plan (exact region); full FI and INT8 history plans unchanged'}
    result['worker_boundary_observations']=observations
    result['worker_boundary_exclusivity_passed']=len(observations)==2 and all(
        os.getpid() in r['pids'] and not r['unexpected_pids'] for r in observations)
    torch.save(captured,out.with_suffix('.pt'))
    result['split_output_sha256']=base.sha256_file(out.with_suffix('.pt'))
    base.atomic_json(out,result)
    print(f'B{batch} split{split}: '+json.dumps(result['speedup_over_fp16_p50']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--batch',type=int);p.add_argument('--split',type=int)
    a=p.parse_args()
    if a.worker:worker(a.worker,a.batch,a.split);return
    import fcntl
    idle=base.idle_preflight(0);gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads((ROOT/'results/mlsys2027_baselines_v1/common_engine_page_gauge_20260908T114547Z_6931ac52/manifest.json').read_text())
        paths={Path(x) for x in old['source_sha256']};paths.add(Path(__file__))
        hashes={str(x):base.sha256_file(x) for x in paths}
        cases=[{'batch':batch,'exact_split_pages':split} for batch in (1,4) for split in (128,32,64)]
        out=ROOT/'results/mlsys2027_baselines_v1'/('rtx_exact_split_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'manifest.json',{'cases':cases,'source_sha256':hashes,'idle':idle,'orchestrator_pid':os.getpid(),
            'numerical_relative_l2_max_tolerance':0.005,'numerical_absolute_tolerance':0.0001,
            'scope':'Synthetic B1/B4 length22016; same2880 exact tokens, arranged prefix2112/tail768. Not an actual S4/A128 model-cache trajectory. Numerical execution check compares the same quantized cache across split settings (control128); raw original-FP16 error is descriptive, not a predictive-quality cutoff.'})
        print('RTX exact split probe: '+str(out),flush=True)
        import torch
        rows=[];controls={}
        for i,case in enumerate(cases):
            for path,sha in hashes.items():
                if base.sha256_file(Path(path))!=sha:raise RuntimeError('Source changed '+path)
            result_path=out/f'case_{i}.json'
            command=[sys.executable,'-u',str(Path(__file__)),'--worker',str(result_path),
                '--batch',str(case['batch']),'--split',str(case['exact_split_pages'])]
            completion=previous.run_process(command,out,{'index':i},gpu)
            base.atomic_json(out/f'case_{i}_completion.json',completion)
            if completion['return_code']:raise RuntimeError('Probe execution failed')
            r=json.loads(result_path.read_text())
            telemetry=json.loads((out/f'block_{i}_telemetry.json').read_text())
            combined=r['worker_boundary_exclusivity_passed'] and all(
                not x.get('error') and not x.get('unexpected_pids') for x in telemetry['observations'])
            completion['worker_boundary_exclusivity_passed']=r['worker_boundary_exclusivity_passed']
            completion['combined_sampled_exclusivity_passed']=combined
            base.atomic_json(out/f'case_{i}_completion.json',completion)
            if not combined:raise RuntimeError('Short probe exclusivity checks failed')
            tensor_path=result_path.with_suffix('.pt')
            if base.sha256_file(tensor_path)!=r['split_output_sha256']:raise RuntimeError('Changed output')
            observed=torch.load(tensor_path,weights_only=True,map_location='cpu')['output'].double()
            if case['exact_split_pages']==128:controls[case['batch']]=observed
            control=controls[case['batch']]
            error=float(((observed-control).norm(dim=-1)/control.norm(dim=-1).clamp_min(1e-20)).max())
            absolute=float((observed-control).abs().max())
            if not math.isfinite(error) or error>0.005 or absolute>0.0001:raise RuntimeError('Split numerical execution check failed')
            rows.append({**case,'result_sha256':base.sha256_file(result_path),'relative_l2_max':error,
                'absolute_max_vs_same_quantized_control':absolute,
                'original_fp16_error_descriptive':r['correctness'],
                'modes':{mode:{'fi_ms':v['timings']['flashinfer_fp16']['p50_ms'],
                    'pg_ms':v['timings']['page_gauge_int8_exact_segments']['p50_ms']} for mode,v in r['timing_modes'].items()}})
        for path,sha in hashes.items():
            if base.sha256_file(Path(path))!=sha:raise RuntimeError('Source changed '+path)
        base.atomic_json(out/'analysis.json',{'rows':rows,'scope':'Synthetic split-only pilot; no automatic production promotion or e2e claim'})
        print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
