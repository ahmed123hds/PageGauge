"""Five fresh processes, one matched TRAIN workload: timing pilot, not final CI."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
from common_engine import DEFAULT


def main():
    order=['flashinfer_fp16','kivi_int4','page_gauge','bitdecode_int4','kivi_int2']
    pilot=json.loads((DEFAULT/'manifest.json').read_text())
    sources={Path(p) for p in pilot['source_sha256']}
    sources.update([Path(__file__),Path(__file__).with_name('common_engine.py'),Path(__file__).with_name('paged_mistral_adapter.py')])
    hashes={str(p):base.sha256_file(p) for p in sources}
    out=ROOT/'results/mlsys2027_baselines_v1'/('common_engine_pilot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json',{'order':order,'fixture':str(DEFAULT),'fixture_manifest_sha256':base.sha256_file(DEFAULT/'manifest.json'),
        'source_sha256':hashes,'orchestrator_pid':os.getpid(),'context':20480,'decode_steps':1536,'batch_size':1,
        'timed_repeats':3,'warmup_steps':1536,
        'scope':'One exposed TRAIN workload and one fresh process per backend. Shared eager model body, no explicit cache eviction; pilot point estimates only, no hierarchical CI or optimized-system claim.'})
    print('Common-engine pilot suite: '+str(out),flush=True)
    results={}
    for i,backend in enumerate(order):
        for p,sha in hashes.items():
            if base.sha256_file(Path(p))!=sha:raise RuntimeError('Source drift '+p)
        command=[sys.executable,'-u',str(Path(__file__).with_name('common_engine.py')),
                 '--backend',backend,'--fixture',str(DEFAULT),'--steps','1536','--repeats','3']
        target=None
        with (out/f'{i}_{backend}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    if line.startswith('Common-engine run: '):target=Path(line.strip().split(': ',1)[1])
                code=process.wait()
            finally:
                if process.poll() is None:process.terminate();process.wait()
        base.atomic_json(out/f'{i}_{backend}_completion.json',{'return_code':code,'run':str(target) if target else None,'command':command})
        if code or target is None:raise RuntimeError('Pilot worker failed; inspect preserved evidence')
        process_result=json.loads((target/'completion.json').read_text())
        if process_result['return_code'] or not process_result['sampled_exclusivity_passed']:
            raise RuntimeError('Invalid worker completion')
        result=json.loads((target/'analysis.json').read_text())
        m=json.loads((target/'manifest.json').read_text())
        if m['tokens_sha256']!=pilot['tokens_sha256'] or result['validation_only'] or len(result['rows'])!=3:
            raise ValueError('Unmatched/incomplete timed run')
        rows=result['rows']
        results[backend]={'run':str(target),'analysis_sha256':base.sha256_file(target/'analysis.json'),
            'wall_ms_per_step':statistics.median(r['wall_ms_per_step'] for r in rows),
            'cuda_ms_per_step':statistics.median(r['cuda_ms_per_step'] for r in rows),
            'cache':rows[-1]['cache'],'cache_setup_seconds_by_repeat':[r['cache_setup_seconds'] for r in rows],
            'warmup':result['warmup'],'shared_prefill_seconds':result['shared_prefill_seconds']}
    for p,sha in hashes.items():
        if base.sha256_file(Path(p))!=sha:raise RuntimeError('Source drift '+p)
    pg=results['page_gauge']['wall_ms_per_step']
    for name,r in results.items():r['wall_latency_over_page_gauge']=r['wall_ms_per_step']/pg
    base.atomic_json(out/'analysis.json',{'rows':results,
        'scope':'Single-workload shared-eager-engine pilot. Independent fresh process per backend, three within-process repeats. These are point ratios, not final CIs, optimized graph performance, or prefill-inclusive serving results.'})
    print(json.dumps({k:{field:r[field] for field in ('wall_ms_per_step','cuda_ms_per_step','wall_latency_over_page_gauge')} for k,r in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
