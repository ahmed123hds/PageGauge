"""Eight matched optimized B4 fresh-process blocks: PG exact128 versus exact32."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
sys.path.insert(0,str(ROOT/'experiments/mlsys2027/factorization_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from full_model import assess


def validate_experiment(p,b,m):
    r=p['exact_split_experiment'];o=r['observed_planning'];require=previous.require
    require(r['exact_split_pages']==b['exact_split_pages'],'Wrong exact split')
    require(r['adapter_sha256']==m['source_sha256']['experiments/mlsys2027/baselines_v1/optimized_split_worker.py'],'Wrong adapter')
    require(not r['production_sources_modified'] and not r['quantization_or_kernel_changed'],'Changed representation/kernel')
    require(o['decoder_instances']>0 and o['exact_plan_calls']>=1536 and o['old_plan_calls']==o['exact_plan_calls'],'Missing observed plans')
    require(o['requested_exact_splits']==[128] and o['effective_exact_splits']==[b['exact_split_pages']] and o['old_splits']==[128],'Plan override drift')
    capacity=p['scheduler_capacity']['wrappers']
    require(capacity['exact_fp16']['fixed_split_pages']==b['exact_split_pages'] and capacity['old_int8']['fixed_split_pages']==128,'Wrong actual scheduler report')


def reduce_results(payloads,blocks):
    if len(payloads)!=8:raise ValueError('All eight blocks required')
    endpoints={}
    for mode in previous.MODES:
        for metric in previous.METRICS:
            rows=[]
            for i in range(0,8,2):
                a=next(j for j in (i,i+1) if blocks[j]['exact_split_pages']==128)
                b=next(j for j in (i,i+1) if blocks[j]['exact_split_pages']==32)
                for key in previous.MATCHED_FIELDS:
                    previous.require(payloads[a]['pairing']['configuration'][key]==payloads[b]['pairing']['configuration'][key],'Unmatched fixture '+key)
                previous.require(payloads[a]['timed_work']==payloads[b]['timed_work'],'Changed timed work')
                means=[statistics.fmean(math.log(s[metric]) for s in payloads[j]['timing_modes'][mode]['raw_samples']) for j in (a,b)]
                rows.append({'seed':blocks[i]['seed'],'pair_id':blocks[i]['pair_id'],'pair_order':blocks[i]['pair_order'],
                    'log_speedup':means[0]-means[1],'exact128_ms_per_step':math.exp(means[0])/1536,
                    'exact32_ms_per_step':math.exp(means[1])/1536})
            endpoints[mode+'.'+metric]={'exact128_over_exact32':math.exp(statistics.fmean(r['log_speedup'] for r in rows)),
                **previous._hierarchical_bootstrap_ci(rows,50000,2026090819),'pairs':rows}
    return {'endpoints':endpoints,'fresh_process_blocks':8,'fixture_clusters':2,'adjacent_pairs':4,
        'scope':'Optimized B4/C20480/D1536 PG exact-region split contrast, same mixed cache and history kernel. TRAIN fixtures; not FI-relative speed or final quality. Two fixture clusters limit generality; retain negative/tied outcomes.'}


def main():
    import fcntl
    original=json.loads((ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c/manifest.json').read_text())
    schedule=[dict(b,exact_split_pages=128 if b['backend']=='flashinfer_fp16' else 32,backend='page_gauge') for b in previous.schedule()]
    names=set(original['source_sha256'])|{'scripts/page_gauge_value_conditioning.py',
        'experiments/mlsys2027/factorization_v1/full_model.py','experiments/mlsys2027/factorization_v1/control.py',
        'experiments/mlsys2027/baselines_v1/optimized_split.py','experiments/mlsys2027/baselines_v1/optimized_split_worker.py',
        'experiments/mlsys2027/representation_v2/run.sh',
        'results/mlsys2027_baselines_v1/rtx_exact_split_20260908T115408Z_e0315ebe/analysis.json',
        'results/mlsys2027_baselines_v1/profile_attribution_20260908T120217Z_93cf930d/analysis.json'}
    idle=base.idle_preflight(0);gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m={**original,'experiment':'E2_optimized_exact_region_split','schedule':schedule,
            'source_sha256':{n:base.sha256_file(ROOT/n) for n in sorted(names)},
            'created_utc':datetime.now(timezone.utc).isoformat(),'idle':idle,
            'config':{**original['config'],'min_logits_cosine':-1.0},
            'statistics':{'primary':'cache_neutral.wall_ms','bootstrap_samples':50000,'bootstrap_seed':2026090819,
                          'interpretation':'Exact128/exact32 ratio with hierarchical interval around1, not a forced pass'},
            'scope':'Performance-only exact32 candidate after development microprobe/quality/profile; no kernel/quantizer change, no default promotion or final TEST'}
        m.pop('manifest_sha256',None);m['manifest_sha256']=base.canonical_hash(m)
        previous.verify_frozen(m,full_inputs=True)
        out=ROOT/'results/mlsys2027_baselines_v1'/('optimized_split_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True);base.atomic_json(out/'manifest.json',m)
        base.atomic_json(out/'orchestrator.json',{'pid':os.getpid(),'gpu':gpu})
        print('Optimized split comparison: '+str(out),flush=True)
        os.environ['PAGEGAUGE_VALUE_CONDITIONING']='none';payloads=[]
        for b in schedule:
            previous.verify_frozen(m);base.idle_preflight(0)
            os.environ['PAGEGAUGE_EXPERIMENT_EXACT_SPLIT']=str(b['exact_split_pages'])
            target=out/f"block_{b['index']}.json"
            command=previous.worker_command(b,m['inputs']['model_path'],target)
            command[2]=str(Path(__file__).with_name('optimized_split_worker.py'))
            command[command.index('--min-logits-cosine')+1]='-1'
            base.atomic_json(out/f"block_{b['index']}_invocation.json",{'command':command,'exact_split_pages':b['exact_split_pages']})
            print(f"Optimized block {b['index']+1}/8, exact{b['exact_split_pages']}",flush=True)
            c=previous.run_process(command,out,b,gpu)
            base.atomic_json(out/f"block_{b['index']}_completion.json",c)
            p=json.loads(target.read_text())
            checks=assess(p,b,m,c,experiment_validator=validate_experiment)
            previous.verify_frozen(m)
            base.atomic_json(out/f"block_{b['index']}_assessment.json",{**checks,'result_sha256':base.sha256_file(target)})
            payloads.append(p)
        previous.verify_frozen(m,full_inputs=True)
        result=reduce_results(payloads,schedule);result['manifest_sha256']=m['manifest_sha256']
        base.atomic_json(out/'analysis.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
