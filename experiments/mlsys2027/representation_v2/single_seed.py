"""One-seed adjacent fresh-process PG baseline / folded-value validation."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid
from prototype import ROOT, digest, write_json


def main():
    sys.path.insert(0,str(ROOT/'diagnostics'))
    import mlsys_rtx5090_entry as base
    import mlsys_rtx5090_step02 as previous
    old=ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'
    original=json.loads((old/'manifest.json').read_text())
    command=json.loads((old/'block_1_invocation.json').read_text())['command']
    # Explicit prospective reporting revision; no numerical cosine acceptance gate.
    command[command.index('--min-logits-cosine')+1]='-1'
    idle=base.idle_preflight(0)
    gpu=idle[-1]['uuid']
    import fcntl
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=list(original['source_sha256'])+['scripts/page_gauge_value_conditioning.py',
                 'experiments/mlsys2027/representation_v2/single_seed.py',
                 'experiments/mlsys2027/representation_v2/run.sh']
        source_hashes={name:digest(ROOT/name) for name in paths}
        for name,evidence in original['input_file_evidence'].items():
            if digest(name)!=evidence['sha256']:
                raise RuntimeError('Original model/corpus/ABI input changed: '+name)
        out=ROOT/'results/mlsys2027_representation_v2'/('single_seed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        write_json(out/'manifest.json',{'seed':20260861,'order':['none','folded_prefill_rms'],
                   'command_template':command,'source_sha256':source_hashes,'inputs':original['input_file_evidence'],
                   'practical_latency_tolerance_fraction':0.005,
                   'scope':'One-seed development check; <=0.5% slowdown is operational tolerance, not proof of zero cost or a statistical noninferiority claim',
                   'cosine_reporting':'descriptive; -1 disables former cutoff; top1>=0.99 retained','idle_check':idle})
        print('Single-seed output: '+str(out),flush=True)
        payloads=[]
        processes=[]
        for index,mode in enumerate(('none','folded_prefill_rms')):
            base.idle_preflight(0)
            for name,sha in source_hashes.items():
                if digest(ROOT/name)!=sha:raise RuntimeError('Frozen source changed: '+name)
            cmd=list(command)
            target=out/f'block_{index}.json'
            cmd[cmd.index('--output')+1]=str(target)
            os.environ['PAGEGAUGE_VALUE_CONDITIONING']=mode
            print(f'Starting {mode}, seed 20260861, B4/C20480/D1536',flush=True)
            process=previous.run_process(cmd,out,{'index':index},gpu)
            write_json(out/f'block_{index}_completion.json',process)
            if not target.exists() or process['return_code'] not in (0,2):
                raise RuntimeError('Worker failed; evidence preserved')
            p=json.loads(target.read_text())
            for name,sha in p['source_sha256'].items():
                if source_hashes.get(name)!=sha:raise RuntimeError('Unfrozen worker source: '+name)
            for name,sha in source_hashes.items():
                if digest(ROOT/name)!=sha:raise RuntimeError('Source drift during worker: '+name)
            same=p['correctness']['same_backend_eager_vs_graph']
            recurrence=p['correctness']['runtime_page_finalization_and_consumption']
            if not (same['passed'] and recurrence['passed'] and p['cuda_graph_provenance']['structure_gate_passed']
                    and process['sampled_exclusivity_passed']):
                raise RuntimeError('Execution/exclusivity check failed')
            if p['configuration']['value_conditioning_mode']!=mode:
                raise RuntimeError('Conditioning option not applied')
            payloads.append(p);processes.append(process)
        a,b=payloads
        if a['trajectory']['teacher_inputs_sha256']!=b['trajectory']['teacher_inputs_sha256']:
            raise RuntimeError('Different teacher input chains')
        timing={}
        for mode in ('cache_neutral','cache_hot'):
            first=a['timing_modes'][mode]['wall']['mean_ms_per_decode_step']
            second=b['timing_modes'][mode]['wall']['mean_ms_per_decode_step']
            timing[mode]={'baseline_ms':first,'folded_ms':second,'folded_over_baseline':second/first,
                          'within_0_5_percent_slowdown':second/first<=1.005,
                          'baseline_samples_ms':a['timing_modes'][mode]['wall']['raw_ms'],
                          'folded_samples_ms':b['timing_modes'][mode]['wall']['raw_ms']}
        quality=[]
        for p in payloads:
            hf=p['correctness']['backend_vs_hf_sdpa_fp16']
            quality.append({'conditioning':p['configuration']['value_conditioning_mode'],
                            'min_cosine':hf['logits']['minimum_cosine'],
                            'top1':hf['logits']['top1_agreement_fraction'],'top1_passed':hf['passed']})
        result={'timing':timing,'quality':quality,'value_conditioning':b['value_conditioning'],
                'same_timed_work_counters':a['timed_work']==b['timed_work'],
                'baseline_served_bytes':a['cache_build']['selected_backend_cache_served_bytes_excluding_following_canary'],
                'folded_served_bytes':b['cache_build']['selected_backend_cache_served_bytes_excluding_following_canary'],
                'execution_passed':True,'one_seed_only':True,'production_default_changed':False,
                'scope':'No heldout quality or publication speed claim; single adjacent pair has order/noise limitations'}
        write_json(out/'analysis.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
