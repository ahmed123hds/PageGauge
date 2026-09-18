"""Retain full KIVI pilot and run seven more predeclared TRAIN windows."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
PILOT=ROOT/'results/mlsys2027_baselines_v1/kivi_quality_20260908T094630Z_7cd59654'


def reduce_runs(directories):
    cells={'int2':[],'int4':[]}
    for directory in directories:
        process=json.loads((directory/'completion.json').read_text())
        if process['return_code']!=0 or not process['sampled_exclusivity_passed']:
            raise RuntimeError('Invalid native run '+str(directory))
        result=json.loads((directory/'analysis.json').read_text())
        for name in cells:
            path=directory/(name+'.json')
            if base.sha256_file(path)!=result['results'][name]['sha256']:raise RuntimeError('Changed result')
            p=json.loads(path.read_text())
            unit=p['distribution_quality']['cluster_bootstrap_units']
            if len(unit)!=1:raise ValueError('B1 cohort expected')
            cells[name].append(unit[0])
    summary={}
    for name,units in cells.items():
        if len({u['cluster_unit_id'] for u in units})!=len(units):raise ValueError('Duplicate window')
        count=sum(u['token_count'] for u in units)
        stats={key:sum(u['raw_sufficient_statistics'][key] for u in units)
               for key in units[0]['raw_sufficient_statistics']}
        summary[name]={'windows':len(units),'tokens':count,
            'ppl':math.exp(stats['candidate_nll_sum_nats']/count),
            'native_hf_ppl':math.exp(stats['reference_nll_sum_nats']/count),
            'ppl_ratio_to_native_hf':math.exp(stats['nll_delta_sum_nats']/count),
            'mean_kl_to_native_hf':stats['forward_kl_sum_nats']/count,
            'top1_agreement':stats['top1_agreement_count']/count,
            'true_token_top1_accuracy':stats['candidate_true_token_top1_count']/count}
    return {'summary':summary,'runs':[str(p) for p in directories],
        'scope':'Exposed TRAIN cohort; native KIVI transformers 4.36.2 with shared FP16 eager prefill. No speed or final TEST claim.',
        'cross_stack_warning':'PageGauge uses transformers 4.57.6; retain each native HF reference when comparing.'}


def main():
    offsets=[472000+23600*i for i in range(1,8)]
    old=json.loads((PILOT/'manifest.json').read_text())
    paths={Path(p) for p in old['source_sha256']}
    paths.add(Path(__file__))
    hashes={str(p):base.sha256_file(p) for p in paths}
    out=ROOT/'results/mlsys2027_baselines_v1'/('kivi_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json',{'pilot':str(PILOT),'pilot_analysis_sha256':base.sha256_file(PILOT/'analysis.json'),
        'new_offsets':offsets,'source_sha256':hashes,'orchestrator_pid':os.getpid(),
        'scope':'Seven new baseline runs plus retained pilot on the same eight exposed TRAIN windows as E0'})
    print('KIVI quality suite: '+str(out),flush=True)
    directories=[PILOT]
    for index,offset in enumerate(offsets):
        for name,sha in hashes.items():
            if base.sha256_file(Path(name))!=sha:raise RuntimeError('Source changed '+name)
        command=[sys.executable,'-u',str(Path(__file__).with_name('kivi_quality.py')),'--full','--offset',str(offset)]
        target=None
        with (out/f'window_{index}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    if line.startswith('KIVI pretrained quality: '):target=Path(line.strip().split(': ',1)[1])
                code=process.wait()
            finally:
                if process.poll() is None:process.terminate();process.wait()
            base.atomic_json(out/f'window_{index}_completion.json',{'offset':offset,'return_code':code,
                'run':str(target) if target else None,'command':command})
        if code!=0 or target is None:raise RuntimeError('Native quality window failed; retain prior evidence')
        directories.append(target)
    for name,sha in hashes.items():
        if base.sha256_file(Path(name))!=sha:raise RuntimeError('Source changed '+name)
    result=reduce_runs(directories)
    base.atomic_json(out/'analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
