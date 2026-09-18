"""E0 development expansion: retain pilot plus seven predeclared TRAIN windows."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import uuid
import numpy as np
from prototype import ROOT, digest, write_json

PILOT = ROOT/'results/mlsys2027_representation_v2/ground_truth_20260908T074422Z_8ab9bf75'
VARIANTS = ('original_fi','original_pg','fixed_pg','fixed_fi')


def reduce_runs(directories):
    cells = {name:[] for name in VARIANTS}
    for directory in directories:
        completion=json.loads((directory/'completion.json').read_text())
        if completion['return_code'] != 0 or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('Unsuccessful run '+str(directory))
        analysis=json.loads((directory/'analysis.json').read_text())
        for name in VARIANTS:
            path=directory/(name+'.json')
            if digest(path) != analysis['results'][name]['sha256']:
                raise RuntimeError('Changed variant output')
            p=json.loads(path.read_text())
            units=p['distribution_quality']['cluster_bootstrap_units']
            if len(units)!=1:raise ValueError('B1 window expected')
            cells[name].append(units[0])
    summary={}
    for name,units in cells.items():
        count=sum(u['token_count'] for u in units)
        stats={key:sum(u['raw_sufficient_statistics'][key] for u in units)
               for key in units[0]['raw_sufficient_statistics']}
        summary[name]={'tokens':count,'windows':len(units),
            'ppl':math.exp(stats['candidate_nll_sum_nats']/count),
            'hf_ppl':math.exp(stats['reference_nll_sum_nats']/count),
            'ppl_ratio_to_hf':math.exp(stats['nll_delta_sum_nats']/count),
            'mean_kl_to_hf':stats['forward_kl_sum_nats']/count,
            'top1_agreement':stats['top1_agreement_count']/count,
            'true_token_top1_accuracy':stats['candidate_true_token_top1_count']/count}
    baseline=cells['original_pg'];fixed=cells['fixed_pg']
    for a,b in zip(baseline,fixed):
        if a['cluster_unit_id']!=b['cluster_unit_id']:raise ValueError('Unpaired windows')
    delta=np.array([b['raw_sufficient_statistics']['candidate_nll_sum_nats']-a['raw_sufficient_statistics']['candidate_nll_sum_nats']
                    for a,b in zip(baseline,fixed)])
    counts=np.array([a['token_count'] for a in baseline])
    samples=np.random.default_rng(20260908).integers(0,len(counts),size=(5000,len(counts)))
    ratios=np.exp(delta[samples].sum(1)/counts[samples].sum(1))
    return {'summary':summary,'fixed_over_original_pg_ppl':{
        'point':float(np.exp(delta.sum()/counts.sum())),
        'descriptive_window_bootstrap95':np.quantile(ratios,[.025,.975]).tolist(),
        'windows_with_lower_nll':int((delta<0).sum()),'windows_with_higher_nll':int((delta>0).sum())},
        'scope':'Exposed TRAIN development cohort, including an already-seen pilot; window uncertainty is not independent-document or final TEST inference',
        'production_default_changed':False}


def main():
    manifest=json.loads((PILOT/'manifest.json').read_text())
    hashes=dict(manifest['source_sha256'])
    hashes[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    offsets=[472000+23600*i for i in range(1,8)]
    out=ROOT/'results/mlsys2027_representation_v2'/('quality_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    write_json(out/'manifest.json',{'stage':'E0_eight_window_development',
        'pilot':str(PILOT),'pilot_analysis_sha256':digest(PILOT/'analysis.json'),
        'new_offsets':offsets,'source_sha256':hashes,'orchestrator_pid':os.getpid(),
        'scope':'Seven new TRAIN windows plus retained pilot; no TEST or publication claim'})
    print('Quality suite: '+str(out),flush=True)
    directories=[PILOT]
    for index,offset in enumerate(offsets):
        for name,sha in hashes.items():
            if digest(ROOT/name)!=sha:raise RuntimeError('Frozen source drift '+name)
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('ground_truth_quality.py')),'--offset',str(offset)]
        print('Starting development offset '+str(offset),flush=True)
        target=None
        with (out/f'window_{index}.log').open('x') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                    if line.startswith('Ground-truth output: '):target=Path(line.strip().split(': ',1)[1])
                code=process.wait()
            finally:
                if process.poll() is None:process.terminate();process.wait()
            write_json(out/f'window_{index}_completion.json',{'offset':offset,'return_code':code,
                'run':str(target) if target else None,'command':cmd})
        if code!=0 or target is None:raise RuntimeError('Quality window failed; do not discard previous windows')
        directories.append(target)
    result=reduce_runs(directories)
    result['runs']=[str(p) for p in directories]
    write_json(out/'analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
