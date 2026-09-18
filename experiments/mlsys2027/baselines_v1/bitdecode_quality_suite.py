"""Retain the full BitDecoding-kernel pilot and complete the matched TRAIN cohort."""
import argparse
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


def reduce_runs(directories):
    units=[];token_hashes=[]
    for directory in directories:
        m=json.loads((directory/'manifest.json').read_text())
        if m['backend']!='bitdecode' or (m['context'],m['decode_steps'])!=(20480,1536):
            raise ValueError('Full recurrent BitDecoding configuration required')
        completion=json.loads((directory/'completion.json').read_text())
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Incomplete or nonexclusive run')
        a=json.loads((directory/'analysis.json').read_text())
        if base.sha256_file(directory/'int4.json')!=a['results']['int4']['sha256']:
            raise ValueError('Changed quality result')
        if base.sha256_file(directory/'tokens.json')!=m['tokens_sha256']:
            raise ValueError('Changed token fixture')
        p=json.loads((directory/'int4.json').read_text())
        if p['finalized_blocks_per_layer']!=[12]*32 or p['consumed_finalized_blocks_per_layer']!=[11]*32:
            raise ValueError('Missing recurrent cache closure/consumption')
        records=p['distribution_quality']['cluster_bootstrap_units']
        if len(records)!=1:raise ValueError('B1 windows required')
        units.extend(records);token_hashes.append(m['tokens_sha256'])
    if len({u['cluster_unit_id'] for u in units})!=len(units) or len(set(token_hashes))!=len(units):
        raise ValueError('Duplicate window')
    count=sum(u['token_count'] for u in units)
    stats={k:sum(u['raw_sufficient_statistics'][k] for u in units)
           for k in units[0]['raw_sufficient_statistics']}
    return {'summary':{'int4':{'windows':len(units),'tokens':count,
        'ppl':math.exp(stats['candidate_nll_sum_nats']/count),
        'native_hf_ppl':math.exp(stats['reference_nll_sum_nats']/count),
        'ppl_ratio_to_native_hf':math.exp(stats['nll_delta_sum_nats']/count),
        'mean_kl_to_native_hf':stats['forward_kl_sum_nats']/count,
        'top1_agreement':stats['top1_agreement_count']/count,
        'true_token_top1_accuracy':stats['candidate_true_token_top1_count']/count}},
        'runs':[str(p) for p in directories],
        'scope':'Eight exposed TRAIN windows. BitDecoding INT4 SM120 kernel integration, not upstream Mistral engine or final TEST. Shared FP16 prefill; no timing claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    args=parser.parse_args();pilot=args.pilot.resolve()
    reduce_runs([pilot])
    old=json.loads((pilot/'manifest.json').read_text())
    sources={Path(p) for p in old['source_sha256']};sources.add(Path(__file__))
    hashes={str(p):base.sha256_file(p) for p in sources}
    # Refuse stale pilot code, rather than silently combine changed algorithms.
    for p,sha in old['source_sha256'].items():
        if hashes[p]!=sha:raise RuntimeError('Pilot source no longer matches '+p)
    offsets=[472000+23600*i for i in range(1,8)]
    out=ROOT/'results/mlsys2027_baselines_v1'/('bitdecode_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json',{'pilot':str(pilot),'pilot_analysis_sha256':base.sha256_file(pilot/'analysis.json'),
        'new_offsets':offsets,'source_sha256':hashes,'orchestrator_pid':os.getpid(),
        'scope':'Seven new matched TRAIN windows plus retained full pilot, no method selection or TEST'})
    print('BitDecoding quality suite: '+str(out),flush=True)
    directories=[pilot]
    for i,offset in enumerate(offsets):
        for p,sha in hashes.items():
            if base.sha256_file(Path(p))!=sha:raise RuntimeError('Source drift '+p)
        command=[sys.executable,'-u',str(Path(__file__).with_name('kivi_quality.py')),
                 '--backend','bitdecode','--full','--offset',str(offset)]
        target=None
        with (out/f'window_{i}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    if line.startswith('BitDecoding pretrained quality: '):target=Path(line.strip().split(': ',1)[1])
                code=process.wait()
            finally:
                if process.poll() is None:process.terminate();process.wait()
        base.atomic_json(out/f'window_{i}_completion.json',{'offset':offset,'return_code':code,
            'run':str(target) if target else None,'command':command})
        if code or target is None:raise RuntimeError('Baseline window failed; prior evidence preserved')
        directories.append(target)
    for p,sha in hashes.items():
        if base.sha256_file(Path(p))!=sha:raise RuntimeError('Source drift '+p)
    result=reduce_runs(directories)
    base.atomic_json(out/'analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
