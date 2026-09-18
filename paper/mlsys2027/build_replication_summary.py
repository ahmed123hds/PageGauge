"""Verify completed regional replication evidence before typesetting its claims."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/mlsys2027/ablation_v1'))
from regional_replication import previous, base, assess, reduce_results
from recover_fi_a0 import match_work

RUNS = [('regional_reference_a0_20260909T035751Z_48404aa1','analysis.json'),
        ('regional_fi_a0_20260909T053631Z_bcb13816','recovered_analysis.json')]

def validate(run, filename):
    m=json.loads((run/'manifest.json').read_text()); previous.verify_frozen(m)
    result=json.loads((run/filename).read_text())
    previous.require(result['status']=='complete' and result['execution_passed'], 'Incomplete replication')
    previous.require(result['manifest_sha256']==m['manifest_sha256'], 'Wrong manifest')
    payloads=[]; hashes={}
    for block in m['schedule']:
        i=block['index']; p=json.loads((run/f'block_{i}.json').read_text())
        c=json.loads((run/f'block_{i}_completion.json').read_text())
        a=json.loads((run/f'block_{i}_assessment.json').read_text())
        digest=base.sha256_file(run/f'block_{i}.json')
        previous.require(digest==a['result_sha256'], 'Raw result changed')
        previous.require(c['own_pid_seen'] and c['sampled_exclusivity_passed'], 'Ownership failed')
        for suffix,key in (('.log','log_sha256'),('_telemetry.json','telemetry_sha256')):
            previous.require(base.sha256_file(run/f'block_{i}{suffix}')==c[key], 'Monitoring changed')
        if block['backend']=='page_gauge': assess(p,block,m,c)
        else: previous.validate_worker(p,block,m,c['return_code'])
        payloads.append(p); hashes[f'block_{i}.json']=digest
    if filename=='recovered_analysis.json':
        recovery=result['recovery']
        previous.require(base.sha256_file(ROOT/'experiments/mlsys2027/ablation_v1/recover_fi_a0.py')==recovery['source_sha256'], 'Recovery source changed')
        for name,digest in recovery['input_sha256'].items():
            previous.require(base.sha256_file(run/name)==digest, 'Recovery input changed')
        # Validate all non-identical overhead fields, then make equal copies only
        # for the original reducer's comparison; timing samples remain untouched.
        import copy
        normalized=copy.deepcopy(payloads)
        for i in range(0,8,2):
            match_work(payloads[i],payloads[i+1])
            normalized[i+1]['timed_work']=normalized[i]['timed_work']
        calculated=reduce_results(normalized,m['schedule'])
    else: calculated=reduce_results(payloads,m['schedule'])
    previous.require(calculated['endpoints']==result['endpoints'], 'Reduction does not reproduce')
    return result, {'analysis_sha256':base.sha256_file(run/filename),'raw_sha256':hashes}

def main():
    results=[]; evidence={}
    for name,filename in RUNS:
        r,e=validate(ROOT/'results/mlsys2027_ablation_v1'/name,filename)
        results.append(r); evidence[name]=e
    def display(r):
        v=r['endpoints']['cache_neutral.wall_ms']; lo,hi=v['speedup_95_ci']
        return f"{v['reference_over_a0']:.5f} [{lo:.5f}, {hi:.5f}]"
    paragraph=(r'\paragraph{Balanced A0 replication.} '
        'Two subsequent contrasts each use eight fresh processes in ABBA/BAAB order, '
        'four adjacent pairs and two exposed TRAIN fixture clusters. With history160/exact32, '
        'the neutral-wall reference-PageGauge/A0 ratio is '+display(results[0])+', '
        'and the directly measured FlashInfer/A0 ratio is '+display(results[1])+'. '
        'Brackets are hierarchical 95\\% intervals, conditional on these two fixtures. '
        'These ratios are not products of earlier speedups; prefill, capture, restoration '
        'and scrub remain outside timing. A0 retains 46.16\\% served-KV reduction versus FP16. '
        'The FI/A0 launcher finished measurement but rejected unequal wrapper counters at reduction. '
        'A separately hashed CPU recovery validates the exact one-versus-two-wrapper counts '
        'and equal remaining work, preserving all raw measurements. Two FI blocks retained '
        'auxiliary HF minimum-cosine warnings (0.988675 versus 0.995), despite top-1 agreement '
        'of 1.0; execution checks pass. This is development decoder performance, not a new '
        'quality gate pass, independent TEST, serving result or automatic policy promotion.\n')
    out=Path(__file__).resolve().parent/'generated'
    (out/'replication_summary.tex').write_text(paragraph)
    (out/'replication_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Revalidated sixteen blocks and reproduced both regional bootstrap reductions.')

if __name__=='__main__': main()
