"""Matched TRAIN quality/cache frontier; preserves each implementation's HF reference."""
from datetime import datetime,timezone
import argparse
import json
import math
from pathlib import Path
import sys
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
PG=ROOT/'results/mlsys2027_representation_v2/quality_suite_20260908T075133Z_92588ba6/analysis.json'
KIVI=ROOT/'results/mlsys2027_baselines_v1/kivi_suite_20260908T095326Z_13eb0281/analysis.json'


def summarize(units,cache_bytes):
    if len(set(cache_bytes))!=1:raise ValueError('Cache capacities differ across this fixed-shape cohort')
    if len({u['cluster_unit_id'] for u in units})!=len(units):raise ValueError('Duplicate quality window')
    counts=np.array([u['token_count'] for u in units])
    stats={key:sum(u['raw_sufficient_statistics'][key] for u in units)
           for key in units[0]['raw_sufficient_statistics']}
    count=int(counts.sum())
    delta=np.array([u['raw_sufficient_statistics']['nll_delta_sum_nats'] for u in units])
    samples=np.random.default_rng(2026090817).integers(0,len(units),size=(5000,len(units)))
    bootstrap=np.exp(delta[samples].sum(1)/counts[samples].sum(1))
    return {'windows':len(units),'tokens':count,
        'ppl':math.exp(stats['candidate_nll_sum_nats']/count),
        'native_hf_ppl':math.exp(stats['reference_nll_sum_nats']/count),
        'ppl_ratio_to_native_hf':math.exp(stats['nll_delta_sum_nats']/count),
        'descriptive_window_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist(),
        'top1_agreement_to_native_hf':stats['top1_agreement_count']/count,
        'mean_kl_to_native_hf':stats['forward_kl_sum_nats']/count,
        'true_token_top1_accuracy':stats['candidate_true_token_top1_count']/count,
        'cache_bytes':cache_bytes[0]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bitdecode',type=Path,help='Completed eight-window BitDecoding suite analysis.json')
    args=parser.parse_args()
    pg=json.loads(PG.read_text());native=json.loads(KIVI.read_text())
    pg_dirs=[Path(p) for p in pg['runs']];native_dirs=[Path(p) for p in native['runs']]
    if len(pg_dirs)!=8 or len(native_dirs)!=8:raise ValueError('Full eight-window cohorts required')
    names=('flashinfer_fp16','pagegauge_int8','kivi_int2','kivi_int4')
    bitdecode_dirs=None
    if args.bitdecode:
        bitdecode_dirs=[Path(p) for p in json.loads(args.bitdecode.read_text())['runs']]
        if len(bitdecode_dirs)!=8:raise ValueError('Full BitDecoding cohort required')
        names+=('bitdecode_int4',)
    cells={name:[] for name in names};memory={name:[] for name in names}
    evidence={str(PG):base.sha256_file(PG),str(KIVI):base.sha256_file(KIVI)}
    if args.bitdecode:evidence[str(args.bitdecode)]=base.sha256_file(args.bitdecode)
    staging=[]
    for index,(pg_dir,native_dir) in enumerate(zip(pg_dirs,native_dirs)):
        pg_tokens=json.loads((pg_dir/'token_provenance.json').read_text())
        native_manifest=json.loads((native_dir/'manifest.json').read_text())
        if pg_tokens['token_ids_sha256']!=native_manifest['token_provenance']['token_ids_sha256']:
            raise ValueError('Different actual token IDs across stacks')
        for directory in (pg_dir,native_dir):
            completion=json.loads((directory/'completion.json').read_text())
            if completion['return_code']!=0 or not completion['sampled_exclusivity_passed']:
                raise RuntimeError('Invalid constituent run')
        for name,key,directory in (('flashinfer_fp16','original_fi',pg_dir),('pagegauge_int8','original_pg',pg_dir),
                                   ('kivi_int2','int2',native_dir),('kivi_int4','int4',native_dir)):
            path=directory/(key+'.json')
            parent=json.loads((directory/'analysis.json').read_text())
            digest=base.sha256_file(path)
            if digest!=parent['results'][key]['sha256']:raise RuntimeError('Changed result')
            evidence[str(path)]=digest
            result=json.loads(path.read_text())
            unit=result['distribution_quality']['cluster_bootstrap_units']
            if len(unit)!=1:raise ValueError('Expected B1 quality unit')
            cells[name].append(unit[0])
            memory[name].append(result['cache_accounting']['final']['unique_storage_bytes']
                if name.startswith('kivi') else result['cache_storage_bytes'])
        if bitdecode_dirs:
            directory=bitdecode_dirs[index]
            m=json.loads((directory/'manifest.json').read_text())
            c=json.loads((directory/'completion.json').read_text())
            if m['backend']!='bitdecode' or m['token_provenance']['token_ids_sha256']!=pg_tokens['token_ids_sha256']:
                raise ValueError('Mismatched BitDecoding fixture')
            if c['return_code'] or not c['sampled_exclusivity_passed']:raise ValueError('Invalid BitDecoding run')
            path=directory/'int4.json';digest=base.sha256_file(path)
            if digest!=json.loads((directory/'analysis.json').read_text())['results']['int4']['sha256']:
                raise ValueError('Changed BitDecoding result')
            evidence[str(path)]=digest
            r=json.loads(path.read_text())
            if r['finalized_blocks_per_layer']!=[12]*32 or r['consumed_finalized_blocks_per_layer']!=[11]*32:
                raise ValueError('Missing BitDecoding recurrent consumption')
            u=r['distribution_quality']['cluster_bootstrap_units']
            if len(u)!=1:raise ValueError('Expected B1 unit')
            cells['bitdecode_int4'].append(u[0])
            memory['bitdecode_int4'].append(r['cache_accounting']['final']['unique_storage_bytes'])
            staging.append(r['cache_accounting']['final']['persistent_staging_bytes'])
        ids={cells[name][-1]['cluster_unit_id'] for name in names}
        if len(ids)!=1:raise ValueError('Unpaired windows')
    rows={name:summarize(cells[name],memory[name]) for name in names}
    expected_fp16=32*22016*8*128*4
    if rows['flashinfer_fp16']['cache_bytes']!=expected_fp16:raise ValueError('Unexpected FP16 cache capacity')
    for name,row in rows.items():
        row['kv_reduction_fraction']=1-row['cache_bytes']/expected_fp16
        row['native_stack']='transformers 4.36.2 / HF eager' if name.startswith(('kivi','bitdecode')) else 'transformers 4.57.6 / HF SDPA'
    if bitdecode_dirs:
        if len(set(staging))!=1:raise ValueError('Staging capacity drift')
        rows['bitdecode_int4']['persistent_staging_bytes']=staging[0]
        rows['bitdecode_int4']['implementation']='SM120-port kernel integration into shared Mistral harness, not upstream model engine'
    out=ROOT/'results/mlsys2027_baselines_v1'/('quality_frontier_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    result={'rows':rows,'matched_actual_token_ids':True,'input_sha256':evidence,
        'reducer_sha256':base.sha256_file(Path(__file__)),
        'scope':'Same eight exposed TRAIN windows, B1/C20480/D1536, not independent books or final TEST. No performance comparison.',
        'cache_scope':'Served-cache tensors/storage only; excludes weights, reference fixtures, activations and allocator reserve.',
        'uncertainty':'Paired-to-own-HF window bootstrap, descriptive only; no prospective non-inferiority margin tested.',
        'cross_stack_warning':'Same checkpoint/token IDs, different HF numerical references; keep native HF PPL alongside each row.'}
    base.atomic_json(out/'analysis.json',result)
    print('Quality frontier: '+str(out),flush=True)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
