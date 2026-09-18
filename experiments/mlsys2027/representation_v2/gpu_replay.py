"""Untimed GPU prototype: scale the historical output before exact-state merge.

Same INT8 kernels; same FP16 exact cache. Extra output scaling is NOT free.
Only validates the value-conditioned development candidate, not a production selection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid
from prototype import ROOT, digest, write_json, encode, output_metrics, attention


def replay(path, plan):
    import numpy as np
    import torch
    import flashinfer
    sys.path.insert(0, str(ROOT/'scripts'))
    import benchmark_page_gauge_transformer as pg
    with np.load(path, allow_pickle=False) as f:
        q_cpu = f['query']
        key = encode(f['key'],f['key_center'],plan['initial_context_tokens'],plan['policy'],False)
        value = encode(f['value'],f['value_center'],plan['initial_context_tokens'],plan['policy'],True)
        exact_key = (f['key'].astype(np.float32)-f['key_center'].astype(np.float32)).astype(np.float16)
        exact_value = (f['value'].astype(np.float32)-f['value_center'].astype(np.float32)).astype(np.float16)
        original_key, original_value = f['key'],f['value']
    q = torch.from_numpy(q_cpu.copy()).cuda()[None]
    policy=plan['policy']
    n,page,heads,dim=len(exact_key),16,8,128
    pages=(n+page-1)//page
    last=(n-1)%page+1
    initial=plan['initial_context_tokens']//page
    tail=policy['exact_tail_tokens']//page
    prefix,suffix=policy['exact_prefix_pages'],policy['exact_static_suffix_pages']
    storage=prefix+suffix+tail
    part=pg.page_gauge_logical_partition(pages,tail,prefix,last,suffix,initial)
    ring=pg.request_major_exact_page_table(1,pages,tail,prefix,suffix,initial)
    logical=torch.tensor(part['exact_logical_pages'],dtype=torch.long,device='cuda')
    physical=ring.index_select(1,logical).flatten().long()
    codes,scales,exact,reconstructed=[],[],[],[]
    for rep,exact_cpu in ((key,exact_key),(value,exact_value)):
        code=torch.zeros((pages*page,heads,dim),dtype=torch.int8,device='cuda')
        code[:n]=torch.from_numpy(rep['codes'].copy()).cuda()
        codes.append(code.reshape(pages,page,heads,dim))
        scale=torch.ones((pages,heads),dtype=torch.float16,device='cuda')
        old_logical=torch.tensor(part['old_logical_pages'],dtype=torch.long,device='cuda')
        scale.index_copy_(0,old_logical,torch.from_numpy(rep['scales'].copy()).cuda())
        scales.append(scale)
        padded=torch.zeros((pages*page,heads,dim),dtype=torch.float16,device='cuda')
        padded[:n]=torch.from_numpy(exact_cpu.copy()).cuda()
        exact_buffer=torch.zeros((storage,page,heads,dim),dtype=torch.float16,device='cuda')
        exact_buffer.index_copy_(0,physical,padded.reshape(pages,page,heads,dim).index_select(0,logical))
        exact.append(exact_buffer)
        recon=torch.zeros_like(padded)
        recon[:n]=torch.from_numpy((rep['real']-rep['center'].astype(np.float64)).astype(np.float16)).cuda()
        reconstructed.append(recon.reshape(pages,page,heads,dim))
    old_wrapper=pg.GraphDecodeWrapper(flashinfer,pages,torch.int8,True)
    exact_wrapper=pg.GraphDecodeWrapper(flashinfer,storage,torch.float16,False)
    reference=pg.GraphDecodeWrapper(flashinfer,pages,torch.float16,False)
    table=pg.request_major_page_table(1,pages)
    old_table=pg.write_page_gauge_old_page_table(torch.empty_like(table),table,part['old_logical_pages'])
    exact_table=pg.write_exact_prefix_tail_page_table(torch.empty((1,storage),dtype=torch.int32,device='cuda'),ring,
                                                   part['tail_logical_begin'],pages,prefix,suffix,initial)
    old_wrapper.plan(old_table,part['old_token_count'],page,128)
    exact_wrapper.plan(exact_table,part['exact_token_count'],last,128)
    reference.plan(table,n,last,256)
    old_out=torch.empty_like(q)
    exact_out=torch.empty_like(q)
    old_lse=torch.empty((1,32),dtype=torch.float32,device='cuda')
    exact_lse=torch.empty_like(old_lse)
    old_wrapper.wrapper.run(q,tuple(codes),*scales,1.0/np.sqrt(dim),out=old_out,lse=old_lse,return_lse=True)
    # Restore V channel units on OLD output only. Exact FP16 values remain unchanged.
    gain=torch.from_numpy(value['gain'].copy()).cuda().repeat_interleave(4,dim=0)[None]
    old_out.mul_(gain)
    exact_wrapper.wrapper.run(q,tuple(exact),out=exact_out,lse=exact_lse,return_lse=True)
    flashinfer.merge_state_in_place(old_out,old_lse,exact_out,exact_lse)
    center=torch.from_numpy(value['center'].copy()).cuda().repeat_interleave(4,dim=0)[None]
    old_out.add_(center)
    explicit=reference.wrapper.run(q,tuple(reconstructed))
    explicit.add_(center)
    observed=old_out[0].cpu().numpy()
    explicit_cpu=explicit[0].cpu().numpy()
    rows=[]
    for h in range(heads):
        qs=q_cpu[4*h:4*h+4]
        truth=attention(qs,original_key[:,h],original_value[:,h])
        mathematical=attention(qs,key['real'][:,h],value['real'][:,h])
        row={'head':h,'gpu_vs_original':output_metrics(observed[4*h:4*h+4],truth),
             'gpu_vs_reconstructed_fp64':output_metrics(observed[4*h:4*h+4],mathematical),
             'gpu_vs_explicit_fp16':output_metrics(observed[4*h:4*h+4],explicit_cpu[4*h:4*h+4])}
        rows.append(row)
    return observed,explicit_cpu,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-run',required=True,type=Path)
    parser.add_argument('--gpu-index',type=int,default=0)
    args=parser.parse_args()
    source=args.capture_run.resolve()
    sys.path.insert(0,str(ROOT/'diagnostics'))
    import mlsys_rtx5090_entry as base
    manifest=json.loads((source/'manifest.json').read_text())
    captures=json.loads((source/'capture_summary.json').read_text())
    if captures['manifest_sha256']!=manifest['manifest_sha256'] or manifest['plan']['wikitext_member']!='wikitext-2-raw/wiki.train.raw':
        raise ValueError('Requires original TRAIN capture identity')
    for relative,sha in manifest['source_sha256'].items():
        if digest(ROOT/relative)!=sha:
            raise ValueError('Original source drift: '+relative)
    for row in captures['records']:
        if Path(row['path']).name!=row['path'] or digest(source/row['path'])!=row['sha256']:
            raise ValueError('Capture identity mismatch')
    snapshots=base.idle_preflight(args.gpu_index)
    gpu_uuid=snapshots[-1]['uuid']
    import fcntl
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu_uuid}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES']=gpu_uuid
        import torch
        import numpy as np
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        out=ROOT/'results/mlsys2027_representation_v2'/('gpu_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        write_json(out/'manifest.json',{'capture_manifest':manifest['manifest_sha256'],'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),Path(__file__).with_name('prototype.py'))},
                   'original_source_sha256':manifest['source_sha256'],'idle_check':snapshots,
                   'candidate':'value_only_power_two_channel_conditioning','scope':'development candidate validation, no speed or heldout claim'})
        print(str(out),flush=True)
        rows=[]
        for record in captures['records']:
            observed,explicit,metrics=replay(source/record['path'],manifest['plan'])
            target=out/record['path']
            np.savez(target,observed=observed,explicit=explicit)
            rows.append({'capture':record['path'],'sha256':digest(target),'heads':metrics})
            print('Replayed '+record['path'],flush=True)
        summary={}
        for name in ('gpu_vs_original','gpu_vs_reconstructed_fp64','gpu_vs_explicit_fp16'):
            metrics=[h[name] for r in rows for h in r['heads']]
            summary[name]={'mean_l2':float(np.mean([m['l2_per_query'] for m in metrics])),
                           'min_cosine':min(c for m in metrics for c in m['cosine_per_query'] if c is not None),
                           'max_abs':max(m['max_abs'] for m in metrics)}
        write_json(out/'analysis.json',{'rows':rows,'summary':summary,'speed_claim':False,'heldout_quality_claim':False})
        print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
