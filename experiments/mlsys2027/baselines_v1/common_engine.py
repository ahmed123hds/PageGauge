"""Shared eager Mistral body, native low-bit or FI/PG attention adapters.

Decode-only development comparison: no graph capture, generated-task score,
native-prefill performance, or optimized-production speedup claim.
"""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from kivi_quality import verify,PYTHON
DEFAULT=ROOT/'results/mlsys2027_baselines_v1/bitdecode_quality_20260908T105048Z_28ae38c3'
BACKENDS=('flashinfer_fp16','page_gauge','kivi_int2','kivi_int4','bitdecode_int4')


def initialize(model,initial,backend,maximum,exact_split_pages=None):
    if backend in ('flashinfer_fp16','page_gauge'):
        from paged_mistral_adapter import configure_and_pack
        return configure_and_pack(model,initial,backend,maximum,exact_split_pages)
    if exact_split_pages is not None:raise ValueError('Exact split applies only to PageGauge')
    if backend=='bitdecode_int4':
        from bitdecode_adapter import configure_decode,pack_initial_cache
    else:
        from kivi_adapter import configure_decode,pack_initial_cache
    bits=2 if backend=='kivi_int2' else 4
    configure_decode(model,bits)
    return pack_initial_cache(initial,bits)


def memory(past,backend):
    if backend in ('flashinfer_fp16','page_gauge'):
        from paged_mistral_adapter import cache_accounting
    else:
        from kivi_quality import cache_accounting
    return cache_accounting(past)


def check_cache(past,backend,context,steps):
    if len(past)!=32 or any(layer[-1]!=context+steps for layer in past):
        raise RuntimeError('Incomplete layer cache recurrence')
    if backend=='bitdecode_int4':
        if any(layer[0].finalized_blocks!=steps//128 or layer[0].consumed_finalized_blocks!=(steps-1)//128 for layer in past):
            raise RuntimeError('Missing BitDecoding closure/consumption')
    if backend in ('flashinfer_fp16','page_gauge'):
        driver=past[0][0]
        if driver.logical_lengths!=[context+steps]*32 or driver.decoder_plan_calls!=steps:
            raise RuntimeError('Paged driver recurrence failed')


def worker(out):
    import torch
    import transformers
    from transformers import MistralForCausalLM
    import benchmark_pg19_external_quality as quality
    m=json.loads((out/'manifest.json').read_text());verify(m)
    if base.sha256_file(out/'tokens.json')!=m['tokens_sha256']:raise RuntimeError('Token drift')
    torch.set_grad_enabled(False);torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING']='none'
    tokens=torch.tensor(json.loads((out/'tokens.json').read_text())['ids'],device='cuda',dtype=torch.long)
    context,steps=m['context'],m['decode_steps'];backend=m['backend']
    profiling=m.get('profile',False)
    if tokens.shape!=(1,context+steps+1):raise ValueError('Unpadded B1 fixture required')
    print('Loading shared ungraphed Mistral body: '+backend,flush=True)
    model=MistralForCausalLM.from_pretrained(m['model'],torch_dtype=torch.float16,
        local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda:0'},attn_implementation='eager').eval()
    if model.config.sliding_window is not None:raise ValueError('Full-context fixture required')
    with torch.inference_mode():
        past=None
        torch.cuda.synchronize();begin=time.perf_counter()
        for start in range(0,context,256):
            outputs=model(tokens[:,start:min(start+256,context)],past_key_values=past,use_cache=True)
            past=outputs.past_key_values
        torch.cuda.synchronize();prefill=time.perf_counter()-begin
        legacy=past.to_legacy_cache() if hasattr(past,'to_legacy_cache') else past
        initial=tuple((k.clone(),v.clone()) for k,v in legacy)
        reference=[]
        if m['validate']:
            print('Untimed native HF reference...',flush=True)
            for p in range(context,context+steps):
                outputs=model(tokens[:,p:p+1],past_key_values=past,use_cache=True)
                past=outputs.past_key_values
                reference.append(outputs.logits[:,-1].float().cpu())
        del outputs,past,legacy
        gc.collect();torch.cuda.empty_cache()
        rows=[];warmup=None
        rounds=1 if m['validate'] or profiling else m['repeats']+1
        for repeat in range(rounds):
            torch.cuda.synchronize();begin=time.perf_counter()
            past=initialize(model,initial,backend,context+steps,m.get('exact_split_pages'))
            torch.cuda.synchronize();setup=time.perf_counter()-begin
            observed=[];n=steps
            print(('Validation' if m['validate'] else 'Warmup' if repeat==0 else 'Timed repeat '+str(repeat))+': '+backend,flush=True)
            start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) if profiling else None
            torch.cuda.synchronize();begin=time.perf_counter();start_event.record()
            for p in range(context,context+n):
                if profiler is not None and p==context+n-129:
                    torch.cuda.synchronize();profiler.start()
                outputs=model(tokens[:,p:p+1],past_key_values=past,use_cache=True)
                past=outputs.past_key_values
                if m['validate']:observed.append(outputs.logits[:,-1].float().cpu())
            end_event.record();torch.cuda.synchronize();wall=(time.perf_counter()-begin)*1000
            cuda=start_event.elapsed_time(end_event)
            if profiler is not None:
                profiler.stop()
                events=[{'name':e.key,'device_type':str(e.device_type),'count':e.count,
                    'self_cpu_us':e.self_cpu_time_total,'total_cpu_us':e.cpu_time_total,
                    'self_device_us':e.self_device_time_total,'total_device_us':e.device_time_total}
                    for e in profiler.key_averages()]
                base.atomic_json(out/'profile.json',{'backend':backend,
                    'model_position_start':context+n-129,'profiled_steps':129,'events':events,
                    'scope':'Instrumented last129 steps after earlier recurrent warmup. Diagnostic attribution only, not headline latency. Nested CPU/device rows can overlap; do not sum them blindly.'})
            check_cache(past,backend,context,n)
            if not torch.isfinite(outputs.logits).all():raise RuntimeError('Nonfinite final logits')
            record={'repeat':repeat,'steps':n,'cache_setup_seconds':setup,'cache':memory(past,backend)}
            if not m['validate'] and repeat==0:
                warmup={**record,'wall_seconds':wall/1000,'cuda_seconds':cuda/1000}
            if not m['validate'] and repeat:
                record.update(wall_ms_per_step=wall/n,cuda_ms_per_step=cuda/n)
                rows.append(record)
            if m['validate']:
                labels=tokens[:,context+1:context+steps+1].T.cpu()
                windows=quality.request_window_metadata(m['token_provenance'],context,steps,1)
                comparison=quality.compare_logits(reference,observed,labels,
                    reference_name='HF eager FP16 transformers 4.36.2',candidate_name='Common eager body '+backend,
                    predicted_position_start=context+1,request_windows=windows)
                comparison['execution']=record
                base.atomic_json(out/'quality.json',comparison)
                rows.append(record)
            del past,outputs,observed
            gc.collect();torch.cuda.empty_cache()
    verify(m)
    base.atomic_json(out/'analysis.json',{'backend':backend,'validation_only':m['validate'],'profile_only':profiling,'rows':rows,'warmup':warmup,
        'profile_sha256':base.sha256_file(out/'profile.json') if profiling else None,
        'shared_prefill_seconds':prefill,'transformers':transformers.__version__,'torch':torch.__version__,
        'quality_sha256':base.sha256_file(out/'quality.json') if m['validate'] else None,
        'execution_contract':m['execution_contract'],'scope':m['scope']})
    print('Common-engine complete: '+str(out),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--fixture',type=Path,default=DEFAULT)
    p.add_argument('--backend',choices=BACKENDS,default='flashinfer_fp16')
    p.add_argument('--validate',action='store_true');p.add_argument('--steps',type=int,default=1536)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--exact-split-pages',type=int,choices=(32,64,128))
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.worker:
        try:worker(args.worker)
        except BaseException as e:
            base.atomic_json(args.worker/'failure.json',{'error':str(e),'traceback':traceback.format_exc()})
            raise
        return
    if not 129<=args.steps<=1536 or args.repeats<1:raise ValueError('Invalid recurrent run length/repeats')
    if args.profile and args.validate:raise ValueError('Separate quality validation from profiling')
    if args.exact_split_pages is not None and args.backend!='page_gauge':raise ValueError('Exact split override requires PageGauge')
    import fcntl
    idle=base.idle_preflight(0);gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=json.loads((args.fixture/'manifest.json').read_text())
        if source['context']!=20480 or source['token_provenance']['split']!='train':
            raise ValueError('Only exposed 20K TRAIN fixtures allowed')
        if base.sha256_file(args.fixture/'tokens.json')!=source['tokens_sha256']:raise ValueError('Changed fixture')
        for name,e in source['input_file_evidence'].items():
            if base.sha256_file(Path(name))!=e['sha256']:raise RuntimeError('Changed input '+name)
        paths={Path(x) for x in source['source_sha256']}
        paths.update([Path(__file__),Path(__file__).with_name('paged_mistral_adapter.py')])
        selection=ROOT/'results/mlsys2027_baselines_v1/rtx_exact_split_20260908T115408Z_e0315ebe/analysis.json'
        if args.exact_split_pages is not None:paths.add(selection)
        out=ROOT/'results/mlsys2027_baselines_v1'/('common_engine_'+args.backend+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        ids=json.loads((args.fixture/'tokens.json').read_text())['ids']
        ids=[row[:source['context']+args.steps+1] for row in ids]
        base.atomic_json(out/'tokens.json',{'ids':ids})
        m={k:source[k] for k in ('context','seed','model','input_file_evidence','token_provenance')}
        import benchmark_pg19_external_quality as quality
        provenance=dict(source['token_provenance'])
        provenance.update(shape=[1,source['context']+args.steps+1],
            corpus_tokens_per_request=source['context']+args.steps,
            corpus_window_end_offsets_exclusive=[x+source['context']+args.steps for x in provenance['corpus_window_start_offsets']],
            token_ids_sha256=quality.canonical_json_sha256(ids),
            teacher_forced_decode_ids_step_major=[[ids[0][x]] for x in range(source['context'],source['context']+args.steps)],
            decode_label_ids_step_major=[[ids[0][x]] for x in range(source['context']+1,source['context']+args.steps+1)])
        m['token_provenance']=provenance
        m.update(backend=args.backend,decode_steps=args.steps,repeats=args.repeats,validate=args.validate,profile=args.profile,
            exact_split_pages=args.exact_split_pages,launch_probe=str(selection) if args.exact_split_pages is not None else None,
            source_sha256={str(x):base.sha256_file(x) for x in sorted(paths)},
            tokens_sha256=base.sha256_file(out/'tokens.json'),source_fixture=str(args.fixture),
            idle=idle,orchestrator_pid=os.getpid(),
            execution_contract='Same native Mistral body, FP16 weights, separate eager projections, no CUDA graphs/torch.compile, shared HF prefill. Native cache policies. Decode includes attention, all layer/LM-head work, append/finalization and planning. No explicit cache eviction. Host/CUDA full-loop timing; no per-step output transfer when timed.',
            scope='Exposed TRAIN development, B1. Decode-only common-engine contrast, NOT optimized-system or prefill-inclusive serving speed. Warmup and cache setup excluded from decode timing and reported separately. Retained initial FP16 fixture prevents treating process peak allocation as deployed memory.')
        if args.profile:m['scope']='Instrumented final129 recurrent steps of a full1536-step warmup, TRAIN B1. Diagnostic attribution only; no performance-acceptance result.'
        base.atomic_json(out/'manifest.json',m)
        print('Common-engine run: '+str(out),flush=True)
        command=[PYTHON,'-u',str(Path(__file__)),'--worker',str(out)]
        base.atomic_json(out/'invocation.json',{'command':command})
        result=previous.run_process(command,out,{'index':0},gpu)
        base.atomic_json(out/'completion.json',result);verify(m)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise RuntimeError('Common-engine execution failed; evidence retained')


if __name__=='__main__':main()
