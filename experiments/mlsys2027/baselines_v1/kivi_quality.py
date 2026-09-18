"""Pretrained native KIVI corpus-label quality, shared FP16 prefill, TRAIN only."""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
NATIVE=Path('/home/anonymous/pagegauge_baselines/KIVI')
PYTHON='/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/bin/python'


def verify(manifest):
    for name,sha in manifest['source_sha256'].items():
        if base.sha256_file(Path(name))!=sha:raise RuntimeError('Source drift '+name)
    for name,evidence in manifest['input_file_evidence'].items():
        stat=Path(name).stat()
        if stat.st_size!=evidence['size'] or stat.st_mtime_ns!=evidence['mtime_ns']:
            raise RuntimeError('Input metadata changed '+name)


def cache_accounting(cache):
    if cache and hasattr(cache[0][0],'memory'):
        records=[layer[0].memory() for layer in cache]
        serving=sum(r['packed_bytes']+r['fp16_append_capacity_bytes'] for r in records)
        return {'logical_tensor_bytes':serving,'unique_storage_bytes':serving,
            'persistent_staging_bytes':sum(r['persistent_staging_bytes'] for r in records),
            'scope':'BitDecoding packed cache plus allocated FP16 append capacity; staging separate; not process peak memory'}
    tensors=[t for layer in cache for t in layer if hasattr(t,'untyped_storage')]
    stores={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
    return {'logical_tensor_bytes':sum(t.numel()*t.element_size() for t in tensors),
        'unique_storage_bytes':sum(stores.values()),'tensor_count':len(tensors),
        'scope':'Served cache only; excludes retained reference fixture, weights, activations and allocator reserve'}


def worker(out):
    import torch
    import transformers
    from transformers import MistralForCausalLM
    import benchmark_pg19_external_quality as quality
    m=json.loads((out/'manifest.json').read_text())
    backend=m.get('backend','kivi')
    if backend=='bitdecode':
        from bitdecode_adapter import configure_decode,pack_initial_cache
        label='BitDecoding INT4 SM120 kernel integration'
    else:
        from kivi_adapter import configure_decode,pack_initial_cache
        label='Native KIVI'
    verify(m)
    if base.sha256_file(out/'tokens.json')!=m['tokens_sha256']:raise RuntimeError('Token artifact changed')
    tokens=torch.tensor(json.loads((out/'tokens.json').read_text())['ids'],dtype=torch.long,device='cuda')
    context,steps=m['context'],m['decode_steps']
    if tokens.shape!=(1,context+steps+1):raise ValueError('Missing input or final label')
    torch.set_grad_enabled(False)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    print('Loading pinned Mistral with native-stack HF eager attention...',flush=True)
    model=MistralForCausalLM.from_pretrained(m['model'],torch_dtype=torch.float16,
        local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda:0'},attn_implementation='eager').eval()
    if model.config.sliding_window is not None:raise ValueError('This fixture requires full context')
    if len(model.model.layers)!=32 or model.config.num_key_value_heads!=8:raise ValueError('Wrong model')
    outputs=None;past=None
    with torch.inference_mode():
        # This released Mistral stack predates HF SDPA support. Chunk the eager
        # prefill to bound attention-matrix memory; this runner makes no timing claim.
        for start in range(0,context,256):
            outputs=model(tokens[:,start:min(start+256,context)],past_key_values=past,use_cache=True)
            past=outputs.past_key_values
        legacy=past.to_legacy_cache() if hasattr(past,'to_legacy_cache') else past
        initial=tuple((k.clone(),v.clone()) for k,v in legacy)
        if any(k.shape[-2]!=context for k,v in initial):raise RuntimeError('Incomplete prefill history')
        reference=[]
        print('HF corpus-label reference decode...',flush=True)
        for position in range(context,context+steps):
            outputs=model(tokens[:,position:position+1],past_key_values=past,use_cache=True)
            past=outputs.past_key_values
            reference.append(outputs.logits[:,-1].float().cpu())
        del outputs,past,legacy
        gc.collect();torch.cuda.empty_cache()
        labels=tokens[:,context+1:context+steps+1].T.cpu()
        windows=quality.request_window_metadata(m['token_provenance'],context,steps,1)
        results={}
        for bits in m['bits']:
            configure_decode(model,bits)
            past=pack_initial_cache(initial,bits)
            before=cache_accounting(past)
            observed=[]
            print(label+' INT'+str(bits)+' corpus-label decode...',flush=True)
            for position in range(context,context+steps):
                outputs=model(tokens[:,position:position+1],past_key_values=past,use_cache=True)
                past=outputs.past_key_values
                observed.append(outputs.logits[:,-1].float().cpu())
            for layer in past:
                if backend=='bitdecode':
                    if layer[-1]!=context+steps or layer[0].finalized_blocks!=steps//128 or layer[0].consumed_finalized_blocks!=(steps-1)//128:
                        raise RuntimeError('BitDecoding cache close/consumption failed')
                elif layer[-1]!=context+steps or layer[0].dtype!=torch.int32 or layer[4].dtype!=torch.int32:
                    raise RuntimeError('KIVI recurrence failed')
            comparison=quality.compare_logits(reference,observed,labels,
                reference_name='HF eager FP16 transformers 4.36.2',candidate_name=label+' INT'+str(bits),
                predicted_position_start=context+1,request_windows=windows)
            comparison['cache_accounting']={'initial':before,'final':cache_accounting(past)}
            comparison['native_recurrence_checked_steps']=steps
            if backend=='bitdecode':
                comparison['finalized_blocks_per_layer']=[layer[0].finalized_blocks for layer in past]
                comparison['consumed_finalized_blocks_per_layer']=[layer[0].consumed_finalized_blocks for layer in past]
            path=out/f'int{bits}.json'
            base.atomic_json(path,comparison)
            results['int'+str(bits)]={'sha256':base.sha256_file(path),
                'top1':comparison['top1_agreement_fraction'],
                'distribution_quality':{k:v for k,v in comparison['distribution_quality'].items()
                    if k!='per_token_metrics_step_major'},'cache_accounting':comparison['cache_accounting']}
            del past,outputs,observed,comparison
            gc.collect();torch.cuda.empty_cache()
    verify(m)
    base.atomic_json(out/'analysis.json',{'stage':'E2_'+backend+'_development_quality','results':results,'backend':backend,
        'transformers':transformers.__version__,'torch':torch.__version__,
        'scope':'One TRAIN window; shared FP16 prefill and '+label+' decode. Native-stack HF reference; no timing or final TEST claim.',
        'cross_stack_warning':'Do not silently pool with PageGauge transformers 4.57.6 results.'})
    print('Complete '+str(out),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--offset',type=int,default=472000)
    parser.add_argument('--backend',choices=('kivi','bitdecode'),default='kivi')
    args=parser.parse_args()
    if args.worker:
        try:worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json',{'error':str(error),'error_type':type(error).__name__,
                'traceback':traceback.format_exc()})
            raise
        return
    import fcntl
    import subprocess
    import benchmark_pg19_external_quality as quality
    old=ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c/manifest.json'
    original=json.loads(old.read_text())
    idle=base.idle_preflight(0);gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['git','-C',str(NATIVE),'rev-parse','HEAD'],text=True).strip()!='876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6':
            raise RuntimeError('Native revision changed')
        if subprocess.check_output(['git','-C',str(NATIVE),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise RuntimeError('Native source modified')
        context,steps=(20480,1536) if args.full else (1024,64)
        if args.backend=='bitdecode' and not args.full:steps=129
        token_args=SimpleNamespace(context=context,decode_steps=steps,batch_size=1,token_source='wikitext2',
            wikitext_member='wikitext-2-raw/wiki.train.raw',wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip',
            seed=2026090815,token_offset=args.offset,token_stride=23600,model=original['inputs']['model_path'])
        tokens,provenance=quality.build_token_matrix(token_args,32768)
        e0=json.loads((ROOT/'results/mlsys2027_representation_v2/ground_truth_20260908T074422Z_8ab9bf75/manifest.json').read_text())
        paths={ROOT/name for name in e0['source_sha256']}
        paths.update([Path(__file__),Path(__file__).with_name('kivi_adapter.py'),NATIVE/'models/mistral_kivi.py',
            NATIVE/'quant/new_pack.py',NATIVE/'quant/matmul.py',NATIVE/'quant/csrc/gemv_cuda.cu',
            Path('/home/anonymous/pagegauge_baselines/kivi_sm120_env/lib/python3.12/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so'),
            Path('/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py')])
        if args.backend=='bitdecode':
            paths.update([Path(__file__).with_name('bitdecode_adapter.py'),Path(__file__).with_name('bitdecode_cache.py'),
                Path('/home/anonymous/pagegauge_baselines/bitdecode_sm120_env/lib/python3.12/site-packages/bit_decode_cuda.cpython-312-x86_64-linux-gnu.so'),
                Path('/home/anonymous/pagegauge_baselines/bitdecode_sm120_env/lib/python3.12/site-packages/bit_decode/bit_decode_interface.py')])
        for name,evidence in original['input_file_evidence'].items():
            if base.sha256_file(Path(name))!=evidence['sha256']:raise RuntimeError('Input changed '+name)
        out=ROOT/'results/mlsys2027_baselines_v1'/(args.backend+'_quality_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json',{'ids':tokens.tolist()})
        m={'context':context,'decode_steps':steps,'seed':token_args.seed,'model':token_args.model,
            'input_file_evidence':original['input_file_evidence'],
            'source_sha256':{str(p):base.sha256_file(p) for p in sorted(paths)},
            'tokens_sha256':base.sha256_file(out/'tokens.json'),'token_provenance':provenance,
            'backend':args.backend,'bits':[4] if args.backend=='bitdecode' else [2,4],'group':32,
            'residual':128 if args.backend=='bitdecode' else 32,'prefill_attention':'HF eager FP16','prefill_chunk':256,
            'idle':idle,'orchestrator_pid':os.getpid(),
            'scope':'TRAIN development; pilot first, then recurrent full configuration; no acceptance cutoff on quality'}
        base.atomic_json(out/'manifest.json',m)
        print(('KIVI' if args.backend=='kivi' else 'BitDecoding')+' pretrained quality: '+str(out),flush=True)
        command=[PYTHON,'-u',str(Path(__file__)),'--worker',str(out)]
        base.atomic_json(out/'invocation.json',{'command':command})
        completion=previous.run_process(command,out,{'index':0},gpu)
        base.atomic_json(out/'completion.json',completion)
        verify(m)
        if completion['return_code']!=0 or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('Quality worker/exclusivity failed; inspect preserved evidence')


if __name__=='__main__':main()
