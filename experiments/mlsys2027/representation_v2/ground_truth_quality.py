"""E0 TRAIN-only recurrent quality: HF, original FI/PG, folded PG/FI.

Separate from historical workshop workers. No timing or held-out quality claim.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
from prototype import ROOT, digest, write_json

CALIBRATION = ROOT/'results/mlsys2027_representation_v2/fixed_model_20260908T071637Z_0ad135f3/calibration_gain.pt'
CALIBRATION_SHA = '6a6d13ff9ff0a52d9db3a16f038263331500debb0b341a0b30373f39964642f9'
OLD = ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'


def verify(manifest):
    for name, sha in manifest['source_sha256'].items():
        if digest(ROOT/name) != sha:raise RuntimeError('Source drift: '+name)
    if digest(CALIBRATION) != CALIBRATION_SHA:raise RuntimeError('Calibration drift')


def worker(directory):
    import torch
    import flashinfer
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(ROOT/'diagnostics'))
    import benchmark_pg19_external_quality as quality
    pg = quality.PG
    manifest = json.loads((directory/'manifest.json').read_text())
    verify(manifest)
    torch.set_grad_enabled(False)
    torch.manual_seed(manifest['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
    os.environ.pop('PAGEGAUGE_EXPORT_GAIN_PATH', None)
    args = SimpleNamespace(context=20480, decode_steps=1536, batch_size=1,
        token_source='wikitext2', wikitext_member='wikitext-2-raw/wiki.train.raw',
        wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip', seed=manifest['seed'],
        token_offset=manifest['offset'], token_stride=23600, model=manifest['model'])
    print('Loading HF ground-truth fixture...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').eval()
    layers,hq,hkv,hidden = pg.BASE_E2E.check_model(model)
    if (layers,hq,hkv) != (32,32,8):raise ValueError('Mistral-only pilot')
    tokens, provenance = quality.build_token_matrix(args, int(model.config.vocab_size))
    write_json(directory/'token_provenance.json', provenance)
    model.cuda()
    maximum = args.context+args.decode_steps
    pages, initial, tail = math.ceil(maximum/16), args.context//16, 48
    baseline_cache = quality.allocate_baseline_cache(layers,pages,1,hkv)
    hf,_,prefill_records,kv_hash = quality.prefill_model_cache(model,tokens,baseline_cache,
        pages,args.context,args.decode_steps,1024)
    if not all(r['sampled_boundary_copy_bitwise_identical'] for r in prefill_records):
        raise RuntimeError('HF cache copy mismatch')
    pg.BASE_E2E.pack_model_projections(model)
    extension = pg.RUNTIME.load_append_extension()
    with torch.inference_mode():
        positions = torch.arange(maximum,device='cuda',dtype=torch.long)[None]
        cos,sin = model.model.rotary_emb(torch.empty(1,1,hidden,device='cuda',dtype=torch.float16),positions)
        if cos.dim() == 3:cos,sin=cos[0],sin[0]
        cos,sin=cos.half().contiguous(),sin.half().contiguous()
    decode = tokens[:,args.context:maximum].T.tolist()
    labels = tokens[:,args.context+1:maximum+1].T.contiguous()
    windows = quality.request_window_metadata(provenance,args.context,args.decode_steps,1)
    results = {}
    def run(name, backend, cache):
        print('Ground-truth recurrent decoding: '+name,flush=True)
        decoder = pg.TransformerDecoder(model,flashinfer,extension,backend,cache,maximum,
            768,256,128,cos,sin,'attention_add',tail_attention='flashinfer_merge',batch_size=1,
            old_value_scale_placement='probability',exact_sink_pages=4 if backend=='page_gauge' else 0,
            exact_static_suffix_pages=128 if backend=='page_gauge' else 0,
            initial_context_pages=initial if backend=='page_gauge' else None)
        logits = quality.collect_decoder_logits_on_cpu(decoder,decode,args.context)
        comparison = quality.compare_logits(hf,logits,labels,reference_name='Original HF SDPA FP16',
            candidate_name=name,predicted_position_start=args.context+1,request_windows=windows)
        comparison['conditioning'] = decoder.value_conditioning
        comparison['cache_storage_bytes'] = sum(t.numel()*t.element_size() for t in vars(cache).values() if isinstance(t,torch.Tensor))
        write_json(directory/(name+'.json'), comparison)
        results[name] = {'path':name+'.json','sha256':digest(directory/(name+'.json')),
            'top1':comparison['top1_agreement_fraction'],
            'distribution_quality':{k:v for k,v in comparison['distribution_quality'].items()
                                    if k != 'per_token_metrics_step_major'}}
        del decoder,logits,comparison
        gc.collect();torch.cuda.empty_cache()

    run('original_fi','flashinfer_fp16',baseline_cache)
    gauge = quality.build_gauge_cache_from_baseline(baseline_cache,layers,pages,initial,tail,1,hkv,4,128)
    run('original_pg','page_gauge',gauge)
    del gauge
    gc.collect();torch.cuda.empty_cache()
    # All original-coordinate/HF decoding is complete before mutating weights or
    # initial V storage. The same immutable calibration serves every request.
    os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'folded_fixed_rms'
    os.environ['PAGEGAUGE_FIXED_GAIN_PATH'] = str(CALIBRATION)
    os.environ['PAGEGAUGE_FIXED_GAIN_SHA256'] = CALIBRATION_SHA
    holder = SimpleNamespace(value_center=torch.empty(layers,1,hkv,128,device='cuda',dtype=torch.float16))
    for layer in range(layers):
        view = baseline_cache.value[layer,:initial].view(args.context,hkv,128)
        view.copy_(pg.VALUE_CONDITIONING.condition_prefill(holder,layer,0,view))
    gauge = quality.build_gauge_cache_from_baseline(baseline_cache,layers,pages,initial,tail,1,hkv,4,128)
    for name in ('value_channel_gain','value_conditioning_fitted','value_conditioning_setup_seconds','value_conditioning_artifact_sha256'):
        setattr(gauge,name,getattr(holder,name))
    run('fixed_pg','page_gauge',gauge)  # Constructor folds V/O exactly once.
    del gauge
    gc.collect();torch.cuda.empty_cache()
    # FI's constructor does not fold again. Its initial V cache is already scaled;
    # sequential appends replace each future slot before that position is read.
    run('fixed_fi','flashinfer_fp16',baseline_cache)
    verify(manifest)
    write_json(directory/'analysis.json', {'stage':'E0_ground_truth_development',
        'results':results,'prefill_records':prefill_records,'initial_kv_sample_sha256':kv_hash,
        'calibration_sha256':CALIBRATION_SHA,'labels':args.decode_steps,'batch':1,
        'scope':'One TRAIN window; recurrent corpus-label PPL, not final quality or timing confirmation',
        'production_default_changed':False})
    print('Completed '+str(directory),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset',type=int,default=472000)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker);return
    if args.offset < 472000:raise ValueError('Pilot restricted to new TRAIN offset >=472000')
    sys.path.insert(0,str(ROOT/'diagnostics'))
    import mlsys_rtx5090_entry as base
    import mlsys_rtx5090_step02 as previous
    import fcntl
    idle=base.idle_preflight(0)
    gpu=idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=json.loads((OLD/'manifest.json').read_text())
        for name,evidence in original['input_file_evidence'].items():
            if digest(name)!=evidence['sha256']:raise RuntimeError('Input drift '+name)
        if digest(CALIBRATION)!=CALIBRATION_SHA:raise RuntimeError('Calibration artifact drift')
        names=set(original['source_sha256']) | {'scripts/page_gauge_value_conditioning.py',
            'diagnostics/benchmark_pg19_external_quality.py', 'diagnostics/aggregate_pg19_external_quality.py',
            'diagnostics/aggregate_heldout_quality.py','diagnostics/benchmark_token_step_graphs.py',
            'diagnostics/benchmark_generated_sequence_graph.py',
            'experiments/mlsys2027/representation_v2/ground_truth_quality.py',
            'experiments/mlsys2027/representation_v2/prototype.py','experiments/mlsys2027/representation_v2/run.sh'}
        out=ROOT/'results/mlsys2027_representation_v2'/('ground_truth_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        cmd=[sys.executable,'-u',str(Path(__file__)),'--worker',str(out)]
        invocation=json.loads((OLD/'block_1_invocation.json').read_text())['command']
        write_json(out/'manifest.json',{'stage':'E0_ground_truth_development','offset':args.offset,
            'seed':2026090804,'model':invocation[invocation.index('--model')+1],
            'source_sha256':{n:digest(ROOT/n) for n in sorted(names)},'inputs':original['input_file_evidence'],
            'command':cmd,'calibration_sha256':CALIBRATION_SHA,'idle_check':idle,
            'order':['original_fi','original_pg','fixed_pg','fixed_fi'],
            'scope':'TRAIN only; all four variants reported; no quality-based early selection'})
        print('Ground-truth output: '+str(out),flush=True)
        completion=previous.run_process(cmd,out,{'index':0},gpu)
        write_json(out/'completion.json',completion)
        if completion['return_code']!=0 or not completion['sampled_exclusivity_passed']:
            raise RuntimeError('Quality worker failed; preserve logs and partial results')


if __name__=='__main__':main()
