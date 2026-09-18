"""Matched recurrent CPU/CUDA diagnostic attribution; never a speed-acceptance CI."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
RUNS={
    'fi': 'common_engine_flashinfer_fp16_20260908T114348Z_dae29b42',
    'pg_exact128': 'common_engine_page_gauge_20260908T114547Z_6931ac52',
    'pg_exact32': 'common_engine_page_gauge_20260908T115837Z_eb061126',
}


def category(name):
    if 'gemvx' in name:return 'projection_and_lm_gemv'
    if 'BatchPrefillWithPagedKVCacheKernel' in name:
        return 'int8_history_attention' if 'PageAffineInt8Attention' in name else 'fp16_attention'
    if 'MergeState' in name:return 'merge'
    if 'rope_append' in name.lower():return 'rope_append'
    return 'other'


def main():
    rows={};evidence={};tokens=set();positions=set()
    for label,name in RUNS.items():
        directory=ROOT/'results/mlsys2027_baselines_v1'/name
        m=json.loads((directory/'manifest.json').read_text())
        a=json.loads((directory/'analysis.json').read_text())
        c=json.loads((directory/'completion.json').read_text())
        if not a['profile_only'] or c['return_code'] or not c['sampled_exclusivity_passed']:
            raise ValueError('Invalid profile worker')
        path=directory/'profile.json';sha=base.sha256_file(path)
        if sha!=a['profile_sha256']:raise ValueError('Changed profile')
        evidence[str(path)]=sha
        p=json.loads(path.read_text());steps=p['profiled_steps']
        if steps!=129:raise ValueError('Mismatched profile length')
        tokens.add(m['tokens_sha256']);positions.add(p['model_position_start'])
        gpu=[e for e in p['events'] if e['device_type']=='DeviceType.CUDA']
        if not gpu:raise ValueError('No CUDA attribution captured')
        groups={};counts={}
        for e in gpu:
            key=category(e['name'])
            groups[key]=groups.get(key,0)+e['self_device_us']/1000/steps
            counts[key]=counts.get(key,0)+e['count']/steps
        launch=[e for e in p['events'] if e['device_type']=='DeviceType.CPU' and e['name'] in ('cudaLaunchKernel','cudaLaunchKernelExC')]
        rows[label]={'gpu_self_ms_per_step':groups,'gpu_total_self_ms_per_step':sum(groups.values()),
            'gpu_events_per_step':counts,'cpu_launch_api_calls_per_step':sum(e['count'] for e in launch)/steps,
            'instrumented_cpu_launch_self_ms_per_step':sum(e['self_cpu_us'] for e in launch)/1000/steps,
            'exact_split_pages':m.get('exact_split_pages',128) if label!='fi' else None}
    if len(tokens)!=1 or len(positions)!=1:raise ValueError('Unmatched fixtures or profile positions')
    out=ROOT/'results/mlsys2027_baselines_v1'/('profile_attribution_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    result={'rows':rows,'input_sha256':evidence,'reducer_sha256':base.sha256_file(Path(__file__)),
        'scope':'Same B1/C20480/D1536 TRAIN trajectory, final129 steps. GPU kernel self-times only, CPU launch API rows separately. Profiling perturbs host execution; totals are not wall time, final CIs or optimized serving throughput. Quantized attention differs numerically across paths.'}
    base.atomic_json(out/'analysis.json',result)
    print('Profile attribution: '+str(out),flush=True)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
