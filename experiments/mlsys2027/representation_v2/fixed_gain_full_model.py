"""E0 three-worker development pilot: calibrate, new-fixture baseline, fixed gain.

Full-model teacher-forced logits and warmed decode cost, not PPL/task or final CI.
"""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid
from prototype import ROOT, digest, write_json


def main():
    sys.path.insert(0, str(ROOT/'diagnostics'))
    import mlsys_rtx5090_entry as base
    import mlsys_rtx5090_step02 as previous
    import fcntl
    original_dir = ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'
    original = json.loads((original_dir/'manifest.json').read_text())
    template = json.loads((original_dir/'block_1_invocation.json').read_text())['command']
    template[template.index('--min-logits-cosine')+1] = '-1'
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = set(original['source_sha256']) | {'scripts/page_gauge_value_conditioning.py',
            'experiments/mlsys2027/representation_v2/fixed_gain_full_model.py',
            'experiments/mlsys2027/representation_v2/prototype.py',
            'experiments/mlsys2027/representation_v2/run.sh'}
        hashes = {name:digest(ROOT/name) for name in sorted(names)}
        for name, evidence in original['input_file_evidence'].items():
            if digest(name) != evidence['sha256']:
                raise RuntimeError('Changed model/corpus/ABI: '+name)
        out = ROOT/'results/mlsys2027_representation_v2'/('fixed_model_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        schedule = [('calibration', 'folded_prefill_rms', 0),
                    ('transfer_baseline', 'none', 283200),
                    ('transfer_fixed', 'folded_fixed_rms', 283200)]
        write_json(out/'manifest.json', {'stage':'E0_full_model_development_pilot', 'schedule':schedule,
            'source_sha256':hashes, 'inputs':original['input_file_evidence'], 'command_template':template,
            'seed':20260861, 'calibration_scope':'Request zero initial TRAIN prefill, all layers; no evaluation-query fitting',
            'evaluation_scope':'One new B4 TRAIN fixture, not four independent documents or held-out TEST',
            'quality':'All HF logits/top1 checks reported; cosine descriptive; no PPL/task claim',
            'timing':'Adjacent baseline/fixed pair, order confounded, not final CI; all workers retain full recurrence',
            'idle_check':idle, 'production_default_changed':False})
        write_json(out/'orchestrator.json', {'pid':os.getpid(), 'gpu':gpu, 'source_dir':str(ROOT)})
        print('Fixed-gain full-model output: '+str(out), flush=True)
        payloads = []
        artifact = out/'calibration_gain.pt'
        gain_sha = None
        for index, (label, mode, offset) in enumerate(schedule):
            base.idle_preflight(0)
            for name, sha in hashes.items():
                if digest(ROOT/name) != sha:
                    raise RuntimeError('Frozen source drift: '+name)
            for variable in ('PAGEGAUGE_EXPORT_GAIN_PATH','PAGEGAUGE_FIXED_GAIN_PATH','PAGEGAUGE_FIXED_GAIN_SHA256'):
                os.environ.pop(variable, None)
            os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = mode
            if label == 'calibration':
                os.environ['PAGEGAUGE_EXPORT_GAIN_PATH'] = str(artifact)
            if mode == 'folded_fixed_rms':
                if gain_sha is None or digest(artifact) != gain_sha:
                    raise RuntimeError('Calibration artifact changed')
                os.environ['PAGEGAUGE_FIXED_GAIN_PATH'] = str(artifact)
                os.environ['PAGEGAUGE_FIXED_GAIN_SHA256'] = gain_sha
            cmd = list(template)
            target = out/f'block_{index}.json'
            cmd[cmd.index('--output')+1] = str(target)
            cmd[cmd.index('--token-offset')+1] = str(offset)
            write_json(out/f'block_{index}_invocation.json', {'command':cmd, 'conditioning':mode,
                'calibration_sha256':gain_sha, 'label':label})
            print('Starting '+label+' B4/C20480/D1536', flush=True)
            process = previous.run_process(cmd, out, {'index':index}, gpu)
            write_json(out/f'block_{index}_completion.json', process)
            if process['return_code'] not in (0,2) or not target.exists():
                raise RuntimeError('Worker failed; inspect preserved log')
            p = json.loads(target.read_text())
            for name, sha in p['source_sha256'].items():
                if hashes.get(name) != sha:
                    raise RuntimeError('Unfrozen worker source '+name)
            for name, sha in hashes.items():
                if digest(ROOT/name) != sha:
                    raise RuntimeError('Source drift during worker '+name)
            checks = p['correctness']
            if not (checks['same_backend_eager_vs_graph']['passed']
                and checks['runtime_page_finalization_and_consumption']['passed']
                and p['cuda_graph_provenance']['structure_gate_passed'] and process['sampled_exclusivity_passed']):
                raise RuntimeError('Execution/exclusivity gate failed')
            if p['configuration']['value_conditioning_mode'] != mode:
                raise RuntimeError('Conditioning mode mismatch')
            if label == 'calibration':
                gain_sha = digest(artifact)
                write_json(out/'calibration_artifact.json', {'path':str(artifact),'sha256':gain_sha,
                    'source_worker':str(target), 'source_worker_sha256':digest(target),
                    'calibration_token_offset':0,'initial_tokens':20480,'request':0})
            elif mode == 'folded_fixed_rms' and p['value_conditioning']['calibration_artifact_sha256'] != gain_sha:
                raise RuntimeError('Worker did not consume frozen gain')
            payloads.append(p)
        _, a, b = payloads
        if a['trajectory']['teacher_inputs_sha256'] != b['trajectory']['teacher_inputs_sha256']:
            raise RuntimeError('Transfer fixture mismatch')
        timing = {}
        for mode in ('cache_neutral','cache_hot'):
            x = a['timing_modes'][mode]['wall']['mean_ms_per_decode_step']
            y = b['timing_modes'][mode]['wall']['mean_ms_per_decode_step']
            timing[mode] = {'baseline_ms':x, 'fixed_gain_ms':y, 'latency_change_fraction':y/x-1}
        result = {'quality':[{'label':label,'metrics':p['correctness']['backend_vs_hf_sdpa_fp16']}
                            for (label,_,_),p in zip(schedule,payloads)],
            'timing':timing,'conditioning':b['value_conditioning'],'same_timed_work':a['timed_work']==b['timed_work'],
            'execution_passed':True,'production_default_changed':False,
            'scope':'One new TRAIN fixture; not independent-corpus or final quality/speed confirmation'}
        write_json(out/'analysis.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
