"""Qualify pinned Mistral-Instruct own-generation on synthetic data only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
import synthetic_generation as synthetic
from synthetic_generation import ROOT, base, previous, wsl_gpu_monitor, verify


def main():
    from transformers import AutoTokenizer
    import fcntl
    downloads = list((ROOT/'results/mlsys2027_tasks_v1').glob('instruct_download_*/analysis.json'))
    if len(downloads) != 1:
        raise ValueError('Expected one completed pinned download')
    download = json.loads(downloads[0].read_text())
    model = Path(download['snapshot'])
    if download['revision'] != 'c170c708c41dac9275d15a8fff4eca08d52bab71':
        raise ValueError('Wrong instruction checkpoint')
    source = ROOT/'results/mlsys2027_tasks_v1/synthetic_20260909T025247Z_a44bf775/manifest.json'
    m = json.loads(source.read_text()); verify(m)
    evidence = {}
    for name, expected in download['file_evidence'].items():
        path = model/name
        if base.sha256_file(path) != expected['sha256']:
            raise ValueError('Changed instruction model file')
        stat = path.stat()
        evidence[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': expected['sha256']}
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    cases = [synthetic.make_case(tokenizer, 8192, .5, 2026091016, 'mistral')]
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_tasks_v1'/('instruct_synthetic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir()
        base.atomic_json(out/'fixtures.json', {'cases': cases})
        m.update(model=str(model), model_family='mistral', input_file_evidence=evidence,
            fixtures_sha256=base.sha256_file(out/'fixtures.json'), seed=2026091016,
            prompt_budgets=[8192], max_new_tokens=32,
            scope='One synthetic Mistral-Instruct retrieval prompt; independent HF/FI/PG generation with first-case native HF-generate comparison. Reference policy S4/A128/T768. Not public-task score or optimized layer-runtime performance.')
        for path in (Path(__file__).resolve(), downloads[0], Path(synthetic.__file__).resolve()):
            m['source_sha256'][str(path)] = base.sha256_file(path)
        base.atomic_json(out/'manifest.json', m)
        command = [sys.executable, '-u', synthetic.__file__, '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        print('Instruction synthetic validation: '+str(out), flush=True)
        c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', c); verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed']:
            raise ValueError('Instruction qualification failed; retain and diagnose')


if __name__ == '__main__':
    main()
