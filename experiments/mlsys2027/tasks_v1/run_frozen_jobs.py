"""Execute an ordered frozen task matrix; never silently restart partial jobs."""
import argparse
import json
from pathlib import Path
from synthetic_generation import base, previous, wsl_gpu_monitor, verify


def validate_plan(plan, root):
    if plan.get('schema_version') != 1 or plan.get('purpose') not in ('synthetic_integration', 'frozen_public_evaluation'):
        raise ValueError('Explicit plan purpose/schema required')
    jobs = plan['jobs']
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Unique nonempty job list required')
    for job in jobs:
        folder = (root/job['directory']).resolve()
        if folder == root or root not in folder.parents:
            raise ValueError('Job directory escapes plan root')
        for filename in ('manifest.json', 'fixtures.json'):
            if base.sha256_file(folder/filename) != job['sha256'][filename]:
                raise ValueError('Frozen job input changed')
        manifest = json.loads((folder/'manifest.json').read_text())
        verify(manifest)
        if manifest['fixtures_sha256'] != job['sha256']['fixtures.json']:
            raise ValueError('Worker fixture attestation differs')
        if base.sha256_file(Path(job['worker'])) != job['sha256']['worker']:
            raise ValueError('Worker changed')
        if not Path(job['python']).is_file():
            raise ValueError('Missing pinned environment')
    return jobs


def run(plan_path):
    import fcntl
    root = plan_path.resolve().parent
    plan_hash = base.sha256_file(plan_path)
    plan = json.loads(plan_path.read_text())
    jobs = validate_plan(plan, root)
    completed = []
    for job in jobs:
        folder = (root/job['directory']).resolve()
        receipt = folder/'matrix_completion.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['plan_sha256'] != plan_hash or old['analysis_sha256'] != base.sha256_file(folder/'analysis.json'):
                raise ValueError('Completed matrix result changed')
            completed.append(job['id'])
            continue
        if any((folder/name).exists() for name in ('matrix_invocation.json', 'completion.json', 'analysis.json', 'failure.json', 'progress.json')):
            raise RuntimeError('Existing attempt needs explicit diagnosis/recovery, not automatic restart: '+job['id'])
        idle = base.idle_preflight(0)
        gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if base.sha256_file(plan_path) != plan_hash:
                raise ValueError('Plan changed during execution')
            validate_plan(plan, root)
            command = [job['python'], '-u', job['worker'], '--worker', str(folder)]
            base.atomic_json(folder/'matrix_invocation.json', {'command': command, 'plan_sha256': plan_hash, 'idle': idle})
            print('Starting frozen job '+job['id'], flush=True)
            completion = wsl_gpu_monitor.run_process(previous, command, folder, {'index': 0}, gpu)
            base.atomic_json(folder/'completion.json', completion)
            if completion['return_code'] or not completion['sampled_exclusivity_passed']:
                raise RuntimeError('Failed matrix job retained: '+job['id'])
            analysis = json.loads((folder/'analysis.json').read_text())
            if analysis.get('execution_complete') is not True:
                raise ValueError('Worker did not complete declared cohort')
            validate_plan(plan, root)
            base.atomic_json(receipt, {'plan_sha256': plan_hash,
                'analysis_sha256': base.sha256_file(folder/'analysis.json')})
            completed.append(job['id'])
    base.atomic_json(root/'matrix_execution.json', {'plan_sha256': plan_hash,
        'completed_jobs': completed, 'execution_complete': True,
        'scope': 'Generation execution only; requires complete-cohort quality reduction.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
