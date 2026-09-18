"""Prepare separate post-exposure matrix; never relabel the original freeze."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/tasks_v1'))
from run_frozen_jobs import validate_plan


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--original-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old_path, out = args.original_plan.resolve(), args.output.resolve()
    old = json.loads(old_path.read_text())
    validate_plan(old, old_path.parent)
    if out.exists():
        raise ValueError('New output directory required')
    amended_worker = Path(__file__).with_name('task_generation_worker.py').resolve()
    additions = [amended_worker, Path(__file__).resolve(),
                 ROOT/'diagnostics/merge_center_candidate.py',
                 ROOT/'diagnostics/install_merge_center_candidate.py']
    plan = copy.deepcopy(old)
    amendment = {'version': 2, 'post_exposure_correction': True,
        'original_plan_sha256': sha(old_path), 'original_plan': str(old_path),
        'original_freeze_sha256': old['freeze_sha256'],
        'change': 'FP32 merge plus center restoration with one FP16 store; unchanged cache, cohort, metrics and numerical limits',
        'independent_test_claim': False,
        'source_sha256': {str(p): sha(p) for p in additions}}
    out.mkdir(parents=True)
    write(out/'amendment.json', amendment)
    plan['amendment_sha256'] = sha(out/'amendment.json')
    plan['post_exposure_correction'] = True
    for job in plan['jobs']:
        source = old_path.parent/job['directory']
        target = out/job['directory']
        target.mkdir()
        m = json.loads((source/'manifest.json').read_text())
        (target/'fixtures.json').write_bytes((source/'fixtures.json').read_bytes())
        if Path(job['worker']).name == 'task_generation_worker.py':
            m['merge_center_amendment'] = amendment
            m['source_sha256'].update(amendment['source_sha256'])
            job['worker'] = str(amended_worker)
            job['sha256']['worker'] = sha(amended_worker)
        write(target/'manifest.json', m)
        job['sha256']['manifest.json'] = sha(target/'manifest.json')
        if sha(target/'fixtures.json') != job['sha256']['fixtures.json']:
            raise ValueError('Fixture changed during copy')
    write(out/'plan.json', plan)
    validate_plan(plan, out)
    print('Prepared post-exposure matrix:', out)


if __name__ == '__main__':
    main()
