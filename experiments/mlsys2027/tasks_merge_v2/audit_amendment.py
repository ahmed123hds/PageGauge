"""Audit the exact allowed delta from the original public matrix."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(plan_path):
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    amendment_path = plan_path.parent/'amendment.json'
    amendment = json.loads(amendment_path.read_text())
    if (plan.get('post_exposure_correction') is not True or
            plan.get('amendment_sha256') != sha(amendment_path) or
            amendment.get('version') != 2 or
            amendment.get('independent_test_claim') is not False):
        raise ValueError('Invalid amendment provenance')
    original_path = Path(amendment['original_plan'])
    if sha(original_path) != amendment['original_plan_sha256']:
        raise ValueError('Original plan changed')
    original = json.loads(original_path.read_text())
    if original['freeze_sha256'] != amendment['original_freeze_sha256']:
        raise ValueError('Original freeze mismatch')
    for path, digest in amendment['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Amended source changed')
    restored = copy.deepcopy(plan)
    restored.pop('post_exposure_correction')
    restored.pop('amendment_sha256')
    if len(plan['jobs']) != len(original['jobs']):
        raise ValueError('Changed cohort size')
    count = 0
    for i, (job, old) in enumerate(zip(plan['jobs'], original['jobs'])):
        before = original_path.parent/old['directory']
        after = plan_path.parent/job['directory']
        if sha(after/'fixtures.json') != old['sha256']['fixtures.json']:
            raise ValueError('Changed task inputs')
        m = json.loads((after/'manifest.json').read_text())
        previous = json.loads((before/'manifest.json').read_text())
        if sha(after/'manifest.json') != job['sha256']['manifest.json']:
            raise ValueError('Changed amended manifest')
        if Path(old['worker']).name == 'task_generation_worker.py':
            if m.pop('merge_center_amendment', None) != amendment:
                raise ValueError('Missing worker amendment')
            expected = copy.deepcopy(previous)
            expected['source_sha256'].update(amendment['source_sha256'])
            if m != expected:
                raise ValueError('Changed settings beyond merge amendment')
            expected_worker = Path(__file__).with_name('task_generation_worker.py').resolve()
            if job['worker'] != str(expected_worker) or job['sha256']['worker'] != sha(expected_worker):
                raise ValueError('Wrong amended worker')
            count += 1
        elif m != previous or job != old:
            raise ValueError('Unchanged baseline was modified')
        normalized = copy.deepcopy(job)
        normalized['worker'] = old['worker']
        normalized['sha256'] = old['sha256']
        if normalized != old:
            raise ValueError('Changed job order or configuration')
        restored['jobs'][i] = old
    if restored != original or count != 16:
        raise ValueError('Unexpected matrix delta')
    return {'audited_jobs': len(plan['jobs']), 'amended_pg_jobs': count,
            'post_exposure_correction': True, 'independent_test_claim': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('plan', type=Path)
    print(json.dumps(audit(p.parse_args().plan)))
