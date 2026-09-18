import copy
import json
from pathlib import Path
import tempfile
import unittest
from audit_amendment import audit, sha


def write(p, v):
    p.write_text(json.dumps(v))


class Tests(unittest.TestCase):
    def test_allowed_delta_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root/'old', root/'new'
            old.mkdir(); new.mkdir()
            worker = Path(__file__).with_name('task_generation_worker.py').resolve()
            jobs = []
            for i in range(17):
                d = old/str(i); d.mkdir()
                write(d/'fixtures.json', {'cases': []})
                write(d/'manifest.json', {'source_sha256': {}, 'seed': 1})
                jobs.append({'id': str(i), 'directory': str(i),
                    'worker': '/original/task_generation_worker.py' if i < 16 else '/original/native.py',
                    'sha256': {'worker': 'old', 'manifest.json': sha(d/'manifest.json'),
                               'fixtures.json': sha(d/'fixtures.json')}})
            original = {'jobs': jobs, 'freeze_sha256': 'original'}
            write(old/'plan.json', original)
            amendment = {'version': 2, 'post_exposure_correction': True,
                'independent_test_claim': False, 'original_plan': str(old/'plan.json'),
                'original_plan_sha256': sha(old/'plan.json'),
                'original_freeze_sha256': 'original', 'source_sha256': {str(worker): sha(worker)}}
            write(new/'amendment.json', amendment)
            plan = copy.deepcopy(original)
            plan.update(post_exposure_correction=True, amendment_sha256=sha(new/'amendment.json'))
            for i, job in enumerate(plan['jobs']):
                d = new/str(i); d.mkdir()
                (d/'fixtures.json').write_bytes((old/str(i)/'fixtures.json').read_bytes())
                m = {'source_sha256': {}, 'seed': 1}
                if i < 16:
                    m.update(merge_center_amendment=amendment, source_sha256=amendment['source_sha256'])
                    job['worker'] = str(worker)
                    job['sha256']['worker'] = sha(worker)
                write(d/'manifest.json', m)
                job['sha256']['manifest.json'] = sha(d/'manifest.json')
            path = new/'plan.json'
            write(path, plan)
            self.assertEqual(audit(path)['amended_pg_jobs'], 16)
            wrong = copy.deepcopy(plan)
            wrong['jobs'].reverse()
            write(path, wrong)
            with self.assertRaises(ValueError): audit(path)
            write(path, plan)
            m = json.loads((new/'0/manifest.json').read_text())
            m['seed'] = 2
            write(new/'0/manifest.json', m)
            plan['jobs'][0]['sha256']['manifest.json'] = sha(new/'0/manifest.json')
            write(path, plan)
            with self.assertRaises(ValueError): audit(path)
            amendment['independent_test_claim'] = True
            write(new/'amendment.json', amendment)
            plan['amendment_sha256'] = sha(new/'amendment.json')
            write(path, plan)
            with self.assertRaises(ValueError): audit(path)


if __name__ == '__main__': unittest.main()
