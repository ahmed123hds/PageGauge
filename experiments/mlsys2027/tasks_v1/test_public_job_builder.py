import json
from pathlib import Path
import tempfile
import unittest
import build_public_jobs as builder
from materialize_longbench import sha, ARCHIVE_SHA
from public_matrix_spec import GROUPS, POLICY, jobs
from task_contract import TASK_OUTPUT_LIMITS


class Tests(unittest.TestCase):
    def test_eighty_filesystem_jobs_and_policy_mapping(self):
        with tempfile.TemporaryDirectory(prefix='pagegauge_builder_test_') as temporary:
            root = Path(temporary)
            freeze = {'status': 'frozen_before_public_data', 'policy': POLICY, 'job_specs': jobs(),
                'source_sha256': {str(Path(builder.__file__).resolve()): sha(Path(builder.__file__))},
                'models': {f: {'path': '/synthetic/'+f, 'input_file_evidence': {}} for f in ('mistral', 'qwen3')},
                'control_groups': {g['id']: {'python': '/synthetic/python', 'template_manifest': {}}
                                   for g in GROUPS}}
            freeze_path = root/'synthetic_freeze.json'
            builder.write_json(freeze_path, freeze)
            digest = sha(freeze_path)
            prepared = root/'prepared'
            prepared.mkdir()
            hashes = {}
            for task in TASK_OUTPUT_LIMITS:
                for family in ('mistral', 'qwen3'):
                    path = prepared/(family+'_'+task+'_fixtures.json')
                    builder.write_json(path, {'cases': [{'case_id': task+':synthetic', 'task': task,
                        'prompt_ids': [1, 2], 'original_prompt_tokens': 2, 'removed_tokens': 0}]})
                    hashes[path.name] = sha(path)
            builder.write_json(prepared/'materialization.json', {'freeze_sha256': digest,
                'archive_sha256': ARCHIVE_SHA, 'output_sha256': hashes,
                'task_counts': {task: 1 for task in TASK_OUTPUT_LIMITS}})
            plan_path = builder.build(freeze_path, digest, prepared, root/'matrix')
            plan = json.loads(plan_path.read_text())
            self.assertEqual(len(plan['jobs']), 80)
            for job, spec in zip(plan['jobs'], jobs()):
                folder = plan_path.parent/job['directory']
                manifest = json.loads((folder/'manifest.json').read_text())
                self.assertEqual(manifest['fixtures_sha256'], sha(folder/'fixtures.json'))
                self.assertEqual(manifest['model'], '/synthetic/'+spec['model'])
                if spec['worker'] == 'task_generation_worker.py':
                    self.assertEqual((manifest['S'], manifest['A'], manifest['T']), (4, 128, 768))
                else:
                    self.assertEqual(manifest['task_backend'], spec['arms'][0])
            with self.assertRaises(ValueError):
                builder.build(freeze_path, digest, prepared, root/'matrix')


if __name__ == '__main__':
    unittest.main()
