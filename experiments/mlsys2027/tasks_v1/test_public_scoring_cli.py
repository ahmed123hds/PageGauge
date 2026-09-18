"""Synthetic filesystem integration; never represents measured model outputs."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import subprocess
import sys
import score_public_matrix
from materialize_longbench import ARCHIVE_SHA
from task_contract import TASK_OUTPUT_LIMITS
from build_public_jobs import write_json
from load_public_results import load_predictions
from materialize_longbench import sha
from public_matrix_spec import jobs
from task_contract import prepare


class Tests(unittest.TestCase):
    def test_all_normalizers_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix='pg_result_test_') as temporary:
            root = Path(temporary)
            matrix, central = [], {}
            for spec in jobs():
                folder = root/spec['id']
                folder.mkdir()
                case = {'case_id': spec['task']+':fake', 'task': spec['task'], 'prompt_ids': [1]*17,
                        'original_prompt_tokens': 17, 'removed_tokens': 0}
                fixture = {'cases': [case]}
                central[spec['task'], spec['model']] = fixture
                contract = asdict(prepare(case['task'], case['prompt_ids'], 32768))
                arm = {'generated_ids': [2], 'generated_text': 'fake', 'stop_reason': 'eos',
                       'fallback': True, 'executed_backend': 'native_hf_fp16', 'quantized_tokens_served': 0}
                row = {'case_id': case['case_id'], 'task': case['task'], 'prompt_contract': contract}
                if spec['worker'] == 'task_generation_worker.py':
                    row['arms'] = {a: dict(arm) for a in spec['arms']}
                    analysis = {'execution_complete': True, 'results': [row]}
                elif spec['worker'] == 'native_task_worker.py':
                    row.update(requested_backend=spec['arms'][0], result=arm)
                    analysis = {'execution_complete': True, 'results': [row]}
                else:
                    family = 'nsn' if spec['control_group'] == 'mistral_nsn' else 'kitty'
                    arm.update(fallback=False, executed_backend=spec['arms'][0], final_lengths=[17]*(32 if family == 'nsn' else 36))
                    row['result'] = arm
                    analysis = {'execution_complete': True, 'results': [row], 'family': family, 'backend': spec['arms'][0]}
                for name, value in [('fixtures.json', fixture), ('manifest.json', {}), ('analysis.json', analysis),
                                    ('completion.json', {'return_code': 0, 'sampled_exclusivity_passed': True})]:
                    write_json(folder/name, value)
                matrix.append({'id': spec['id'], 'directory': spec['id'],
                    'sha256': {n: sha(folder/n) for n in ('manifest.json', 'fixtures.json')}})
            plan = root/'plan.json'
            prepared = root/'prepared'
            prepared.mkdir()
            for (task, family), fixture in central.items():
                write_json(prepared/(family+'_'+task+'_fixtures.json'), fixture)
            for task in TASK_OUTPUT_LIMITS:
                write_json(prepared/(task+'_references.json'), [{'id': task+':fake', 'task': task, 'answers': ['fake']}])
            cache = '/home/anonymous/.cache/huggingface/hub/'
            freeze = {'status': 'frozen_before_public_data',
                'source_sha256': {str(Path(score_public_matrix.__file__).resolve()): sha(Path(score_public_matrix.__file__))},
                'models': {
                    'mistral': {'path': cache+'models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71', 'context_limit': 32768, 'eos_ids': [2]},
                    'qwen3': {'path': cache+'models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218', 'context_limit': 32768, 'eos_ids': [2]}}}
            # Artificial EOS/output records are solely test inputs, not real model claims.
            freeze_path = root/'synthetic_freeze.json'
            write_json(freeze_path, freeze)
            write_json(prepared/'materialization.json', {'freeze_sha256': sha(freeze_path),
                'archive_sha256': ARCHIVE_SHA, 'output_sha256': {p.name: sha(p) for p in prepared.iterdir()}})
            write_json(plan, {'jobs': matrix, 'purpose': 'frozen_public_evaluation',
                'freeze_sha256': sha(freeze_path), 'materialization_sha256': sha(prepared/'materialization.json')})
            for job in matrix:
                folder = root/job['directory']
                write_json(folder/'matrix_completion.json', {'plan_sha256': sha(plan), 'analysis_sha256': sha(folder/'analysis.json')})
            output = root/'synthetic_report.json'
            subprocess.run([sys.executable, score_public_matrix.__file__, '--freeze', str(freeze_path),
                '--freeze-sha256', sha(freeze_path), '--plan', str(plan), '--materialized', str(prepared),
                '--metrics', '/home/anonymous/pagegauge_baselines/longbench_scoring_2e00731/metrics.py',
                '--output', str(output)], check=True)
            report = json.loads(output.read_text())
            self.assertEqual(len(report['tasks']), 8)
            self.assertEqual(sum(len(groups) for groups in report['tasks'].values()), 40)
            tokenizer = SimpleNamespace(decode=lambda ids, **kw: 'fake')
            args = (plan, central, {f: tokenizer for f in ('mistral', 'qwen3')},
                    {f: 32768 for f in ('mistral', 'qwen3')}, {f: [2] for f in ('mistral', 'qwen3')})
            predictions, evidence = load_predictions(*args)
            self.assertEqual(len(predictions), 40)
            self.assertEqual(sum(map(len, predictions.values())), 112)
            self.assertEqual(len(evidence), 240)
            (root/matrix[0]['directory']/'analysis.json').write_text('{}')
            with self.assertRaises(ValueError):
                load_predictions(*args)


if __name__ == '__main__':
    unittest.main()
