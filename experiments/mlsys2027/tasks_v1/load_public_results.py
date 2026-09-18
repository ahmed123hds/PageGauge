"""Load attested completed jobs into the declared quality reduction matrix."""
import json
from pathlib import Path
from materialize_longbench import sha
from public_matrix_spec import jobs
from result_adapter import normalize as normalize_pg
from native_result_adapter import normalize as normalize_native
from full_native_result_adapter import normalize as normalize_full


def load_predictions(plan_path, central_fixtures, tokenizers, context_limits, eos_ids):
    root = plan_path.resolve().parent
    plan = json.loads(plan_path.read_text())
    expected = {j['id']: j for j in jobs()}
    if len(plan['jobs']) != len(expected) or {j['id'] for j in plan['jobs']} != set(expected):
        raise ValueError('Wrong public job matrix')
    plan_hash = sha(plan_path)
    predictions, evidence = {}, {}
    for job in plan['jobs']:
        spec = expected[job['id']]
        folder = (root/job['directory']).resolve()
        if root not in folder.parents:
            raise ValueError('Job escapes matrix root')
        receipt = json.loads((folder/'matrix_completion.json').read_text())
        completion = json.loads((folder/'completion.json').read_text())
        if receipt['plan_sha256'] != plan_hash or receipt['analysis_sha256'] != sha(folder/'analysis.json'):
            raise ValueError('Unattested result')
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Failed execution/exclusivity')
        for name in ('manifest.json', 'fixtures.json'):
            if sha(folder/name) != job['sha256'][name]:
                raise ValueError('Job input drift')
        fixture = json.loads((folder/'fixtures.json').read_text())
        central = central_fixtures[spec['task'], spec['model']]
        if fixture != central:
            raise ValueError('Different inputs from frozen central prompts')
        analysis = json.loads((folder/'analysis.json').read_text())
        family = spec['model']
        if spec['worker'] == 'task_generation_worker.py':
            rows = normalize_pg(analysis, fixture, spec['arms'])
        elif spec['worker'] == 'native_task_worker.py':
            rows = normalize_native(analysis, fixture, spec['arms'][0], tokenizers[family], context_limits[family], eos_ids[family])
        else:
            native_family = 'nsn' if spec['control_group'] == 'mistral_nsn' else 'kitty'
            rows = normalize_full(analysis, fixture, spec['arms'][0], native_family,
                tokenizers[family], context_limits[family], eos_ids[family])
        predictions.setdefault((spec['task'], spec['control_group']), []).extend(rows)
        for name in ('analysis.json', 'completion.json', 'matrix_completion.json'):
            path = folder/name
            evidence[str(path)] = sha(path)
    return predictions, evidence
