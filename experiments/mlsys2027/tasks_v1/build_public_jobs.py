"""Create the declared public matrix from a pre-exposure freeze and prompts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from materialize_longbench import sha, ARCHIVE_SHA
from public_matrix_spec import jobs, POLICY


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def build(freeze_path, digest, materialized, out):
    if out.exists() or sha(freeze_path) != digest:
        raise ValueError('Existing matrix or changed freeze')
    freeze = json.loads(freeze_path.read_text())
    if freeze['status'] != 'frozen_before_public_data' or freeze['policy'] != POLICY or freeze['job_specs'] != jobs():
        raise ValueError('Wrong public specification')
    if freeze['source_sha256'].get(str(Path(__file__).resolve())) != sha(Path(__file__)):
        raise ValueError('Public builder not frozen')
    for name, expected in freeze['source_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Frozen source changed: '+name)
    receipt = json.loads((materialized/'materialization.json').read_text())
    if receipt['freeze_sha256'] != digest or receipt['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Different dataset origin')
    out.mkdir(parents=True)
    matrix = []
    for spec in jobs():
        name = spec['model']+'_'+spec['task']+'_fixtures.json'
        fixture_path = materialized/name
        if sha(fixture_path) != receipt['output_sha256'][name]:
            raise ValueError('Changed prepared prompts')
        fixture = json.loads(fixture_path.read_text())
        if len(fixture['cases']) != receipt['task_counts'][spec['task']]:
            raise ValueError('Incomplete task fixture')
        group = freeze['control_groups'][spec['control_group']]
        manifest = deepcopy(group['template_manifest'])
        model = freeze['models'][spec['model']]
        manifest.update(model=model['path'], model_family=spec['model'],
            source_sha256=freeze['source_sha256'], input_file_evidence=model['input_file_evidence'],
            seed=2026091017, scope='Frozen public generated-task quality; not timing or final PG19 TEST.')
        if spec['worker'] == 'task_generation_worker.py':
            manifest.update({k: POLICY['pagegauge'][k] for k in ('S', 'A', 'T', 'exact_split_pages')})
        else:
            manifest['task_backend'] = spec['arms'][0]
            if spec['worker'] == 'native_full_task_worker_v2.py':
                manifest.update(native_family='nsn' if spec['control_group'] == 'mistral_nsn' else 'kitty',
                                cache_capacity=POLICY['execution']['cache_capacity'])
        folder = out/spec['id']
        folder.mkdir()
        write_json(folder/'fixtures.json', fixture)
        manifest['fixtures_sha256'] = sha(folder/'fixtures.json')
        write_json(folder/'manifest.json', manifest)
        worker = str(Path(__file__).with_name(spec['worker']).resolve())
        matrix.append({'id': spec['id'], 'directory': spec['id'], 'worker': worker,
            'python': group['python'], 'control_group': spec['control_group'],
            'sha256': {'worker': sha(Path(worker)), **{n: sha(folder/n) for n in ('manifest.json', 'fixtures.json')}}})
    write_json(out/'plan.json', {'schema_version': 1, 'purpose': 'frozen_public_evaluation',
        'freeze_sha256': digest, 'materialization_sha256': sha(materialized/'materialization.json'), 'jobs': matrix})
    return out/'plan.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('freeze', 'materialized', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    print(build(args.freeze, args.freeze_sha256, args.materialized, args.out))
