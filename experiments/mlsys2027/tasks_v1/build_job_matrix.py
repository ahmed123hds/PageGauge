"""Build immutable worker directories from qualified source manifests."""
import json
from pathlib import Path
from synthetic_generation import base, verify


def build(root, specifications, purpose, metadata):
    if root.exists():
        raise FileExistsError('Use a new matrix directory')
    if purpose != 'synthetic_integration':
        raise ValueError('Public evaluation requires the separate dataset/source freeze builder')
    if not specifications or len({s['id'] for s in specifications}) != len(specifications):
        raise ValueError('Unique nonempty jobs required')
    root.mkdir(parents=True)
    jobs = []
    for spec in specifications:
        if not spec['id'].replace('_', '').isalnum():
            raise ValueError('Simple job IDs required')
        source = Path(spec['source'])
        manifest = json.loads((source/'manifest.json').read_text())
        verify(manifest)
        fixtures_path = Path(spec['fixtures'])
        fixtures = json.loads(fixtures_path.read_text())
        folder = root/spec['id']
        folder.mkdir()
        base.atomic_json(folder/'fixtures.json', fixtures)
        manifest.update(spec.get('updates', {}))
        manifest['fixtures_sha256'] = base.sha256_file(folder/'fixtures.json')
        for path in (Path(__file__).resolve(), Path(spec['worker']), Path(__file__).with_name('run_frozen_jobs.py')):
            manifest['source_sha256'][str(path)] = base.sha256_file(path)
        manifest['scope'] = 'Synthetic unified-launcher integration only, no benchmark quality claim.'
        base.atomic_json(folder/'manifest.json', manifest)
        jobs.append({'id': spec['id'], 'directory': spec['id'], 'worker': spec['worker'],
            'python': spec['python'], 'control_group': spec['control_group'],
            'sha256': {**{n: base.sha256_file(folder/n) for n in ('manifest.json', 'fixtures.json')},
                       'worker': base.sha256_file(Path(spec['worker']))}})
    plan = {'schema_version': 1, 'purpose': purpose, 'jobs': jobs, 'metadata': metadata}
    base.atomic_json(root/'plan.json', plan)
    return root/'plan.json'
