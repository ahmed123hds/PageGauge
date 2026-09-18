"""Read-only candidate model/config/source audit before final publication freeze."""
import argparse
import json
from pathlib import Path
from materialize_longbench import sha
from public_matrix_spec import jobs, POLICY


def audit(path):
    candidate = json.loads(path.read_text())
    assert candidate['job_specs'] == jobs() and candidate['policy'] == POLICY
    for name, digest in candidate['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Source drift: '+name)
    for family, model in candidate['models'].items():
        config = json.loads((Path(model['path'])/'config.json').read_text())
        generation = json.loads((Path(model['path'])/'generation_config.json').read_text())
        eos = generation['eos_token_id']
        if model['eos_ids'] != (eos if isinstance(eos, list) else [eos]):
            raise ValueError('EOS mismatch: '+family)
        if model['context_limit'] > config['max_position_embeddings']:
            raise ValueError('Context exceeds checkpoint')
    required = [Path(__file__).with_name(j['worker']).resolve() for j in jobs()]
    required.append(Path('/home/anonymous/pagegauge_baselines/kivi_sm120_env/lib/python3.12/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so'))
    missing = sorted({str(p) for p in required if str(p) not in candidate['source_sha256']})
    print(json.dumps({'candidate_sha256': sha(path), 'missing_required_sources': missing,
                      'passed': not missing, 'scope': 'Read-only audit; does not authorize dataset exposure.'}, indent=2))
    return not missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('candidate', type=Path)
    if not audit(parser.parse_args().candidate):
        raise SystemExit(1)
