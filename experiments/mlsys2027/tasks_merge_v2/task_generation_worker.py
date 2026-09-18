"""Explicit post-exposure merge correction; same task cohort and metrics."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/tasks_v1'))
import task_generation_worker as original


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    out = parser.parse_args().worker.resolve()
    try:
        m = json.loads((out/'manifest.json').read_text())
        amendment = m.get('merge_center_amendment', {})
        if amendment.get('version') != 2 or amendment.get('post_exposure_correction') is not True:
            raise ValueError('Explicit post-exposure amendment required')
        required = [Path(__file__).resolve(), ROOT/'diagnostics/merge_center_candidate.py',
                    ROOT/'diagnostics/install_merge_center_candidate.py']
        for path in required:
            if m['source_sha256'].get(str(path)) != original.base.sha256_file(path):
                raise ValueError('Missing or changed amended implementation hash')
        original.verify(m)
        import benchmark_pg19_external_quality as quality
        from install_merge_center_candidate import install
        install(quality.PG.TransformerDecoder)
        original.worker(out)
        original.base.atomic_json(out/'amendment_execution.json', {
            'version': 2, 'post_exposure_correction': True,
            'manifest_sha256': original.base.sha256_file(out/'manifest.json'),
            'scope': __doc__})
    except BaseException as error:
        original.base.atomic_json(out/'failure.json', {'type': type(error).__name__,
            'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
