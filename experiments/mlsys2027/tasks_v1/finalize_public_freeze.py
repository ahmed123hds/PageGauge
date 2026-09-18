"""Finalize the audited generated-task protocol before public-data access."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from audit_freeze_candidate import audit
from build_public_jobs import write_json
from materialize_longbench import sha

ROOT = Path(__file__).resolve().parents[3]


def main():
    directory = ROOT/'results/mlsys2027_tasks_v1'
    candidate = directory/'public_freeze_candidate_20260910_v4.json'
    target = directory/'public_evaluation_freeze_20260910.json'
    if target.exists():
        raise FileExistsError('Final freeze already exists')
    if sha(candidate) != '77db303f30a9bc062a0388f5d19c9277b3784402eed2bbfc211185c8dfe97caa' or not audit(candidate):
        raise ValueError('Wrong or failed audited candidate')
    tests = ['test_public_matrix_spec', 'test_public_job_builder', 'test_public_result_files',
             'test_public_scoring_cli', 'test_materialization', 'test_cohort_scoring',
             'test_result_adapter', 'test_full_native_result_adapter']
    subprocess.run([sys.executable, '-m', 'unittest', *tests], cwd=Path(__file__).parent, check=True)
    result = json.loads(candidate.read_text())
    result['status'] = 'frozen_before_public_data'
    result['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    result['candidate_sha256'] = sha(candidate)
    result['source_sha256'][str(Path(__file__).resolve())] = sha(Path(__file__))
    result.pop('remaining')
    result['freeze_scope'] = ('LongBench eight-task, two-checkpoint quality evaluation only. '
        'Not final PG19 TEST, serving evaluation, A100 runtime confirmation or entire-paper completion. '
        'No retuning on resulting public scores; retain failures and amendments explicitly.')
    result['integration_tests'] = tests
    write_json(target, result)
    print(json.dumps({'freeze': str(target), 'sha256': sha(target), 'status': result['status']}, indent=2))


if __name__ == '__main__':
    main()
