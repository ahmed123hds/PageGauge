"""Original metrics with mandatory post-exposure amendment provenance."""
import argparse
import json
from pathlib import Path
import sys
from audit_amendment import audit, sha

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/tasks_v1'))
import score_public_matrix as original


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('freeze', 'plan', 'materialized', 'metrics', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--freeze-sha256', required=True)
    args = p.parse_args()
    evidence = audit(args.plan)
    plan = json.loads(args.plan.read_text())
    if plan['freeze_sha256'] != args.freeze_sha256:
        raise ValueError('Wrong original freeze')
    # Require amended-worker execution receipts, not just the original
    # worker's completion JSON, for every affected job.
    for job in plan['jobs']:
        folder = args.plan.resolve().parent/job['directory']
        m = json.loads((folder/'manifest.json').read_text())
        if 'merge_center_amendment' in m:
            receipt = json.loads((folder/'amendment_execution.json').read_text())
            if (receipt.get('version') != 2 or receipt.get('post_exposure_correction') is not True
                    or receipt.get('manifest_sha256') != sha(folder/'manifest.json')):
                raise ValueError('Missing valid amended execution receipt')
    raw_output = args.output.with_name(args.output.name+'.metrics.json')
    if args.output.exists() or raw_output.exists():
        raise ValueError('New output required')
    saved = sys.argv
    sys.argv = [str(Path(original.__file__))]
    for name in ('freeze', 'plan', 'materialized', 'metrics'):
        sys.argv.extend(['--'+name, str(getattr(args, name))])
    sys.argv.extend(['--output', str(raw_output), '--freeze-sha256', args.freeze_sha256])
    try:
        original.main()
    finally:
        sys.argv = saved
    result = json.loads(raw_output.read_text())
    result['evaluation_provenance'] = dict(evidence,
        amendment_sha256=plan['amendment_sha256'],
        scope='Post-exposure correction with unchanged full cohort and original metrics; not untouched independent evaluation',
        original_metrics_report_sha256=sha(raw_output))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
