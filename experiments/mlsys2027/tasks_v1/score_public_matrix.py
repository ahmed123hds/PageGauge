"""Frozen public matrix quality report; requires every completed declared job."""
import argparse
import importlib.util
import json
from pathlib import Path
from materialize_longbench import sha, ARCHIVE_SHA
from load_public_results import load_predictions
from public_result_reduction import reduce_complete
from score_saved_run import METRIC_SHA
from task_contract import TASK_OUTPUT_LIMITS


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('freeze', 'plan', 'materialized', 'metrics', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.freeze) != args.freeze_sha256:
        raise ValueError('Existing output or wrong freeze')
    freeze = json.loads(args.freeze.read_text())
    if freeze.get('status') != 'frozen_before_public_data':
        raise ValueError('Candidate evidence is not a final public freeze')
    plan = json.loads(args.plan.read_text())
    if plan.get('purpose') != 'frozen_public_evaluation' or plan.get('freeze_sha256') != args.freeze_sha256:
        raise ValueError('Execution plan belongs to a different freeze')
    for path, digest in freeze['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen source drift')
    if freeze['source_sha256'].get(str(Path(__file__).resolve())) != sha(Path(__file__)):
        raise ValueError('Scoring entry point not frozen')
    materialization = json.loads((args.materialized/'materialization.json').read_text())
    if plan.get('materialization_sha256') != sha(args.materialized/'materialization.json'):
        raise ValueError('Execution plan used different materialization')
    if materialization['freeze_sha256'] != args.freeze_sha256 or materialization['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Different materialization origin')
    expected_files = {task+'_references.json' for task in TASK_OUTPUT_LIMITS}
    expected_files.update(family+'_'+task+'_fixtures.json' for task in TASK_OUTPUT_LIMITS for family in freeze['models'])
    if set(materialization['output_sha256']) != expected_files:
        raise ValueError('Incomplete materialized file attestation')
    for name, digest in materialization['output_sha256'].items():
        if Path(name).name != name or sha(args.materialized/name) != digest:
            raise ValueError('Materialized file changed')
    references = {task: json.loads((args.materialized/(task+'_references.json')).read_text()) for task in TASK_OUTPUT_LIMITS}
    fixtures = {(task, family): json.loads((args.materialized/(family+'_'+task+'_fixtures.json')).read_text())
                for task in TASK_OUTPUT_LIMITS for family in freeze['models']}
    from transformers import AutoTokenizer
    tokenizers = {family: AutoTokenizer.from_pretrained(spec['path'], local_files_only=True, trust_remote_code=False)
                  for family, spec in freeze['models'].items()}
    predictions, evidence = load_predictions(args.plan, fixtures, tokenizers,
        {f: s['context_limit'] for f, s in freeze['models'].items()},
        {f: s['eos_ids'] for f, s in freeze['models'].items()})
    if sha(args.metrics) != METRIC_SHA:
        raise ValueError('Wrong official metric code')
    spec = importlib.util.spec_from_file_location('official_metrics', args.metrics)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    if metrics.fuzz.SequenceMatcher.__module__ != 'difflib':
        raise ValueError('Changed code similarity backend')
    result = reduce_complete(references, fixtures, predictions, metrics)
    result.update(freeze_sha256=args.freeze_sha256, plan_sha256=sha(args.plan),
                  materialization_sha256=sha(args.materialized/'materialization.json'), input_sha256=evidence)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
