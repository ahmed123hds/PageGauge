"""Prepare pinned public prompts only after an explicit source/protocol freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from prompt_preparation import load_templates, prepare_prompt
from task_contract import TASK_OUTPUT_LIMITS

ARCHIVE_SHA = 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
REVISION = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_task(archive, task):
    """No filesystem extraction; read only the exact predeclared task member."""
    name = 'data/'+task+'.jsonl'
    if archive.namelist().count(name) != 1:
        raise ValueError('Missing/duplicate task archive member')
    records = [json.loads(line) for line in archive.read(name).decode('utf-8').splitlines() if line.strip()]
    if not records or len({r['_id'] for r in records}) != len(records):
        raise ValueError('Empty task or duplicate dataset IDs')
    return records


def materialize(freeze_path, expected_freeze_sha, archive_path, out):
    if sha(freeze_path) != expected_freeze_sha or out.exists():
        raise ValueError('Wrong freeze or existing output directory')
    freeze = json.loads(freeze_path.read_text())
    if freeze.get('status') != 'frozen_before_public_data' or freeze['dataset_revision'] != REVISION:
        raise ValueError('Explicit pre-exposure freeze required')
    if freeze['tasks'] != list(TASK_OUTPUT_LIMITS) or freeze['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Wrong declared task cohort/archive')
    sources = freeze['source_sha256']
    if sources.get(str(Path(__file__).resolve())) != sha(Path(__file__)):
        raise ValueError('Materializer not included in frozen source closure')
    for path, digest in sources.items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen source drift: '+path)
    if sha(archive_path) != ARCHIVE_SHA:
        raise ValueError('Archive content differs from pinned LFS object')
    from transformers import AutoTokenizer
    templates = load_templates(freeze['prompt_templates'])
    tokenizers = {family: AutoTokenizer.from_pretrained(spec['path'], local_files_only=True,
                   trust_remote_code=False) for family, spec in freeze['models'].items()}
    out.mkdir(parents=True)
    counts = {}
    with zipfile.ZipFile(archive_path) as archive:
        for task in freeze['tasks']:
            records = read_task(archive, task)
            references = [{'id': task+':'+r['_id'], 'task': task, 'answers': r['answers']} for r in records]
            with (out/(task+'_references.json')).open('x', encoding='utf-8') as f:
                json.dump(references, f, ensure_ascii=False)
            for family, tokenizer in tokenizers.items():
                cases = []
                for record in records:
                    contract = prepare_prompt(task, record['context'], record['input'], tokenizer,
                        family, templates, freeze['models'][family]['context_limit'], freeze['prompt_budget'])
                    cases.append({'case_id': task+':'+record['_id'], 'task': task,
                        'prompt_ids': list(contract.prompt_ids),
                        'original_prompt_tokens': contract.original_prompt_tokens,
                        'removed_tokens': contract.removed_tokens})
                with (out/(family+'_'+task+'_fixtures.json')).open('x', encoding='utf-8') as f:
                    json.dump({'cases': cases}, f)
            counts[task] = len(records)
    evidence = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    with (out/'materialization.json').open('x') as f:
        json.dump({'freeze_sha256': expected_freeze_sha, 'archive_sha256': ARCHIVE_SHA,
                   'task_counts': counts, 'output_sha256': evidence}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    materialize(args.freeze, args.freeze_sha256, args.archive, args.out)
