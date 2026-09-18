"""Collect candidate public freeze evidence without accessing benchmark data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from materialize_longbench import sha, ARCHIVE_SHA, REVISION
from public_matrix_spec import POLICY, jobs
from task_contract import TASK_OUTPUT_LIMITS
from build_public_jobs import write_json

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    'mistral_pg': ('task_fallback_smoke_20260910T081503Z_3aaa0d41', '/home/anonymous/page_gauge_env_protocol_v2/bin/python'),
    'qwen_pg': ('qwen_task_cohort_20260910T095720Z_b7c7d75f', '/home/anonymous/page_gauge_env_protocol_v2/bin/python'),
    'mistral_kivi': ('kivi4_cohort_smoke_20260910T093433Z_95c55494', '/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/bin/python'),
    'mistral_nsn': ('nsn_nsn_int2_cohort_20260910T094104Z_045085c9', '/home/anonymous/pagegauge_baselines/nsn_sm120_env/bin/python'),
    'qwen_kitty': ('kitty_kitty_pro_eos_v2_observed_20260910T100104Z_aa1af6c9', '/home/anonymous/pagegauge_baselines/kitty_sm120_env/bin/python'),
}


def collect(out):
    if out.exists():
        raise FileExistsError('Never overwrite freeze evidence')
    sources, groups, models, environments = {}, {}, {}, {}
    for group, (name, python) in SOURCES.items():
        directory = ROOT/'results/mlsys2027_tasks_v1'/name
        manifest = json.loads((directory/'manifest.json').read_text())
        completion = json.loads((directory/'completion.json').read_text())
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Unqualified source run: '+name)
        for path, digest in manifest['source_sha256'].items():
            if sha(Path(path)) != digest:
                raise ValueError('Historical source drift: '+path)
            sources[path] = digest
        groups[group] = {'python': python, 'template_manifest': manifest,
                         'qualification_completion_sha256': sha(directory/'completion.json')}
        family = 'qwen3' if group.startswith('qwen') else 'mistral'
        if family not in models:
            evidence = manifest['input_file_evidence']
            for path, expected in evidence.items():
                if sha(Path(path)) != expected['sha256']:
                    raise ValueError('Model input changed: '+path)
            models[family] = {'path': manifest['model'], 'input_file_evidence': evidence,
                              'context_limit': 32768, 'eos_ids': [151645, 151643] if family == 'qwen3' else [2]}
        if python not in environments:
            environments[python] = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True).splitlines()
    # KIVI's group template alone does not include the BitDecoding extension.
    extra = ROOT/'results/mlsys2027_tasks_v1/bitdecode4_cohort_smoke_20260910T093451Z_6412bbce'
    extra_manifest = json.loads((extra/'manifest.json').read_text())
    for name, digest in extra_manifest['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Supplemental baseline source drift: '+name)
        sources[name] = digest
    for group in groups.values():
        for name in ('model',):
            if not group['template_manifest'].get(name):
                raise ValueError('Incomplete control template')
    scoring = '/home/anonymous/pagegauge_baselines/longbench_scoring_env/bin/python'
    environments[scoring] = subprocess.check_output([scoring, '-m', 'pip', 'freeze'], text=True).splitlines()
    for path in Path(__file__).parent.glob('*.py'):
        sources[str(path.resolve())] = sha(path)
    official = Path('/home/anonymous/pagegauge_baselines/longbench_scoring_2e00731')
    for name in ('metrics.py', 'eval.py', 'dataset2prompt.json'):
        sources[str(official/name)] = sha(official/name)
    result = {'status': 'candidate_not_authorized_for_data',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset_revision': REVISION,
        'archive_sha256': ARCHIVE_SHA, 'tasks': list(TASK_OUTPUT_LIMITS),
        'prompt_budget': POLICY['prompt_budget'], 'prompt_templates': str(official/'dataset2prompt.json'),
        'policy': POLICY, 'job_specs': jobs(), 'models': models, 'control_groups': groups,
        'source_sha256': sources, 'environments': environments,
        'remaining': ['Review candidate/source coverage then write distinct final pre-exposure freeze']}
    write_json(out, result)
    print(str(out), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    collect(parser.parse_args().out)
