"""Pre-exposure run matrix. Specification only until source freeze is built."""
from task_contract import TASK_OUTPUT_LIMITS

GROUPS = (
    {'id': 'mistral_pg', 'model': 'mistral', 'worker': 'task_generation_worker.py',
     'environment': 'page_gauge_env_protocol_v2', 'arms': ['hf', 'flashinfer_fp16', 'page_gauge']},
    {'id': 'qwen_pg', 'model': 'qwen3', 'worker': 'task_generation_worker.py',
     'environment': 'page_gauge_env_protocol_v2', 'arms': ['hf', 'flashinfer_fp16', 'page_gauge']},
    {'id': 'mistral_kivi', 'model': 'mistral', 'worker': 'native_task_worker.py',
     'environment': 'kivi_mistral_sm120_env', 'arms': ['hf', 'kivi_int2', 'kivi_int4', 'bitdecode_int4']},
    {'id': 'mistral_nsn', 'model': 'mistral', 'worker': 'native_full_task_worker_v2.py',
     'environment': 'nsn_sm120_env', 'arms': ['hf', 'nsn_int2']},
    {'id': 'qwen_kitty', 'model': 'qwen3', 'worker': 'native_full_task_worker_v2.py',
     'environment': 'kitty_sm120_env', 'arms': ['hf', 'kitty_pro']},
)


def jobs():
    result = []
    for task in TASK_OUTPUT_LIMITS:
        for group in GROUPS:
            # Existing PG worker generates three independent arms per model
            # load; native workers use one arm per fresh process.
            partitions = [group['arms']] if group['worker'] == 'task_generation_worker.py' else [[a] for a in group['arms']]
            for arms in partitions:
                result.append({'id': task.replace('-', '_')+'__'+group['id']+'__'+('_'.join(arms)),
                    'task': task, 'control_group': group['id'], 'model': group['model'],
                    'worker': group['worker'], 'environment': group['environment'], 'arms': arms})
    return result


POLICY = {
    'pagegauge': {'S': 4, 'A': 128, 'T': 768, 'exact_split_pages': 32,
                  'selection': 'Previously qualified reference generation policy, not A0 promotion'},
    'prompt_budget': 30720, 'cohort': 'All examples in each of the eight declared tasks',
    'generation': 'Independent greedy continuation, one beam, EOS or official task limit',
    'bootstrap': {'unit': 'paired example within task and software-stack control group',
                  'draws': 10000, 'seed': 2026091017},
    'execution': {'gpu': 'RTX5090', 'concurrent_gpu_jobs': 1, 'batch': 1,
                  'cache_capacity': 32768, 'resource_failure': 'Retain partial result and diagnose; no silent sample exclusion'},
    'interpretation': 'Quality evaluation only; no serving-speed claim from these workers',
    'test_policy': 'Do not retune on public evaluation scores; final PG19 TEST remains separate',
}
