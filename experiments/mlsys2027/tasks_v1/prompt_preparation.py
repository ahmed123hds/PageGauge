"""Pinned LongBench text templates and per-model final token preparation."""
import hashlib
import json
from pathlib import Path
from task_contract import TASK_OUTPUT_LIMITS, RAW_COMPLETION_TASKS, prepare

PROMPT_SHA = '56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30'


def load_templates(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != PROMPT_SHA:
        raise ValueError('Prompt configuration differs from pinned official source')
    templates = json.loads(data)
    return {task: templates[task] for task in TASK_OUTPUT_LIMITS}


def prepare_prompt(task, context, query, tokenizer, family, templates,
                   context_limit, prompt_budget=30720):
    if family not in ('mistral', 'qwen3') or task not in TASK_OUTPUT_LIMITS:
        raise ValueError('Unsupported model family/task')
    if not isinstance(context, str) or not isinstance(query, str):
        raise ValueError('Context/query must be strings; never pass answers')
    text = templates[task].format(context=context, input=query)
    if task in RAW_COMPLETION_TASKS:
        ids = tokenizer(text, add_special_tokens=True)['input_ids']
    else:
        kwargs = {'enable_thinking': False} if family == 'qwen3' else {}
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                    tokenize=True, add_generation_prompt=True, **kwargs)
    contract = prepare(task, ids, context_limit, prompt_budget)
    return contract
