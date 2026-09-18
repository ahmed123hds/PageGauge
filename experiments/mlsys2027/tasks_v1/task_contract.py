"""Dataset-independent generated-task contract; does not read benchmark data."""
from dataclasses import dataclass
from generation_contract import aligned_prefix

TASK_OUTPUT_LIMITS = {'qasper': 128, 'multifieldqa_en': 64, 'hotpotqa': 32,
                      'gov_report': 512, 'qmsum': 512, 'triviaqa': 32,
                      'lcc': 64, 'repobench-p': 64}
RAW_COMPLETION_TASKS = frozenset(('triviaqa', 'lcc', 'repobench-p'))


@dataclass(frozen=True)
class PromptContract:
    task: str
    prompt_ids: tuple
    original_prompt_tokens: int
    max_new_tokens: int
    context_limit: int
    prefill_tokens: int
    fp16_fallback: bool
    fallback_reason: str | None

    @property
    def removed_tokens(self):
        return self.original_prompt_tokens-len(self.prompt_ids)


def prepare(task, formatted_ids, context_limit, prompt_budget=30720,
            minimum_quantized_prefix=4096):
    """Accept final native-template IDs, never re-tokenize or inspect answers.

    Deterministic middle truncation preserves both sequence ends. Caller must
    validate chat-template integrity for its tokenizer before benchmark freeze.
    Fallback retains the identical prompt and output budget, using native FP16.
    """
    if task not in TASK_OUTPUT_LIMITS:
        raise ValueError('Undeclared task')
    ids = tuple(formatted_ids)
    if not ids or any(type(v) is not int or v < 0 for v in ids):
        raise ValueError('Nonempty nonnegative integer token IDs required')
    output = TASK_OUTPUT_LIMITS[task]
    budget = min(prompt_budget, context_limit-output)
    if budget < 2 or minimum_quantized_prefix < 16 or minimum_quantized_prefix % 16:
        raise ValueError('Invalid context/prefix budget')
    if len(ids) > budget:
        left = (budget+1)//2
        ids = ids[:left]+ids[-(budget-left):]
    prefix = aligned_prefix(len(ids)) if len(ids) >= 17 else 0
    fallback = prefix < minimum_quantized_prefix
    return PromptContract(task, ids, len(formatted_ids), output, context_limit,
                          prefix, fallback, 'short_prompt_fixed_policy' if fallback else None)


def validate_result(contract, generated_ids, stop_reason, eos_ids):
    ids = tuple(generated_ids)
    if not ids or len(ids) > contract.max_new_tokens:
        raise ValueError('Invalid output length')
    if any(type(v) is not int or v < 0 for v in ids):
        raise ValueError('Invalid output token IDs')
    eos = set(eos_ids)
    if any(v in eos for v in ids[:-1]):
        raise ValueError('Generation continued after EOS')
    if stop_reason == 'eos':
        if ids[-1] not in eos:
            raise ValueError('EOS reason without EOS token')
    elif stop_reason == 'max_new_tokens':
        if len(ids) != contract.max_new_tokens or ids[-1] in eos:
            raise ValueError('Wrong token-limit termination')
    else:
        raise ValueError('Undeclared stop reason')
