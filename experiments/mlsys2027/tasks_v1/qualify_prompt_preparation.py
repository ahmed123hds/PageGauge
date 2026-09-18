"""Tokenizer-only synthetic qualification; no benchmark data or model weights."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from prompt_preparation import load_templates, prepare_prompt
from task_contract import RAW_COMPLETION_TASKS


def main():
    cache = Path('/home/anonymous/.cache/huggingface/hub')
    specs = [('mistral', 'mistralai--Mistral-7B-Instruct-v0.3', 'c170c708c41dac9275d15a8fff4eca08d52bab71'),
             ('qwen3', 'Qwen--Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218')]
    templates = load_templates('/home/anonymous/pagegauge_baselines/longbench_scoring_2e00731/dataset2prompt.json')
    rows = []
    for family, repo, revision in specs:
        tokenizer = AutoTokenizer.from_pretrained(cache/('models--'+repo)/'snapshots'/revision,
                                                  local_files_only=True, trust_remote_code=False)
        for task in templates:
            for repetitions in (1, 14000):
                context = 'Synthetic archive entry contains ordinary words.\n'*repetitions
                full = prepare_prompt(task, context, 'What is recorded?', tokenizer, family, templates, 200000, 190000)
                limited = prepare_prompt(task, context, 'What is recorded?', tokenizer, family, templates, 32768)
                assert len(limited.prompt_ids)+limited.max_new_tokens <= 32768
                assert limited.prompt_ids[:64] == full.prompt_ids[:64]
                assert limited.prompt_ids[-64:] == full.prompt_ids[-64:]
                if task not in RAW_COMPLETION_TASKS:
                    # All native control tokens must survive middle truncation in order.
                    specials = set(tokenizer.all_special_ids)
                    assert [x for x in full.prompt_ids if x in specials] == [x for x in limited.prompt_ids if x in specials]
                rows.append({'family': family, 'task': task, 'original_tokens': len(full.prompt_ids),
                             'final_tokens': len(limited.prompt_ids), 'removed_tokens': limited.removed_tokens})
    print(json.dumps({'passed': True, 'synthetic_tokenizer_cases': rows,
                      'scope': 'Prompt boundaries/control tokens only; no public examples, model quality or generation claim.'}, indent=2))


if __name__ == '__main__':
    main()
