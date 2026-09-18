# Generated-task evaluation implementation plan

This is an implementation plan, **not the final evaluation manifest**. No
LongBench examples, answers, or final PG19 TEST books have been opened here.
Finish development policy/cost decisions and freeze the runnable source closure
before benchmark exposure. Do not optimize a method on the resulting TEST scores.

## Model and runner validation first

Public Mistral-7B-Instruct-v0.3 revision
`c170c708c41dac9275d15a8fff4eca08d52bab71` is downloaded separately from the
base model used for PPL. Never label base-model and instruction-model results
as the same checkpoint. Qwen3-8B is already local; use its native chat template
with thinking explicitly disabled for this direct-answer protocol, identical
for all its cache backends. Do not use remote model code.

Validate full-model projection/RoPE/dtype compatibility and prompt generation
on synthetic development retrieval cases before reading benchmark examples.
Each HF, FI and PG arm generates its **own** greedy continuation. No HF
teacher-forced continuation is counted as generated-task performance.

For a non-page-aligned prompt, prefill only its largest page-aligned prefix
strictly shorter than the prompt, then recurrently process the remaining
1--16 real prompt tokens. The last prompt token's logits produce output token
zero. Do not pad with visible filler, omit prompt tokens or reuse HF's first
answer. Validate this partition against unmodified HF generation. Short prompts
that cannot support the fixed exact-region policy need a declared FP16 fallback,
reported separately; never silently shorten the exact regions to make them fit.

## Planned eight-task subset

### Native-stack comparison control

The KIVI/BitDecoding Mistral quality adapters use transformers4.36.2 and
256-token eager FP16 prefill chunks, followed by one-token prompt remainder and
own-token recurrence. Include a separate HF control with exactly that schedule;
do not silently identify it with the PageGauge transformers4.57.6 HF control.
Use identical externally prepared prompt IDs across stacks. Report both HF task
scores and paired differences within each stack before interpreting cross-stack
method differences. These adapters do not establish native-prefill speed.

Synthetic qualification found that single-pass and chunked HF prefill can produce
different greedy continuations: the512-token diagnostic first differed at output
index53. Reusing the identical prefill schedule in native model.generate restored
all128 output IDs exactly. This isolates schedule dependence in that case, not
universal numerical equivalence or a quantization-quality result. Preserve both
outputs and the failed single-pass comparison. Do not require cross-stack token
identity as a substitute for measured task scores, and do not attribute an HF
schedule difference to a cache quantizer. Full evaluation still needs the declared
cohort, native controls, truncation/fallback reporting and pinned source closure.

Qasper, MultiFieldQA-en, HotpotQA, GovReport, QMSum, TriviaQA, LCC, RepoBench-P:
three QA tasks, two summarization tasks, one few-shot task, two code tasks.
This is our specified eight-task subset, not a claim that LongBench defines a
unique official eight-task benchmark. Use all examples in each selected task
for final claims, report each score and paired-example bootstrap intervals,
and retain raw generated text, token IDs, EOS reason and cache/backend records.

Use the official LongBench v1 prompts and metrics from commit
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`. Preserve max-new-token budgets
128/64/32/512/512/32/64/64 in the task order above. Apply native model chat
formatting to QA/summarization; retain the official raw completion format for
TriviaQA/LCC/RepoBench-P. Greedy generation, one beam, EOS or task token limit.
No answer-dependent retries, prompt optimization or discarded disagreements.

Model context limits apply to **prompt plus generated tokens**. Keep a fixed
prompt budget (proposed30720 for both local models) including chat formatting.
If needed, deterministic middle truncation preserves both ends. Audit final
token length after formatting; never change the model's RoPE/context settings.
Declare truncation fraction and short-prompt fallback fraction per task.

These output budgets are mostly shorter than T768 and do not prove that
newly generated tokens age into INT8 history. Keep the existing D1536 recurrent
quality tests and add a separate controlled long-generation diagnostic if a
new generation-specific recurrence claim is made. Do not extend official
benchmark output limits just to manufacture page-aging coverage.

Add controlled retrieval and GSM8K as separate panels, with their own frozen
prompt, sampling and scoring definitions. Validate GSM8K integration using
its training split only; do not call it a long-context result. Native low-bit
comparators must receive exactly the same per-model prompt token IDs, generation
settings and scoring, and retain their actual method-specific cache behavior.

## Sources inspected before data exposure

- [LongBench v1 official protocol](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench)
- [Official generation code](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py)
- [Official task output limits](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2maxlen.json)
- [Mistral instruction checkpoint](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71)

Remaining before final freeze: executable generation adapter and scoring tests,
synthetic smoke, model/config attestation, method policy decision, native baseline
generation adapters, exact dataset revision/input manifest, and final resource
budget. This document does not assert those pieces are already implemented.
