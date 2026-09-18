# Generated-task integration status

These are synthetic implementation checks, not public-task scores or final
evaluation approval. Historical failures remain in their original directories.

| Execution path | Verified integration | Remaining scope |
| --- | --- | --- |
| PageGauge/FI/HF, Mistral-Instruct | Independent long generation, native short fallback, packed model reuse | Frozen public cohort and end-to-end reduction |
| Native HF/KIVI4/BitDecoding4 | Resident model, long+short requests, normalized raw outputs | Full public cohort; KIVI2 cohort worker integration |
| KIVI2 | Independent long synthetic generation and unified-launcher long/short cohort completed | Normalized cohort reduction and public evaluation |
| Native HF/NSN2 | Resident model, native full-prefix long+short requests, normalization | Full public cohort |
| Native HF/KittyPro | Qwen resident model, native full-prefix long+short requests, normalization | Full public cohort |

Evidence: results/mlsys2027_tasks_v1/native_cohort_qualification.json and
full_native_cohort_qualification.json preserve normalized synthetic outputs and
input hashes. NSN/Kitty complete-cohort processes all exited0 with sampled GPU
exclusivity; each retained one model load. Short native inputs were not replaced
by PageGauge fallback. Identical strings on these examples do not establish
benchmark equivalence or quantized computation on every short token.

The HF partition investigation found different continuations after53 identical
tokens when changing prefill scheduling; matched prefill restored all128 output
IDs. Keep native-stack HF controls and report separate stack scores, rather than
attributing this scheduling difference to quantization. Preserve single-pass
mismatch as a limitation, not a passing check.

## Next critical path

1. Finish one reusable launcher that freezes models, method policies, tokenizer,
   task prompts, dataset revision, generation/scoring sources, environments,
   run order and resource budget. Do not open public examples before this freeze.
2. Generate immutable per-model prompt fixtures centrally and pass identical IDs
   to each applicable backend. Keep answer references separate from workers.
3. Run all examples of each predeclared task. Preserve failures/partial progress;
   never score a surviving subset as a complete task. Decode saved IDs with the
   pinned modern tokenizer and reduce with official metrics/paired intervals.
4. Report native controls, method-specific fallback, truncation and cache
   allocation policy. Long-generation aging, actual serving, A100 runtime
   confirmation and final PG19 TEST remain separate unfinished requirements.

The benchmark is not yet run-ready solely because these synthetic checks pass.

## Dataset metadata candidate for final freeze

Read repository metadata only, not data.zip or examples: the THUDM/LongBench
API resolves to zai-org/LongBench, revision
5e628be450b7e67fb7ae6e201bd6d8f7056f7672 (last modified2024-12-18).
Repository files are .gitattributes, LongBench.py, README.md, data.zip.
The final builder must pin the archive object/hash and extract only the eight
declared tasks after source/protocol freeze. Repository metadata discovery is
not dataset exposure or a completed freeze.
