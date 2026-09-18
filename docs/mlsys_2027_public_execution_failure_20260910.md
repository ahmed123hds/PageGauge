# Frozen public evaluation: Qwen HotpotQA execution failure

The original matrix stopped at job 22, after 21 finalized jobs. Preserve its
plan, source closure, completion receipts, 166 case files, and failure.json.
Do not restart over this directory or count the partial task as completed.

Failed case: `hotpotqa:a3859ad1c8ffbce9bb2952f83cb463cc7051072c38eb8d70`.
The exception is from `ablation_v1/regional_quality.py:95`, called by the
PageGauge arm of `tasks_v1/task_generation_worker.py`. At recurrent step 0,
layer 31, KV head 0, relative L2 was 0.0065430765971541405 and maximum
absolute error was 0.006743431091308594. The unchanged check rejects either
relative L2 > 0.005 or maximum absolute error > 0.02. Thus the relative
criterion failed; the absolute criterion did not.

This oracle reconstructs the *same quantized cache* using FP32 code-times-scale
products, FP32 attention and common-center restoration. It is not a comparison
against the original unquantized HF cache, and is not the old 0.995 cosine gate.
Consequently INT8 information loss alone does not explain this discrepancy.
Finite-precision execution, reference/indexing mistakes, and implementation
errors remain hypotheses; none is established yet. Attention mass in these
four selected queries was history 0.27840939, prefix 0.67284691, static suffix
0.04828528, tail 0.00045846; this is not whole-model attention attribution.

## Next diagnostic, separate from evaluation

Use an explicitly versioned diagnostic outside the frozen source closure.
Replay only the failed input with unchanged model/cache/kernel settings and
capture the selected query, code/scales, exact-region cache, common center and
observed output before the original assertion. Keep the assertion and original
failed evaluation intact. No answer-based selection or threshold adjustment.
On the captured tensors, compare FP64 versus FP32 reference, centered versus
restored output error, and separate historical/exact-region outputs and merge.
These checks distinguish reference conditioning/rounding from region/indexing
or kernel errors. Any correction needs a documented amendment and a new
evaluation version; exposed examples cannot become independent final evidence.

No task-quality conclusion can be drawn from this failed partial job. The
remaining frozen matrix, independent PG19 evaluation, serving validation,
cross-GPU confirmation and paper update are still outstanding.

## Isolated replay and precision decomposition

Separate captures `qwen_hotpot_failure_capture_v1` and `v2` reproduced the
original error exactly without changing the assertion. FP32 versus FP64
reference relative error was 2.0281e-6. Historical and exact-region output
relative errors were respectively 0.00020159 and 0.00030229; their log2 LSE
maximum errors were 4.3810e-6 and 6.3924e-6. The FP64 partitioned reference
agrees with the concatenated reference to 1.43e-14 relative error.

With final-output normalization, ideal merging/restoration of the captured
rounded region outputs gives 0.00413359 relative error. Restoring the center
to the captured FP16 merged output gives 0.00654338. This localizes a material
contribution to the intermediate merge rounding under center cancellation.
The next candidate is a fused higher-precision merge and center restoration,
retaining the same algebra/cache representation. It is not yet implemented,
timed, or validated across development cases. Do not claim a resolved failure.
