# Fused merge-and-center development candidate

Status: validated on the development checks below, not promoted to the frozen
public evaluation or final paper claims. Original public matrix stopped after
21 completed jobs at Qwen HotpotQA; retain the failed partial job and all inputs.

## Change and rationale

The candidate merges the historical and exact-region attention outputs and
restores the common value center in FP32, with one final FP16 output conversion.
It does not change quantization, scales, cache policy, attention algebra, or
the existing .005 relative/.02 absolute execution checks. The original code
rounded the merged centered output to FP16 before adding the center. Strong
center cancellation can amplify this intermediate rounding.

Implementation remains in diagnostics/merge_center_candidate.py with an
explicit process-local adapter in diagnostics/install_merge_center_candidate.py.
Production source files have not been edited. Adapter installation is a real
implementation amendment, not evidence that the original frozen method passed.

## Evidence

- Original failed input reproduced exactly in two isolated captures. Candidate
  full-decoder replay completed with six unchanged probes, maximum relative
  error 0.00385423. This exposed-input replay is diagnostic, not independent.
- Existing synthetic 8K Qwen and Mistral prompts completed with six probes each:
  maximum relative errors 0.000586506 and 0.000449636 respectively.
- Twelve synthetic GPU configurations covered B1/B4/B16, region emptiness,
  cancellation and output aliasing against a double-precision reference.
- Qwen TRAIN C20480/D1536 completed: 18 probes, maximum relative 0.00150095,
  all 48 expected newly aged pages consumed. One window, not broad quality proof.
- B4/C20480/D1536 layer-graph validation v2 completed under sampled exclusive
  GPU ownership: 49,152 attention replays and 352 graph/eager checks. These
  compare graph and eager execution, not independent FP32 reconstruction.
  Validation v1 failed because the new launcher used the wrong environment;
  it remains retained. v2 used the harness's pinned KIVI/Mistral environment.

## Matched development timing

results/mlsys2027_baselines_v1/merge_center_timing_v1 contains three fresh
processes in fixed order: FlashInfer, original PG, candidate PG. Each has three
timed repeats after warmup, matching retained tokens and B4/C20480/D1536.
All completion receipts report return code zero and sampled exclusivity pass.

| Arm | Median wall ms/step |
| --- | ---: |
| FlashInfer | 18.015692543619824 |
| Original PageGauge | 15.530409373046913 |
| Candidate PageGauge | 15.482590852213521 |

FI/candidate = 1.16360968; original/candidate = 1.00308853. This pilot showed no
speed penalty, but cannot establish a statistically significant 0.31% gain.
The timing boundary is full teacher-forced decode with cache maintenance,
excluding prefill, restoration and graph capture; it is not request/serving
latency. Synthetic merge microbenchmarks are not substitutes for these results.

## Required continuation

Prepare and audit an explicit new evaluation version with candidate source
hashes and unchanged cohort/metrics. Preserve the original failure and disclose
the correction after exposure; do not label reruns of exposed examples an
untouched test. Do not pool old and amended PG outputs as one implementation.
Use matched replicated timing for any confidence-interval speed claim.
Independent final PG19 evaluation, actual serving, A100 confirmation, broader
runtime/quality coverage and paper revision remain outstanding.
