# Next experiment: attribute the error before changing the precision policy

Prepared 2026-09-08. This is a separate development experiment. Do not modify,
stop, or add workload to the running Step 02 benchmark. No production kernel,
quantizer, or frozen benchmark file has been changed for this work. The future
quality-reporting policy was revised below; existing frozen criteria are untouched.

## Research question and decision order

Is the observed quality discrepancy caused primarily by representation loss,
FP16 execution of the factorized algebra, or a non-quantized decoder/reference
discrepancy? An exact real-arithmetic identity does not answer that question.

1. **R03: Fixed-real-activation attribution (implemented, GPU validation pending).**
   Capture one predeclared B1 Mistral TRAIN window, offset 188800 (after the
   Step 02 timing fixtures), with a 20480-token prefill and 1536 ground-truth
   teacher-forced updates. Capture post-RoPE Q and full HF K/V at layers 0,15,31
   and generated lengths 1,784,1536: nine fixtures, all eight KV heads and all
   32 query heads. Freeze hashes before reading text. No TEST access or model
   download. Ground-truth forcing is deliberately distinct from Step 02's HF
   teacher trajectory; do not pool these experiments.
2. **R04: Fix the dominant numerical mechanism (not implemented).** If production
   factorization differs materially from an explicit reconstruction on the SAME
   Q/codes/scales, investigate scaled probability precision, exponent range,
   accumulation, and center restoration. Measure actual internal fragments
   before claiming underflow; normalized final probabilities are not those
   fragments. Keep storage/quantization unchanged for this contrast. A safe
   rescaling prototype must preserve the numerator and denominator algebra.
3. **R05: Improve representation only if its loss is substantial (CPU algebra
   reference implemented; learned/production implementation not implemented).**
   Compare the fixed scalar baseline to shared channel conditioning and then
   separable row/channel scaling. Shared transforms must preserve RoPE semantics
   and have their inference cost counted. Keep K-only, V-only, and joint changes
   separate. No assertion that rotations or output-aware loss are new.
4. **R06: Mechanism and speed ablations (pending).** Implement matched fused
   reconstruction on exactly the same format/layout/policy. Only after numerical
   and representation decisions, test removing static S/A protection; do not
   search TEST or repeatedly extend FP16 regions to pass a threshold. Measure
   retained duplicate cache storage, launch/merge overhead and the fraction of
   full-decoder time that attention can actually improve.
5. **R07: External systems and transfer (pending).** Prioritize BitDecoding,
   KIVI and Kitty; evaluate OScaR/OptR as recent representation competitors.
   Report QServe separately because it changes weights/activations too. Compare
   at matched quality and at matched KV bytes, with same model, context, batch,
   trajectory and timing boundary. Freeze new configs before validation/test.
   Then cover Mistral/Llama/Qwen families, task quality and long context; 5090
   first, A100 second. 1.2--1.5x is a workload-specific objective, not a promise.

R03 is one diagnostic window, not evidence of generality, held-out quality,
free-running generation fidelity, or end-to-end speed. Do not tune and evaluate
on the same window. All 72 KV-head/snapshot cases are reported, not just outliers.

## Exact comparison definitions

Every component comparison fixes the *same* post-RoPE query and original HF
history. Centers come from the original initial-prefill cache, stored FP16.

| Output | Cache and execution | Difference isolates |
|---|---|---|
| A | Original captured K/V, FP64 attention | Numerical reference |
| C | FP16-centered K/V, center restored in FP64 | Centered-storage rounding relative to A |
| B | Historical INT8 codes and stored scales reconstructed in FP64; exact regions as C | Mixed representation error relative to A; B-C measures the effect of quantizing historical pages |
| B16 | Historical residuals reconstructed and rounded to FP16, FP64 attention | Reconstruction-rounding effect relative to B |
| E | B16 centered residuals through stock FP16 FlashInfer, then center add | FP16 execution relative to B16 |
| F | Same fixed-input cache through production PageGauge | Factorized execution relative to E and B |
| H | Observed HF SDPA attention output from the original capture | HF attention numerical difference relative to A |

Errors are vectors: differences telescope, but their norms/cosines do not add.
No comparison here attributes full-model packing/MLP/RoPE implementation errors;
that requires a separate matched full-FP16/HF decoder diagnosis after Step 02.
The new quantizer is a reference emulation with round-half-away-from-zero, not
an assertion that the runtime finalization kernel is identical. GPU replay
uses the unchanged production attention, but not its append/finalize trajectory.

## Safe CPU command now

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/robustness_v1/run.sh check
```

This reads small source/metadata files, does not query or use the GPU, and does
not tokenize the corpus or load the model. The CPU test command is:

```powershell
wsl.exe -d Ubuntu -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/anonymous/page_gauge_env_protocol_v2/bin/python -m unittest discover -s /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/tests -p test_pagegauge_robustness_v1.py
```

## Commands only after the current benchmark finishes

Capture (first GPU validation of this new worker):

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/robustness_v1/run.sh capture
```

The command prints a directory under `results/mlsys2027_robustness_v1`. Expect
roughly 1 GB of fixture storage. It refuses a busy GPU and shares the existing
experiment's device lock; it never stops another job. Only its own child is
terminated if the user interrupts this command. No automatic retry or overwrite.

Then, one command at a time, substitute that printed Linux directory for RUN:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/robustness_v1/run.sh replay --run RUN
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/robustness_v1/run.sh analyze --run RUN
```

Replay uses the GPU without model weights. Analysis uses CPU NumPy/FP64 and
one BLAS thread. Do not run either while collecting performance evidence.
The final `attribution.json` contains per-head comparisons and the fixed-query
bound. No synthetic pass is a quality claim. For future experiments, cosine is
descriptive, not an acceptance gate. The top-1 agreement threshold remains 0.99,
but passing it alone does not establish downstream task quality.

### Quality-policy revision (2026-09-08)

At the user's request, after inspecting related papers and after seeing interim
Step 02 results, the prospective plan removes the hard 0.995 logit-cosine cutoff.
This is a disclosed protocol revision, not a retroactive pass of Step 02.
Preserve original manifests, measured cosine values, and original pass/fail flags.
Do not edit the active runner until its frozen experiment has finished.

The inspected papers do not report this hard logit-cosine gate:

- [KIVI, Section 4](https://arxiv.org/html/2402.02750v2): generation-task scores,
  LongBench, and needle-in-a-haystack retrieval.
- [KVQuant, Section 4 and Appendix M](https://arxiv.org/html/2401.18079v4):
  perplexity on WikiText-2/C4 and long-context evaluations.
- [BitDecoding, Section VI-C, Table I](https://arxiv.org/html/2503.18773v3):
  LongBench scores alongside throughput across bit widths.

This is evidence from these papers, not a claim that no paper uses cosine gates.
For the MLSys study, report task-score and perplexity changes against FP16 with
the quantized cache actually exercised. Keep cosine and top-1 as supplementary
fidelity metrics. Define any task-level acceptance tolerances before fresh
held-out evaluation; do not infer them from the currently observed results.

## Theory and references

`research_notes.tex` contains a proved generalized factorization, fixed-query
error bound, and conditional propagation/margin statements. These are development
notes, not new publication results or an established novelty claim. The original
workshop source and PDF remain unchanged.

Reference correction for the new manuscript: HeadQ (arXiv:2605.03562) is withdrawn
according to its current arXiv record, checked 2026-09-08. Do not cite its empirical
claims as established support. See `research_notes.tex` for primary-source links.
