# Hostile reviewer audit

## Is the general principle clear?

**Yes.** The introduction and Figure 1 lead with the operator-first thesis, not
with INT8 as novelty. The precise novelty claim is that token-independent
centers and page/head scalar scales are selected so affine metadata factors
through the consuming QK/softmax/PV algebra.

## Is the mixed-segment derivation mathematically complete?

**Yes, within its stated assumptions.** Proposition 1 defines centered exact
K/V, centered quantized reconstruction, the globally shared centers, scalar
page/head scales, and the online `(m,l,o)` merge. It shows why the key center is
a domain-wide common logit shift and why the value center is restored once
only after the complete merged probability distribution sums to one. The paper
does not apply these arguments independently to unaligned segment centers.

## Is factorization error separated from quantization error?

**Yes.** The paper distinguishes original FP16 to reconstructed cache error
from reconstructed attention to factorized kernel error. It never says that
PageGauge is exact relative to FP16. The explicit-reconstruction experiment is
labeled a development diagnostic and is not used as final predictive evidence.

## Are exact-region bytes counted honestly?

**Yes.** The main table reports raw served allocation: 7,288,520,704 bytes.
It includes 64 exact prefix tokens, a 2,048-token fixed exact original-prefill
suffix, and a 768-token exact recent tail. It also states that INT8 code/scale
storage for exact pages remains allocated; no compact-layout claim is made.

## Is the FlashInfer comparison fair and fresh-process?

**Yes, with explicit scope.** Each of eight blocks starts a fresh Python/CUDA
process and allocates only the selected backend cache. The schedule is ABBA then
BAAB across two disjoint fixture/seed clusters. The claim excludes model load,
prefill, graph capture, cache restoration, and scrub, but includes all 1,536
continuous decode steps, page replanning, 32 layer-graph replays per step, final
norm, LM head, and GPU argmax. This is a complete-decoder benchmark, not a full
production serving stack.

## Are timing statistics aggregated at the correct independent level?

**Yes, but the sample of independent fixtures is small.** The primary estimand
is the exponential mean of adjacent-pair log latency ratios. The 95% interval
resamples fixture/seed first and adjacent pair second, not the three within-block
repeats as independent observations. Only two fixture/seed clusters are
available, so the very narrow interval should not be read as cross-workload
uncertainty.

## Is adaptive development disclosed accurately?

**Yes.** The originally preregistered S3/T768 TEST confirmation failed. The
final S4/A128/T768 policy was selected after diagnosis on exposed TEST data.
The six-window result is therefore described as post-failure fixed-policy
confirmation, not untouched held-out selection. The original selection SHA
`D1C8675B3B1F70437023DC8CD56665EA79602AAF6EDA2752717C0CEBC127798F`
and the failure record are preserved in the appendix.

## Does any claim exceed the measured GPU/model evidence?

**No.** Claims are limited to Mistral-7B-v0.3, FP16 arithmetic after code
conversion, batch 4, 20,480-token context plus 1,536 decode steps, and one RTX
5090/SM120 under WSL2. The paper does not claim state of the art, cross-model or
cross-GPU generality, end-to-end server throughput, or novelty for exact
prefix/tail/suffix policies.

## Remaining skeptical reading

The strongest objections are external validity and adaptive evaluation. The
algebraic proposition is general under its assumptions, but the system evidence
is one model/GPU point. The final predictive cohort reuses exposed TEST windows
after policy repair, so it establishes consistency of the fixed repaired policy
on those windows, not unbiased generalization. The performance experiment is
methodologically cleaner, but only two independent fixture/seed clusters bound
chronology and input sensitivity. These limitations are material and appear in
the main paper rather than being hidden in the appendix.
