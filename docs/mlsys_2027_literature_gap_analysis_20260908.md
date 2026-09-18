# PageGauge: published-work comparison and next experiments

Research assessment, 8 September 2026. This is a new planning overlay, not a
replacement for the submitted workshop paper or any frozen result manifest.
No GPU experiments were launched for this assessment.

## Bottom line

PageGauge has a working mechanism, unusually careful fixed-configuration timing,
independent-corpus fidelity evidence, and useful error attribution. It does not
yet establish a competitive low-bit inference system across models and workloads.
The highest-value next work is a causal factorization control, genuine low-bit
comparisons, model/task transfer, and accounting for practical serving costs.
More repetitions of the existing Mistral/B4 speed experiment cannot fill those gaps.

The papers below provide examples of persuasive evidence, not a mandatory MLSys
checklist or a guarantee of acceptance. A strong paper needs a defensible new
insight and evidence proportional to its claims, not every experiment in every
related paper. Headline speedups with different denominators are not comparable.

## 1. Relevant published work, 2024–2026

Publication status was checked against proceedings or the conference program.
Where proceedings PDF access was unreliable, an author manuscript was consulted
and is linked separately. This is a targeted comparison, not an exhaustive survey.

| Work / venue | What its evidence actually establishes | Consequence for PageGauge (our assessment) |
|---|---|---|
| [KIVI, ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html) | Asymmetric key/value quantization; quality evaluation includes generation and long-context tasks, with memory/throughput evaluation. It retains a full-precision residual region. | Implement a real KIVI comparison. Residual FP16 is not inherently disqualifying, but its budget must be disclosed. |
| [KVQuant, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf) | Per-channel/pre-RoPE keys, nonuniform quantization and outlier treatment; perplexity across model families, long-context evaluation, and kernel measurements. | Compare representation quality and effective bytes. Do not present its matrix-vector kernel gains as complete-serving gains. |
| [QuaRot, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf) | Rotation-based quantization of weights, activations and KV; perplexity/zero-shot quality. Performance evaluation uses a transformer block on RTX 3090. | Conditioning/weight folding needs prior-art attribution. Consumer-GPU evidence can be useful, but block timing is not full-model timing. |
| [QServe, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html) | W4A8KV4 algorithm/runtime co-design, quality tables, hardware analysis and throughput under memory budgets on A100/L40S. | Copy the bottleneck-to-design-to-system-evidence reasoning, not its headline speedup: it changes weights and activations as well as KV. |
| [AQUA-KV, ICML 2025](https://proceedings.mlr.press/v267/shutova25a.html) | Inter-layer prediction plus residual compression; broad perplexity/LongBench evaluation, calibration and component ablations. | A better representation needs cross-domain/model evidence and explicit calibration costs, not just better local cosine. |
| [CommVQ, ICML 2025](https://proceedings.mlr.press/v267/li25du.html) | Learned vector codes, RoPE-commutative codebooks, reordered attention computation; LongBench/InfiniteBench/GSM8K, domain-shift and optimized-versus-naive execution evaluation. | Closest conceptual warning: “use algebra to avoid reconstructing the cache” is not sufficiently specific novelty. Distinguish affine page metadata and demonstrate its incremental benefit. |
| [NSNQuant, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/3d8ee933c215fbb7b4d1948ff4906299-Paper-Conference.pdf) | Double normalization and low-bit vector quantization; multi-model PPL, long-context/generation tasks and custom kernels. Its quality comparison controls the full-precision residual policy. | Separate representation ablations with matched residual policy from native-system comparisons. Normalization alone is not a fresh contribution claim. |
| [Kitty, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/e4d8d1b5120be349d3fff8878650cf45-Abstract-Conference.html) | Mixed precision, sink/channel treatments, packing/runtime design; reasoning/code quality and memory-constrained throughput. Quality uses a simulation framework, separately from its inference engine. | Show precisely which results use simulated quantization versus the actual production kernel; study memory-enabled batching as well as fixed-batch latency. |
| [BitDecoding, HPCA 2026](https://2026.hpca-conf.org/details/hpca-2026-main-conference/92/BitDecoding-Unlocking-Tensor-Cores-for-Long-Context-LLMs-with-Low-Bit-KV-Cache) | Low-bit decode kernels designed around Tensor Cores, dequantization, reductions and scheduling; quality plus cross-GPU efficiency evaluation. | A priority native kernel baseline, particularly on A100 if its released implementation does not support SM120. |

Supporting author manuscripts: [KIVI](https://arxiv.org/html/2402.02750v2),
[KVQuant](https://arxiv.org/html/2401.18079v4),
[AQUA-KV](https://arxiv.org/html/2501.19392v2),
[CommVQ](https://arxiv.org/html/2506.18879v1),
[BitDecoding](https://arxiv.org/html/2503.18773v3).

Two particularly useful details:

- QServe's Section 4.3 reports that a naive KV4 path can be slower than KV8 on
  A100 but faster on L40S, then analyzes ALU/dequantization overhead. Thus our
  A100 difficulty is a plausible systems problem, not evidence that valid algebra
  must yield hardware-independent speed. [QServe paper](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf).
- CommVQ's value path already reassociates attention probabilities, codes and a
  codebook; its key path exploits RoPE commutativity. PageGauge must distinguish
  shared-center cancellation, page/head scalar correction and online exact-region
  merging from that construction. This is a novelty assessment, not a claim that
  the two methods implement identical mathematics. [CommVQ Sections 4–5](https://arxiv.org/html/2506.18879v1).

Published papers make different compromises. We should not infer that every
accepted method provides production serving, eliminates every FP16 residual,
beats every competitor, or enforces a universal minimum-cosine cutoff.

## 2. What PageGauge has actually completed

| Evidence | Status and scope | Still not established |
|---|---|---|
| Affine attention factorization | Exact identity in real arithmetic for the reconstructed mixed cache under shared-center/scalar-scale assumptions | Exact equality to the original unquantized model; a universally fastest implementation |
| Production correctness | Latest Step 01 passed seven synthetic snapshots; earlier reconstruction/split diagnostics also exist | An optimized matched fused-reconstruction speed control |
| RTX 5090 complete-decoder speed | Latest Step 02: 1.132281x versus FP16 FlashInfer, 95% bootstrap interval [1.131358, 1.133021], eight fresh processes | Workload/model generality; the interval is conditional on only two fixture clusters |
| Workshop A100 performance | Submitted configuration: approximately 1.1093x, with eight-process confirmation | A new A100 quality claim; its separate auxiliary HF cosine gate failed |
| Served KV storage | Original configuration: 7,288,520,704 versus 11,542,724,608 bytes, 36.856% reduction | Competitiveness against real 4-bit/2-bit storage; total device-memory reduction of the same percentage |
| Independent PG19 fidelity | Six B1 books, 9,216 scored logits; top-1 0.9982639; upper 95% PPL-ratio bound 1.0000263 in the submitted study | Multi-model/task quality; untouched evaluation for a newly tuned representation |
| Error attribution | Nine development captures; representation error dominates execution error in all 72 KV-head case averages | Every layer/query/domain behaves the same way |
| Folded V/O conditioning | One development seed: min-logit cosine 0.981797 to 0.986912; top-1 unchanged at 0.9996745; neutral latency +0.04647% | Statistically zero cost, improved task quality, a new FI-relative speed result, or final policy selection |
| Broad experimental framework | Protocol/planning infrastructure exists | Baseline adapters, multi-model workers and serving evaluation being finished |

Local evidence:

- `results/mlsys2027_rtx5090/01_correctness/20260908T035252Z_ae8e7138`
- `results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c/analysis.json`
- `results/mlsys2027_robustness_v1/20260908T052332Z_37fb839c/attribution.json`
- `results/mlsys2027_representation_v2/gpu_20260908T053519Z_cb1b89a1/analysis.json`
- `results/mlsys2027_representation_v2/single_seed_20260908T054708Z_398a7e36/analysis.json`
- Submitted historical claims: `C:/Users/noorg/Downloads/32_PageGauge_Factoring_Affine_.pdf`.

Do not pool the workshop configuration, later reproduction and opt-in conditioned
development result as if they were a single frozen experiment. We already have
PPL evidence; the gap is its breadth and applicability to the next method.

## 3. A practical issue to resolve before scaling conditioning

`scripts/page_gauge_value_conditioning.py` fits gains from request zero's initial
prefill, shares them across requests, and changes packed V/O weights in memory.
The most recent run spent about 1.054 seconds on fitting/conditioning and folding,
outside decode timing, added 64 KiB of gains, and recorded FP16 weight-roundtrip
changes. Zero added decode operations does not mean zero request-level cost.

Inference from this implementation: arbitrary continuous batching is not yet
supported by this construction. Refitting model-wide gains for a newly arriving
request would invalidate existing requests' cache coordinates unless state were
converted or separated. Test a model-wide gain fitted once on TRAIN calibration,
then fixed for all future requests, before building a serving story around it.
This fixed-gain alternative is proposed, not implemented by this document.

## 4. Ordered next experiments

Use the 5090 first. Each stage starts with a small implementation/feasibility
pilot, then seals its own configuration and evaluation manifest. Do not launch
the full model x corpus x method x shape Cartesian product.

### E0 — Fixed-gain transfer and true cost (next short development experiment)

Question: does the conditioning improvement transfer without per-batch weight
changes or substantial request setup cost?

- Mistral first; compare original PG, current request-zero conditioning and one
  fixed calibration-only gain. Fit only on declared TRAIN data, never later tokens.
- Use eight new development windows across general text and books. Include
  different first requests/batch orderings for the request-zero variant, holding
  the evaluated requests constant. This is selection data, not final TEST.
- Add FP16 attention with the same folded weights/cache coordinates to isolate
  floating-point reparameterization drift from quantization error.
- Measure token-weighted NLL/PPL ratio, top-1, KL and descriptive cosine, including
  per-window regressions. Do not choose from the best cosine alone.
- Measure setup, prefill/TTFT and complete decode separately. Retain counters and
  gain/cache bytes. One timing pair is a screening result, not equivalence proof.
- Decision: retain at most one candidate if quality transfers and complete cost
  is acceptable; otherwise keep unconditioned PG. Do not expand scale searches.

### E1 — Causal factorization control (highest-priority systems experiment)

Question: how much benefit is due to factoring affine metadata, rather than merely
compressing the cache or using a different implementation?

- Implement an optimized fused affine-reconstruction attention control. Match
  INT8 codes, centers, scales, exact regions, cache layout, workload and graph
  boundary. Reconstruct in registers; do not materialize FP16 history for this
  primary control. Permit fair resource-aware scheduling for both kernels.
- Explicit materialized reconstruction remains a diagnostic, not the strongest
  baseline. Confirm numerical equivalence to a common reference before timing.
- Cross factorization on/off with conditioning on/off in a 2x2 development
  ablation. Each timed contrast is separately paired; do not misuse the existing
  two-treatment ABBA reducer for a four-treatment schedule.
- First profile B4/C20480/D1536, then a short- and a longer-context cell that fit.
  Record latency, DRAM bytes, registers/spills, occupancy, launches and scalar
  work where profiler support permits. Profiling runs are not timing samples.
- Stop expanding if a fairly optimized control removes the gain. That result
  changes the claim and directs implementation work; it must not be hidden.

### E2 — Native low-bit competitors and the quality/memory/latency frontier

Question: when would someone choose PageGauge instead of existing KV compression?

- First establish FP16 FI, KIVI 4-bit/2-bit, and one recent execution baseline:
  BitDecoding, plus Kitty when its released engine/model support is validated.
- Include a quality-oriented representation comparison with KVQuant or NSNQuant.
  Review CommVQ analytically in detail; add its released checkpoint comparison
  if calibration/model compatibility permits. Not every paper needs a port.
- Keep weights/activations identical for KV-only contrasts. QServe is a separate
  full-system comparison because W4A8KV4 changes more than the KV cache.
- Report native defaults with actual metadata/residual bytes. Separately perform
  matched-residual representation ablations; label modifications to baselines.
- Show fixed-batch latency AND maximum feasible throughput under the same memory
  budget. Different batch sizes cannot establish a fixed-work speedup.
- Use quality-versus-bytes and quality-versus-latency plots. If PG is dominated
  at comparable quality, investigate one justified lower-bit branch or narrow
  the target regime. Do not promise a 1.2–1.5x win before measurement.
- Unsupported native SM120 kernels are compatibility gaps, not slow results.
  Run the relevant comparison on A100 rather than timing a Python fallback.

### E3 — Model, corpus and generated-task generalization

Question: does the frozen method preserve useful behavior beyond one fixture?

- Target three families: existing Mistral, Llama-3.1-8B, and a Qwen 7–8B model.
  Pin revisions and audit projection/head/RoPE/backend support first. These are
  proposed model additions, not currently verified drop-in implementations.
- Preserve base-versus-instruct identity across each comparison. Use appropriate
  instruction checkpoints for generated tasks; do not conflate them with base
  checkpoints used in existing PPL experiments.
- PPL: WikiText-2/C4 plus new PG19 books; compute aggregate token NLL before
  exponentiation. State whether evaluation uses production recurrent decoding
  or quantize/dequantize simulation. Do not mix those protocols in one result.
- Task suite: LongBench's eight-task subset for QA, summarization, few-shot and
  code; add controlled retrieval and GSM8K generation. Start with a declared
  development subset to validate the runner; use the frozen full suite for claims.
- Reserve an initial target of 24 disjoint new held-out PG19 books after an
  eight-window development pilot; determine statistical precision before final
  exposure. Previous six workshop books are now exposed and cannot be the sole
  untouched test for this tuned revision.
- Bootstrap paired documents/problems, not correlated token rows. Report each
  task and uncertainty, not only an average. Cosine stays descriptive; top-1 is
  supporting fidelity, not a substitute for PPL and generated-task quality.
- Any chosen non-inferiority margins must have a task/application rationale and
  be fixed before final TEST. No retrospective cutoff adjustment to obtain a pass.

### E4 — Shape coverage and GPU transfer

Question: where does the benefit hold, disappear or reverse?

- Mistral/5090 pilot: B in {1,4,8}, initial context in {8192,20480,32768}; use
  D1536 for recurrence-valid complete runs where memory permits. Run three
  representative cells first, then complete feasible coverage for the main
  comparisons. Verify model context semantics, not only allocated length.
- Preflight total memory, not KV alone. Log OOM/infeasible cells. Do not shard
  batches to make a nominal unsharded latency comparison fit.
- On A100, confirm a matched short/central/long workload set with the final
  quality protocol as well as performance. Record exact SKU, VRAM, SM count,
  software and device exclusivity; do not assume Colab A100 equals A100-80GB.
- Report each GPU separately. Profile at least a win and a weak/losing case to
  validate the proposed bandwidth-versus-instruction cost explanation.
- H100 is optional strengthening, not a prerequisite for 5090/A100 progress.
  L4/L40S is optional after verifying the actual device. T4 is not a priority.

### E5 — Residual-region ablation and realistic serving

Question: are the memory savings and speed useful once the entire request and
changing batch composition are included?

- On development data, remove prefix/static-suffix/recent-tail protection one at
  a time, with all other policy fixed. Report affected attention mass, quality,
  actual bytes and latency. This is a small causal ablation, not a new giant sweep.
- Integrate one real serving engine first. Use a fixed-gain policy if E0 supports
  it, or keep conditioning off; model-wide request-specific refolding is not a
  valid general continuous-batching implementation.
- Evaluate fixed concurrency and one mixed-length arrival workload at several
  offered loads, with identical prompts, stopping rules and memory budget.
- Report TTFT, time per output token, p50/p95 latency, output tokens/s, peak
  allocated/reserved memory, cache-finalization work and admission capacity.
  Include setup amortization and prefill, not just warmed graph replay.
- Final confirmation: freeze code, policies and inputs; use fresh-process paired
  blocks for selected headline contrasts, plus independent workload fixtures.
  More bootstrap resamples do not replace more independent fixtures.

## 5. Scope control, readiness and resource budget

The old `decoder_matrix_v1.json` requires H100 and missing controls and is still
non-runnable. Do not edit its frozen outputs or pretend its runners are complete.
Create a new accessible-GPU stage manifest when implementing E0/E1. Updated source
closure also requires a fresh correctness attestation before reusing the old
Step 01/02 execution pipeline.

Implementation order: E0 -> E1 -> E2 -> E3 -> E4 -> E5. E1 is the decisive causal
claim; E0 comes first only because it cheaply resolves the current development
branch. Download/pin models and baseline environments once. Record pilot GPU time
and use it to budget later stages; no unmeasured runtime estimate is a guarantee.
Use paid Colab only after local runners and source bundles are validated.

Minimum credible main-paper target (our judgment, not conference rules): causal
control; actual low-bit comparisons; multiple model families and generated-task
quality; feasible shape coverage on 5090/A100; complete cost accounting and one
serving experiment. More compelling work adds a validated predictive cost model
and a clear non-dominated operating region. A third expensive GPU or another
cosine threshold is not the main missing contribution.

## 6. Suggested paper flow and result artifacts

1. **Measured problem:** same-format fused reconstruction still spends material
   work on affine metadata. Motivate with E1, not an artificially slow baseline.
2. **Precise distinction from prior work:** scalar page/head affine structure,
   shared-center invariances and streaming mixed-cache execution. Discuss CommVQ,
   rotations and normalization explicitly.
3. **Mathematical contract:** exact identity relative to reconstructed cache;
   clearly separate quantization error, FP16 execution error and model drift.
4. **Representation and runtime:** explain only improvements supported by E0–E2,
   including conditioning setup and exact-region costs.
5. **Evaluation by questions:** does it preserve quality; does factorization cause
   speedup; when is it competitive; does it transfer; does serving benefit?
6. **Limitations and artifact:** losing regimes, untested models, residual policy,
   GPU compatibility, calibration/test boundaries and reproducible scripts.

Target artifacts: (a) prior-work/assumptions table, (b) causal control table and
profile breakdown, (c) per-model/task quality tables, (d) effective-bytes/latency
frontier, (e) shape/GPU plots, (f) serving latency-throughput curve, (g) residual and
conditioning ablations. Build each from machine-readable results, not hand-picked
figures. Keep the submitted workshop archive unchanged.
