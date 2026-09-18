# Exact-prefix S3 Williams protocol

The `exact_prefix_s3_d1024_t768` preset is fail-closed. It freezes the matched
Mistral B4/C20480/D1024 policy, PageGauge exact prefix S3, exact tail T768,
segmented `flashinfer_merge`, probability-side old-value scaling, strict
0.995 logit-cosine and 0.99 top-1 gates, and frozen-HF WikiText-2 TRAIN input
trajectories. The FP16 FlashInfer worker honestly records S0; the PageGauge
worker records S3 in its own hashed pairing configuration. The analyzer
normalizes only these two backend-physical aliases and requires every shared
token, corpus, HF-reference, model, source, ABI, and environment identity to
match.

Run the two-cohort publication pilot from the repository root:

```bash
python diagnostics/orchestrate_sustained_dynamic_graphs.py \
  --policy-preset exact_prefix_s3_d1024_t768 \
  --profile pilot \
  --output-dir results/exact_prefix_s3_v1/train_williams_d1024_t768_pilot \
  --model mistralai/Mistral-7B-v0.3 \
  --batch-size 4 \
  --context 20480 \
  --decode-steps 1024 \
  --exact-tail 768 \
  --exact-sink-pages 3 \
  --baseline-split-pages 256 \
  --candidate-split-pages 256 \
  --tail-attention flashinfer_merge \
  --old-value-scale-placement probability \
  --trajectory-mode frozen_hf_teacher_forced \
  --token-source wikitext2 \
  --wikitext-member wikitext-2-raw/wiki.train.raw \
  --token-offset 100000 \
  --token-stride 21984 \
  --seed-token-offset-stride 87456 \
  --seeds 20260861 20260862 \
  --pairs-per-seed 2 \
  --warmups 1 \
  --repeats 2 \
  --bootstrap-samples 2000 \
  --min-logits-cosine 0.995 \
  --min-top1-agreement 0.99 \
  --quality-diagnostics-top-k 0
```

This schedule produces one atomic ABBA quartet for the first seed/offset
cohort and one atomic BAAB quartet for the second. The half-open corpus spans
are disjoint. Every block is a fresh subprocess bracketed by GPU-idle checks
and sampled by `nvidia-smi` telemetry. D1024 closes 64 generated pages and
T768 retains 48, so the final attention layout must consume exactly 16
runtime-generated pages through INT8 recurrence.

The exact full publication command uses the preset's frozen four-seed,
four-pair-per-seed, three-warmup, ten-repeat defaults:

```bash
python diagnostics/orchestrate_sustained_dynamic_graphs.py \
  --policy-preset exact_prefix_s3_d1024_t768 \
  --profile publication \
  --output-dir results/exact_prefix_s3_v1/train_williams_d1024_t768_publication
```

A resume must use the identical command and add `--resume`; changed
config/source hashes, escaped artifact paths, changed worker-result hashes,
or a partial quartet fail closed. Use `--rerun-failed` only with `--resume`,
which reruns the entire affected Williams quartet.
