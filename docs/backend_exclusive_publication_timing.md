# Backend-exclusive publication timing

The full-model performance comparison must not allocate FlashInfer and
PageGauge caches in the same CUDA process. The publication driver launches a
fresh process for every backend block and waits for system GPU memory and
utilization to return to the recorded idle envelope before continuing.

The diagnostic pilot is deliberately small: one corpus seed, two adjacent
pairs, Williams order `ABBA`, three warmups, and four raw repetitions per cache
mode. It uses the validated `flashinfer_merge` exact-tail path; `fused_kernel`
is retained only as an explicitly selected diagnostic alternative.

```bash
python diagnostics/orchestrate_backend_exclusive.py \
  --profile pilot \
  --output-dir results/backend_exclusive_pilot
```

This expands to four fresh workers: FlashInfer, PageGauge, PageGauge,
FlashInfer. Each worker measures both cache-neutral and cache-hot modes and
records raw wall-clock and CUDA-event samples. The orchestrator also records
its command, source hashes, idle-gate snapshots, stdout/stderr, and periodic
`nvidia-smi` telemetry.

Inspect the pilot's per-layer residency gate before scaling. PageGauge's old
cache scan must remain in the resident-latency regime through the final layer;
a layer cliff invalidates the performance block even if correctness passes.

The publication preset uses four disjoint corpus-window seeds, four pairs per
seed, alternating `ABBA` and `BAAB` Williams quartets, ten warmups, thirty raw
repetitions, and 20,000 bootstrap draws.

```bash
python diagnostics/orchestrate_backend_exclusive.py \
  --profile publication \
  --output-dir results/backend_exclusive_publication
```

Explicit `--seeds`, `--pairs-per-seed`, `--warmups`, `--repeats`, and
`--seed-token-offset-stride` values override a preset. Pairs per seed and raw
repetitions must be positive and even. Use a new output directory for a changed
configuration; `--resume` accepts only the exact recorded configuration and
source hashes.

The primary estimate is the exponentiated mean adjacent-pair log latency ratio,
`exp(mean(log(FI latency) - log(PageGauge latency)))`, for cache-neutral wall
time. CUDA-event and cache-hot estimates are secondary. Results include:

- `orchestration_manifest.json`: immutable schedule, commands, hashes, idle
  gates, and block status;
- `blocks/*/attempt_*/worker_result.json`: backend-owned raw measurements and
  correctness/memory evidence;
- `blocks/*/attempt_*/nvidia_smi_telemetry.jsonl`: chronological system GPU
  telemetry;
- `raw_latency_samples.csv`: every wall/CUDA sample;
- `paired_log_speedups.csv`: adjacent-pair experimental units;
- `publication_analysis.json`: geometric speedups, order strata, paired-block
  bootstrap intervals, and hierarchical seed-then-pair bootstrap intervals.

The analysis fails closed on missing/truncated raw samples, sample counts that
do not equal the declared repetitions, failed worker gates, result hash
mismatches, non-adjacent pairs, mismatched model/token/KV provenance, or
different HF/backend generated-token trajectory hashes across a pair. A
one-seed pilot cannot estimate between-seed variation and is not a publication
confidence interval.
