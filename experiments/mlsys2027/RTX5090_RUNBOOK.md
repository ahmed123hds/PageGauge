# PageGauge MLSys 2027: RTX 5090 first

Run one experiment at a time. Keep the workshop results unchanged. A100 comes
after the 5090 configuration and experiment code are stable; no Colab upload is
needed for this step. The mathematics, quantizer, and production kernel are not
changed by this launcher.

## 0. Check the existing environment (safe while ICLR uses the GPU)

Run in **Windows PowerShell**:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/run_rtx5090.sh check
```

Expect `"passed": true`, `"cpu_only": true`, and `"gpu_queried": false`.
This checks source syntax, dependency metadata, the existing patched production
header, and the compiler path. It neither imports CUDA libraries nor queries or
uses the GPU. It does not install packages or download a model.

## 1. Production-attention correctness (run only after ICLR finishes)

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/run_rtx5090.sh 01
```

Default device: GPU index 0. To select another *idle RTX 5090*, append
`--gpu-index 1` (or its actual index). The launcher checks device identity, compute
processes, memory use, and utilization before importing GPU libraries. It refuses
busy or uncertain states and never stops an existing job. It is a start-time
check, not a scheduler reservation: do not start another GPU job during this run.
If WSL cannot report process/memory telemetry, keep the refusal and report the
error; there is no force-start flag.

What runs:

- Synthetic B=4, Hq=32, Hkv=8, d=128 tensors; 20,480 initial context tokens.
- The existing S4/A128/T768 production attention path at D=0,1,15,16,768,784,1536
  static cache snapshots, including a partial page and generated historical pages.
- Comparisons with FP16 explicitly reconstructed attention and an independent
  FP64 dense-attention reference; partition coverage and graph replay checks.
- Fixed tolerances: max absolute output error <=0.0002, minimum output cosine
  >=0.999999, centered log-sum-exp error <=0.0002, and bitwise graph/eager equality.

These are **kernel-correctness tolerances on synthetic attention outputs**, not
the workshop's 0.995 model-logit fidelity gate. No held-out corpus is used here.
First-use compilation may take several minutes. No model download is required.

Each invocation creates a new directory (never reuses an old passing result):

```text
results/mlsys2027_rtx5090/01_correctness/<UTC timestamp>_<run id>/
  manifest.json    frozen configuration, source hashes, environment, idle check
  worker.log       progress and any traceback
  result.json      seven snapshot results (or an error)
  completion.json worker exit status and artifact/source verification
```

Wait for `STEP 01 PASS`. Send `result.json` and `completion.json`; if the run fails,
send `worker.log` as well. A failure is diagnosed before proceeding; do not change
tolerances or repeatedly rerun in search of a pass.

## 2. Eight fresh-process FP16/PageGauge full-decoder performance blocks

Step 02 is implemented. Its local prerequisites can be checked without using
the GPU by replacing `02` below with `check02`. The pinned Mistral snapshot and
WikiText archive are already present on this machine; no download is needed.

Run in **Windows PowerShell**, with the RTX 5090 idle:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/run_rtx5090.sh 02
```

This is one experiment comprising eight sequential processes: **FI, PG, PG, FI;
PG, FI, FI, PG**, two fixtures, four adjacent pairs. Each process has B=4,
20,480 prompt tokens, 1,536 decode steps, one warmup and three measured sequences
per cache mode. PageGauge uses S4/A128/T768, split128; FP16 uses split256. Neither
kernel nor the quantizer is changed. The script freezes model/source/corpus
hashes before reading corpus text, and uses **WikiText TRAIN performance
windows**, not a new held-out test. Request windows use stride 23,600 and fixture
offsets 0 and 94,400. These windows are not new quality confirmation.

Each new invocation creates:

```text
results/mlsys2027_rtx5090/02_fp16_performance/<UTC timestamp>_<run id>/
  manifest.json
  block_<n>_<backend>.json       unchanged production worker's complete raw result
  block_<n>.log
  block_<n>_invocation.json
  block_<n>_telemetry.json       sampled compute-process checks every 10 seconds
  block_<n>_completion.json
  analysis.json                 written only after all eight valid blocks
  failure.json                  written instead if execution/identity checking fails
```

The runner checks matched input/model/ABI hashes, all 96 generated page closes,
48 generated historical INT8 pages for PG (zero INT8 pages is correct for FI),
eager/graph/cache equality, no eager fallback, and exact cache-memory formulas.
It checks idle state before each process and samples other compute processes
during execution. Sampling is not continuous proof of exclusive device use; do
not start other GPU work. There is no automatic retry, tuning, or skip of old
results. If interrupted or failed, preserve the directory and diagnose it first.

The primary endpoint is cache-neutral wall-time FI/PG speedup. Three repeats
are combined within each process; the four adjacent log-speedup pairs feed the
unchanged seed-fixture/adjacent-pair hierarchical bootstrap (50,000 draws).
Only two fixture clusters are available, so this is a narrow anchor result.
The strict speed gate is point estimate **and** lower 95% bound >1.10. All four
cache-mode/timing endpoints are reported, including slowdowns. Exit 3 means the
complete measured speed gate failed, not that the measurement is discarded.

The existing worker exits 2 if its HF quality diagnostic fails. This launcher
accepts that exit **only** for a complete schema-3 result passing independent
execution checks, and preserves/reports the HF failure separately. Neither a
passing performance gate nor a passing TRAIN diagnostic establishes held-out
quality. Thresholds are not weakened and failed values are not hidden.

The timed interval includes the full GPU-resident decoder loop and host planner,
but excludes loading, prefill, graph capture, restore/scrub and preconditioning.
It is not request-level online serving. KV accounting includes retained INT8
storage for exact regions, scales, centers and the expanded output center;
the following-page canary is excluded from served-cache bytes and reported
separately. Peak process memory is reported separately from KV-cache savings.

Wait for `STEP 02 COMPLETE`, then send `analysis.json`. On failure, send
`failure.json` and the last block log. This multi-process/model run is much
longer than Step 01; the command does not stop after the first backend.

## Subsequent 5090 order (still not runnable through this launcher)

1. Matched explicit and genuinely fused reconstruction controls with the same
   quantized cache. Step 02's FI comparison does not isolate reconstruction cost
   and must not be described as that ablation.
2. Compact batch/context scaling and latency decomposition; Step 02 already
   checks dynamic append/finalize recurrence and cache accounting at its anchor.
3. Additional model families and independent corpora, with frozen validation/test
   boundaries and task-level quality measurements.
4. Optimized external baseline and realistic serving: mixed request lengths,
   concurrency, throughput, and tail latency under stated latency targets.
5. Freeze 5090 code/results, then export the same supported experiments to A100
   and run independently there. Architecture-specific launch settings must be
   identified explicitly, not silently substituted.

The broad decoder matrix is a planning inventory, not a launch-all job. Its
general controlled reducer remains provisional until all controls and source
closures exist. The separate Step 02 runner integrates only the FP16 contrast
and reuses its bootstrap estimator; it does not enable the broad matrix or
claim the other controls are ready. Step 01 establishes neither runtime append
correctness, task quality, nor an end-to-end speedup.
