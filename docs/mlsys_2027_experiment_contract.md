# PageGauge MLSys 2027 experiment contract

The first implementation stage is intentionally CPU-only. It seals the desired
experiment surface before any new GPU worker is written or launched. The broad
matrix is a planning inventory, not an execution manifest. Each experiment
family receives its own smaller immutable manifest so incompatible statistical
units are never pooled.

## Files

- `docs/mlsys_2027_experiment_matrix.json`: model, GPU, method, workload, and
  shape axes.
- `diagnostics/mlsys_experiment_protocol.py`: strict validation, deterministic
  expansion, source closure, and tamper-evident manifest construction.
- `diagnostics/prepare_mlsys_experiment_matrix.py`: dry-run manifest CLI.
- `tests/test_mlsys_experiment_protocol.py`: CPU-only negative and integrity
  tests.
- `experiments/mlsys2027/decoder_matrix_v1.json`: the first executable design,
  containing nine explicit controlled-decoder cells.
- `diagnostics/mlsys_controlled_protocol.py`: exact cell identities and the
  eight-fresh-process ABBA/BAAB schedule.
- `diagnostics/reduce_mlsys_controlled.py`: raw-sample validation and the
  unchanged seed-fixture/adjacent-pair hierarchical bootstrap.
- `tests/test_mlsys_controlled_protocol.py`: synthetic schedule, reducer, and
  fail-closed tests.

## Safety boundary

The preparer cannot launch a worker. It does not import Torch, CUDA, FlashInfer,
SGLang, or vLLM. A manifest is `runnable=true` only after every model revision,
GPU access requirement, comparison implementation, and suite runner is marked
ready. The current manifest is expected to remain a design manifest.

The controlled manifest is also deliberately non-runnable until H100 access and
both reconstruction controls exist. Its 9 cells are three GPUs times three
separate pairwise contrasts: factorized PageGauge versus FP16 FlashInfer,
explicit affine reconstruction, and fused conventional reconstruction. Every
cell has two seed fixtures, ABBA then BAAB, four physical adjacent pairs, and
eight unique fresh processes. No four-treatment schedule is permitted.

## Current unresolved gates

- pin exact Llama and Qwen model revisions;
- secure H100-SXM access;
- implement explicit and fused reconstruction controls;
- validate/install BitDecoding, QServe, KIVI, and Kitty;
- implement controlled-factorization, serving, quality, and profiling runners.

## CPU-only command

```powershell
python diagnostics/prepare_mlsys_experiment_matrix.py `
  --spec docs/mlsys_2027_experiment_matrix.json `
  --output results/mlsys_2027_design/experiment_manifest.json
```

The command only validates and writes JSON. It performs no GPU discovery or
execution. Later execution orchestrators must consume the sealed run identities
rather than reconstructing a matrix from mutable CLI defaults.

The first controlled design is sealed separately with:

```powershell
python diagnostics/prepare_mlsys_controlled_matrix.py `
  --matrix experiments/mlsys2027/decoder_matrix_v1.json `
  --output results/mlsys_2027_design/controlled_decoder_manifest.json
```

Do not use `--require-runnable` until the reported unresolved requirements are
closed. The future orchestrator must acquire an exclusive idle device and exit
without launching or killing anything when another workload owns the GPU.
