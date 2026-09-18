# Generalization model audit (2026-09-08)

Evidence: `results/mlsys2027_generalization_v1/model_audit_20260908T130343Z_01063d11/analysis.json`.
Only small configs/API metadata were requested. No weights, corpus, or TEST
examples were downloaded; no GPU was used. Script:
`experiments/mlsys2027/audit_generalization_models.py`.

## Qwen3-8B

Official [config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json),
pinned revision `b968826d9c46dd6066d109eabc6255188de91218`.
Config accessible with existing environment; ungated.

- 36 decoder layers, hidden4096, intermediate12288, Hq32/Hkv8, head128.
- Context limit40960, RoPE theta1e6, no sliding window/scaling in this config.
- Checkpoint dtype BF16. Existing PageGauge path is FP16: any FP16 evaluation
  must convert both candidate and matched HF/competitor references identically,
  disclose that dtype choice, and not call it native-BF16 performance.
- Installed Transformers4.57.6 Qwen3Attention applies per-head `q_norm` and
  `k_norm` after projection and before RoPE. The current PageGauge
  `_execute_decoder_layer` omits these because its supported model families
  do not need them. Merely extending the model-type allowlist is incorrect.
- `benchmark_e2e_transformer.check_model` currently permits Mistral/Llama/
  Ministral3 only. It reads the actual layer count, rather than assuming32.
  Cache allocation/reduction must also use36; do not carry over Mistral bytes.
- Required implementation: faithful Q/K head normalization on both FI and PG
  paths; correct 36-layer packing/accounting; HF RoPE tables and tokenizer;
  same-weight FP16 numerical/control checks before quality/performance.
  Keep Qwen thinking/chat-template behavior fixed for generated tasks. Teacher-
  forced PPL does not by itself validate generated instruction performance.
- The current production graph wrappers hardcode Hq32/Hkv8. The opt-in Qwen
  adapter now enforces that actual limitation. Cached Qwen3-1.7B must not be
  routed through those wrappers without a separately validated head-shape
  generalization. It is not a substitute for the planned 8B experiment.

## Llama-3.1-8B

Official [model](https://huggingface.co/meta-llama/Llama-3.1-8B), API revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
Model metadata is public, but the config request using currently configured
access returned `GatedRepoError`, HTTP401. Config contents were not obtained;
do not assume this revision has been tested or downloaded. The user needs
approved HF access and a local login before this original target can run.
No agreement accepted, contact details sent, access request submitted, or
credentials displayed. Other experiment stages can continue meanwhile.

This audit is not proof of model compatibility, quality, or speed. It prevents
silently dropping architecture-specific operations or replacing a gated target
with a smaller model while claiming the originally planned coverage.
