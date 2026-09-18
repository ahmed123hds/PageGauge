# E2 native-baseline preparation — not results

Local source inspection and isolated build, 2026-09-08. No packages have been
installed into the PageGauge environment and no baseline GPU measurements exist yet.

| Baseline | Official source | Pinned revision | Local WSL source |
|---|---|---|---|
| BitDecoding | https://github.com/OpenBitSys/BitDecoding | ae0d83630d6292453355ced498db2ac87f56ec62 | /home/anonymous/pagegauge_baselines/BitDecoding |
| KIVI main | https://github.com/jy-yuan/KIVI | 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6 | /home/anonymous/pagegauge_baselines/KIVI |

KIVI also advertises an optimized `develop` branch; its remote revision was
8c3bdf1f83d5b548d1a6ad4fbef8f1bbc5686804 at inspection. Inspect/pin that branch
before choosing a latency baseline, rather than knowingly selecting a slower
implementation. Main is useful for the published model-quality implementation.

## Compatibility findings

- BitDecoding `setup.py` emits SM80/SM90 cubins, not SM120 or forward PTX.
  `csrc/bit_decode/decode_api.cpp:345` accepts only device major 8 or 9 in the
  examined decoding entry point. RTX 5090 therefore needs an explicit, disclosed
  portability patch and numerical validation before any timing; this is not a
  native out-of-box 5090 result. Its pinned CUTLASS submodule is
  3fe62887d8dd75700fdaf57f9c181878701b0802 (fetched and checked out).
- Do not run BitDecoding's `install.sh`: it contains broad cleanup operations.
  Use a separate environment and reviewed build commands instead.
- KIVI main pins torch 2.4.1 and transformers 4.43.1; current PageGauge uses
  torch 2.12.1+cu130 / transformers 4.57.6. Do not downgrade the shared runtime.
  First test an isolated environment using existing Blackwell-capable torch,
  declared transformers compatibility, and source-built `kivi_gemv` for SM120.
  Any API compatibility changes must be saved as a patch, not concealed.
- KIVI Mistral implementation exists, but exact compatibility with our pinned
  Mistral-7B-v0.3 checkpoint is not verified. BitDecoding release examples cover
  Llama and Qwen3; do not assume a drop-in Mistral wrapper.

## Execution order after E1

1. Isolated build/import and numerical smoke, with compiled-kernel identity.
2. Validate native quantization and actual cache allocation against declared
   INT2/INT4 groups/residual policy. No Python dequantization fallback timing.
3. Same frozen development token fixtures and weights for quality contrasts.
   Record actual corpus-label NLL, not just agreement with HF-generated tokens.
4. Compare full-model systems under the same execution contract where possible.
   If engine/graph implementations differ, label it as a native-system comparison
   and separately report attention-kernel contrasts; do not imply factorization
   alone explains that difference.
5. Include published native residual defaults and actual metadata/cache bytes;
   separately label any matched-residual modifications.
6. Run unsupported native cases on A100 once accessible; never convert an
   unsupported result into a speedup for PageGauge.

No automatic all-model install/download or large grid is authorized by this
file. Follow the staged project plan, the user's resource authority, and live
GPU ownership checks.

## Completed isolated build / next smoke

KIVI main's unmodified native extension compiled successfully with CUDA 13 and
`TORCH_CUDA_ARCH_LIST=12.0`. Environment:
`/home/anonymous/pagegauge_baselines/kivi_sm120_env`; it shares the existing torch
via a `.pth` appended search path, while keeping its own transformers 4.43.1 and
tokenizers 0.19.1. Original environment packages were not uninstalled or changed.
The first build attempt lacked inherited setuptools; adding that explicit shared
runtime path resolved it. Native extension SHA256:
`d155b98084a580bf535a72c5e1106e057ba878fd052f4073c6aeeff48f356c0f`.
CPU import of `MistralForCausalLM_KIVI` passed. This is not a model execution test.

The pinned develop branch is also checked out at
`/home/anonymous/pagegauge_baselines/KIVI_develop`. Its GEMV interface uses an
older MHA/MQA boolean, whereas main supports explicit `nh_kv` GQA. Develop pins
transformers 4.36.2 / torch 2.1.2. Do not assume it is uniformly faster or directly
compatible with our GQA model; evaluate compatibility explicitly.

`kivi_smoke.py` is implemented and Python-compiled, not yet GPU-run. It tests
native INT2/INT4 QK/PV GQA GEMV on B1/B4 against explicit reconstructed operands.
Run using the isolated KIVI interpreter **after** the active E0/E1 GPU jobs:

```
/home/anonymous/pagegauge_baselines/kivi_sm120_env/bin/python /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/experiments/mlsys2027/baselines_v1/kivi_smoke.py
```

The smoke is numerical only, with no timed fallback or end-to-end claim.
Native full-model prefill support still needs auditing (the released Mistral
path has eager or FlashAttention prefill, not our existing HF-SDPA wrapper).

09:29 UTC update: the native smoke above has now PASSED all eight cases.
Result: `results/mlsys2027_baselines_v1/kivi_smoke_20260908T092841Z_aafe2a1b/analysis.json`.
Runtime ~7.5 seconds including preflight and first-use Triton compilation.
Relative L2 against reconstructed operands 0.000204--0.000208. No full-model
or performance claim follows from this smoke.

## Mistral integration and shared-prefill adapter

The first actual tiny-model run under 4.43.1 failed at the obsolete RoPE
`seq_len` keyword. Failure retained in
`kivi_model_smoke_20260908T093302Z_2d355914/failure.json`.
Use `/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/bin/python` for
native Mistral: it has transformers 4.36.2/tokenizers 0.15.2, with shared torch
and the already built native kernel. Neither quantization nor KIVI source changed.
INT2/INT4 C128/D64 random tiny-model integration then passed in
`kivi_model_smoke_20260908T093342Z_45de6353` (prefill logits bitwise equal to HF).

`kivi_adapter.py` supports shared FP16 prefill followed by upstream KIVI decode
without duplicating or transforming weights. Its cache packing and 33 recurrent
steps were bitwise equal to native KIVI for both widths in
`kivi_adapter_smoke_20260908T093737Z_0f76b131`. This is a tested small-model adapter,
not yet pretrained-model quality, native-prefill timing, or a serving benchmark.
