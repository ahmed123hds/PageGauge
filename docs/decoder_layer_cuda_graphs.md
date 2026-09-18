# Opt-in complete decoder-layer CUDA graphs

The production default remains one attention-only CUDA graph per decoder
layer. The `decoder_layer` scope is a diagnostic designed to test whether
WDDM host-launch starvation, rather than attention arithmetic, hides the
resident PageGauge advantage.

For an aligned 16-token decode page, each backend captures 16 graph banks.
Each bank contains 32 graphs, one for each Mistral decoder layer. A graph
contains input normalization, packed QKV projection, RoPE plus cache append
and possible page finalization, the selected attention implementation, output
projection and residual, post-attention normalization, packed gate/up
projection, SiLU times up, down projection and residual, and one final copy to
a persistent inter-layer output buffer. The copy is a real, symmetric graph
node and is included in latency.

Token embedding, final model normalization, LM head, GPU argmax feedback, the
current-position planner call, and wrapper `last_page_len` fills stay outside
the layer graphs but inside the timed generated-token block. There is no host
synchronization between tokens. The worker reports all of these operations.

Each offset owns an independent graph memory pool. Graphs inside an offset
bank are always replayed serially in layer order; all inter-layer values live
in explicitly allocated persistent buffers outside graph-private storage.
This avoids retaining graph-private outputs or depending on concurrent pool
lifetimes.

The replay guard binds graphs to both the exact captured logical page and the
invariant FlashInfer plan signature. This is necessary because RoPE views,
append positions, physical page destinations, and ring destinations are
captured arguments. A token outside the exact captured page falls back to the
eager layer path. No future-offset preplanning occurs in timed execution.

The backend-exclusive worker validates, for all 16 offsets including page
close:

- eager versus graph logits and generated token IDs are bitwise identical;
- every mutated cache tensor is bitwise identical, including PageGauge codes
  and scales produced by finalization;
- a second graph replay without cache restoration is bitwise identical;
- exactly 512 complete-layer graphs replay with 100% coverage;
- no nested attention graph is dispatched.

It also restores the mutated page after capture, records capture memory, and
temporarily disables complete-layer graphs for the event-instrumented
attention residency profile before restoring them for timing.

Example opt-in publication-pilot invocation:

```bash
python diagnostics/orchestrate_backend_exclusive.py \
  --profile pilot \
  --cuda-graph-scope decoder_layer \
  --output-dir results/backend_exclusive_decoder_layer_pilot
```

This is still fixed-batch, fixed-context, one-page generated-token serving.
Graph capture, prefill, cache restoration, cache scrubbing, and hot
preconditioning are excluded from timed latency and disclosed separately.
