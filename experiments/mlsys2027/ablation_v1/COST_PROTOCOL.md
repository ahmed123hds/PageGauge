# Clean regional-policy cost pilots

Run `regional_cost.py` only after the quality cohort releases the GPU. Four
fresh processes in the fixed order reference, no-prefix, no-static-suffix,
minimum-page-tail. Same first retained WikiText TRAIN B4 fixture, C20480,
D1536, one warmup and three repeats per cache-neutral/hot mode. All four run
regardless of predictive quality or timing direction, unless execution fails.

Use the existing audited sustained decoder worker and exact32 adapter.
INT8 history split128, no conditioning, unchanged production sources and
quantizer. Each policy's expected FP16 storage, new-history consumption and
wrapper rebuild counts are calculated explicitly. Eager/graph recurrence,
every page-close digest, final metadata, scheduler capacity and sampled GPU
exclusivity remain required. HF cosine is descriptive; preserve top1 failures.

Timed work includes the entire resident decoder, planner, append/finalize,
embedding, all layers, final normalization, LM head and argmax. It excludes
load, coherent HF prefill, graph capture, cache restore and scrub. No regional
attention oracle or per-token CPU logit-copy instrumentation is installed.

Report per-arm medians and repeat ranges, not a confidence interval or a
general speed claim. One process per arm and fixed ordering leave temporal
confounding. Any headline policy contrast needs balanced fresh-process
replication on multiple fixtures after development selection. This does not
modify defaults or historical workshop gates, and does not inspect final TEST.
