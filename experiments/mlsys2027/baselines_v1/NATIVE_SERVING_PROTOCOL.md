# Native request-cost pilot (development)

Frozen pilot order: NSN own-stack HF, native NSN INT2, Kitty own-stack HF,
corrected-native Kitty-Pro. One fresh process per arm; B1/C20480/D1536 on the
first already-completed quality fixture for that family. These are timing
development data, not independent final quality data or a PageGauge comparison.

Use `nsn_quality_20260908T133908Z_a67611e0` and
`kitty_quality_20260908T150715Z_8cf039f7`, preserving exact tokens and native
source hashes. No representation/default change, parameter search, or timing
selection. Each arm uses the declared expandable allocator and 28 GiB PyTorch
budget. Failures remain failures; no silent fallback or selective repeats.

One full-request warmup, then three full-request repeats. Rebuild native cache
via native full-prefix prefill each time; no CPU snapshot or retained reference
KV. Prefix timing includes native cache allocation/packing and the last prompt
LM head. Decode timing includes all model layers, every native append/packing
transition, and LM head. It does not include sampling, networking, admission,
or scheduler overhead. Model load/static rotations are separate one-time costs.
Report prefill and decode peaks separately, cache storage separately from both,
and the sum of timed segments, not a falsely labelled outer request latency.

Link earlier full recurrent native quality evidence; validate final lengths,
finite output, source/token stability and sampled GPU exclusivity. Numerical
quality comparisons never run inside the timing interval. Three repeat ranges
are not confidence intervals. Use measured pilot costs to size later balanced
fresh-process replication. Different Transformers stacks mean the two families'
HF ratios and previous PageGauge timings cannot be pooled into one speed claim.
