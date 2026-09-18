# Fixed layer-runtime shape extension

All three predeclared development extensions completed full correctness
validation before timing. Same reference S4/A128/T768 policy, kernels and
layer-segment implementation; no mid-extension tuning. Each timing point uses
one fresh process per backend, full warmup and three repeats. These are not CIs.

| Batch/context | FI median ms | PG median ms | FI/PG | FI/PG served KV bytes |
|---|---:|---:|---:|---:|
| B1/8192 | 12.490728 | 12.692126 | 0.984132 | 1275068416 / 1016037376 |
| B1/30720 | 13.378546 | 12.784250 | 1.046487 | 4227858432 / 2493874176 |
| B4/8192 | 14.144919 | 13.880718 | 1.019034 | 5100273664 / 4064149504 |

The separate eight-process B4/20480 replication remains1.159561 with95% interval
[1.157992,1.161633]. Do not attach that interval to these other shapes. None of
the three extension points reaches1.1, and B1/8K remains slower. All results
exclude prefill/capture/restoration and include timed copies/runtime guards.
The data support a workload-dependent benefit, not universal acceleration.

## Evidence directories

Under `results/mlsys2027_baselines_v1/`:

- B1/8K: `layer_segment_timing_v2_flashinfer_fp16_20260910T004705Z_4d7f0428`
  and `layer_segment_timing_v2_page_gauge_20260910T004850Z_1c7f9257`.
- B1/30K: `layer_segment_timing_v2_flashinfer_fp16_20260910T021300Z_56e4ebc4`
  and `layer_segment_timing_v2_page_gauge_20260910T021511Z_2545406b`.
- B4/8K: `layer_segment_timing_v2_flashinfer_fp16_20260910T032735Z_dde1dec0`
  and `layer_segment_timing_v2_page_gauge_20260910T032945Z_5a472456`.

`reduce_layer_shape_timing.py` revalidated each pair's source/input metadata,
token match, execution counters, log/telemetry hashes, medians and KV accounting.
No TEST data or workshop files changed. Keep these negatives in subsequent
paper updates. Next prioritize generated-task and actual-serving integration;
any new short-context optimization must be a separately declared development
experiment and must not overwrite this completed extension.
