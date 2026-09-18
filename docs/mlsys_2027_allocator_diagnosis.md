# Allocator diagnosis after the fixed capacity grid

Do not change the running original grid. The recorded PG B8 failure occurs at
the exact attention wrapper's 128 MiB workspace allocation before decode.
KIVI INT4 B8 fails at a 320 MiB contiguous temporary in the native value matmul.
Both report reserved-but-unused memory, but this alone does not prove that an
alternative allocator will succeed. Keep both original failures.

PyTorch 2.12 documents expandable segments as an experimental allocator option
intended to reduce unusable allocation slices as sizes change:
https://docs.pytorch.org/docs/2.12/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf
The same documentation distinguishes allocated tensor storage from allocator
reserve and provides memory_stats/memory_snapshot for diagnosis. Freeing cached
memory cannot release live tensor storage. Do not claim the entire OOM is
fragmentation from the error message alone.

After the original grid completes, use `frontier_allocator_probe.py` on the
retained PG B8 fixture: first explicit default (expandable_segments:False),
then expandable_segments:True. This is one binary systems diagnosis, not an
allocator-parameter search. Keep the 28 GiB budget, all token IDs, representation,
native kernels and full recurrence unchanged. Capture allocator state/snapshot
even if it fails. Successful runs require bitwise restored-state checks, full
D1536 warmup, and one D1536 timed recurrence; timings remain pilot-only.

If the policy helps PG, apply the same diagnostic to failed native KIVI cells
before claiming a comparative capacity advantage. Any later deployment frontier
must give all backends the same declared allocator policy and remeasure timing.
No silent budget increase, replacement of original outcomes, math change or
default promotion is authorized by this diagnosis itself.

## Explicit-default replay, 2026-09-08 17:32 UTC

`allocator_default_page_gauge_20260908T173051Z_8fd203dd` reproduces the exact
PG B8 workspace OOM under explicit expandable_segments:False. Sampled GPU
exclusivity passes; there are no completed restored-state checks or decode
trajectory. Snapshot SHA256:
`1a3ce086273096cba846d458dee97eb9b20c5c71faed7b91bdb9ddbf0f7f32bb`.

- Allocated/active: 29,788,426,752 bytes.
- Reserved: 30,054,285,312 bytes, below the 30,064,771,072-byte budget.
- Inactive split storage: 265,858,560 bytes (253.54 MiB).
- Snapshot: 67 inactive blocks; largest block only 4,194,304 bytes (4 MiB).
- Failing workspace request: 134,217,728 bytes (128 MiB).
- Allocator retries: 25; OOM count: 1; expandable segment count: 0.

Thus this particular failed allocation has direct fragmentation evidence:
allocated storage plus the requested workspace fits under the budget, but no
inactive block is large enough and reserved memory leaves insufficient space
for a new allocation. This does not prove that all later operations fit, that
every other backend's OOM has the same cause, or that changing the allocator
has zero runtime cost. The expandable-policy run remains a required check.
