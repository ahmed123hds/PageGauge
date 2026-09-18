# Frozen layer-runtime shape extension

Keep the replicated S4/A128/T768 layer implementation unchanged. Evaluate
previously declared TRAIN shape-grid fixtures, not final TEST. The first
extension is B1/C8192/D1536, original grid cells15(FI) and16(PG), a configuration
where the eager reference had a negative point result. No selection by new
timing outcomes. Validate both backends fully before a timing pilot; retain
failures and all repeats. Do not alter the kernel, quantizer or policy to rescue
this shape within the extension.

Both workers must pass full recurrence, selected bitwise pre/post layer segment
checks, attention replay/same-cache oracle and initial cache restoration.
HF perplexity/top1 are descriptive development results. If execution passes,
use fresh FI then PG workers, full warmup and three repeats. This extension is
a point/range characterization, not a replicated confidence-bound claim.

After this cell, extend to B1/C30720, then B4/C8192; keep results separated by
shape. Larger-capacity graph-planning cases require explicit feasibility checks
and must not silently fall back to a different execution path.
