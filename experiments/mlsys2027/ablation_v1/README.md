# Exact-region ablation

Development-only, fixed four-policy comparison, no optimization sweep:

| Policy | Prefix pages S | Static prefill-suffix pages A | Recent-tail tokens T |
|---|---:|---:|---:|
| Reference | 4 | 128 | 768 |
| Without prefix | 0 | 128 | 768 |
| Without static suffix | 4 | 0 | 768 |
| Minimum supported tail | 4 | 128 | 16 |

The implementation requires a nonempty exact page for unfinished-token append.
T16 is **not** a zero-residual ablation. No kernel/quantizer change is introduced
to manufacture T0 support. All page/head scalars, shared centers, INT8 codes and
split settings are unchanged; only selected exact-region policy changes.

First pilot: completed Mistral PG19 TRAIN book
`book_mistral_20260908T162400Z_d12033cd`. One model load, common coherent HF
prefill/full1536-label reference, then four independent cache initializations
and full1536-step PageGauge trajectories. Initial sampled KV must reproduce
the retained fixture. No per-arm quality-driven stopping or replacement.
An execution failure stops for diagnosis; poor predictive quality is reported.

Report model-token PPL ratio, top-1, KV bytes, new-history consumption and selected
attention-region mass. Selected probes cover layers0/15/31, steps0/768/1535,
KVheads0/7 (18 groups of four queries/policy), with a same-cache FP32 reference.
Existing execution tolerances remain .005 relativeL2/.02 absolute. These are
execution checks, not a retrospective predictive-quality cutoff. Region mass
uses the actual candidate's reconstructed keys, not original-HF attention;
overlap precedence is prefix, then static suffix, then remaining tail. It is
not a claim about all model attention. Pilot quality is one book, not a cohort.

No speed is inferred from the quality worker's instrumented per-token CPU copies.
After this pilot, finish the same policies on the remaining seven preselected
TRAIN books if execution is valid, regardless of whether ablations help/hurt.
Then use separate clean fresh-process timing before any quality/cost tradeoff
or production policy decision. Final TEST remains unopened.
