# PageGauge AXIOM 2026 submission package

This directory is the final, experiment-frozen paper package. No number in the
paper was produced by the document build.

Files:

- `claim_evidence_ledger.md`: one-page internal claim/evidence ledger.
- `deliverables.md`: title candidates, abstract, four-page outline, figure
  specification, and populated main table.
- `main.tex`: double-blind four-page main paper in the official NeurIPS 2026
  workshop template, followed by references and appendix.
- `neurips_2026.sty`: unmodified official NeurIPS 2026 style file.
- `checklist.tex`: completed official NeurIPS 2026 paper checklist.
- `appendix.tex`: extended derivation, protocol, adaptation disclosure,
  statistical details, artifact hashes, and negative evidence.
- `references.bib`: bibliography.
- `hostile_reviewer_audit.md`: final claim-scope audit.
The publication PDF is compiled directly from LaTeX and written to
`output/pdf/pagegauge_axiom2026_submission.pdf`. From this repository root,
the equivalent Tectonic build is:

```text
cd paper/axiom2026
../../tmp/tex/tectonic-0.17.0/tectonic.exe main.tex --outdir ../../output/pdf
```

## Frozen headline artifacts

- Predictive quality: `results/final_fixed_suffix_s4_a128_t768/quality/aggregate.json`
  (`7adbbdc36f14c985004fe22349087210427f3831ee52d92e5a537667c81a64f9`).
- External-corpus predictive quality:
  `results/pg19_external_s4_a128_t768/aggregate.json`
  (`35a10df9ccc70deb8efd01fb249d820040c100928bc706188bc6fbef3d8518fb`),
  using six PG-19 test books frozen before content access.
- Fresh-process performance:
  `results/final_fixed_suffix_s4_a128_t768/performance/fresh_process_williams_split128_v2/final_performance_analysis.json`
  (`42c3553e3c652601c0fabacb6e59f430fb5589ff5f927c440592d72e584a19fc`).
- Explicit-reconstruction diagnostic:
  `results/page_gauge_int8/heterogeneous_fa2_correctness.json`
  (`20c64fc5b9413a6888287a0c8e6ae29fa2fafbd1c307d6e2d5ef51ea49db367d`).

## Scope

The final policy is S4/A128/T768: four exact 16-token prefix pages, 128 exact
static prefill-suffix pages, and a 768-token exact recent tail. The remaining
history uses pagewise affine INT8 KV codes. The predictive-quality cohort is a
post-failure B1 fixed-policy confirmation, not untouched held-out selection.
The B4 performance experiment uses eight fresh Python/CUDA processes in
ABBA/BAAB order and is independently eligible under same-backend execution
gates.
