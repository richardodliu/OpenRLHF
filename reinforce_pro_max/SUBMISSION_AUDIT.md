# ICLR submission audit

This is an evidence record for submission readiness; it is not an acceptance
claim.

## Current branch artifact

The `codex/iclr` branch contains the official ICLR 2027 template obtained from
`ICLR/Master-Template` (upstream commit `46ed6f4`, 2026-09-01). The submission
entry point is `tex/iclr2027_submission.tex`; the exact style and bibliography
files are under `tex/iclr2027/`. The author block is anonymous and
`\\iclrfinalcopy` remains commented, as required for initial review.

The compiled artifact `tex/iclr2027_submission.pdf` has 4 pages: 3 pages of
main text and 1 page of appendix. The main text is below the ICLR 2027 initial
submission limit of 9 pages. Citations resolve with the official ICLR 2027
bibliography style. The build has no overfull boxes; two underfull boxes occur
in an appendix paragraph and do not change the page boundary.

The submission includes the required AI use statement and the recommended
reproducibility and ethics statements. The appendix records the exact theorem
scope, formalization boundary, and source locations.

## Evidence already checked

- `LEAN_JOBS=3 bash formal/leanw build` succeeds; the AxiomAudit passes for 911
  project theorems and allows only `propext`, `Classical.choice`, and
  `Quot.sound`.
- The CPU structural suite passes: 51 tests passed in 5.93 seconds, with one
  unrelated deepspeed deprecation warning.
- The complete proof manuscript remains available as `tex/main.tex` and
  `tex/main.pdf`; it is intentionally separate from the page-limited ICLR
  entry point.
- The current experiments are deterministic synthetic structural checks only.
  There is no matched LLM training, multi-seed comparison, independent
  AIME/MATH-style evaluation, or benchmark performance claim.

## Submission limitations

The ICLR-format and content structure are now prepared, but empirical
readiness is incomplete. GPU training, matched baselines, ablations, and
independent evaluation are required before making empirical claims. The Lean
formalization uses exact finite models and does not certify floating-point
kernels, Ray/vLLM scheduling, optimizer dynamics, or general neural-policy
assumptions. No acceptance claim is made.
