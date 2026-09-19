# ICLR submission audit

This is an evidence record for submission readiness; it is not an acceptance
claim.

## Current branch artifact

The `codex/iclr` branch contains the official ICLR 2027 template obtained from
`ICLR/Master-Template` (upstream commit `46ed6f4`, 2026-09-01). The submission
entry point is `tex/iclr2027_submission.tex`; the exact style and bibliography
files are under `tex/iclr2027/`. The author block is anonymous and
`\\iclrfinalcopy` remains commented, as required for initial review.

The compiled artifact `tex/iclr2027_submission.pdf` has 43 pages. The main
text now ends with the complete Conclusion at the bottom of page 9. Statements
and references start on page 10; the proof appendix starts on page 11. The
end-of-main-text label is placed after a paragraph break and before the page
break, excluding the AI use, reproducibility, and ethics statements.

PDF text extraction and visual inspection verify the final main-text line on
page 9 at y = 732.27 pt (coordinates from the top of the US Letter page),
matching the bottom text line on other full pages. No conclusion text spills
onto page 10. The official template's font sizes and page geometry are unchanged.
The final build has no overfull boxes. Existing underfull-box and unresolved
cross-reference warnings remain; this pagination check is not a claim that
all submission issues are resolved.

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
- The paper is intentionally theory-focused. Deterministic structural checks are
  included only as consistency checks for the implementation and formal claims;
  no benchmark performance claim is made.

## Submission limitations

The ICLR-format and content structure are prepared for a theory submission; the main text is exactly 9 pages.
The central evidence is the theorem/proof development, the implementation
correspondence, and the explicit assumptions and counterexamples. The Lean
formalization uses exact finite models and does not certify floating-point
kernels, Ray/vLLM scheduling, optimizer dynamics, or general neural-policy
assumptions. No empirical performance claim or acceptance claim is made.
