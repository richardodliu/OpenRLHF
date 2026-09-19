# ICLR submission audit

This is an evidence record for submission readiness; it is not an acceptance
claim.

## Current artifact

The current LaTeX build uses `article`, includes the complete appendix in the
same PDF, and produces 77 pages. The build log shows the main text ending at
page 37; `main/appendix.tex` starts on page 38 and runs through page 77.
Thus the current artifact has approximately 37 pages of main text before the
appendix, far beyond a normal ICLR main-paper limit. The formal proofs being
present in the appendix do not make the main text page-limit compliant.
No ICLR style file or submission skeleton is currently present in this
repository or in the checked local project tree, so a compliant submission
entry point has not been fabricated from an unverified template.

## Evidence already checked

- `python reinforce_pro_max/check_submission_readiness.py` reproduces the
  current counts: 77 PDF pages, appendix first page 38, 37 main-text pages,
  and 31 labeled mathematical environments.
- `PUSH=0 COMMIT_PDF=0 ./compile.sh` succeeds.
- The PDF build has no LaTeX, BibTeX, overfull-box, or underfull-box warnings.
- The current Lean audit passes for 911 project theorems and allows only
  `propext`, `Classical.choice`, and `Quot.sound`.
- The experiments section explicitly reports synthetic structural checks only;
  it does not report an LLM benchmark.

## Remaining submission work

`tex/iclr_main_draft.tex` is now a separate three-page content skeleton. It
compiles without warnings and includes citations for RLOO, GRPO, PPO, and the
trust-region motivation. It keeps the analyzed configuration and theorem
boundaries explicit, but it is not yet an ICLR-template submission: it omits
figures and experimental results.

An ICLR submission artifact still needs a real ICLR template and a separately
reviewed main paper. The main text must move detailed proofs and secondary
variants to the appendix while retaining the definitions, assumptions, key
theorem statements, and enough method detail to make the contribution
reviewable. It also needs matched LLM experiments, independent evaluation,
multiple training seeds, and ablations. No acceptance claim is made until
those artifacts and results exist.
