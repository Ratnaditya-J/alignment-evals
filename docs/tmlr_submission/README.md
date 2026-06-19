# TMLR submission build — "Probing Is Not Enough"

TMLR-formatted, anonymized build of `docs/probing-is-not-enough.md`. The content is the
verbatim draft (pandoc-converted); only the format wrapper and the double-blind
anonymization differ.

## Submission status

**Submitted to TMLR (OpenReview) on 2026-06-18 — under double-blind review.**

- Single author: Ratnaditya Jonnalagadda. License CC BY 4.0. Standard PDF submission (no Beyond-PDF, no supplementary material).
- Built on **controlled holdout v2**; the optional `controlled_holdout_v3` surface-robustness run was deliberately deferred — claims stand on v2 (§3.3/§5, §7.2(A)), and the frozen probes + activations are not retained locally, so v3 would require a full GPU reproduction (retrain probe + reproduce-v2 gate + extract + score), not warranted for submission.
- Before submission, Related Work + intro + abstract were updated to cite and distinguish the three nearest-neighbour papers: **Nguyen et al. (arXiv:2507.01786)** — closest prior result, *corroborated* (their probe-direction steering null on Llama-3.3-70B), strengthened here with a matched random/orthogonal/wrong-layer control suite, three behaviours, and a second architecture; **The Elicitation Game (arXiv:2502.02180)**; **Boxo et al. (arXiv:2509.21344)**.
- Extra camera-ready TODO (beyond the section below): add formal BibTeX entries for those three (currently cited inline by arXiv ID, consistent with the other 2026 cites).

## To compile

1. Put the official **`tmlr.sty`** in this directory (from the TMLR author guide /
   jmlr.org/tmlr, or the OpenReview submission page; it usually ships with a small `.bst`
   too if you later switch to BibTeX).
2. `pdflatex main` then `pdflatex main` again (second pass resolves the table of contents /
   refs). Validated to compile cleanly with `tmlr` stubbed out; only the official style
   file is needed to produce the final TMLR look.

`main.tex` is self-contained (figures in `figures/`). No `\input` files, no BibTeX
required — the draft's References are a manual section.

## Anonymization (double-blind review)

Done in this build:
- Author name and email removed from the title block (no `\author`).
- The code-repository link replaced with "an anonymized repository (link provided in the
  camera-ready version)" in the Contributions and Appendix C.

## For the camera-ready (after acceptance)

- Switch `\usepackage{tmlr}` to `\usepackage[accepted]{tmlr}` near the top of `main.tex`.
- Restore the author block and the real repository link.

## Notes

- Compiles under **pdflatex** (TMLR's default); a Unicode-glyph mapping block in the
  preamble handles the math symbols in the text so xelatex is not required.
- TMLR has no hard page limit, but values concision.
