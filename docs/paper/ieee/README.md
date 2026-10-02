# CADENCE — IEEE conference paper (Overleaf-ready)

This folder is a self-contained IEEE (`IEEEtran`) paper project.

## Files
- `cadence.tex` — the paper (IEEEtran, `conference` mode).
- `references.bib` — 56 real citations (all cited in-text).
- `figures/` — all figures the paper needs (PDF where available, else PNG).
- `evidence_audit.md` — honest map of every reviewer priority to what we
  have (with proof) vs what we don't. Read this before editing claims.

## Build on Overleaf (recommended)
1. Zip this folder (or upload it) — there is a prebuilt `cadence_ieee.zip`
   one level up in `docs/paper/`.
2. Overleaf → **New Project → Upload Project** → pick the zip.
3. Menu → **Compiler: pdfLaTeX**, **Main document: `cadence.tex`**.
4. Recompile. Overleaf runs BibTeX automatically; if references show as
   `[?]`, hit Recompile once more (pdfLaTeX → BibTeX → pdfLaTeX ×2).

## Build locally (if you install TeX Live / MiKTeX)
```bash
pdflatex cadence
bibtex   cadence
pdflatex cadence
pdflatex cadence
```

## Before camera-ready
- **Verify bibliographic details** in `references.bib` against the original
  sources (author/venue/year are accurate; exact page numbers/months should
  be confirmed — standard practice).
- Update the author block / affiliations as needed.
- Confirm the target venue's exact template (this uses the standard IEEE
  `IEEEtran` conference class; some IEEE venues ship a customized `.cls` —
  if so, drop `cadence.tex`'s body into their template).
- The claims are scoped to current evidence. If the Stage-2 Colab run lands
  STRONG, upgrade the H2 wording in §6.3 and the abstract accordingly (and
  only then).

## Honesty contract
Every quantitative number in the paper traces to a measured entry in
`../../results.md`. Do not add a number to the paper that is not in the
ledger. Unrun experiments belong in §8 (Limitations), not in §6 (Results).
