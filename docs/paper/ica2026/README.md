# CADENCE — IEEE ICA 2026 submission (Overleaf project)

Self-contained IEEEtran conference paper targeting **6 pages**.

## Files
- `cadence_ica2026.tex` — the paper (single file; bibliography is inline, so
  no `.bib` needed). References are in **ascending order of publication year**,
  and the Related Work (Sec. II) is written chronologically to match.
- `figures/` — the four evidence figures (architecture + H1 + Elec2 + H3), PDF.

## Compile on Overleaf
1. Overleaf → **New Project → Upload Project** → this zip.
2. **Compiler: pdfLaTeX**, main document `cadence_ica2026.tex`. Recompile.
   (No BibTeX step — the bibliography is a `thebibliography` block.)

## Locking it to exactly 6 pages
Built to land near 6, but page count can only be confirmed by compiling.
- **If it runs to 6.5–7 pages**, apply in this order until it fits:
  1. change the three result figures from `0.82\columnwidth` to `0.7\columnwidth`;
  2. delete Fig. 4 (H3) but keep Table III;
  3. delete Table I (comparison matrix) — its content is in the Related-Work prose;
  4. trim the "System analysis" subsection (Sec. V-F) to three sentences.
- **If it runs under 6 pages**, enlarge the result figures to `\columnwidth`
  and restore fuller wording in Sec. V-F.

## Honesty contract
Every quantitative value traces to the project's measured results ledger
(`docs/results.md`). Unrun experiments (full n=10 Stage 2, CARA/RCCDA baselines,
harder CL benchmarks) are stated as limitations, not results.
