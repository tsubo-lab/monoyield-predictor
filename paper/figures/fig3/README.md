# Fig. 3 family (Fig3 + ExtFig6-9): scripts that read only `paper/tables`

Run any script with `python paper/figures/fig3/<script>.py`; PDFs go to `paper/outputs/fig3/`. Each script prints the key statistics (n, r, slope).
Shared code: `dms_data.py` (data preparation from S04 / S23a / S23b) and `dms_plots.py` (per-protein scatter and substitution heatmaps).

| Panel | Script | Tables | Output | Status |
|---|---|---|---|---|
| Fig3a Full DMS vs 50 mutations | `fig3a_dms-count.py` | S04 | `Fig3a.pdf` | done (counts 9 / 56 derived from the WT entries: dms-1..5 vs denovo-8 + muts-1) |
| Fig3b DMS yield vs dG (hexbin) | `fig3b_yield-vs-dG.py` | S04 | `Fig3b.pdf` | done (deltaG_95CI <= 0.5: n = 14,551, r = 0.555) |
| Fig3c aggregated AA effects (target / source) | `fig3c_aa-effects.py` | S04 | `Fig3c-target.pdf`, `Fig3c-source.pdf` | done |
| Fig3e 1PGA, Fig3f 1AOY: scatter + dG / yield heatmaps | `fig3ef_1pga-1aoy.py` | S04 | `Fig3e-1PGA-{scatter,dG-heatmap,yield-heatmap}.pdf`, `Fig3f-1AOY-...` | done (red boxes of Fig3e were drawn by hand; structure renders out of scope) |
| Fig3g Pearson R with slope (buried_np_AFILMVWY dropped, 2026-11-02) | `fig3gh_extfig9_slope-features.py` | S04, S23a, S23b | `Fig3g.pdf` | done |
| Fig3h slope vs hydrophobicity | `fig3gh_extfig9_slope-features.py` | S04, S23a, S23b | `Fig3h.pdf` | done |
| ExtFig6a delta yield / ddG boxes (target, source) | `extfig6a_aa-boxes.py` | S04 | `ExtFig6a-{target,source}-{yield,ddG}.pdf` | done |
| ExtFig7 A-E (9 proteins) | `extfig7_dms-summaries.py` | S04 | `ExtFig7{A..E}-<PDB>.pdf` | partly: scatter and both heatmaps done; the structure image (top-left) is a rendered PDB and is left blank |
| ExtFig8 per-protein dG vs yield grid (56 proteins; 2026-11-02 layout) | `extfig8_dg-vs-yield-grid.py` | S04 | `ExtFig8.pdf` | done |
| ExtFig9 slope vs hydrophobicity metrics | `fig3gh_extfig9_slope-features.py` | S04, S23a, S23b | `ExtFig9.pdf` | done |
| Fig3d, structures of Fig3e/f and ExtFig7, Fig3i | - | - | - | not in scope (renders / schematic) |

## Agreement with the reference PDFs (`reproduced_figures/Fig3`)
Text tokens are identical for all 27 PDFs, and the rendered pages are pixel-identical (fraction of differing pixels 0.00000 at 100 dpi) for every panel except ExtFig7 (0.4-0.7 %, exactly the blank structure image).
Key numbers: Fig3b n = 14,551, r = 0.555; Fig3e 1PGA n = 1171, r = 0.302, slope 0.038; Fig3f 1AOY n = 1451, r = 0.863, slope 0.650;
Fig3g/ExtFig9 hydrophobicity r = 0.402, buried nonpolar area (FAMILYVW) 0.371, buried nonpolar residues (AFILMVWY) 0.370, buried nonpolar residues 0.368, hydrophobic residues 0.359 / 0.359 (excl. Ala) (58 proteins, 666 features).

## Notes
* Fig3b is drawn with `deltaG_95CI <= 0.5` (n = 14,551, r = 0.555), as in the published figure (r = 0.55). The exported reference PDF in the archive (`reproduced_figures/Fig3/Fig3b_reproduced.pdf`, n = 15,145, r = 0.53) was made without this filter, so the two differ.
* Filters of the original notebooks kept as they are: yield clipped at -1.5 log10 mg/L; `.pdb` entries removed; Fig3c/e/f, ExtFig6a/7 use dG without CI filter (CI >= 0.5 cells are hatched grey in the dG heatmaps); ExtFig8 / Fig3g/h / ExtFig9 use dG only where `deltaG_95CI <= 0.5`, regress yield on dG over variants with yield > -1.5 and dG > -2, and require > 20 valid variants for Fig3g/h and ExtFig9.
* Fig3c / ExtFig6a: `deltaG_pred` (S05) is not needed for the printed panels (the notebook also computes a predicted-ddG delta that is not exported), so S05 is not read.
* The WT Rosetta features come from S23a (`pdb_wt/*`) and S23b (`rosetta_relax/*`); the protein name is the file stem as in the notebook; columns with missing values and per-site / per-residue / 3D-contact columns are dropped as in the notebook.
* Gaps: none for the quantitative panels.
