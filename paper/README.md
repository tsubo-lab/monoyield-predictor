# Paper figures from the supplementary tables

`tables/` holds the supplementary tables of the paper (csv; the largest as csv.gz) and the raw HPLC chromatograms (`tables/hplc_chromatograms/`).
`figures/fig1 ... fig5/` hold one script per figure (group of panels) that draws the panels **from these tables only**
(Fig. 1 + Extended Data 1-4, Fig. 2 + ED 5, Fig. 3 + ED 6-9, Fig. 4 + ED 10-12, Fig. 5 + ED 13-14).
Each family folder has a README with the panel -> script -> tables map and the agreement with the figures of the paper.

```bash
conda env create -f paper/environment.yml && conda activate monoyield-paper
bash paper/run_all.sh            # all panels; or: bash paper/run_all.sh fig4
```
PDFs are written to `paper/outputs/<fig>/` (not tracked by git). Plotting uses `common/style.py` (Arial 6 pt; install Arial or Matplotlib falls back to another font).

Not included: schematics, rendered structures and hand-annotated panels (Fig. 1a,f,i; Fig. 3d and the structure panels of Fig. 3e,f and ED 7; Fig. 4b,g; Fig. 5a,b,e-h; ED 1b, ED 5a); these are listed in the family READMEs.
The tables and their columns are described in `tables/README.md`.

`common/tables.py` loads a table by key, e.g. `load("S04")` (main library, protein level), `load("S15a")` (Fig. 4 predictions).

Update 2026-11-02 (figure version of `original_figs_261102`): Fig1h, ExtFig1a (now one 3-panel figure; ExtFig1a's total-soluble panel removed), ExtFig1c (was ExtFig2a, one 3 x 5 figure), ExtFig2 (was ExtFig2b, one 3 x 5 figure), ExtFig3a labels, Fig2b/e/g, ExtFig5c (one figure), Fig3g, ExtFig8, ExtFig9, Fig4d/e, ExtFig11 and Fig4hi were redrawn to follow the changed notebooks. A full run now writes 17 / 24 / 27 / 23 / 18 PDFs for fig1-fig5 (fig1 had 47 files while ExtFig2a/b were 30 separate PDFs). `common/style.py: set_style(arial_math=True)` reproduces the Arial math text of the new Fig1e/g/h, ExtFig1c/2/3a notebooks.
