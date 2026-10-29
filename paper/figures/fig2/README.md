# Fig 2 and Extended Data Fig 5 (de novo backbones, structure-prediction filters)

Run any script with `python figures/fig2/<script>.py`; PDFs go to `outputs/fig2/`. `_fig2_data.py` is the shared helper
(merges S17 design list + S04 yield, clips yield to [-1.5, 3.5] as in the analysis, adds S13 backbone features).

Data conventions (checked against the original analysis):
- One row per design = S17 (19,506 designs); yield from S04 by (`sublib`, `name`). Designs whose `sublib` is "a;b" (present in two
  sublibraries) have no yield (they were removed in the analysis), leaving 19,072 designs with a yield (19,071 without the `wt` entries).
- AF3 panels use the **AF3-fast** columns (`*_af3_fast`) of S17 (the version used in the original Fig 2 / ExtFig 5); AF2 panels use `*_af2`.
- Per-backbone panels (Fig 2b-d, 2i, ExtFig 5b) exclude the parent-sequence entries (`<backbone>_wt_*`, method `wt`); the filter panels
  (Fig 2e-h, ExtFig 5c-g) keep them, exactly as in the original notebooks.
- Positive = yield > 10 mg/L (log10 > 1); filter = pLDDT > 90 and TM-score > 0.8 unless stated.

| panel | script | tables | status |
|---|---|---|---|
| Fig 2a | - | - | out of scope (schematic + methods table) |
| Fig 2b | `fig2b_yield_per_backbone.py` | S04, S13, S17 | done |
| Fig 2c | `fig2cd_extfig5b_backbone_features.py` | S04, S13, S17 | done (bootstrap, seed 0, 1000 resamples; backbones sorted by label as in the original so the CIs are identical) |
| Fig 2d (Rg, contact order) | `fig2cd_extfig5b_backbone_features.py` | S04, S13, S17 | done |
| Fig 2e | `fig2e_extfig5de_af_boxplot_scatter.py` | S04, S17 | done |
| Fig 2f | `fig2fgh_extfig5fg_filter_performance.py` | S04, S17 | done |
| Fig 2g | `fig2fgh_extfig5fg_filter_performance.py` | S04, S17 | done (numeric labels of the notebook output kept, as in the reference PDF) |
| Fig 2h | `fig2fgh_extfig5fg_filter_performance.py` | S04, S17 | done |
| Fig 2i top / bottom | `fig2i_stacked_bars.py` | S04, S17 | done |
| ExtFig 5a | - | (S13 `origin`) | out of scope: typeset table. Counts per origin are in S13 (SALAD 14, RFdiffusion2 6, ...; see README of source_data_v2, note 3) |
| ExtFig 5b | `fig2cd_extfig5b_backbone_features.py` | S04, S13, S17 | done |
| ExtFig 5c (AF3 / AF2, pLDDT / TM-score) | `extfig5c_filter_ecdf.py` | S04, S17 | done |
| ExtFig 5d (AF3 / AF2) | `fig2e_extfig5de_af_boxplot_scatter.py` | S04, S17 | done |
| ExtFig 5e | `fig2e_extfig5de_af_boxplot_scatter.py` | S04, S17 | done |
| ExtFig 5f | `fig2fgh_extfig5fg_filter_performance.py` | S04, S17 | done |
| ExtFig 5g (before / after AF3 filter) | `fig2fgh_extfig5fg_filter_performance.py` | S04, S17 | done |

Agreement with `reproduced_figures/Fig2/*_reproduced.pdf`: text tokens identical for all 24 PDFs; pixel difference (100 dpi, |diff| > 40)
0.000 for all except ExtFig 5f (0.012, sub-pixel marker positions). No data gaps: S13 is not needed beyond the backbone features, labels
and SS groups.
