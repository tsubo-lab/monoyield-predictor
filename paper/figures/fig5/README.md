# Fig 5 / ExtFig 13-14 (nanobody) - figure scripts

All scripts read only `paper/tables/` (via `common/tables.py`) and write PDFs to `paper/outputs/fig5/`.
Run e.g. `python paper/figures/fig5/fig5i_rule_score.py` (the scripts `extfig13cde_*` import the loader in `fig5i_rule_score.py`, keep them in the same folder).
Agreement = text tokens of the PDF vs the reference (`reproduced_figures/Fig5`) and pixel fraction differing (100 dpi, threshold 40).

| Panel | Script | Tables | Status | Agreement |
|---|---|---|---|---|
| Fig5c (R with monomer yield) | `fig5cd_feature_correlation.py` | S07, S24, S24b, S08 | done | text identical, pixdiff 0.0026 |
| Fig5d (R with monomer/dimer ratio) | `fig5cd_feature_correlation.py` | S07, S24, S24b, S08 | done | text identical, pixdiff 0.0047 |
| Fig5i (rule count vs yield / ratio) | `fig5i_rule_score.py` | S07, S18 | done | text identical, pixdiff 0.0012 |
| Fig5j-yield / -ratio (ESMC-Nb-MonoYield / -MDRatio) | `fig5j_extfig14bd_prediction_hexbin.py` | S16a, S16b | done | identical (pixdiff 0) |
| ExtFig13a-yield / -ratio | `extfig13ab_methods_and_scatter.py` | S07 | done | identical (pixdiff 0) |
| ExtFig13b-1 / -2 | `extfig13ab_methods_and_scatter.py` | S07 | done | identical (pixdiff 0) |
| ExtFig13c (CDR aa frequency) | `extfig13cde_cdr_heatmaps.py` | S07, S18 | done | identical (pixdiff 0) |
| ExtFig13d / e (median yield / ratio) | `extfig13cde_cdr_heatmaps.py` | S07, S18 | done | identical (pixdiff 0) |
| ExtFig14a-yield / -ratio | `extfig14a_cdna_display_enrichment.py` | S19 | done | identical (pixdiff 0) |
| ExtFig14b (ESMC-MonoYield on nanobodies) | `fig5j_extfig14bd_prediction_hexbin.py` | S16a | done | identical (pixdiff 0) |
| ExtFig14c (positive rate vs selection) | `extfig14c_topk_precision.py` | S16a, S16b | done | identical (pixdiff 0) |
| ExtFig14d-expression / -ratio (omnilib-ml) | `fig5j_extfig14bd_prediction_hexbin.py` | S16a, S16b, S07 | done | identical (pixdiff 0) |

## Notes
- Yields: S07 non-spike rows (2,590), `monomer`/`dimer` clipped at -1.5 and the ratio recomputed (as in the notebook), CDRs / method from S18 (matched by lower-case name). Fig5c/d use 2,589 nanobodies (those with features in S24).
- Fig5c/d: representatives of the pooled correlation clusters (S24b, `freq_CA` relabelled `cdr3_len`) + AF3/ESMFold2 ipTM/pLDDT and ESM3dG (S08); bootstrap 1000x, seed 0, as in the notebook.
- Fig5i: rule scores are computed from the CDR2/CDR3 sequences (S18) with the rules of the notebook (6 McMahon rules; v2 = Cpos1 Y/W + 3 additional rules). Values above boxes: 6/29, 14/33 (yield), 17/37, 28/44 (ratio).
- ExtFig13a: the notebook also draws total / supernatant / fraction boxplots; the paper panels show only monomer yield and ratio (the other fractions are not in the tables and not needed).
- ExtFig14d-expression: the unclipped observed yield (S07) is used because S16a stores the observed value clipped at -1.5 (omnilib scores are missing for 200 sequences).
- Key statistics (printed by the scripts): Fig5j-yield r=0.806 / R2=0.646; Fig5j-ratio r=0.853 / R2=0.728; ExtFig14b r=0.226; omnilib-ml r=0.24 / 0.27 (yield), 0.28 / 0.31 (ratio).
- Not drawn (not in the paper panels): the notebook's extra plots (lib82 histograms/ECDFs, ESMC-MonoYield_ft hexbin).
