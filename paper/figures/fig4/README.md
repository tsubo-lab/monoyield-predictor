# Fig. 4 and Extended Data Figs 10-12: figure scripts

Run `python paper/figures/fig4/<script>.py` (any directory); PDFs are written to `paper/outputs/fig4/`. Only the tables in `paper/tables/` are read. `fig4_common.py` holds shared helpers (model names, colours, hexbin panel, Wilson CI).

| Panel | Script | Tables | Output | Status |
|---|---|---|---|---|
| Fig4a, ExtFig10a | `fig4a_extfig10a_zeroshot.py` | S04, S05, S05_feature_list, S25 | `Fig4a.pdf`, `ExtFig10a.pdf` | done (pixel diff 0) |
| Fig4b, Fig4g | - | - | - | out of scope (schematic / rendered structure) |
| Fig4c, ExtFig10c | `fig4c_extfig10c_model_comparison.py` | S15a, S15b | `Fig4c.pdf`, `ExtFig10c.pdf` | done (pixel diff 0) |
| Fig4d, Fig4e | `fig4de_extfig11_hexbin.py` | S15a | `Fig4d_{ESMC-MonoYield,SaProt-LoRA,Aggrescan3D}.pdf`, `Fig4e_{ESMC-MonoYield,SaProt-LoRA}.pdf` | done (pixel diff 0) |
| ExtFig11A-C | `fig4de_extfig11_hexbin.py` | S15a, S15b | `ExtFig11A.pdf`, `ExtFig11B.pdf`, `ExtFig11C.pdf` | done (pixel diff 0) |
| Fig4f | `fig4f_extfig12ad_topk.py` | S15a | `Fig4f.pdf` | done (pixel diff 0.2 %, rounding) |
| ExtFig12a | `fig4f_extfig12ad_topk.py` | S15a | `ExtFig12a.pdf` | done (pixel diff 0) |
| ExtFig12d (left, right) | `fig4f_extfig12ad_topk.py` | S15b | `ExtFig12d_left.pdf`, `ExtFig12d_right.pdf` | done (pixel diff 0.006 % / 0) |
| Fig4h, Fig4i | `fig4hi_extfig12bc_hplc_cov.py` | S14d | `Fig4hi.pdf` (h left, i right) | done (pixel diff 0) |
| ExtFig12b | `fig4hi_extfig12bc_hplc_cov.py` | S14d, `hplc("cov_binder")` | `ExtFig12b.pdf` | done (pixel diff 0) |
| ExtFig12c | `fig4hi_extfig12bc_hplc_cov.py` | S14d | `ExtFig12c.pdf` | done (pixel diff 0) |
| ExtFig10b | `extfig10b_esm3dg_vs_dg.py` | S04, S05 | `ExtFig10b.pdf` | done (pixel diff 0) |
| ExtFig12e | `extfig12e_pross.py` | S04 | `ExtFig12e_1_yield.pdf`, `ExtFig12e_2_dG.pdf`, `ExtFig12e_3_ddG_vs_dyield.pdf` | done (pixel diff 0) |

## Gaps and notes
1. **Fig4a / ExtFig10a**: ESMFold2 pLDDT / pTM and the contact order / radius of gyration / core fraction of each design are in S25 (added after the first version of this script); with them the top-10 features, the bootstrap CIs and the hexbins are identical to the reference figures.
2. Fig4c / ExtFig10c show point estimates only (no bootstrap CI in the original notebook cells); bootstrap CIs exist only for the Fig4a feature bars (1000 resamples, seed 0, as in the notebook).
3. The zero-shot ranking and the hexbins use the `sup`-free monomer yield only (the sup/total yields are not in the tables, and are not part of the paper panels).
4. ExtFig10b reproduces the original filter `~name.str.contains('.pdb')` (regex), which removes 991 dms-5 rows.
5. Fig4h/i, ExtFig12b/c use the final head-B labels (`selection_fig4h`, `in_fig4h_fig4i`) of S14d; Aggrescan3D is stored sign-flipped in S14d and S15a, so the raw score of ExtFig12c is -S14d value.
