# Models, checkpoints and figures

| model | checkpoint dir (on the cluster, not shipped) | paper panels |
|---|---|---|
| ESMC-MonoYield | `261004_Lora_bbs_dg/outputs/nodg_6b_mmseq30_merged` (folds 1-5) | Fig. 4c-f, ExtFig. 10-12, Fig. 5j / ExtFig. 14 (applied to nanobodies) |
| SaProt LoRA | `261004_Lora_bbs_dg/outputs/saprot_B_mlp128_mmseq30` | Fig. 4d,e, ExtFig. 11, 12c |
| ESMC-Nb-MonoYield | `260623_LoRA/outputs/out_nanobody_monomer` | Fig. 5j, ExtFig. 14 |
| ESMC-Nb-MDRatio | `260623_LoRA/outputs/out_nanobody_monomer_dimer_ratio` | Fig. 5j |
| ESMC-MonoYield_ft | `260623_LoRA/outputs/nanobody_ft_5fold` | ExtFig. 14 |
| linear-head baselines | `260929_LoRA_mmseq30/outputs/{out_6b_mmseq30_merged, saprot_one_stage_..._clip_mmseq30, esm3_..._postnorm_head}` | Fig. 4c, ExtFig. 10c (not part of this repository) |

Predictions of all models for every test sequence are in the paper's source data (`TableS15a/b`, `TableS16a/b`).

## Open items before publication
- Export LoRA adapter + head only from `best_model.pt` (currently the full ~25 GB fp32 state per fold) and deposit (Hugging Face / Zenodo); fill `download_weights.sh`.
- Check that `predict_*` run end-to-end from the exported weights (the scripts import the model classes from `train_esmc_lora_dg.py`).
- Decide on a package interface (single `monoyield.predict(seqs)`), optional.
- Fill the license holder, citation, and paper title.
