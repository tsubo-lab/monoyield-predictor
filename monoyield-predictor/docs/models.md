# Models, checkpoints and figures

Checkpoints are 5-fold ensembles (`<model>_1.pt ... <model>_5.pt`, adapter-only: LoRA + head, 7-12 MB per fold; the frozen base model is loaded from Hugging Face).

| model | exported weights | paper panels |
|---|---|---|
| ESMC-MonoYield | `weights/esmc_monoyield` | Fig. 4c-f, ExtFig. 10-12; applied to nanobodies in ExtFig. 14b-c |
| SaProt LoRA | `weights/saprot_lora` | Fig. 4d,e, ExtFig. 11, 12c |
| ESMC-Nb-MonoYield | `weights/esmc_nb_monoyield` | Fig. 5j, ExtFig. 14c |
| ESMC-Nb-MDRatio | `weights/esmc_nb_mdratio` | Fig. 5j |

The linear-head LoRA baselines of Fig. 4c / ExtFig. 10c are not part of this repository.
Predictions of all models for every test sequence are in the paper's source data (`TableS15a/b`, `TableS16a/b`).

## Export and check
`scripts/export_adapter_weights.py` strips a training checkpoint (full fp32 base-model state, ~25 GB for ESMC-6B) to the LoRA + head tensors.
Predictions from the exported weights were compared with the original checkpoints on example sequences (fold group_1):
ESMC-MonoYield, ESMC-Nb-MonoYield and ESMC-Nb-MDRatio give identical values (max |diff| = 0); SaProt LoRA reproduces the predictions stored with the paper (max |diff| < 1e-6).

## Open items before publication
- Deposit the exported weights (Hugging Face / Zenodo) and fill `download_weights.sh`.
- Optionally provide a single `monoyield.predict(seqs)` interface.
- Fill the license holder, citation, and paper title.
