# monoyield-predictor

Sequence-based predictors of the **monomeric expression yield in *E. coli*** (log10 mg/L) used in
*Massively parallel quantification and prediction of monomeric protein yield in Escherichia coli* ([bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.11.02.763507v1)). LoRA-fine-tuned protein language models trained on the pooled-barcode MS yield assay
(~35,000 de novo / DMS designs; nanobody library of 2,600 sequences).

> **Status (draft):** code, data, example inputs and documentation are in place. The trained weights (adapter-only, about 190 MB) are hosted on Hugging Face (`Tsuboyama-Lab/monoyield-predictor`; see "Weights"). Scripts are the ones used for the paper; they have not been refactored into a package.

## Models

| name in the paper | base model | head | trained on | input | output | script |
|---|---|---|---|---|---|---|
| **ESMC-MonoYield** | ESMC-6B (`biohub/ESMC-6B`) + LoRA | masked-mean pooling -> MLP (hidden -> 128 -> ReLU -> 1) | de novo + DMS, monomer log10 mg/L, 5-fold (mmseq cluster groups) | amino-acid sequence | log10 monomer yield (mg/L) | `scripts/predict_esmc_mlp_head.py` |
| **SaProt LoRA** | SaProt 1.3B (`westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI`) + LoRA | same MLP head | same | structure-aware sequence (AA + 3Di, `scripts/prepare_saprot_structure_seqs.py`) | same | `scripts/predict_saprot_mlp_head.py` |
| **ESMC-Nb-MonoYield** | ESMC-6B + LoRA | per-residue regression, mean-pooled (`train_esmc_lora_denovo_only_residue_mean_clip.py`) | nanobody library (monomer log10 mg/L), 5-fold | nanobody sequence | log10 monomer yield | `scripts/predict_esmc_residue.py` / `predict_nanobody_model_on_denovo.py` |
| **ESMC-Nb-MDRatio** | ESMC-6B + LoRA | same as above | nanobody library (log10 monomer/dimer ratio) | nanobody sequence | log10 monomer/dimer ratio | same |

All predictions are 5-fold ensembles (mean of the five fold checkpoints). Yield targets (monomer log10 mg/L) were clipped to [-1.5, 3.5] for training; the monomer/dimer ratio target was not clipped.
See `docs/models.md` for the mapping between models, checkpoints, training commands and the paper's figures.

## Installation

**Environment used for the paper (record).** The models were trained and evaluated in a conda environment (`esmc-6000`) built as follows:
Python 3.12, then `pip install esm@git+https://github.com/Biohub/esm.git@main` (June 2026; `esm` 3.3.0, commit `f652b471d29da828b31e9b7a9cf7d0a7803240f5`),
which at that time pulled in Biohub's own `transformers` build (`github.com/Biohub/transformers`, commit `f9a5a374be135f63b3019c1cefb91ea9e2d27e10`, reports itself as `4.57.6`) with native ESMC support.
PyTorch 2.12.0.dev (CUDA 12.8), `peft==0.19.1`.
If you already have such an environment, the scripts in `scripts/` run as they are.
**This environment cannot be rebuilt from scratch today**: the Biohub `transformers` repository is no longer publicly accessible, so the pinned install fails.

**New installation.** Use the scripts in `scripts_esm34/` (see Quick start):

```bash
conda create -n monoyield python=3.12 -y && conda activate monoyield
pip install "esm==3.4.2" "peft==0.19.1" pandas scipy scikit-learn     # also installs torch 2.11 and transformers 4.57
```

If your GPU driver does not support CUDA 13 (the default torch 2.11 wheel), install torch first from a matching index, e.g.
`pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128`.

## Weights

```bash
HF_REPO=Tsuboyama-Lab/monoyield-predictor bash download_weights.sh weights/     # weights hosted on Hugging Face
```
Each model is a directory of five adapter-only checkpoints (LoRA + head, 7-12 MB each; the frozen base model is downloaded from Hugging Face):

```
weights/
├── esmc_monoyield/   esmc_monoyield_1.pt ... esmc_monoyield_5.pt
├── saprot_lora/      saprot_lora_1.pt ... saprot_lora_5.pt
├── esmc_nb_monoyield/ esmc_nb_monoyield_1.pt ... _5.pt
└── esmc_nb_mdratio/  esmc_nb_mdratio_1.pt ... _5.pt
```
`<model>_k.pt` is the checkpoint of fold k (test group k). The prediction scripts take the directory (`--model_dir weights/esmc_monoyield`) and average the five folds.
`scripts/export_adapter_weights.py` creates these files from the full training checkpoints (~25 GB per fold for ESMC-6B).

## Quick start

```bash
# ESMC-MonoYield on the example sequences (columns: name, aa_seq)
python scripts/predict_esmc_mlp_head.py \
    --external_csv examples/example_main_library.csv \
    --model_dir weights/esmc_monoyield \
    --model_name biohub/ESMC-6B --model_revision 45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a \
    --dtype float32 --batch_size 1 \
    --y_col observed_monomer_log10_mgL \
    --out_csv predictions_esmc_monoyield.csv
# -> columns: pred_group_1..5, pred_mean, ... (+ metrics if --y_col is given)

# nanobodies (sequence column is `seq`)
python scripts/predict_esmc_mlp_head.py --external_csv examples/example_nanobody.csv --seq_col seq ...
```
The commands above are for the paper environment. In an environment made with `esm==3.4.2` (see Installation) run the same commands with
`scripts_esm34/predict_esmc_mlp_head.py` / `scripts_esm34/predict_esmc_residue.py` instead of `scripts/...` (same arguments, same output).
`examples/` contains five de novo designs and five nanobodies with their observed values.
SaProt needs a structure-derived sequence (AA + 3Di) in the `saprot_seq` column:
`scripts/prepare_saprot_structure_seqs.py` builds it from AF3/PDB structures (requires Foldseek).

## Training / reproducing the paper

`run/*.sh` contain the exact training and prediction commands used for the paper (run them from the repository root; ESMC-6B in fp32 needs a GPU with >= 48 GB). Training data are in `data/` (see `docs/data.md`).

| step | script |
|---|---|
| ESMC-MonoYield (5 folds) | `run/train_esmc_monoyield.sh` -> `scripts/train_esmc_lora_mlp_head.py` |
| SaProt LoRA | `run/train_saprot_lora.sh` -> `scripts/train_saprot_lora_mlp_head.py` |
| ESMC-Nb-MonoYield / MDRatio | `run/train_esmc_nb_monoyield.sh`, `run/train_esmc_nb_mdratio.sh` -> `scripts/train_esmc_lora_denovo_only_residue_mean_clip.py` |
| prediction examples | `run/predict_cov_binders.sh`, `run/predict_nanobody.sh` |

## Repository layout

```
monoyield-predictor/
├── README.md  LICENSE  NOTICE  pyproject.toml  download_weights.sh  .gitignore
├── scripts/      training and prediction scripts (flat: predict_* import train_esmc_lora_mlp_head.py / pooling_heads.py)
├── run/          training / prediction commands used for the paper
├── data/         training / input csv files (sequences, yields, 5-fold assignment)
├── examples/     example inputs with observed values
└── docs/         models.md, data.md
```

## Citation

If you use this code, data or the trained weights, please cite the preprint:

> Setsu S*, Kiyokawa K*, Lam AYW, Mishiro-Sato E, Tomari Y, Tsuboyama K.
> Massively parallel quantification and prediction of monomeric protein yield in *Escherichia coli*.
> *bioRxiv* (2026). https://doi.org/10.64898/2026.11.02.763507
> \*These authors contributed equally.

Source data, tables and code are archived on Zenodo: https://doi.org/10.5281/zenodo.23874456

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`; Copyright 2026 Kotaro Tsuboyama and Selena Setsu). Base models keep their own licenses (ESMC-6B: EvolutionaryScale/Biohub, SaProt: Westlake).
