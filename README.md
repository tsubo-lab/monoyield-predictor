# monoyield-predictor

Sequence-based predictors of the **monomeric expression yield in *E. coli*** (log10 mg/L) used in
*[paper title - to be filled]*. LoRA-fine-tuned protein language models trained on the pooled-barcode MS yield assay
(~35,000 de novo / DMS designs; nanobody library of 2,600 sequences).

> **Status (draft):** code, example inputs and documentation are in place. The trained checkpoints are **not deposited yet**
> (`download_weights.sh` is a stub). Scripts are the ones used for the paper; they have not been refactored into a package.

## Models

| name in the paper | base model | head | trained on | input | output | script |
|---|---|---|---|---|---|---|
| **ESMC-MonoYield** | ESMC-6B (`biohub/ESMC-6B`) + LoRA | masked-mean pooling -> MLP (hidden -> 128 -> ReLU -> 1) ("head B") | de novo + DMS, monomer log10 mg/L, 5-fold (mmseq cluster groups) | amino-acid sequence | log10 monomer yield (mg/L) | `scripts/predict_esmc_mlp_head.py` |
| **SaProt LoRA** | SaProt 1.3B (`westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI`) + LoRA | same MLP head | same | structure-aware sequence (AA + 3Di, `scripts/prepare_saprot_structure_seqs.py`) | same | `scripts/predict_saprot_mlp_head.py` |
| **ESMC-Nb-MonoYield** | ESMC-6B + LoRA | per-residue regression, mean-pooled (the "residue-mean" head of `train_esmc_lora_denovo_only_residue_mean_clip.py`; not the MLP head B) | nanobody library (monomer log10 mg/L), 5-fold | nanobody sequence | log10 monomer yield | `scripts/predict_esmc_residue.py` / `predict_nanobody_model_on_denovo.py` |
| **ESMC-Nb-MDRatio** | ESMC-6B + LoRA | same as above | nanobody library (log10 monomer/dimer ratio) | nanobody sequence | log10 monomer/dimer ratio | same |
| ESMC-MonoYield_ft | ESMC-MonoYield fine-tuned on the nanobody library | head B | nanobody 5-fold | nanobody sequence | log10 monomer yield | `scripts/predict_esmc_residue_nanobody_ft.py` |

All predictions are 5-fold ensembles (mean of the five fold checkpoints). Targets were clipped to [-1.5, 3.5] (log10 mg/L) for training.
See `docs/models.md` for the mapping between models, checkpoints, training commands and the paper's figures.

## Installation

```bash
conda create -n monoyield python=3.12 -y && conda activate monoyield
# install PyTorch for your CUDA version first, e.g. (Blackwell, CUDA 12.8):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .          # transformers==4.57.6, peft==0.19.1, ...
```

* **GPU memory.** ESMC-6B in fp32 needs ~25 GB (48 GB-class card, e.g. RTX 6000 Pro). `--dtype bfloat16` halves this
  (fits a 24 GB card) but should be validated against fp32 before use (bf16 vs fp32 deviations were checked on the paper's data).
* **Pin the Hugging Face revision.** The `main` revision of `biohub/ESMC-6B` / `ESMC-600M` changed after the models were trained and can break loading.
  Use the training revisions: ESMC-6B `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`, ESMC-600M: the revision recorded with the training run (pass `--model_revision`; the full hash still has to be copied here).
* If the model repository is gated, `export HF_TOKEN=...` (the scripts call `huggingface_hub.login`).

## Weights

```bash
WEIGHTS_URL=<url> bash download_weights.sh weights/     # not available yet
```
Each model is a directory with `group_1 ... group_5/best_model.pt`.
**Note:** the checkpoints produced by the training scripts contain the full fp32 ESMC-6B state (~25 GB per fold, 5 folds per model).
Before deposition, export only the LoRA adapter + head parameters (TODO, see `docs/models.md`).

## Quick start

```bash
# ESMC-MonoYield on the example sequences (columns: name, aa_seq)
python scripts/predict_esmc_mlp_head.py \
    --external_csv examples/example_main_library.csv \
    --model_dir weights/esmc6b_head_b \
    --model_name biohub/ESMC-6B --model_revision 45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a \
    --dtype float32 --batch_size 1 \
    --y_col observed_monomer_log10_mgL \
    --out_csv predictions_esmc_monoyield.csv
# -> columns: pred_group_1..5, pred_mean, ... (+ metrics if --y_col is given)

# nanobodies (sequence column is `seq`)
python scripts/predict_esmc_mlp_head.py --external_csv examples/example_nanobody.csv --seq_col seq ...
```
`examples/` contains five de novo designs and five nanobodies with their observed values.
SaProt needs a structure-derived sequence (AA + 3Di) in the `saprot_seq` column:
`scripts/prepare_saprot_structure_seqs.py` builds it from AF3/PDB structures (requires Foldseek).

## Training / reproducing the paper

`scripts/train_*.py` and `slurm/*.sh` are the commands used on our SLURM cluster (paths are replaced by `${DATA_ROOT}`;
adapt GPU/partition lines). Training data (sequence, yield, cluster-based 5-fold split) are provided as source data
with the paper (see `docs/data.md`): the tables `TableS04` (main library), `TableS07` / `TableS18` (nanobody).

| step | script |
|---|---|
| ESMC-MonoYield (head B, 5 folds) | `slurm/run_nodg_6b_mmseq30.sh` -> `scripts/train_esmc_lora_dg.py` |
| SaProt LoRA (head B) | `slurm/run_saprot_B_mlp128_mmseq30.sh` -> `scripts/train_saprot_lora_mlp_head.py` |
| ESMC-Nb-MonoYield / MDRatio | `slurm/run_6b_all_clip_nanobody_monomer_log.sh`, `..._mono_di_ratio.sh` -> `scripts/train_esmc_lora_denovo_only_residue_mean_clip.py` |
| nanobody fine-tuning of ESMC-MonoYield | `slurm/run_nanobody_finetune_5fold.sh` -> `scripts/finetune_esmc_lora_nanobody_5fold_from_pt.py` |
| prediction examples | `slurm/run_predict_*.sh` |

## Repository layout

```
monoyield-predictor/
├── README.md  LICENSE  pyproject.toml  download_weights.sh  .gitignore
├── scripts/      training and prediction scripts (flat: predict_* import train_esmc_lora_dg.py / pooling_heads.py)
├── slurm/        example job scripts (cluster specific)
├── examples/     example inputs with observed values
├── docs/         models.md, data.md
└── notebooks/    (planned) minimal prediction notebooks
```

## Citation

*[to be filled]*

## License

MIT (see `LICENSE`; copyright holder to be filled). Base models keep their own licenses (ESMC-6B: EvolutionaryScale/Biohub, SaProt: Westlake).
