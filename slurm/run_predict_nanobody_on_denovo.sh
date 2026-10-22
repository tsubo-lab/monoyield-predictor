#!/bin/bash
#SBATCH --job-name=nb-predict
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"

nvidia-smi
nvcc -V

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esmc-6000

# export HF_TOKEN=...   (only needed for gated Hugging Face repos)


python scripts/predict_nanobody_model_on_denovo.py \
  --checkpoint outputs/nanobody_ft/group_1/best_nanobody_finetuned_model.pt \
  --denovo_csv denovo_5fold.csv \
  --out_csv outputs/nanobody_ft/group_1/denovo_holdout_predictions.csv \
  --batch_size 4
