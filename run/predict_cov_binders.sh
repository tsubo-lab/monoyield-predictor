#!/bin/bash
# Predict the SARS-CoV-2 binder designs with ESMC-MonoYield and SaProt LoRA (5-fold ensembles).
# Run from the repository root after download_weights.sh.
python scripts/predict_esmc_mlp_head.py \
  --external_csv data/cov_binders.csv --seq_col aa_seq \
  --model_dir weights/esmc_monoyield \
  --model_name biohub/ESMC-6B --model_revision 45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a \
  --batch_size 1 --out_csv predictions_cov_binders_esmc_monoyield.csv

python scripts/predict_saprot_mlp_head.py \
  --external_csv data/cov_binders_3di.csv --seq_col saprot_seq \
  --model_dir weights/saprot_lora \
  --model_revision 3e000715b428e69a8ea166716727c8596bc3ddcf \
  --batch_size 1 --out_csv predictions_cov_binders_saprot_lora.csv
