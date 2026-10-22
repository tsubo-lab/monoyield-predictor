#!/bin/bash
# ESMC-MonoYield: ESMC-6B + LoRA, MLP head, 5 folds. Run from the repository root; needs a GPU with >= 48 GB memory (fp32).
python scripts/train_esmc_lora_mlp_head.py \
  --denovo_csv data/main_library_denovo_5fold.csv \
  --dms_csv data/main_library_dms_5fold.csv \
  --out_dir outputs/esmc_monoyield \
  --model_name biohub/ESMC-6B \
  --model_revision 45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a \
  --y_col monomer_log_mgL --wt_y_col wt_monomer_log_mgL --rep_col rep \
  --epochs 10 --patience 5 --batch_size 2 --grad_accum_steps 8 \
  --lambda_abs 1.0 --lambda_delta 0.5 --abs_loss_type huber --delta_loss_type mse \
  --clip_y_min -1.5 --clip_y_max 3.5 \
  --no_deltag
