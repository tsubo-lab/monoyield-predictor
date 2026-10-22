#!/bin/bash
# SaProt LoRA: SaProt 1.3B + LoRA, MLP head, 5 folds (input: amino acid + Foldseek 3Di sequence). Run from the repository root.
python scripts/train_saprot_lora_mlp_head.py \
  --denovo_csv data/main_library_denovo_5fold_3di.csv \
  --dms_csv data/main_library_dms_5fold_3di.csv \
  --out_dir outputs/saprot_lora \
  --architecture mean_mlp128 \
  --model_name westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI \
  --model_revision 3e000715b428e69a8ea166716727c8596bc3ddcf \
  --seq_col saprot_seq --wt_seq_col wt_saprot_seq \
  --y_col monomer_log_mgL --wt_y_col wt_monomer_log_mgL --rep_col rep \
  --epochs 10 --patience 5 --batch_size 2 --grad_accum_steps 8 \
  --lora_r 4 --lora_alpha 8 --lora_dropout 0.15 --lora_lr 5e-5 --head_lr 1e-4 \
  --weight_decay 0.01 --grad_clip 1.0 --max_length 512 --seed 42 \
  --lambda_abs 1.0 --lambda_delta 0.5 --abs_loss_type huber --delta_loss_type mse \
  --clip_y_min -1.5 --clip_y_max 3.5
