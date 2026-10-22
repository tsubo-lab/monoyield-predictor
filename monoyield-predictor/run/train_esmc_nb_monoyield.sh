#!/bin/bash
# ESMC-Nb-MonoYield: ESMC-6B + LoRA trained on the nanobody library (monomer yield), 5 folds. Run from the repository root.
python scripts/train_esmc_lora_denovo_only_residue_mean_clip.py \
  --denovo_csv data/nanobody_5fold.csv \
  --out_dir outputs/esmc_nb_monoyield \
  --model_name biohub/ESMC-6B \
  --y_col monomer_log_mgL --seq_col seq --rep_col cluster \
  --epochs 10 --patience 5 --batch_size 2 --grad_accum_steps 8 \
  --clip_y_min -1.5 --clip_y_max 3.5
