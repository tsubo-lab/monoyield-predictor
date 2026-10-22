#!/bin/bash
# ESMC-Nb-MDRatio: same as train_esmc_nb_monoyield.sh, target = log10 monomer/dimer ratio. Run from the repository root.
python scripts/train_esmc_lora_denovo_only_residue_mean_clip.py \
  --denovo_csv data/nanobody_5fold.csv \
  --out_dir outputs/esmc_nb_mdratio \
  --model_name biohub/ESMC-6B \
  --y_col monomer_dimer_ratio_log --seq_col seq --rep_col cluster \
  --epochs 10 --patience 5 --batch_size 2 --grad_accum_steps 8
