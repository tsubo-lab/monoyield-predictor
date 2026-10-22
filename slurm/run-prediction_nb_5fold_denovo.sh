#!/bin/bash
#SBATCH --job-name=esm3        
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"

nvidia-smi
nvcc -V

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esmc-6000


# export HF_TOKEN=...   (only needed for gated Hugging Face repos)


python scripts/predict_esmc_residue_nanobody_ft.py \
  --external_csv ${DATA_ROOT}/260623_LoRA/260723_denovo_group1_2.csv \
  --model_dir ${DATA_ROOT}/260623_LoRA/outputs/nanobody_ft_5fold \
  --out_csv predict_mgL_denovo_nb_model.csv \
  --name_col name \
  --seq_col aa_seq \
  --y_col monomer_log_mgL \
  --checkpoint_name best_nanobody_finetuned_model.pt \
  --pt_folder_names fold_1_test_group_1 fold_2_test_group_2 fold_3_test_group_3 fold_4_test_group_4 fold_5_test_group_5 \
  --batch_size 2