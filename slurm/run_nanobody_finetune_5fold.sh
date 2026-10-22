#!/bin/bash
#SBATCH --job-name=esm3        
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"

nvidia-smi
nvcc -V

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esmc-6000



# export HF_TOKEN=...   (only needed for gated Hugging Face repos)


python scripts/finetune_esmc_lora_nanobody_5fold_from_pt.py \
  --init_pt outputs/out_one_stage_abs_wt_delta_residue_6b_clip-1/group_1/best_model.pt \
  --nanobody_csv ${DATA_ROOT}/260702_mmseq_cluster_nanobody/nanobody_5fold.csv \
  --out_dir outputs/nanobody_ft_5fold \
  --model_name biohub/ESMC-6B \
  --name_col name \
  --seq_col seq \
  --y_col monomer_log_mgL \
  --split_col split \
  --run_5fold \
  --fold_values group_1 group_2 group_3 group_4 group_5 \
  --epochs 20 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --loss_type mse \
  --range_loss_weight 0.1 \
  --early_stop_metric rmse \
  --lora_lr 1e-5 \
  --head_lr 5e-5 \
  --train_head \
  --train_lora