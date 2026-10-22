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


python scripts/train_esmc_lora_denovo_only_residue_mean_clip.py \
  --denovo_csv ${DATA_ROOT}/260702_mmseq_cluster_nanobody/nanobody_5fold_v2.csv \
  --out_dir outputs/out_nanobody_monomer \
  --model_name biohub/ESMC-6B \
  --y_col monomer_log_mgL \
  --seq_col seq \
  --rep_col cluster \
  --epochs 10 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --patience 5 \
  --clip_y_min -1.5 \
  --clip_y_max 3.5