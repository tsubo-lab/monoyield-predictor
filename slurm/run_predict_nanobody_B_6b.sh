#!/bin/bash
#SBATCH --job-name=pred_nb_B
#SBATCH --cpus-per-task=2
# fp32 ESMC-6B weights are ~25GB, so a 6000pro is required, not a preference.
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esmc-6000
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TOKENIZERS_PARALLELISM=false
# export HF_TOKEN=...   (only needed for gated Hugging Face repos)

MODEL_REVISION=45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a
NB_CSV=${DATA_ROOT}/260506_lib77_add4_mean_log/260605_nanobody_aa_seq.csv

# 9,284 nanobody sequences, head B (mean_mlp128), 5-fold ensemble.
# The sequence column is `seq`, not the usual `aa_seq` (checked against the file).
# dtype float32 and batch_size 1 are kept identical to Track A's nanobody run
# (261007_predict_nanobody_6b_mmseq30.csv) so head A and head B differ only in the head.
python -u scripts/predict_esmc_mlp_head.py \
  --external_csv "$NB_CSV" \
  --model_dir ckpt_B_6b_mmseq30 \
  --out_csv 261013_predict_nanobody_esmc6b_Bhead.csv \
  --model_revision "${MODEL_REVISION}" \
  --dtype float32 \
  --name_col name \
  --seq_col seq \
  --batch_size 1
RC=$?
echo "[run] predict exit code: $RC"
exit $RC
