#!/bin/bash
#SBATCH --job-name=saprotB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

STAGE=/tmp/setsu_saprotB_mmseq30_${SLURM_JOB_ID}
FINAL=${SLURM_SUBMIT_DIR}/outputs/saprot_B_mlp128_mmseq30
DENOVO_CSV=denovo_5fold_saprot.csv
DMS_CSV=dms_pair_saprot.csv

nvidia-smi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SaProt
# export HF_TOKEN=...   (only needed for gated Hugging Face repos)

source "${SLURM_SUBMIT_DIR}/scripts/_stage_common.sh"
stage_setup
stage_start_partial_sync >/dev/null 2>&1 &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

# Same snapshot the paper run resolved (2026-05-15).
MODEL_REVISION=3e000715b428e69a8ea166716727c8596bc3ddcf

# Condition B head + EVERY training hyperparameter spelled out so the match with
# the ESMC-6B B run (run_nodg_6b_mmseq30.sh) can be checked by eye. The previous
# SaProt run differed on three of these; two are now aligned:
#   epochs            20   -> 10   (aligned)
#   batch/grad_accum  1/16 -> 2/8  (aligned; same effective batch of 16, ~2x faster)
#   lora_dropout      0.15 -> KEPT AT 0.15
# lora_dropout stays SaProt-specific on purpose: it was chosen for this model's
# parameter count and adapter placement (LoRA on query/key/value/attention.output.dense
# = 2.70M trainable, vs 1.64M on ESMC's out_proj alone). Forcing ESMC's 0.05 here
# would cut the regularisation to a third of what this model was tuned for, which
# is a model-design change, not a hyperparameter alignment.
python -u "${SLURM_SUBMIT_DIR}/scripts/train_saprot_lora_mlp_head.py" \
  --denovo_csv "${STAGE}/${DENOVO_CSV}" \
  --dms_csv "${STAGE}/${DMS_CSV}" \
  --out_dir "${STAGE}/out" \
  --architecture mean_mlp128 \
  --model_name westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI \
  --model_revision "${MODEL_REVISION}" \
  --seq_col saprot_seq \
  --wt_seq_col wt_saprot_seq \
  --y_col monomer_log_mgL \
  --wt_y_col wt_monomer_log_mgL \
  --rep_col rep \
  --epochs 10 \
  --patience 5 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lora_r 4 \
  --lora_alpha 8 \
  --lora_dropout 0.15 \
  --lora_lr 5e-5 \
  --head_lr 1e-4 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --max_length 512 \
  --seed 42 \
  --lambda_abs 1.0 \
  --lambda_delta 0.5 \
  --abs_loss_type huber \
  --delta_loss_type mse \
  --clip_y_min -1.5 \
  --clip_y_max 3.5
RC=$?
echo "[run] training exit code: $RC"
kill $SYNC_PID 2>/dev/null
stage_finalize
exit $RC
