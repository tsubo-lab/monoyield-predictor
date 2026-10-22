#!/bin/bash
#SBATCH --job-name=nodg_mm30
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
#SBATCH --exclude=tsubolab-gpu01,tsubolab-gpu02
#SBATCH --output=slurm-%j.out
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

STAGE=/tmp/setsu_nodg_mmseq30_${SLURM_JOB_ID}
FINAL=${SLURM_SUBMIT_DIR}/outputs/nodg_6b_mmseq30
DENOVO_CSV=denovo_5fold_mmseq30.csv
DMS_CSV=dms_pair_mmseq30.csv

nvidia-smi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esmc-6000
# export HF_TOKEN=...   (only needed for gated Hugging Face repos)

source "${SLURM_SUBMIT_DIR}/scripts/_stage_common.sh"
stage_setup
stage_start_partial_sync >/dev/null 2>&1 &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

cp "${SLURM_SUBMIT_DIR}/deltaG_pred.csv" "$STAGE/"

python "${SLURM_SUBMIT_DIR}/scripts/train_esmc_lora_dg.py" \
  --denovo_csv "${STAGE}/${DENOVO_CSV}" \
  --dms_csv "${STAGE}/${DMS_CSV}" \
  --deltag_csv "${STAGE}/deltaG_pred.csv" \
  --out_dir "${STAGE}/out" \
  --model_name biohub/ESMC-6B \
  --model_revision 45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a \
  --y_col monomer_log_mgL \
  --wt_y_col wt_monomer_log_mgL \
  --rep_col rep \
  --epochs 10 \
  --lambda_abs 1.0 \
  --lambda_delta 0.5 \
  --abs_loss_type huber \
  --delta_loss_type mse \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --patience 5 \
  --clip_y_min -1.5 \
  --clip_y_max 3.5 \
  --no_deltag
RC=$?
echo "[run] training exit code: $RC"
kill $SYNC_PID 2>/dev/null
stage_finalize
exit $RC
