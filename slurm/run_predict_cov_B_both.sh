#!/bin/bash
#SBATCH --job-name=pred_cov_B2
#SBATCH --cpus-per-task=2
# ESMC-6B fp32 checkpoints are ~25GB, so a 24GB 4090 is out; this GRES pins to gpu03/gpu04.
#SBATCH --gres=gpu:nvidia_geforce_rtx_6000pro:1
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# DATA_ROOT: directory holding the input csv files / checkpoints (see docs/data.md)
DATA_ROOT="${DATA_ROOT:-./data}"
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$(conda info --base)/etc/profile.d/conda.sh"
# export HF_TOKEN=...   (only needed for gated Hugging Face repos)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR/scripts:${PYTHONPATH:-}"   # set -u: PYTHONPATH may be unset

MM30=${DATA_ROOT}/260929_LoRA_mmseq30
COV_AA=${DATA_ROOT}/260616_prospective_validation_cov/260701_cov_binder_ref_iptm.csv
COV_SAPROT=${MM30}/261007_cov_binder_saprot_input.csv   # saprot_seq from real Foldseek 3Di

# ============================================================
# 1. ESMC-6B, head B (mean_mlp128), 5-fold ensemble
# ============================================================
conda activate esmc-6000
echo "############ ESMC-6B head B ############"
python -u scripts/predict_esmc_mlp_head.py \
  --external_csv "$COV_AA" \
  --model_dir ckpt_B_6b_mmseq30 \
  --out_csv 261010_predict_cov_binder_esmc6b_Bhead.csv \
  --model_revision 45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a \
  --name_col name --seq_col aa_seq --batch_size 2 || exit 1
conda deactivate

# ============================================================
# 2. SaProt 1.3B, head B — sanity check first, then the real run
# ============================================================
conda activate SaProt
echo "############ SaProt head B: fold-1 sanity ############"
# 60 rows from fold 1's own held-out TEST set. The training run stored its
# predictions for these exact rows, so reproducing them proves the predictor
# rebuilds the model faithfully (LoRA targets, head shape, masking, tokenizer).
python -u scripts/predict_saprot_mlp_head.py \
  --external_csv 261010_saprotB_sanity_fold1.csv \
  --model_dir outputs/saprot_B_mlp128_mmseq30 \
  --folds group_1 --pt_folder_names group_1 \
  --out_csv 261010_saprotB_sanity_fold1_pred.csv \
  --model_revision 3e000715b428e69a8ea166716727c8596bc3ddcf \
  --name_col name --seq_col saprot_seq --batch_size 1 || exit 1

python - <<'PY' || exit 1
import pandas as pd, numpy as np, sys
d = pd.read_csv("261010_saprotB_sanity_fold1_pred.csv")
diff = (d["saprot_pred_group_1"] - d["ref_y_pred"]).abs()
r = np.corrcoef(d["saprot_pred_group_1"], d["ref_y_pred"])[0, 1]
print(f"SANITY  n={len(d)}  max|diff|={diff.max():.6f}  mean|diff|={diff.mean():.6f}  r={r:.6f}")
if diff.max() > 1e-3:
    print("SANITY FAILED: predictor does not reproduce the stored fold-1 predictions.")
    sys.exit(1)
print("SANITY PASSED")
PY

echo "############ SaProt head B: cov binder, 5-fold ############"
python -u scripts/predict_saprot_mlp_head.py \
  --external_csv "$COV_SAPROT" \
  --model_dir outputs/saprot_B_mlp128_mmseq30 \
  --out_csv 261010_predict_cov_binder_saprot_Bhead.csv \
  --model_revision 3e000715b428e69a8ea166716727c8596bc3ddcf \
  --name_col name --seq_col saprot_seq --batch_size 1
RC=$?
echo "[run] exit code: $RC"
exit $RC
