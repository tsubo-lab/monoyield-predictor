#!/bin/bash
# Download the trained LoRA / head checkpoints (about 190 MB) from Hugging Face into weights/ (or the directory given as $1).
#   HF_REPO=<user-or-org>/<repo> bash download_weights.sh [out_dir]
set -euo pipefail
HF_REPO="${HF_REPO:?set HF_REPO to the Hugging Face repository id of the weights, e.g. HF_REPO=<user>/monoyield-predictor}"
OUT="${1:-weights}"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="${HF_REPO}", local_dir="${OUT}", allow_patterns=["*/*.pt"])
print("downloaded to", p)
PY
