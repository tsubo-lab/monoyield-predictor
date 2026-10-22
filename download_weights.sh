#!/bin/bash
# Download the trained LoRA / head checkpoints.
# NOTE: the weights have not been deposited yet. Fill in WEIGHTS_URL (Hugging Face, Zenodo, ...) when they are.
set -euo pipefail
WEIGHTS_URL="${WEIGHTS_URL:-}"
OUT="${1:-weights}"
if [ -z "$WEIGHTS_URL" ]; then
  echo "WEIGHTS_URL is not set: the trained checkpoints are not published yet (see README)." >&2
  exit 1
fi
mkdir -p "$OUT"
echo "Downloading $WEIGHTS_URL -> $OUT"
curl -L --fail -o "$OUT/weights.tar.gz" "$WEIGHTS_URL"
tar -xzf "$OUT/weights.tar.gz" -C "$OUT"
echo "done: $OUT"
