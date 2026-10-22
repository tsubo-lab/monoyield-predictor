#!/bin/bash
# Predict nanobody designs: ESMC-MonoYield (applied to nanobodies) and ESMC-Nb-MonoYield / ESMC-Nb-MDRatio (nanobody-specific models).
# Run from the repository root after download_weights.sh. The sequence column of the nanobody tables is `seq`.
REV=45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a
python scripts/predict_esmc_mlp_head.py \
  --external_csv data/nanobody_designs.csv --seq_col seq \
  --model_dir weights/esmc_monoyield \
  --model_name biohub/ESMC-6B --model_revision $REV --dtype float32 --batch_size 1 \
  --out_csv predictions_nanobody_esmc_monoyield.csv

for m in esmc_nb_monoyield esmc_nb_mdratio; do
  python scripts/predict_esmc_residue.py \
    --external_csv data/nanobody_designs.csv --seq_col seq \
    --model_dir weights/$m --model_revision $REV --batch_size 1 \
    --out_csv predictions_nanobody_$m.csv
done
