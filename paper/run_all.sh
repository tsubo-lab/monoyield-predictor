#!/bin/bash
# Draw all figure panels from the supplementary tables (paper/tables). PDFs go to paper/outputs/<fig>/.
# Usage (from anywhere):  bash paper/run_all.sh [fig1 fig2 ...]     (default: all five families)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPERS="_fig2_data.py dms_data.py dms_plots.py fig4_common.py"      # imported by the other scripts, not figure scripts
FAMS=("$@"); [ ${#FAMS[@]} -eq 0 ] && FAMS=(fig1 fig2 fig3 fig4 fig5)
for fam in "${FAMS[@]}"; do
  for s in "$HERE"/figures/$fam/*.py; do
    b=$(basename "$s"); case " $HELPERS " in *" $b "*) continue;; esac
    echo "[$fam] $b"; python "$s"
  done
done
