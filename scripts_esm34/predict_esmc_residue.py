"""predict_esmc_residue.py for the esm 3.4.x environment: same command line and output as scripts/predict_esmc_residue.py (see esm34_compat.py)."""
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, ORIG)
import esm34_compat  # noqa: E402,F401  (must come before the original script imports transformers.AutoModel)

sys.argv[0] = os.path.join(ORIG, "predict_esmc_residue.py")
runpy.run_path(sys.argv[0], run_name="__main__")
