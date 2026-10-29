"""Access to the supplementary tables in paper/tables (csv or csv.gz) by key, e.g. load("S04"), load("S12a"), load("S05_feature_list")."""
import glob, os
import pandas as pd

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(PAPER, "tables")

def path(key):
    files = sorted(glob.glob(os.path.join(TABLES, f"Table{key}_*.csv*")) + glob.glob(os.path.join(TABLES, f"Table{key}.csv*")))
    if "feature_list" not in key:
        files = [f for f in files if "feature_list" not in os.path.basename(f)]
    if len(files) != 1:
        raise FileNotFoundError(f"key {key!r} matches {[os.path.basename(f) for f in files]} in {TABLES}")
    return files[0]

def load(key, **kw):
    return pd.read_csv(path(key), **kw)

def hplc(name):
    """Raw HPLC chromatograms, wide csv (R_Time + one column per sample): 'individual', 'cov_binder' or 'cytochromeC'."""
    return pd.read_csv(os.path.join(TABLES, "hplc_chromatograms", f"hplc_{name}_chromatograms.csv"))

def outdir(name):
    d = os.path.join(PAPER, "outputs", name)
    os.makedirs(d, exist_ok=True)
    return d
