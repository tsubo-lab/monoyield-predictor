"""ExtFig3b: the 48 individual HPLC-SEC chromatograms (baseline-subtracted, rolling 3 % quantile over 10,001 points); red lines = 33 and 40 min
integration window; samples are numbered in the row order of S14a (= order of the original panels).
Tables: hplc individual wide chromatograms (common.tables.hplc), S14a (sample order / protein names)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.tables import hplc, load, outdir
from common.style import set_style

set_style()
RT_WINDOW = (33, 40)
BASELINE_WINDOW, BASELINE_Q = 10001, 0.03

chrom = hplc("individual")
samples = list(load("S14a")["name"])
assert set(samples) == set(chrom.columns) - {"R_Time"}
print("chromatograms:", len(samples), "points:", len(chrom))

fig = plt.figure(figsize=(15 / 2.54, 20 / 2.54), tight_layout=True, dpi=300)
for i, col in enumerate(samples, start=1):
    y = chrom[col].to_numpy(dtype=float)
    baseline = pd.Series(y).rolling(BASELINE_WINDOW, center=True, min_periods=1).quantile(BASELINE_Q).values
    ax = fig.add_subplot(8, 6, i)
    plt.plot(chrom["R_Time"], y - baseline, lw=0.5)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(f"Sample {i}")
    plt.vlines(RT_WINDOW, 0, 4e6, color="red", linestyle="--", lw=0.5)
    ax.tick_params(axis="both", which="major", pad=1)
plt.savefig(os.path.join(outdir("fig1"), "ExtFig3b.pdf"), bbox_inches="tight")
