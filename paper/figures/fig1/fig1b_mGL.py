"""Fig1b: mGL barcode hexbin, replicate 1 (e4) vs replicate 2 (e8), log10(intensity + 1).
Blue = all barcodes, orange = barcodes selected (within the intensity window and < 0.4 from the regression line).
Tables: S22a."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from common.tables import load, outdir
from common.style import set_style

set_style()
DIST_MAX = 0.4
EXTENT = (-0.5, 9, -0.5, 9)

df = load("S22a")
for c in ["e4_old", "e8_old"]:
    df["log_" + c] = np.log10(df[c].fillna(0) + 1)
data_all = df[["log_e4_old", "log_e8_old"]]

logs = df.dropna(subset=["e4_old", "e8_old"]).copy()
slope, intercept, r, p, se = stats.linregress(logs["log_e4_old"], logs["log_e8_old"])
logs["dist"] = np.abs(slope * logs["log_e4_old"] - logs["log_e8_old"] + intercept) / np.sqrt(slope ** 2 + 1)
sel = ((logs["log_e4_old"] < 7) & (logs["log_e4_old"] > 4)
       & (logs["log_e8_old"] < 7 * slope + intercept) & (logs["log_e8_old"] > 4 * slope + intercept)
       & (logs["dist"] < DIST_MAX))
data_sel = logs.loc[sel, ["log_e4_old", "log_e8_old"]]
print(f"all barcodes n={len(data_all)}  r={stats.pearsonr(*data_all.T.values)[0]:.3f}")
print(f"detected in both n={len(logs)}  slope={slope:.3f} intercept={intercept:.3f}")
print(f"selected n={len(data_sel)}  r={stats.pearsonr(*data_sel.T.values)[0]:.3f}")


def truncate_colormap(cmap, minval, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", cmap(np.linspace(minval, maxval, n)))


fig = plt.figure(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
ax = fig.add_subplot(1, 1, 1)
for d, cm, z in [(data_all, truncate_colormap(plt.cm.Blues, 0.7), 1), (data_sel, truncate_colormap(plt.cm.Oranges, 0.30), 2)]:
    ax.hexbin(d["log_e4_old"], d["log_e8_old"], gridsize=60, mincnt=1, cmap=cm, alpha=0.85,
              linewidths=0, extent=EXTENT, zorder=z)
ax.plot([-10, 10], [-10, 10], c="black", ls=":", linewidth=1, zorder=10, alpha=0.7)
ax.set_xlabel("Replicate 1", labelpad=2)
ax.set_ylabel("Replicate 2", labelpad=2)
ax.set_aspect("equal", "box")
ticks = np.arange(0, 11, 2)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xlim(-0.5, 9); ax.set_ylim(-0.5, 9)
ax.tick_params(axis="both", which="major", pad=1)
plt.savefig(os.path.join(outdir("fig1"), "Fig1b.pdf"), bbox_inches="tight")
