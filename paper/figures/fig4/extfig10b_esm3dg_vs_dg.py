"""ExtFig10b: predicted absolute dG (ESM3dG) vs measured dG (de novo / mutational library; measured 95% CI <= 0.5).

Tables: S04 (measured dG, 95% CI, source), S05 (deltaG_pred = ESM3dG).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from scipy import stats
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig4")
s4 = load("S04")[["sublib", "name", "source_dms_denovo", "deltaG_measured_kcal_mol", "deltaG_95CI"]]
s5 = load("S05", usecols=["sublib", "name", "deltaG_pred"])
df = s4.merge(s5, on=["sublib", "name"], how="left")
df = df[~df["name"].str.contains(".pdb")]   # original filter (regex: any character + "pdb")
dark_blues = mcolors.LinearSegmentedColormap.from_list("dark_blues", mpl.colormaps["Blues"](np.linspace(0.25, 1.0, 256)))

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
for i, dt in enumerate(["denovo", "dms"], 1):
    d = df[(df["source_dms_denovo"] == dt) & (df["deltaG_95CI"] <= 0.5)].dropna(subset=["deltaG_pred", "deltaG_measured_kcal_mol"])
    x, y = d["deltaG_pred"].to_numpy(), d["deltaG_measured_kcal_mol"].to_numpy()
    r, rho = stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]
    print(f"ExtFig10b {dt}: n={len(d)} r={r:.3f} rho={rho:.3f}")
    ax = axes[i - 1]
    ax.hexbin(x, y, gridsize=100, mincnt=1, bins="log", cmap=dark_blues, alpha=1.0)
    ax.text(0.05, 0.95, f"r = {r:.2f}\nρ = {rho:.2f}\nn = {len(d)}", transform=ax.transAxes, va="top", fontsize=6)
    ax.set_title({"denovo": "De novo", "dms": "Mutational library"}[dt], fontsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("ΔG (kcal/mol)" if i == 1 else "", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.tick_params(axis="both", which="major", pad=1)
fig.supxlabel("Predicted absolute ΔG (kcal/mol)\nESM3ΔG", fontsize=6)
plt.savefig(os.path.join(OUT, "ExtFig10b.pdf"))
