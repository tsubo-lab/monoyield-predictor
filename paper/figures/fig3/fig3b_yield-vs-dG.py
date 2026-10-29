"""Fig3b: monomeric yield vs measured dG of the DMS library (hexbin, log counts).
Only variants with a reliable dG (deltaG_95CI <= CI_MAX) are shown.
Tables: S04 (monomer_log_mgL, deltaG_measured_kcal_mol, deltaG_95CI)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import load, outdir
from common.style import set_style
import dms_data
import dms_plots

import matplotlib.colors as mcolors
OUT = outdir("fig3")
set_style()

df_gv, _, _ = dms_data.heatmap_view()
CI_MAX = 0.5
plot_df = df_gv[(df_gv["source"] == "dms") & (df_gv["deltaG_95CI"] <= CI_MAX)].dropna(subset=["deltaG", "monomer_log_mgL"])
x, y = plot_df["deltaG"].values, plot_df["monomer_log_mgL"].values
r = stats.pearsonr(x, y)[0]
print(f"DMS: n = {len(plot_df)}, Pearson r = {r:.3f}, Spearman rho = {stats.spearmanr(x, y)[0]:.3f}")

dark_blues = mcolors.LinearSegmentedColormap.from_list("dark_blues", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
fig = plt.figure(figsize=(3 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
ax.hexbin(x, y, gridsize=100, mincnt=1, bins="log", cmap=dark_blues, alpha=1.0)
ax.text(0.05, 0.95, f"r = {r:.2f}\nn = {len(plot_df)}", transform=ax.transAxes, va="top", fontsize=6)
ax.set_title("DMS", fontsize=6)
ax.set_xlabel("ΔG (kcal/mol)", fontsize=6, labelpad=1)
ax.set_ylabel(r"log$_{10}$ (monomeric yield) (mg/L)", labelpad=1)
ax.set_xticks(np.arange(-2, 11, 2))
ax.set_yticks(np.arange(-3, 11, 1))
ax.set_xlim(-2, 6)
ax.set_ylim(-1.7, 2.5)
ax.tick_params(labelsize=6)
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(OUT, "Fig3b.pdf"))
