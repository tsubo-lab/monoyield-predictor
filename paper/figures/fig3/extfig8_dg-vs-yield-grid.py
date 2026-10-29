"""ExtFig8: dG vs monomeric yield for each protein of the partial DMS libraries (denovo-8, muts-1), with per-protein linear fit, r and slope.
Tables: S04 (dG kept only where deltaG_95CI <= 0.5)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from common.tables import outdir
from common.style import set_style
import dms_data
import dms_plots

import numpy as np
import seaborn as sns
from textwrap import fill
OUT = outdir("fig3")
set_style()

df_dms, df_wt, df_pr = dms_data.fit_view()
proteins = df_wt.loc[df_wt["sublib"].str.contains("denovo|mut", na=False), "name"]
ncols = 7
nrows = max(1, int(np.ceil(len(proteins) / ncols)))
fig = plt.figure(figsize=(16 / 2.54, 20 / 2.54), dpi=300)

for i, protein in enumerate(proteins, start=1):
    lib = df_dms.loc[df_dms["name"] == protein, "sublib"].iloc[0]
    sub = df_dms.loc[df_dms["name"].str.contains(protein, na=False) & df_dms["sublib"].eq(lib), ["deltaG", "monomer_log_mgL"]].dropna()
    ax = fig.add_subplot(nrows, ncols, i)
    sns.scatterplot(data=sub, x="deltaG", y="monomer_log_mgL", s=3, alpha=0.7, ax=ax)
    fit = df_pr.loc[df_pr["name"] == protein]
    text = "r = NA\nslope = NA"
    if not fit.empty:
        fit = fit.iloc[0]
        text = f"r = {fit['pearsonR']:.2f}\nslope = {fit['slope']:.2f}"
        if not sub.empty:
            xs = np.array([sub["deltaG"].min(), sub["deltaG"].max()])
            ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="black", linewidth=0.7, zorder=3)
    ax.text(0.05, 1.05, text, transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax.set_title(fill(protein.upper(), width=16), fontsize=6)
    ax.set_xticks(np.arange(-1, 7, 1))
    ax.set_yticks(np.arange(-1, 3, 1))
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1.7, 2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", pad=1, labelsize=6)

print(f"ExtFig8: {len(proteins)} proteins, {len(df_pr)} with a fit")
fig.supxlabel(r"$\Delta G$ (kcal/mol)", fontsize=6, y=0.05)
fig.supylabel(r"Monomeric yield (log$_{10}$ mg/L)", fontsize=6, x=0.05)
fig.tight_layout(rect=(0.04, 0.04, 1, 1), pad=0.6, w_pad=0.7, h_pad=1.0)
fig.savefig(os.path.join(OUT, "ExtFig8.pdf"), bbox_inches="tight")
