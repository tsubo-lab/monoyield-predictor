"""Fig3c: mean change of monomeric yield vs mean ddG (mutant - WT) for each substituted ("target") / original ("source") amino acid.
Tables: S04 (yield, deltaG of DMS variants and WT entries)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import load, outdir
from common.style import set_style
import dms_data
import dms_plots

OUT = outdir("fig3")
set_style()

order = list("qenhdrktsagmclviwyfp") + ["del", "ins", "wt"]
groups = [list("qenhdrk"), list("tsag"), list("mclvi"), list("wyf"), ["p"], ["del", "ins"], ["wt"]]
colors = sns.color_palette("husl", len(groups))
palette = {m: c for c, g in zip(colors, groups) for m in g}
palette = {m: palette[m] for m in order}

df = dms_data.mutation_deltas(dms_data.heatmap_view()[0])
delta_cols = ["delta_deltaG", "delta_monomer_log_mgL"]
mean_by_mut = df.groupby("mut", as_index=False)[delta_cols].mean()
mean_by_ori = df.groupby("ori", as_index=False)[delta_cols].mean()
mean_by_ori = mean_by_ori[~mean_by_ori["ori"].str.contains("del|ins")]

for key, tab, name in [("mut", mean_by_mut, "target"), ("ori", mean_by_ori, "source")]:
    fig = plt.figure(figsize=(3.5 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    sns.scatterplot(data=tab, x="delta_deltaG", y="delta_monomer_log_mgL", hue=key, s=10, palette=palette, alpha=0.8, legend=False, ax=ax)
    for _, row in tab.iterrows():
        ax.text(row["delta_deltaG"] + 0.01, row["delta_monomer_log_mgL"] + 0.01, str(row[key].upper()), fontsize=4)
    sns.regplot(data=tab, x="delta_deltaG", y="delta_monomer_log_mgL", scatter=False, ci=None, line_kws={"lw": 0.5, "color": "grey"}, ax=ax)
    ax.axhline(y=0, lw=0.5, c="grey", ls="--")
    ax.axvline(x=0, lw=0.5, c="grey", ls="--")
    ax.set_title("mean Δ(mutant-WTs)")
    ax.set_xlabel("ΔΔG (kcal/mol)", labelpad=1)
    ax.set_ylabel(r"Δlog$_{10}$(monomeric yield)", labelpad=1)
    ax.tick_params(axis="both", which="major", pad=1)
    r = stats.pearsonr(tab["delta_deltaG"], tab["delta_monomer_log_mgL"])[0]
    print(f"Fig3c-{name}: {len(tab)} groups, Pearson r (mean ddG vs mean dyield) = {r:.3f}")
    fig.savefig(os.path.join(OUT, f"Fig3c-{name}.pdf"))
