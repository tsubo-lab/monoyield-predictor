"""Fig 2e (AF3-fast pLDDT / TM-score boxplots vs yield), ExtFig 5e (same for AF2) and ExtFig 5d (density
scatter, AF3-fast and AF2). All de novo designs with a measured yield (wt entries included). Tables: S04, S17."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from _fig2_data import load_designs, outdir, set_style

set_style()
OUT = outdir("fig2")
PLDDT_BINS = [0] + list(range(50, 105, 10))
TM_BINS = [0] + list(np.round(np.arange(0.5, 1.05, 0.1), 1))
YLABEL = r"log$_{10}$ (monomeric yield)" + "\n(mg/L)"


def bin_labels(bins):
    return [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]


def boxplot(df, name):
    df = df.copy()
    df["bin_plddt"] = pd.cut(df["mean_plddt"], bins=PLDDT_BINS, labels=bin_labels(PLDDT_BINS), include_lowest=True, right=False)
    df["bin_tm"] = pd.cut(df["tmscore"], bins=TM_BINS, labels=bin_labels(TM_BINS), include_lowest=True, right=False)
    fig = plt.figure(figsize=(7.5 / 2.54, 3 / 2.54), tight_layout=True, dpi=300)
    yticks = np.arange(-3, 11, 1)
    for i, (bin_col, xlabel, xtl) in enumerate([
            ("bin_plddt", name + " pLDDT", ["0"] + list(range(50, 105, 10))),
            ("bin_tm", name + " prediction based TM-score", ["0"] + [str(x) for x in TM_BINS[1:]])], start=1):
        ax = fig.add_subplot(1, 2, i)
        sns.boxplot(df.sort_values(bin_col), x=bin_col, y="monomer_log_mgL", palette="crest", linewidth=0.5,
                    fliersize=0.5, gap=0.2, hue=bin_col, legend=False, dodge=False, ax=ax)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylabel(YLABEL if i == 1 else "", labelpad=2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_ylim(-1.7, 2)
        ax.set_xticks(np.arange(-0.5, 6, 1))
        ax.set_xticklabels(xtl)
        ax.tick_params(axis="both", which="major", pad=1)
    return fig


def density_scatter(df, name, size):
    cmap = mcolors.LinearSegmentedColormap.from_list("dark_blues", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    fig = plt.figure(figsize=size, dpi=300)
    for i, (col, xlabel, xlim, ticks) in enumerate([
            ("mean_plddt", f"pLDDT ({name})", (40, 100), np.arange(40, 101, 10)),
            ("tmscore", f"TMscore ({name})", (0.2, 1), np.arange(0.2, 1.1, 0.2))], start=1):
        ax = fig.add_subplot(1, 2, i)
        ax.hexbin(df[col], df["monomer_log_mgL"], gridsize=100, mincnt=1, bins="log", cmap=cmap)
        ax.tick_params(axis="both", which="major", pad=1)
        ax.set_xlabel(xlabel, labelpad=1)
        ax.set_ylabel(r"log$_{10}$ (monomeric yield)" + "\n(mg/L)" if i == 1 else ("" if name == "AF3" else "\n"), labelpad=1)
        ax.set_xticks(ticks)
        ax.set_xlim(*xlim)
        ax.set_ylim(-1.6, 2)
    return fig


for af, name, box_file in [("af3", "AF3", "Fig2e.pdf"), ("af2", "AF2", "ExtFig5e.pdf")]:
    df = load_designs(af)
    print(f"{name}: n designs = {len(df)}, with yield = {df['monomer_log_mgL'].notna().sum()}")
    fig = boxplot(df, name)
    fig.savefig(os.path.join(OUT, box_file))
    fig = density_scatter(df, name, (7 / 2.54, 3 / 2.54))
    fig.savefig(os.path.join(OUT, f"ExtFig5d-{name}.pdf"), bbox_inches="tight")
    plt.close("all")
