"""ExtFig 5c: ECDF of monomer yield after filtering with increasing pLDDT / TM-score cutoffs, for AF3-fast and AF2.
Tables: S04, S17."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _fig2_data import load_designs, outdir, set_style

set_style()
OUT = outdir("fig2")
PLDDT_TH = list(range(10, 100, 10))
TM_TH = [round(x, 2) for x in np.arange(0.1, 1, 0.1)]


def ecdf_by_cutoff(df, col, thresholds, title, fname, lw, legend_kw, xticks=None):
    parts = [df[["monomer_log_mgL"]].dropna().assign(filter="all").rename(columns={"monomer_log_mgL": "value"})]
    for th in thresholds:
        parts.append(df.loc[df[col] > th, ["monomer_log_mgL"]].dropna().assign(filter=f">{th}")
                     .rename(columns={"monomer_log_mgL": "value"}))
    plot_df = pd.concat(parts, ignore_index=True)
    order = ["all"] + [f">{th}" for th in thresholds]
    palette = dict(zip(order, sns.color_palette("crest", n_colors=len(order))))
    fig = plt.figure(figsize=(2.5 / 2.54, 2.5 / 2.54), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    sns.ecdfplot(data=plot_df, x="value", hue="filter", hue_order=order, palette=palette, legend=True, lw=lw, ax=ax)
    sns.move_legend(ax, loc="upper left", title="", bbox_to_anchor=(1, 1), frameon=False, **legend_kw)
    ax.set_xlabel(r"log$_{10}$ (monomeric yield)" + "\n(mg/L)")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.set_ylim(0, 1.01)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.tick_params(axis="both", which="major", pad=1)
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


for af, name, plddt_title, tm_title in [("af3", "AF3", "pLDDT", "TMscore"), ("af2", "AF2", "pLDDT (AF2)", "TMscore (AF2)")]:
    df = load_designs(af)
    ecdf_by_cutoff(df, "mean_plddt", PLDDT_TH, plddt_title, f"ExtFig5c-{name}-plddt.pdf", 0.8, dict(fontsize=6))
    ecdf_by_cutoff(df, "tmscore", TM_TH, tm_title, f"ExtFig5c-{name}-tmscore.pdf", 0.5, dict(title_fontsize=6), xticks=[-1, 0, 1, 2])
