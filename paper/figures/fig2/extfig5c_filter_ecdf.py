"""ExtFig 5c: ECDF of monomer yield after filtering with increasing pLDDT / TM-score cutoffs, AF3-fast (top) and AF2 (bottom), in one figure.
Tables: S04, S17."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from _fig2_data import load_designs, outdir, set_style

set_style()
OUT = outdir("fig2")
col = "monomer_log_mgL"
specs = {
    "pLDDT": dict(key="mean_plddt", thresholds=list(range(10, 100, 10)), palette="crest"),
    "TMscore": dict(key="tmscore", thresholds=[round(x, 2) for x in np.arange(0.1, 1, 0.1)], palette="flare"),
}
designs = {af: load_designs(af) for af in ("af3", "af2")}

fig = plt.figure(figsize=(10 / 2.54, 6.5 / 2.54), dpi=300)
outer = GridSpec(1, 2, figure=fig, wspace=0.2)
for ci, (name, sp) in enumerate(specs.items()):
    gs = outer[0, ci].subgridspec(2, 2, width_ratios=[1, 0.4], wspace=0.05, hspace=0.75)
    order = ["all"] + [f">{th}" for th in sp["thresholds"]]
    pal = dict(zip(order, sns.color_palette(sp["palette"], n_colors=len(order))))
    for ri, af in enumerate(["af3", "af2"]):
        df = designs[af]
        ax = fig.add_subplot(gs[ri, 0])
        parts = [df[[col]].dropna().assign(filter="all").rename(columns={col: "value"})]
        for th in sp["thresholds"]:
            parts.append(df.loc[df[sp["key"]] > th, [col]].dropna().assign(filter=f">{th}").rename(columns={col: "value"}))
        sns.ecdfplot(data=pd.concat(parts, ignore_index=True), x="value", hue="filter", hue_order=order, palette=pal, legend=False, lw=0.8, ax=ax)
        ax.set_title(f"{name} ({af.upper()})", fontsize=6)
        ax.set_ylim(0, 1.01)
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_ylabel("ECDF", labelpad=1)
        ax.set_xlabel(r"log$_{10}$ (monomeric yield)" + "\n(mg/L)", labelpad=1)
        ax.tick_params(axis="both", which="major", pad=1)
    lax = fig.add_subplot(gs[:, 1])
    lax.axis("off")
    handles = [Line2D([0], [0], color=pal[o], lw=0.8) for o in order]
    lax.legend(handles, order, loc="center left", frameon=False, fontsize=6, handlelength=1.5, borderaxespad=0, labelspacing=0.4)
fig.savefig(os.path.join(OUT, "ExtFig5c.pdf"), bbox_inches="tight")
