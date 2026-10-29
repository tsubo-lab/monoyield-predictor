"""ExtFig6a: distribution (median, 20-80 % box) of delta yield and ddG (mutant - WT) per substituted ("target") and original ("source") amino acid.
Tables: S04."""
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

aa = list("QENHDRKTSAGMCLVIWYFP")
groups = [list("qenhdrk"), list("tsag"), list("mclvi"), list("wyf"), ["p"], ["del", "ins"], ["wt"]]
colors = sns.color_palette("husl", len(groups))
all_order = list("qenhdrktsagmclviwyfp") + ["del", "ins", "wt"]
palette = {m: c for c, g in zip(colors, groups) for m in g}
palette = {m: palette[m] for m in all_order}

df = dms_data.mutation_deltas(dms_data.heatmap_view()[0])
for key, name, extra in [("mut", "target", ["del", "ins", "wt"]), ("ori", "source", ["wt"])]:
    order = list("qenhdrktsagmclviwyfp") + extra
    df[key] = pd.Categorical(df[key], categories=order, ordered=True)
    for col in ["delta_monomer_log_mgL", "delta_deltaG"]:
        fig = plt.figure(figsize=(6.5 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        lower = df.groupby(key, observed=False)[col].quantile(0.2).dropna().min()
        upper = df.groupby(key, observed=False)[col].quantile(0.8).dropna().max()
        sns.boxplot(data=df, x=key, y=col, hue=key, dodge=False, order=order, palette=palette, linewidth=0.5, fliersize=0,
                    saturation=1, medianprops={"color": "black", "linewidth": 0.7}, legend=False, whis=0, ax=ax)
        ax.axhline(0, color="grey", lw=0.5, linestyle="--")
        ax.set_xlabel("")
        ax.set_ylabel("Δ(mutant-WTs)")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(aa + extra)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90 if tick.get_text() in ["del", "ins", "wt"] else 0)
            tick.set_ha("center")
        ax.set_ylim(lower * 1.01, upper * 1.01)
        ax.set_title(col)
        ax.tick_params(axis="both", which="major", pad=1)
        med = df.groupby(key, observed=False)[col].median()
        print(f"ExtFig6a-{name}-{'yield' if 'monomer' in col else 'ddG'}: n = {df[col].notna().sum()}, median range {med.min():.2f} to {med.max():.2f}")
        fig.savefig(os.path.join(OUT, f"ExtFig6a-{name}-{'yield' if 'monomer' in col else 'ddG'}.pdf"))
