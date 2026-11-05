"""ExtFig6a: distribution (median, 20-80 % box) of delta yield and ddG (mutant - WT) per substituted ("target") and original ("source") amino acid.
One 2 x 2 panel (columns: target / source amino acid; rows: delta yield / ddG); W, Y, F boxed. Tables: S04."""
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
COLS = [("mut", "target", ["del", "ins", "wt"], "Aggregated by\ntarget AA"), ("ori", "source", ["wt"], "Aggregated by\nsource AA")]
ROWS = [("delta_monomer_log_mgL", "yield", r"log$_{10}$(mutant yield / WT yield)"), ("delta_deltaG", "ddG", r"$\Delta\Delta$G (kcal/mol)")]
fig, axes = plt.subplots(2, 2, figsize=(15 / 2.54, 7.5 / 2.54), dpi=300, gridspec_kw={"hspace": 0.45, "wspace": 0.3})
fig.subplots_adjust(left=0.25, right=0.99, top=0.88, bottom=0.08)
for j, (key, name, extra, ctitle) in enumerate(COLS):
    order = list("qenhdrktsagmclviwyfp") + extra
    df[key] = pd.Categorical(df[key], categories=order, ordered=True)
    for i, (col, rname, rlabel) in enumerate(ROWS):
        ax = axes[i, j]
        lower = df.groupby(key, observed=False)[col].quantile(0.2).dropna().min()
        upper = df.groupby(key, observed=False)[col].quantile(0.8).dropna().max()
        sns.boxplot(data=df, x=key, y=col, hue=key, dodge=False, order=order, palette=palette, linewidth=0.5, fliersize=0,
                    saturation=1, medianprops={"color": "black", "linewidth": 0.7}, legend=False, whis=0, ax=ax)
        ax.axhline(0, color="grey", lw=0.5, linestyle="--")
        ax.set_xlabel("")
        ax.set_ylabel("Δ(mutant-WTs)", labelpad=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(aa + extra)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90 if tick.get_text() in ["del", "ins", "wt"] else 0)
            tick.set_ha("center")
        ax.set_ylim(lower * 1.01, upper * 1.01)
        if i == 0:
            ax.set_title(ctitle, fontsize=6, pad=3)
        ax.tick_params(axis="both", which="major", pad=1)
        med = df.groupby(key, observed=False)[col].median()
        print(f"ExtFig6a-{name}-{rname}: n = {df[col].notna().sum()}, median range {med.min():.2f} to {med.max():.2f}")

# row labels on the left
for i, (_, _, rlabel) in enumerate(ROWS):
    bb = axes[i, 0].get_position()
    fig.text(0.01, (bb.y0 + bb.y1) / 2, rlabel, fontsize=6, ha="left", va="center")

# red boxes around the W Y F tick labels
fig.canvas.draw()
inv = fig.transFigure.inverted()
for ax in axes.ravel():
    boxes = [inv.transform_bbox(t.get_window_extent()) for t in ax.get_xticklabels() if t.get_text() in ("W", "Y", "F")]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    pad = 0.004
    fig.add_artist(plt.Rectangle((x0 - pad, y0 - pad), x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, transform=fig.transFigure,
                                 fill=False, edgecolor="#D7191C", linewidth=0.8))
fig.savefig(os.path.join(OUT, "ExtFig6a.pdf"))
plt.close(fig)
