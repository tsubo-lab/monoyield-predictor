"""Fig 2i: number of designs per method in four yield classes (log10 mg/L: [-2,-1), [-1,0), [0,1), [1,2)),
all designs (top) and designs passing the AF3-fast filter pLDDT > 90 & TM-score > 0.8 (bottom), one two-row panel.
Methods sorted by the fraction (top) / number (bottom) of designs in the highest class. wt entries excluded. Tables: S04, S17."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _fig2_data import load_designs, drop_natural, outdir, set_style

set_style()
OUT = outdir("fig2")
BINS = [-2, -1, 0, 1, 2]
HUE_ORDER = [f"Group{i}" for i in range(4)]
PALETTE = dict(zip(HUE_ORDER, sns.color_palette("Blues", 4)))

df = drop_natural(load_designs("af3"))
df["bin_monomer"] = pd.cut(df["monomer_log_mgL"], bins=BINS, labels=HUE_ORDER, include_lowest=True, right=False)

fig, axes = plt.subplots(2, 1, figsize=(5 / 2.54, 7.8 / 2.54), dpi=300, gridspec_kw={"hspace": 0.3})
for ax, (data, by_fraction) in zip(axes, [(df, True), (df[df["is_filtered"]].copy(), False)]):
    table = pd.crosstab(data["method_a"], data["bin_monomer"])
    print("top" if by_fraction else "bottom", "\n", table.to_string())
    ranking = table.div(table.sum(axis=1), axis=0) if by_fraction else table
    method_order = ranking.sort_values("Group3", ascending=False).index.tolist()
    data["method_a"] = pd.Categorical(data["method_a"], categories=method_order, ordered=True)
    sns.histplot(data=data, y="method_a", hue="bin_monomer", multiple="stack", stat="count", discrete=True, shrink=0.8,
                 hue_order=HUE_ORDER, palette=PALETTE, legend=False, ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("Sequences", labelpad=1)
    ax.set_xlim(0, 1750)
    ax.tick_params(axis="both", which="major", pad=1)
    if by_fraction:
        # class labels above the segments of the top bar (stacked from the highest class)
        w = table.loc[method_order[0], ["Group3", "Group2", "Group1", "Group0"]].to_numpy()
        centers = w.cumsum() - w / 2
        y0, y1 = ax.get_ylim()
        edges = [e for r in ax.patches for e in (r.get_y(), r.get_y() + r.get_height())]
        ytop = max(edges) if y1 > y0 else min(edges)       # upper edge of the top bar
        for c, lab, dx in zip(centers, [">1", "0-1", "-1-0", "<-1"], [0, 0, -5, 3]):
            ax.annotate(lab, xy=(c, ytop), xycoords="data", xytext=(dx, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=0.5, shrinkB=0))
        ax.set_title(r"log$_{10}$(monomeric yield) (mg/L)", fontsize=6, pad=12)
    else:
        ax.text(0.97, 0.45, "pLDDT>90\nTMscore>0.8", transform=ax.transAxes, ha="right", va="center", fontsize=6)
fig.subplots_adjust(left=0.42, right=0.97, top=0.9, bottom=0.08)
fig.savefig(os.path.join(OUT, "Fig2i.pdf"))
plt.close(fig)
