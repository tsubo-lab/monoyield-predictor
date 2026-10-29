"""Fig 2i: number of designs per method in four yield classes (log10 mg/L: [-2,-1), [-1,0), [0,1), [1,2)),
all designs (top) and designs passing the AF3-fast filter pLDDT > 90 & TM-score > 0.8 (bottom).
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

for data, fname, by_fraction in [(df, "Fig2i-top.pdf", True), (df[df["is_filtered"]].copy(), "Fig2i-bottom.pdf", False)]:
    table = pd.crosstab(data["method_a"], data["bin_monomer"])
    print(fname, "\n", table.to_string())
    ranking = table.div(table.sum(axis=1), axis=0) if by_fraction else table
    method_order = ranking.sort_values("Group3", ascending=False).index.tolist()
    data["method_a"] = pd.Categorical(data["method_a"], categories=method_order, ordered=True)
    fig = plt.figure(tight_layout=True, figsize=(5 / 2.54, 4 / 2.54), dpi=300)
    sns.histplot(data=data, y="method_a", hue="bin_monomer", multiple="stack", stat="count", discrete=True, shrink=0.8,
                 hue_order=HUE_ORDER, palette=PALETTE, legend=False)
    plt.ylabel("")
    plt.xlabel("Sequences")
    plt.xlim(0, 1750)
    plt.tick_params(axis="both", which="major", pad=1)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
