"""Fig 2b: monomer yield of every design, per backbone (strip plot, backbones sorted by median yield,
coloured by secondary-structure group). Tables: S04 (yield), S13 (backbone label / SS group), S17 (design list)."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from _fig2_data import load_designs, drop_natural, load_backbones, outdir, set_style

set_style()
df = drop_natural(load_designs())
bb = load_backbones(df)
df = df.merge(bb[["target", "target1", "ss_group"]], on="target")

order = df.groupby("target1")["monomer_log_mgL"].median().sort_values(ascending=False).index
hue_values = ["all alpha", "mixed", "all beta"]
palette = dict(zip(hue_values, sns.color_palette("cool", 3)))

fig = plt.figure(tight_layout=True, figsize=(12.5 / 2.54, 5 / 2.54), dpi=300)
ax = fig.add_subplot(1, 1, 1)
sns.stripplot(data=df, x="target1", y="monomer_log_mgL", hue="ss_group", order=order, hue_order=hue_values,
              palette=palette, alpha=0.1, s=2, dodge=False, legend=True, ax=ax)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=90)
ax.set_ylabel("monomer_log_mgL")
ax.set_xlabel("")
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(outdir("fig2"), "Fig2b.pdf"), bbox_inches="tight")

print(f"n designs with yield = {df['monomer_log_mgL'].notna().sum()}, backbones = {df['target1'].nunique()}")
print(df.groupby("ss_group")["target1"].nunique().to_dict())
