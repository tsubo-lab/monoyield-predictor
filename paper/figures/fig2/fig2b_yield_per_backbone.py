"""Fig 2b: monomer yield of every design, per backbone (strip plot, backbones sorted by median yield,
coloured by secondary-structure group). Tables: S04 (yield), S13 (backbone label / SS group), S17 (design list)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from _fig2_data import load_designs, drop_natural, load_backbones, outdir, set_style

set_style()
df = drop_natural(load_designs())
bb = load_backbones(df)
df = df.merge(bb[["target", "target1", "ss_group"]], on="target")

col, target_col = "monomer_log_mgL", "target1"
medians = df.groupby(target_col, observed=True)[col].median().dropna().sort_values(ascending=False, kind="stable")
order = medians.index.tolist()
groups = ["all alpha", "mixed", "all beta"]
colors = {"all alpha": "#0072B2", "mixed": "#E69F00", "all beta": "#A02C7D"}
labels = ["all α", "αβ", "all β"]

fig, ax = plt.subplots(figsize=(12.5 / 2.54, 5 / 2.54), dpi=300)
sns.stripplot(data=df, x=target_col, y=col, hue="ss_group", order=order, hue_order=groups, palette=colors,
              jitter=0.18, dodge=False, alpha=0.10, size=2, linewidth=0, legend=False, ax=ax)
x = np.arange(len(order))
ax.hlines(medians.to_numpy(), x - 0.32, x + 0.32, color="black", linewidth=0.65, zorder=5, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=90, ha="center", va="top")
ax.set_xlabel("")
ax.set_ylabel(r"log$_{10}$(monomeric yield)" + "(mg/L)", labelpad=0)
ax.set_ylim(-1.7, 2.05)
ax.set_yticks([-1, 0, 1, 2])
ax.set_xlim(-0.6, len(order) - 0.4)
ax.tick_params(axis="both", which="major", pad=1, width=0.5, length=2)
ax.spines[["top", "right"]].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(0.5)
handles = [Line2D([], [], linestyle="none", label=label) for label in labels]
legend = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=False,
                   handlelength=0, handletextpad=0, columnspacing=2.5, borderaxespad=0)
for text, group in zip(legend.get_texts(), groups):
    text.set_color(colors[group])
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.49, top=0.88)
fig.savefig(os.path.join(outdir("fig2"), "Fig2b.pdf"), bbox_inches="tight")

print(f"n designs with yield = {df['monomer_log_mgL'].notna().sum()}, backbones = {df['target1'].nunique()}")
print(df.groupby("ss_group")["target1"].nunique().to_dict())
