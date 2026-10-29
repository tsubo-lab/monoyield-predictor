"""Filter performance of structure-prediction cutoffs (positive = monomer yield > 10 mg/L, i.e. log10 > 1):
Fig 2f (AF3-fast pLDDT cutoffs at TM-score > 0.8), Fig 2g (ECDF all vs filtered), Fig 2h (per-method arrows),
ExtFig 5f (AF2 pLDDT cutoffs) and ExtFig 5g (number of backbones with >10 mg/L designs before / after the AF3 filter).
All designs with a measured yield (wt entries included). Tables: S04, S17."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from _fig2_data import load_designs, outdir, set_style, TWO

set_style()
OUT = outdir("fig2")
COL = "monomer_log_mgL"
POS_TH = 1            # log10(mg/L) > 1  -> >10 mg/L
TM_CUT = 0.80
PLDDT_CUTS = [60, 70, 80, 90, 95]
HUE_ORDER = ["ProteinMPNN-T0.1", "ProteinMPNN-T0.3", "ProteinMPNN-Ca", "SolubleMPNN", "HyperMPNN", "TIMED-Design",
             "ESM3", "ESM-IF1", "Carbon design", "Frame2seq", "Boltz-if"]
HUE_PALETTE = dict(zip(HUE_ORDER, sns.color_palette("husl", 11)))
METHOD_COLOR = {"Boltz-if": "#FF59CD", "ProteinMPNN-T0.3": "#F28E2B", "Carbon design": "#9A88FF", "HyperMPNN": "#16BF55",
                "ProteinMPNN-T0.1": "#FF7099", "ESM-IF1": "#32ABD4", "SolubleMPNN": "#86B820", "Frame2seq": "#EE62F0",
                "ProteinMPNN-Ca": "#B59B16", "ESM3": "#00ADBE", "TIMED-Design": "#20B59E"}
LABEL_POS = {"Boltz-if": (0.80, 0.205), "ProteinMPNN-T0.3": (1.03, 0.181), "Carbon design": (1.03, 0.163),
             "HyperMPNN": (1.03, 0.145), "ProteinMPNN-T0.1": (1.03, 0.127), "ESM-IF1": (1.03, 0.109),
             "SolubleMPNN": (1.03, 0.091), "Frame2seq": (1.03, 0.073), "ProteinMPNN-Ca": (1.03, 0.055),
             "ESM3": (0.55, 0.065), "TIMED-Design": (1.03, 0.026)}


def pct_axes(ax):
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def cutoff_table(df):
    """Positive rate / recall for 'all' and for each pLDDT cutoff combined with TM-score > 0.8."""
    ev = df.dropna(subset=[COL]).copy()
    ev["positive"] = ev[COL] > POS_TH
    total_pos = ev["positive"].sum()
    rows = [dict(condition="all", plddt_filter=np.nan, n=len(ev), positive_rate=ev["positive"].mean(), recall=1.0)]
    for c in sorted(PLDDT_CUTS, reverse=True):   # drawing order of the original table (95 ... 60)
        sub = ev[(ev["mean_plddt"] > c) & (ev["tmscore"] > TM_CUT)]
        rows.append(dict(condition=f"tm80_pl{c}", plddt_filter=c, n=len(sub), positive_n=sub["positive"].sum(),
                         positive_rate=sub["positive"].mean(), recall=sub["positive"].sum() / total_pos,
                         target_n=sub["target"].nunique()))
    return pd.DataFrame(rows)


def method_table(df):
    """Per design method: before / after the pLDDT > 90 & TM-score > 0.8 filter."""
    rows = []
    for method, sub in df.groupby("method_a"):
        if method == "Natural/Rossetta":
            continue
        ev = sub.dropna(subset=[COL]).copy()
        ev["positive"] = ev[COL] > POS_TH
        f = ev[(ev["mean_plddt"] > 90) & (ev["tmscore"] > TM_CUT)]
        rows.append(dict(method=method, n=len(f), target_n=f["target"].nunique(),
                         target_n_positive=f[f["positive"]]["target"].nunique(),
                         target_n_positive_nofilter=ev[ev["positive"]]["target"].nunique(),
                         positive_n=f["positive"].sum(), positive_rate=f["positive"].mean(),
                         recall=f["positive"].sum() / ev["positive"].sum(), base_positive_rate=ev["positive"].mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Fig 2f (AF3-fast)
df3 = load_designs("af3")
res = cutoff_table(df3)
print("AF3-fast cutoff table\n", res.round(3).to_string(index=False))
plot_df = res.query("condition != 'all'")
palette = dict(zip(PLDDT_CUTS, ["#83C5A6", "#63B3A5", "#4C9FA8", "#388CA7", "#4B75A3"]))
base_rate = res.loc[res["condition"] == "all", "positive_rate"].iloc[0]

fig = plt.figure(figsize=(5.2 / 2.54, 4 / 2.54), dpi=300)
ax = fig.add_axes([0.28, 0.23, 0.36, 0.66])
for c in PLDDT_CUTS:
    sub = plot_df.loc[plot_df["plddt_filter"] == c]
    ax.scatter(sub["recall"], sub["positive_rate"], s=30, color=palette[c], edgecolors="none", label=f">{c}", zorder=3)
ax.axhline(base_rate, linestyle="--", linewidth=0.6, color="gray", zorder=1)
ax.annotate("base line", xy=(0.59, base_rate), xytext=(0, -2), textcoords="offset points", ha="left", va="top", fontsize=6)
ax.set(xlim=(0, 1), ylim=(0, 0.20), xticks=[0, 0.5, 1], yticks=[0, 0.05, 0.10, 0.15, 0.20])
pct_axes(ax)
ax.set_xlabel("Recall of sequences with yield >10 mg/L", fontsize=6, labelpad=2)
ax.set_ylabel("Positive rate after filtering\n(yield >10 mg/L)", fontsize=6, labelpad=1)
ax.tick_params(axis="both", labelsize=6, direction="out", pad=1, length=3, width=0.6)
ax.grid(False)
sns.despine(ax=ax)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(0.6)
ax.legend(title="TMscore>0.8\npLDDT cutoff", loc="upper left", bbox_to_anchor=(1.01, 0.98), frameon=False, fontsize=6,
          title_fontsize=6, borderaxespad=0, labelspacing=0.4, handletextpad=0.4)
fig.savefig(os.path.join(OUT, "Fig2f.pdf"), bbox_inches="tight")

# ------------------------------------------------------------------ Fig 2g
mask = (df3["mean_plddt"] > 90) & (df3["tmscore"] > TM_CUT)
plot_df = pd.concat([df3[[COL]].dropna().assign(filter="all").rename(columns={COL: "value"}),
                     df3.loc[mask, [COL]].dropna().assign(filter="filtered").rename(columns={COL: "value"})],
                    ignore_index=True)
order = ["all", "filtered"]
pal = dict(zip(order, TWO))
fig = plt.figure(figsize=(5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.ecdfplot(data=plot_df, x="value", hue="filter", hue_order=order, palette=pal, lw=0.5, ax=ax)
sns.move_legend(ax, loc="upper left", title="", bbox_to_anchor=(0, 1), frameon=False, fontsize=6)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
for name in order:
    values = plot_df.loc[plot_df["filter"] == name, "value"]
    for x0 in [0, 1]:
        y0 = (values <= x0).mean()
        ax.vlines(x=x0, ymin=0, ymax=y0, colors=pal[name], linestyles="--", linewidth=0.5)
        ax.annotate(f"{y0:.3f}", xy=(x0, y0), xytext=(-3, 3) if name == "all" else (3, 3), textcoords="offset points",
                    ha="right" if name == "all" else "left", va="bottom", fontsize=6, color=pal[name])
        print(f"Fig2g ECDF({name}) at log10 yield <= {x0}: {y0:.4f}  (n = {len(values)})")
ax.set_ylim(bottom=0)
ax.tick_params(axis="both", which="major", pad=1)
ax.set_xlabel(r"log$_{10}$ (monomeric yield)" + "\n(mg/L)", labelpad=1)
ax.set_ylabel("ECDF", labelpad=1)
fig.savefig(os.path.join(OUT, "Fig2g.pdf"))

# ------------------------------------------------------------------ Fig 2h and ExtFig 5g (per method)
mres = method_table(df3)
print("per-method table\n", mres.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(5 / 2.54, 4.5 / 2.54), dpi=300)
fig.subplots_adjust(left=0.24, right=0.84, bottom=0.23, top=0.85)
for _, row in mres.iterrows():
    m, color = row["method"], METHOD_COLOR[row["method"]]
    ax.scatter(1.0, row["base_positive_rate"], s=24, color=color, alpha=0.3, edgecolors="none", zorder=3)
    ax.annotate("", xy=(row["recall"], row["positive_rate"]), xytext=(1.0, row["base_positive_rate"]),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7, alpha=0.8, shrinkA=4, shrinkB=4, mutation_scale=8), zorder=2)
    ax.scatter(row["recall"], row["positive_rate"], s=24, color=color, edgecolors="none", zorder=4)
    ax.text(*LABEL_POS[m], m, color=color, fontsize=6, ha="left", va="center", clip_on=False)
ax.annotate("pLDDT>90\nTMscore>0.8", xy=(0.43, 0.22), xytext=(0, 5), textcoords="offset points", fontsize=6,
            ha="center", va="bottom", linespacing=1.05, annotation_clip=False)
ax.set(xlim=(0, 1.3), ylim=(0, 0.25), xticks=[0, 0.5, 1.0], yticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25])
pct_axes(ax)
ax.set_xlabel("Recall of >10 mg/L sequences (%)", fontsize=6, labelpad=2)
ax.set_ylabel("Positive rate after filtering\n(yield >10 mg/L)", fontsize=6, labelpad=2)
ax.tick_params(axis="both", labelsize=6, direction="out", pad=1, length=3, width=0.6)
ax.grid(False)
sns.despine(ax=ax)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(0.6)
fig.savefig(os.path.join(OUT, "Fig2h.pdf"), bbox_inches="tight")

for key, title, fname in [
        ("target_n_positive_nofilter", "Backbones with high-yield designs\nbefore structural filtering", "ExtFig5g-1-before-filtering.pdf"),
        ("target_n_positive", "Backbones with high-yield designs\nafter structural filtering", "ExtFig5g-2-after-AF3-filter.pdf")]:
    fig = plt.figure(figsize=(6 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(data=mres.sort_values(key, ascending=False), x=key, y="method", hue="method", hue_order=HUE_ORDER,
                palette=HUE_PALETTE, legend=False, alpha=0.9, ax=ax)
    ax.set_xlim(0, 70)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=6, pad=1)
    ax.set_xlabel("Number of unique backbones", fontsize=6, labelpad=1)
    ax.tick_params(axis="both", which="major", pad=1)
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")

# ------------------------------------------------------------------ ExtFig 5f (AF2 pLDDT cutoffs)
res2 = cutoff_table(load_designs("af2"))
print("AF2 cutoff table\n", res2.round(3).to_string(index=False))
plot_df = res2.query("condition != 'all'")
order = list(range(60, 96, 10)) + [95]
fig = plt.figure(figsize=(5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.scatterplot(data=plot_df, x="recall", y="positive_rate", hue="plddt_filter", hue_order=order,
                palette=dict(zip(order, sns.color_palette("crest", len(order)))), alpha=0.8, ax=ax)
ax.axhline(res2.loc[res2["condition"] == "all", "positive_rate"].iloc[0], linestyle="--", linewidth=0.5, color="grey")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.2)
pct_axes(ax)
ax.set_xlabel("Recall of >10 mg/L sequences (%)")
ax.set_ylabel("Positive rate after filtering\n(yield >10 mg/L)")
ax.set_title("pLDDT filter performance")
ax.tick_params(axis="both", which="major", pad=1)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.1, 1), frameon=False, fontsize=6, title="pLDDT cutoff", ncol=1,
                columnspacing=1.0)
fig.savefig(os.path.join(OUT, "ExtFig5f.pdf"), bbox_inches="tight")
