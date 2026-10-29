"""ExtFig 14c: positive rate (precision) among the top-k % of nanobodies ranked by predicted score, for omnilib-ml (CNN, LR)
and the ESMC models. Positive: monomer yield >= 0.5 (log10 mg/L, ~3 mg/L) / monomer-dimer ratio >= 1.0 (10-fold).

Tables: S16a (yield), S16b (ratio).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig5")
GREEN = ["#DAF6EB", "#7DE0B9", "#32C189", "#2B9C70", "#257958", "#1F5741"]
PURPLE = ["#E7E5F7", "#C3BEED", "#A097E3", "#7E74C2", "#5E578E", "#403C5D"]
TOP_FRACS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)

yld, rat = load("S16a"), load("S16b")
Y, R = "true_monomer_log_mgL", "true_monomer_dimer_ratio_log"
PANELS = {
    "yield": {"title": "Monomeric yield ≥ ~3 mg/L", "threshold": 0.5, "series": [
        ("omnilib-ml CNN", yld, "omnilib-ml CNN", Y, GREEN[2]),
        ("omnilib-ml LR", yld, "omnilib-ml LR", Y, GREEN[4]),
        ("ESMC-MonoYield", yld, "ESMC-MonoYield", Y, PURPLE[1]),
        ("ESMC-Nb-MonoYield", yld, "ESMC-Nb-MonoYield", Y, PURPLE[5])]},
    "ratio": {"title": "Monomer/dimer ratio ≥ 10 fold", "threshold": 1.0, "series": [
        ("omnilib-ml CNN", rat, "omnilib-ml CNN", R, GREEN[2]),
        ("omnilib-ml LR", rat, "omnilib-ml LR", R, GREEN[4]),
        ("ESMC-Nb-MDRatio", rat, "ESMC-Nb-MDRatio", R, PURPLE[4])]},
}
LABEL_OFFSETS = {"yield": {"omnilib-ml CNN": -2, "omnilib-ml LR": -6}}


def precision_at_k(data, pred_col, true_col, thresh, fracs):
    d = data[[pred_col, true_col]].dropna().copy()
    d["positive"] = (d[true_col] >= thresh).astype(int)
    n_total, base = len(d), d["positive"].mean()
    d = d.sort_values(pred_col, ascending=False).reset_index(drop=True)
    ks, prec = [], []
    for f in fracs:
        k = max(1, int(np.ceil(n_total * f)))
        ks.append(f * 100)
        prec.append(d.iloc[:k]["positive"].mean() * 100)
    return np.array(ks), np.array(prec), base * 100, n_total


fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 4.5 / 2.54), dpi=300)
fig.subplots_adjust(left=0.13, right=0.79, bottom=0.24, top=0.86, wspace=0.9)
for ax, (key, panel) in zip(axes, PANELS.items()):
    base_rates = []
    for label, data, pred_col, true_col, color in panel["series"]:
        ks, prec, base, n = precision_at_k(data, pred_col, true_col, panel["threshold"], TOP_FRACS)
        base_rates.append(base)
        print(f"{key} {label}: n={n} base={base:.1f}% precision@top1/20%={prec[0]:.1f}/{prec[-1]:.1f}")
        ax.plot(ks, prec, marker="o", color=color, ms=3, linewidth=0.9)
        ax.annotate(label, xy=(ks[-1], prec[-1]), xytext=(4, LABEL_OFFSETS.get(key, {}).get(label, 0)),
                    textcoords="offset points", va="center", ha="left", fontsize=6, color=color,
                    annotation_clip=False, clip_on=False)
    ax.axhline(np.mean(base_rates), color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(0, 24); ax.set_xticks([0, 20])
    ax.set_yticks([20, 40, 60]); ax.set_ylim(0, 70)
    ax.set_title(panel["title"], fontsize=6, loc="left", pad=5)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#333333")
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", direction="out", labelsize=6, pad=1, length=2, width=0.6,
                   colors="#222222", top=False, right=False)
axes[0].set_ylabel("Positive rate after\nselection (%)", fontsize=6, labelpad=1)
fig.text(0.46, 0.12, "Fraction of selection (%)", ha="center", fontsize=6)
fig.savefig(os.path.join(OUT, "ExtFig14c.pdf"), bbox_inches="tight", pad_inches=0.04)
