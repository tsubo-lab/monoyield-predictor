"""Fig4h/i (HPLC yield of SARS-CoV-2 binders: random vs top ESMC-MonoYield picks, and HPLC vs predicted yield),
ExtFig12b (baseline-corrected HPLC chromatograms, random vs top) and ExtFig12c (HPLC yield vs SaProt / Aggrescan3D / NetSolP predictions).

Tables: S14d (HPLC yield, head-B selection labels, predictions), hplc('cov_binder') (raw chromatograms; sample = binder name).
The 21 constructs of the final analysis are the rows with in_fig4h_fig4i == True (2 originally top-selected constructs excluded).
Outputs: Fig4hi.pdf (h left, i right), ExtFig12b.pdf, ExtFig12c.pdf
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from common.tables import load, hplc, outdir
from common.style import set_style

set_style()
OUT = outdir("fig4")
PAL = {"random": "tab:blue", "top": "tab:orange"}
ORDER = ["random", "top"]

d = load("S14d")
pool = d[d["in_fig4h_fig4i"]].rename(columns={
    "selection_fig4h": "selection_new", "hplc_est_yield_mgL": "est_amount", "hplc_log10_yield_mgL": "log_est_amount",
    "predicted_monomer_log10_mgL(ESMC-MonoYield)": "pred_monomer_log_mgL",
    "predicted_monomer_log10_mgL(SaProt-LoRA)": "predSaProt"}).reset_index(drop=True)

# ---------------- Fig4h / Fig4i ----------------
top_v = pool.loc[pool["selection_new"] == "top", "est_amount"]
rnd_v = pool.loc[pool["selection_new"] == "random", "est_amount"]
u, pv = mannwhitneyu(top_v, rnd_v, alternative="greater", method="auto")
print(f"Fig4h: top n={len(top_v)} random n={len(rnd_v)}  Mann-Whitney U={u:.1f} one-sided p={pv:.4g}")
for lab, v in [("top", top_v), ("random", rnd_v)]:
    print(f"  {lab}: >=10 mg/L in {(v >= 10).sum()}/{len(v)} ({(v >= 10).mean() * 100:.1f}%)")
r2, p2 = pearsonr(pool["pred_monomer_log_mgL"], pool["log_est_amount"])
print(f"Fig4i: Pearson r = {r2:.4f} (p = {p2:.3g}, n = {len(pool)})")

fig, axes = plt.subplots(1, 2, figsize=(6.8 / 2.54, 3.5 / 2.54), gridspec_kw={"width_ratios": [0.8, 1]}, dpi=300)
ax = axes[0]
sns.boxplot(data=pool, x="selection_new", y="est_amount", hue="selection_new", order=ORDER, hue_order=ORDER, palette=PAL,
            dodge=False, legend=False, width=0.65, fliersize=0, linewidth=0.6, ax=ax)
sns.swarmplot(data=pool, x="selection_new", y="est_amount", hue="selection_new", order=ORDER, hue_order=ORDER, palette=PAL,
              dodge=False, legend=False, alpha=0.8, size=2.5, linewidth=0.4, ax=ax)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Random", "Top"])
ax.set_xlabel(""); ax.set_ylabel("HPLC based individual assay\nMonomeric yield (mg/L)")
y0 = pool["est_amount"].max()
ax.plot([0, 0, 1, 1], [y0 * 1.05, y0 * 1.12, y0 * 1.12, y0 * 1.05], lw=0.6, c="k")
ax.text(0.5, y0 * 1.14, f"p = {pv:.3f}", ha="center", va="bottom", fontsize=6)
ax.axhline(y=10, color="grey", linestyle="--", linewidth=0.8)
ax.set_ylim(-1, y0 * 1.27)
ax.set_yticks([0, 10, 20])
for pos, sel in enumerate(ORDER):
    rate = (pool.loc[pool["selection_new"] == sel, "est_amount"] >= 10).mean() * 100
    ax.text(pos, -0.15, f"{rate:.1f}%", transform=ax.get_xaxis_transform(), ha="center", va="top", color=PAL[sel], fontsize=5, clip_on=False)
ax.text(-0.03, -0.15, "≥10 mg/L:", transform=ax.transAxes, ha="right", va="top", fontsize=5, clip_on=False)

ax = axes[1]
for sel in ORDER:
    s = pool[pool["selection_new"] == sel]
    ax.scatter(s["pred_monomer_log_mgL"], s["log_est_amount"], s=6, alpha=0.8, color=PAL[sel], label=sel, linewidths=0)
ax.axhline(y=1, color="grey", linestyle="--", linewidth=0.8)
ax.text(0.05, 0.95, f"r = {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6)
ax.set_xlabel("Predicted")
ax.set_ylabel("HPLC based individual assay\nlog$_{10}$ (monomeric yield) (mg/L)")
ax.legend(frameon=False, fontsize=4.5, loc="lower right", labelcolor="linecolor", handlelength=0, handletextpad=0, markerscale=0)
ax.tick_params(axis="both", which="major", pad=1)
for ax in axes:
    ax.set_facecolor("white"); ax.set_axisbelow(True)
    ax.grid(True, color="0.94", linewidth=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("0.55"); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=5.5, pad=1, length=2, width=0.5, color="0.55", labelcolor="0.25")
    ax.xaxis.label.set_size(5.5); ax.yaxis.label.set_size(5.5)
    ax.xaxis.labelpad = 2; ax.yaxis.labelpad = 2
axes[1].set_yticks([-1, -0.5, 0, 0.5, 1, 1.5]); axes[1].set_ylim(-1, 1.55); axes[1].set_xticks([0, 0.5])
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.94, wspace=0.82)
plt.savefig(os.path.join(OUT, "Fig4hi.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------- ExtFig12c ----------------
# Aggrescan3D is stored sign-flipped in S14d (higher = better); the panel shows the raw average aggregation score (axis inverted)
pool["aggrescan3d_raw"] = -pool["Aggrescan3D"]
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8 / 2.54, 3.5 / 2.54), dpi=300)
panels = [("SaProt", "predSaProt", "tab:purple", "Predicted log$_{10}$ yield"),
          ("Aggrescan3D", "aggrescan3d_raw", "#83c8aa", "Average aggregation score"),
          ("NetSolP", "NetSolP", "#4C9F70", "Predicted solubility")]
for ax, (tool, col, color, xlabel) in zip(axes, panels):
    r, p = pearsonr(pool[col], pool["log_est_amount"])
    print(f"ExtFig12c {tool}: r={r:.3f} p={p:.3g} n={len(pool)}")
    ax.scatter(pool[col], pool["log_est_amount"], s=6, color=color, alpha=0.8, linewidths=0)
    ax.text(0.04, 0.04, f"r = {r:.2f}\np = {p:.3g}\nn = {len(pool)}", transform=ax.transAxes, va="bottom", fontsize=6)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_title(tool, fontsize=6)
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.5)
    ax.grid(True, color="0.94", linewidth=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6, length=2, width=0.5, pad=1)
axes[1].invert_xaxis()
axes[0].set_ylabel("HPLC log$_{10}$ monomeric yield (mg/L)", fontsize=6)
fig.tight_layout(pad=0.6, w_pad=0.8)
fig.savefig(os.path.join(OUT, "ExtFig12c.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------- ExtFig12b: chromatograms of the same 21 constructs ----------------
RT_MIN, RT_MAX, Y_OFFSET = 33, 40, 1.5e6
raw = hplc("cov_binder")
x = raw["R_Time"].to_numpy()

def baseline_rolling_quantile(y, window_points=10001, q=0.03):
    return pd.Series(y).rolling(window_points, center=True, min_periods=1).quantile(q).values

bl = {n: raw[n].to_numpy(float) - baseline_rolling_quantile(raw[n].to_numpy(float)) for n in pool["name"]}
peak_mask = (x >= RT_MIN) & (x <= RT_MAX)
meta = pool[["name", "selection_new"]].copy()
meta["peak_height"] = [bl[n][peak_mask].max() for n in meta["name"]]
meta = meta.sort_values("peak_height", kind="stable")
fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 6 / 2.54), dpi=300, sharex=True, sharey=True, tight_layout=True)
for ax, sel in zip(axes, ORDER):
    for i, name in enumerate(meta.loc[meta["selection_new"] == sel, "name"]):
        base = i * Y_OFFSET
        y = bl[name] + base
        ax.fill_between(x, base, y, color=PAL[sel], alpha=0.12, linewidth=0)
        ax.plot(x, y, color=PAL[sel], linewidth=0.6, alpha=0.8)
    ax.set_title(sel)
    ax.set_xlim(20, 50); ax.set_ylim(-0.5e6, 1.89e7)
    ax.set_yticks([]); ax.set_yticks([], minor=True)
    for rt in (RT_MIN, RT_MAX): ax.axvline(rt, color="grey", linestyle="--", lw=0.5, alpha=0.5)
    ax.tick_params(axis="both", which="major", pad=1)
fig.supxlabel("Retention time (min)", fontsize=plt.rcParams["axes.labelsize"], y=0.06)
fig.supylabel("Intensity + offset", fontsize=plt.rcParams["axes.labelsize"], x=0.05)
fig.savefig(os.path.join(OUT, "ExtFig12b.pdf"), bbox_inches="tight")
print("ExtFig12b traces:", meta["selection_new"].value_counts().to_dict())
