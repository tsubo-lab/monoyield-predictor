"""ExtFig4: codon-variant library (codon-1; 50 proteins x ~50 synonymous variants, different codon-optimisation methods).
a = monomeric yield of every variant relative to its protein median (fold);  b = geometric SD of yield across the variants of a protein;
c = log10 yield vs CAI / tAI for 5 representative proteins (yield quantiles 0, 25, 50, 75, 100 % of the protein medians);
d = CAI / tAI density of de novo + DMS WT constructs vs the codon-1 variants (grey band = central 95 % of the de novo + DMS WT constructs).
Tables: S04 (codon-1 rows: monomer_mgL, monomer_log_mgL), S20a (codon-1 DNA: target protein, CAI, tAI), S20b (CAI/tAI of de novo and DMS WT constructs)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.tables import load, outdir
from common.style import set_style

set_style(6)
out = outdir("fig1")

s04 = load("S04")
s20a = load("S20a")
# codon-1 variants with their protein (target), CAI and tAI
df_m = s04[s04["sublib"] == "codon-1"].merge(s20a[["name", "target", "cai", "tai", "method"]], on="name")
df_m["name_drop"] = df_m["target"]
print(f"codon-1 variants: {len(df_m)} in {df_m['name_drop'].nunique()} proteins")

# ---- a: fold error relative to the protein median -----------------------------------------------------------
fe = df_m.dropna(subset=["monomer_mgL"]).copy()
fe = fe[fe["monomer_mgL"] > 0]
fe["prot_median"] = fe.groupby("name_drop")["monomer_mgL"].transform("median")
fe["log2_dev"] = np.log2(fe["monomer_mgL"] / fe["prot_median"])
fe["fold_error"] = 2 ** fe["log2_dev"].abs()
print(f"4a: n proteins = {fe['name_drop'].nunique()}, n variants = {len(fe)}; fold error median / 90th / 95th pct: "
      f"{fe['fold_error'].median():.2f} / {fe['fold_error'].quantile(.9):.2f} / {fe['fold_error'].quantile(.95):.2f}; "
      f"within 2-fold: {(fe['fold_error'] <= 2).mean():.3f}")
fig = plt.figure(figsize=(5.5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.histplot(fe["log2_dev"], bins=40, ax=ax, color="tab:blue")
ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
ax.set_xlabel("Monomeric yield relative to\nprotein median (fold)")
ax.set_ylabel("Number of variants")
sns.despine()
fig.savefig(os.path.join(out, "ExtFig4a.pdf"), bbox_inches="tight")

# ---- b: geometric SD per protein ----------------------------------------------------------------------------
sd_df = (df_m.dropna(subset=["monomer_log_mgL"]).groupby("name_drop")["monomer_log_mgL"].std(ddof=1).rename("sd_log10").reset_index())
sd_df["sd_fold"] = 10 ** sd_df["sd_log10"]
print(f"4b: n proteins = {len(sd_df)}; SD fold median / max: {sd_df['sd_fold'].median():.2f} / {sd_df['sd_fold'].max():.2f}; "
      + ", ".join(f"<= {t}-fold: {(sd_df['sd_fold'] <= t).mean():.3f}" for t in (2, 2.5, 3)))
fig = plt.figure(figsize=(5.5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.histplot(sd_df["sd_fold"], bins=15, ax=ax, color="tab:blue")
ax.axvline(3, color="black", linewidth=0.6, linestyle="--")
ax.set_xlabel("Geometric SD of monomeric yield\nacross synonymous variants (fold)")
ax.set_ylabel("Number of proteins")
sns.despine()
fig.savefig(os.path.join(out, "ExtFig4b.pdf"), bbox_inches="tight")

# ---- c: yield vs CAI / tAI for 5 representative proteins ---------------------------------------------------
summary = (df_m.dropna(subset=["monomer_log_mgL", "cai", "tai"]).groupby("name_drop")
           .agg(monomer_median=("monomer_log_mgL", "median")).reset_index())
selected_idx = []
for target in summary["monomer_median"].quantile(np.linspace(0, 1, 5)).values:
    candidates = summary.loc[~summary.index.isin(selected_idx)]
    # the 50 % target lies exactly midway between two proteins (50 medians): resolve the tie towards the lower median, as in the published figure
    dist = candidates["monomer_median"].sub(target).abs().round(9)
    selected_idx.append(candidates.assign(dist=dist).sort_values(["dist", "monomer_median"]).index[0])
selected = summary.loc[selected_idx].sort_values("monomer_median")
plot_df = pd.merge(selected["name_drop"], df_m)
print("4c: representative proteins:", list(selected["name_drop"]))
fig = plt.figure(figsize=(8 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 2, 1)
sns.scatterplot(plot_df, x="cai", y="monomer_log_mgL", hue="name_drop", alpha=0.7, s=6, legend=False)
ax.set_xlabel("CAI")
ax.set_ylabel(r"log$_{10}$ (monomeric yield)" + "\n(mg/L)")
ax = fig.add_subplot(1, 2, 2)
sns.scatterplot(plot_df, x="tai", y="monomer_log_mgL", hue="name_drop", alpha=0.7, s=6, legend=False)
ax.set_xlabel("tAI")
ax.set_ylabel("")
fig.savefig(os.path.join(out, "ExtFig4c.pdf"))

# ---- d: CAI / tAI density, de novo + DMS WT vs codon-1 -----------------------------------------------------
s20b = load("S20b")
reference = s20b[s20b["source"].isin(["main", "dms_wt"])]  # de novo (main) + DMS WT; the BoltzIF designs are not part of the panel
codon1 = s20a.dropna(subset=["cai", "tai"])
plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": .5})
fig, axes = plt.subplots(1, 2, figsize=(8 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
for ax, col, c1col, color, label in zip(axes, ["cai_real", "tai_real"], ["cai", "tai"], ["tab:blue", "tab:orange"], ["CAI", "tAI"]):
    s, old = reference[col], codon1[c1col]
    bins = np.linspace(min(s.min(), old.min()), max(s.max(), old.max()), 31)
    lo, hi = s.quantile([.025, .975])
    print(f"4d {label}: de novo + DMS WT n={len(s)} central 95% {lo:.4f}-{hi:.4f}; codon-1 n={len(old)} {old.quantile(.025):.4f}-{old.quantile(.975):.4f}")
    ax.axvspan(lo, hi, color="grey", alpha=.15, zorder=0)
    sns.histplot(s, bins=bins, stat="density", ax=ax, color=color, alpha=.9, linewidth=0, label="De novo + DMS WT", zorder=2)
    sns.histplot(old, bins=bins, stat="density", ax=ax, color=color, alpha=.3, linewidth=0, label="codon-1", zorder=3)
    ax.tick_params(axis="both", which="both", width=.5, length=2, direction="out")
    ax.set_xlabel(label)
    ax.set_ylabel("Density" if ax is axes[0] else "")
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(0, 1.18), borderaxespad=0)
sns.despine()
fig.savefig(os.path.join(out, "ExtFig4d.pdf"), bbox_inches="tight")
