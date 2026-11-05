"""Fig3g, Fig3h, ExtFig9: which structural features of the wild type explain the per-protein dG-vs-yield slope.
Fig3g: Pearson r of the features most associated with the slope (bootstrap 95 % CI); Fig3h: slope vs hydrophobicity (selected
proteins labelled); ExtFig9: slope vs each of those features.
Tables: S04 (per-protein fits), S23a + S23b (Rosetta score_monomeric of the WT structures)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import outdir
from common.style import set_style
import dms_data

OUT = outdir("fig3")
P_MAX = 0.01                      # features shown: p < 0.01 of the regression feature -> slope
N_TOP = 20                        # features that get a bootstrap CI
N_BOOT = 1000
HIGHLIGHT = ["1pga", "2lgw", "6m3n", "1uzc", "1r69", "2kcm", "1aoy", "3l1x", "2btt"]   # the 9 proteins with a full DMS (Fig3a)
FEATURE_LABEL = {
    "buried_np": "Buried nonpolar area\n(all residues)",
    "buried_npsa": "Buried nonpolar area\n(all residues)",
    "buried_np_AFILMVWY": "Buried nonpolar area\n(AFILMVWY)",
    "buried_npsa_FAMILYVW": "Buried nonpolar area\n(AFILMVWY)",
    "buried_np_per_res": "Buried nonpolar residues per residue",
    "buried_np_AFILMVWY_per_res": "Buried nonpolar residues per residue\n(AFILMVWY)",
    "n_hydrophobic_noA": "Number of hydrophobic residues\n(FILMVWY)",
    "n_hydrophobic": "Number of hydrophobic residues\n(AFILMVWY)",
    "hydrophobicity": "Hydrophobicity",
    "buried_minus_exposed": "Buried − exposed",
    "buried_npsa_core": "Core nonpolar area",
    "buried_over_exposed": "Buried / exposed",
    "p_aa_pp": "PP amino-acid probability",
    "total_score": "Total score",
    "contact_core_SCN": "Core contacts (SCN)",
    "contact_all": "All contacts",
    "contact_core_SASA": "Core contacts (SASA)",
    "helices_freq_L": "Helical Leu frequency",
    "surface_freq_V": "Surface Val frequency",
    "contig_not_hp_avg": "Mean non-hydrophobic stretch",
    "nres": "Number of residues",
    "n_res": "Number of residues",
}
set_style()

df_dms, df_wt, df_pr = dms_data.fit_view()
df_m = dms_data.rosetta_wt_features(df_wt, df_pr)
features = df_m.columns[7:]                       # after name, pearsonR, slope, intercept, p, std_err, valid_muts
print(f"proteins with fit and WT features: {len(df_m)}, features: {len(features)}")


def bootstrap_pearsonr(x, y, n_boot=N_BOOT, ci=95, seed=0):
    """Bootstrap CI of Pearson r (resampling proteins)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        boot.append(stats.pearsonr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


# linear regression of the per-protein slope on every feature
rows = []
for col in features:
    x, y = df_m[col], df_m["slope"]
    if x.sum() == 0:
        continue
    fit = stats.linregress(x, y)
    rows.append({"feature": col, "pearsonR": fit.rvalue, "slope": fit.slope, "p": fit.pvalue})
res = pd.DataFrame(rows)

# bootstrap CI for the 20 best features
top = res.sort_values("p").iloc[:N_TOP].copy()
ci = [bootstrap_pearsonr(df_m[f].to_numpy(), df_m["slope"].to_numpy()) for f in top["feature"]]
top["err_low"] = top["pearsonR"] - [c[0] for c in ci]
top["err_high"] = [c[1] for c in ci] - top["pearsonR"]
shown = top[(~top["feature"].str.contains("freq")) & (top["p"] < P_MAX)].sort_values("p").reset_index(drop=True)
shown = shown[shown["feature"] != "buried_np_AFILMVWY"].reset_index(drop=True)   # dropped from Fig3g / ExtFig9 (2026-11-02)
shown["label"] = shown["feature"].map(FEATURE_LABEL).fillna(shown["feature"])
print(shown[["feature", "pearsonR", "err_low", "err_high", "p"]].to_string())

# ---- Fig3g
n = len(shown)
fig = plt.figure(figsize=(7.5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.barplot(data=shown, x="pearsonR", y="label", orient="h", ax=ax)
ax.errorbar(x=shown["pearsonR"].to_numpy(), y=np.arange(n), xerr=np.vstack([shown["err_low"], shown["err_high"]]),
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson R")
ax.set_ylabel("")
ax.grid(axis="x", linestyle="--", linewidth=0.5)
ax.set_title("Correlation with slope")
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(OUT, "Fig3g.pdf"), bbox_inches="tight")

# ---- Fig3h
targets = df_m[df_m["name"].isin(HIGHLIGHT)]
fig = plt.figure(figsize=(4 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.scatterplot(data=df_m, x="hydrophobicity", y="slope", alpha=0.7, s=6, ax=ax)
sns.scatterplot(data=targets, x="hydrophobicity", y="slope", alpha=1, s=10, ax=ax)
LABEL_OFFSET = {"1uzc": (-1, 1, "right", "bottom"), "2lgw": (1, -1, "left", "top")}   # keep neighbouring labels apart
for _, row in targets.iterrows():
    dx, dy, ha, va = LABEL_OFFSET.get(row["name"], (1, 1, "left", "bottom"))
    ax.annotate(text=row["name"].upper(), xy=(row["hydrophobicity"], row["slope"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=6, ha=ha, va=va)
ax.set_xlabel("Hydrophobicity", labelpad=1)
ax.set_ylabel("Slope", labelpad=1)
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(OUT, "Fig3h.pdf"), bbox_inches="tight")

# ---- ExtFig9
fig = plt.figure(figsize=(10 / 2.54, 7 / 2.54), dpi=300, tight_layout=True)
for i, col in enumerate(shown["feature"]):
    ax = fig.add_subplot(2, 3, i + 1)
    sns.scatterplot(data=df_m, x=col, y="slope", alpha=0.7, s=6, ax=ax)
    r = res.loc[res["feature"] == col, "pearsonR"].values[0]
    ax.text(0.05, 1.05, f"r = {r:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(FEATURE_LABEL.get(col, col), labelpad=1)
    ax.set_ylabel("Slope", labelpad=1)
    ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(OUT, "ExtFig9.pdf"), bbox_inches="tight")
