"""Fig4a and ExtFig10a: zero-shot Pearson R between monomeric yield and single structure/sequence features
(all data = de novo + DMS), top-10 features with bootstrap 95% CI, and hexbin plots of the same features.

Tables: S04 (monomer_log_mgL, measured dG, source), S05 (features incl. ESM3dG deltaG_pred), S05_feature_list (paper labels / groups).
S25 (ESMFold2 pLDDT/pTM, contact order, radius of gyration, core fraction per design).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
from scipy import stats
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig4")
Y = "monomer_log_mgL"
GROUP_ORDER = ["Rosetta", "Packing of core", "Hydrophobicity", "Polarity_hbond", "Structure geometry",
               "Sequence_composition", "Folding stability", "Structure prediction quality"]
EXCLUDE = "_site_|per_residue_|_3d_contacts_|freq|4mer|3mer|2mer|3A_|5A_|net_"
INCLUDE = "sc_|af3_|deltaG|rg|frac_core|CO|esmfold"
DG_LABEL, DG_GROUP = "Absolute ΔG (kcal/mol)", "Folding stability"

# ---- data: S05 features + yield / measured dG from S04 ----
s4 = load("S04")[["sublib", "name", "source_dms_denovo", "monomer_log_mgL", "deltaG_measured_kcal_mol", "deltaG_95CI"]]
s5 = load("S05")
feat_cols = list(s5.columns[2:])
learncols = [c for c in feat_cols if pd.Series(c).str.contains(INCLUDE).iloc[0] and not pd.Series(c).str.contains(EXCLUDE).iloc[0]
             and s5[c].dtype.kind in "fi"]
df = s5[["sublib", "name"] + learncols].merge(s4, on=["sublib", "name"], how="left")
s26 = load("S25").drop_duplicates("name")[["name", "esmfold_plddt", "esmfold_ptm", "contact_order", "radius_of_gyration", "fraction_core_residues"]]
s26 = s26.rename(columns={"contact_order": "CO", "radius_of_gyration": "rg", "fraction_core_residues": "frac_core"})
df = df.merge(s26, on="name", how="left")
df["deltaG"] = df["deltaG_measured_kcal_mol"].where(df["deltaG_95CI"] <= 0.5)   # original filter: 95% CI <= 0.5
learncols = learncols + ["esmfold_plddt", "esmfold_ptm", "CO", "rg", "frac_core", "deltaG"]
fl = load("S05_feature_list").dropna(subset=["paper_label"]).set_index("feature")
label = {**fl["paper_label"].to_dict(), "deltaG": DG_LABEL}
group = {**fl["feature_group"].to_dict(), "deltaG": DG_GROUP}
hue_palette = dict(zip(GROUP_ORDER, sns.color_palette("husl", 8)))

allset = df[df["source_dms_denovo"].isin(["denovo", "dms"])]

def bootstrap_pearsonr(x, y, n_boot=1000, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0: continue
        rs.append(stats.pearsonr(x[idx], y[idx])[0])
    return np.percentile(rs, [(100 - ci) / 2, 100 - (100 - ci) / 2])

# ---- rank features (as in the notebook: the data are progressively NaN-filtered along the feature list) ----
Yv = allset[Y].to_numpy(float)
keep = np.isfinite(Yv)
res = []
for c in learncols:
    x = allset[c].to_numpy(float)
    keep &= np.isfinite(x)
    if keep.sum() < 2 or x[keep].sum() == 0: continue
    res.append((c, abs(stats.linregress(x[keep], Yv[keep]).rvalue)))
top10 = [c for c, _ in sorted(res, key=lambda t: -t[1])[:10]]

# ---- Pearson R with bootstrap CI on all rows with a value (notebook: whole merged frame) ----
rows = []
for c in top10:
    d = df.dropna(subset=[c, Y])
    r = stats.linregress(d[c], d[Y]).rvalue
    lo, hi = bootstrap_pearsonr(d[c].to_numpy(), d[Y].to_numpy())
    rows.append(dict(feature=c, feature2=label[c], group=group[c], pearsonR=r, lo=r - lo, hi=hi - r))
res_df = pd.DataFrame(rows)
print(res_df.round(3).to_string())

# ---- Fig4a ----
plot_df = res_df.reindex(res_df["pearsonR"].abs().sort_values(ascending=False).index)
fig = plt.figure(figsize=(8 / 2.54, 5 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.barplot(data=plot_df, x="pearsonR", y="feature2", orient="h", hue="group", palette=hue_palette, ax=ax)
ax.errorbar(x=plot_df["pearsonR"].values, y=np.arange(len(plot_df)), xerr=np.vstack([plot_df["lo"], plot_df["hi"]]),
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson R to monomeric yield"); ax.set_ylabel(""); ax.set_xlim(-0.3, 0.7); ax.set_xticks([-0.2, 0, 0.2, 0.4])
ax.grid(axis="x", linestyle="--", linewidth=0.5)
leg = ax.legend(loc="lower right", frameon=False, title=None, handlelength=1.0, handleheight=0.8, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)  # legend inside, lower right, no title / frame
ax.tick_params(axis="both", which="major", pad=1)
plt.savefig(os.path.join(OUT, "Fig4a.pdf")); plt.close()

# ---- ExtFig10a: hexbin of the top-10 features (ordered as in the notebook, i.e. by feature ranking order) ----
def group_cmap(base, light=0.92):
    return mcolors.LinearSegmentedColormap.from_list("g", [tuple(light + (1 - light) * c for c in base), base])

fig = plt.figure(figsize=(15 / 2.54, 6 / 2.54), tight_layout=True, dpi=300)
for i, (_, row) in enumerate(res_df.iterrows(), 1):
    c = row["feature"]
    d = allset[[c, Y]].replace([np.inf, -np.inf], np.nan).dropna()
    ax = fig.add_subplot(2, 5, i)
    ax.hexbin(d[c], d[Y], gridsize=100, mincnt=1, bins="log", cmap=group_cmap(hue_palette[row["group"]]), alpha=1.0)
    r = stats.pearsonr(d[c], d[Y])[0]
    print(f"ExtFig10a {c}: n={len(d)} R={r:.3f}")
    ax.text(0.05, 0.95, f"R = {r:.2f}\n", transform=ax.transAxes, ha="left", va="top")
    ax.set_ylabel(r"log$_{10}$ (monomeric yield)" + "\n(mg/L)" if i in (1, 6) else "")
    ax.set_xlabel(row["feature2"])
    ax.tick_params(axis="both", which="major", pad=1)
    sns.despine()
plt.savefig(os.path.join(OUT, "ExtFig10a.pdf")); plt.close()
