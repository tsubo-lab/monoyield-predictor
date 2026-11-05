"""Fig 5c / 5d: Pearson R (with bootstrap 95% CI) of nanobody features with monomer yield (5c) and monomer/dimer ratio (5d).

Tables: S07 (yields), S24 (full feature table), S24b (pooled representative features), S08 (AF3/ESMFold2 ipTM/pLDDT, ESM3dG).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import load, path, outdir
from common.style import set_style

set_style()
OUT = outdir("fig5")
CLIP_LOWER = -1.5
N_BOOT = 1000
EXTRA = ["iptm", "plddt", "esm_iptm", "esm_plddt", "deltaG_pred"]
TARGETS = {"Fig5c": "monomer_log_mgL", "Fig5d": "monomer_dimer_ratio_log"}
YLABEL = {"monomer_log_mgL": "Monomer yield\n(log$_{10}$ mg/L)",
          "monomer_dimer_ratio_log": "Monomer / Dimer\n(log$_{10}$ ratio)"}

FEATURE_NAME = {
    "cdr3_len": "CDR3 length", "total_score": "Total score",
    "buried_minus_exposed": "Buried − exposed\nhydrophobic surface",
    "energy_per_residue_site_101": "Energy at site 101", "fa_atr": "Attractive energy",
    "cdr3_Npos2_is_L": "CDR3 N2 = Leu", "cdr3_Npos1_is_G": "CDR3 N1 = Gly",
    "cdr3_Cpos1_is_Y": "CDR3 C1 = Tyr", "cdr3_Cpos1_is_A": "CDR3 C1 = Ala", "cdr3_Cpos2_is_F": "CDR3 C2 = Phe",
    "avg_per_residue_energies_4mer_100": "4-mer energy at 100", "energy_per_residue_site_40": "Energy at site 40",
    "freq_VK": "Vκ frequency", "percent_hydrophobic_AFILMVWY_surface": "Surface hydrophobicity",
    "avg_per_residue_energies_2mer_100": "2-mer energy at 100",
    "ss_contributes_core": "Core fraction of\nα/β structure",
    "avg_per_res_energy_of_site_100_neighborhood_5A": "Energy near 100 (5 Å)",
    "avg_per_res_energy_of_site_100_neighborhood_3A": "Energy near 100 (3 Å)",
    "buried_npsa_FAMILYVW_surface": "Buried nonpolar area",
    "iptm": "AF3 ipTM", "plddt": "AF3 pLDDT", "esm_iptm": "ESMFold2 ipTM", "esm_plddt": "ESMFold2 pLDDT",
    "deltaG_pred": "Absolute ΔG (ESM3ΔG predicted)",
}
# colours matched to the ChimeraX scheme of the structure figures
CDR1_COLOR, CDR2_COLOR, CDR3_COLOR = "#8FC1E1", "#0072BC", "#0B3754"
SITE40_COLOR, SITE100_COLOR, C1_COLOR, OTHER_COLOR = "#EF5A5A", "#F2B84B", "#3FA796", "#B0B0B0"


def feature_color(name):
    if "site_100" in name or "site_101" in name or "neighborhood_100" in name or name.endswith("_100"):
        return SITE100_COLOR
    if "site_40" in name or name.endswith("_40"):
        return SITE40_COLOR
    if "cdr3_Cpos1" in name:
        return C1_COLOR
    if name.startswith("cdr3"):
        return CDR3_COLOR
    if name.startswith("cdr2"):
        return CDR2_COLOR
    if name.startswith("cdr1"):
        return CDR1_COLOR
    return OTHER_COLOR


def bootstrap_pearsonr(x, y, n_boot=N_BOOT, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        rs.append(stats.pearsonr(xb, yb)[0])
    rs = np.array(rs)
    lo, hi = np.percentile(rs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return np.std(rs, ddof=1), lo, hi


# ---- data ----
top = load("S24b")
top["representative"] = top["representative"].str.replace("freq_CA", "cdr3_len")
learncols = {t: list(top[(top.group == "pooled") & (top.target == t)]["representative"]) + EXTRA
             for t in TARGETS.values()}

y = load("S07")
y = y[~y["is_spike"]].copy()
y["key"] = y["name"].str.lower()
# yields clipped at CLIP_LOWER; the S07 ratio (unclipped monomer - dimer) is used as is
y[["monomer_log_mgL", "dimer_log_mgL"]] = y[["monomer_log_mgL", "dimer_log_mgL"]].clip(lower=CLIP_LOWER)

feat_cols = sorted({c for v in learncols.values() for c in v} - set(EXTRA))
feat = pd.read_csv(path("S24"), usecols=["name"] + feat_cols)
feat["key"] = feat["name"].str.lower()
s08 = load("S08")
s08["key"] = s08["name"].str.lower()
df = (y[["key", "monomer_log_mgL", "dimer_log_mgL", "monomer_dimer_ratio_log"]]
      .merge(feat.drop(columns="name"), on="key").merge(s08[["key"] + EXTRA], on="key"))
print("n nanobodies:", len(df))

# ---- correlations ----
results = {}
for t, cols in learncols.items():
    rows = []
    for xc in cols:
        d = df.dropna(subset=[xc, t])
        if len(d) < 2 or d[xc].sum() == 0:
            continue
        xv, yv = d[xc].to_numpy(), d[t].to_numpy()
        r = stats.linregress(xv, yv).rvalue
        se, lo, hi = bootstrap_pearsonr(xv, yv)
        rows.append(dict(feature=xc, pearsonR=r, ci_low=lo, ci_high=hi, err_low=r - lo, err_high=hi - r))
    results[t] = pd.DataFrame(rows)

plot_dfs = {}
for t, res in results.items():
    res = res.reindex(res["pearsonR"].abs().sort_values(ascending=False).index)
    key = "iptm" if "dimer" in t else "plddt"
    res = pd.concat([res.head(8), res[res["feature"].str.contains(key)]])
    res = res.reindex(res["pearsonR"].abs().sort_values(ascending=False).index)
    plot_dfs[t] = res
    print(t); print(res[["feature", "pearsonR", "ci_low", "ci_high"]].to_string(index=False))


def is_rosetta(f):
    """Rosetta-based structural scores (marked with * in the figure): not a CDR sequence feature, frequency, AF3 / ESMFold2 or ESM3dG score."""
    return not (f.startswith("cdr") or f.startswith("freq_") or f in EXTRA)


def draw_panel(ax, plot_df, t):
    sns.barplot(data=plot_df, x="pearsonR", y="feature", orient="h",
                palette=[feature_color(f) for f in plot_df["feature"]], hue="feature", legend=False, ax=ax)
    ax.errorbar(plot_df["pearsonR"].values, np.arange(len(plot_df)),
                xerr=np.vstack([plot_df["err_low"].values, plot_df["err_high"].values]),
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8)
    labels = [("*" if is_rosetta(l.get_text()) else "") + FEATURE_NAME.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels, linespacing=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson R", labelpad=1)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    ax.set_title(YLABEL[t].split("\n")[0], pad=3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.tick_params(axis="both", which="major", pad=1)


fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 4.5 / 2.54), dpi=300)
for ax, (name, t) in zip(axes, TARGETS.items()):
    draw_panel(ax, plot_dfs[t], t)
axes[0].set_position([0.22, 0.2, 0.25, 0.68])
axes[1].set_position([0.70, 0.2, 0.25, 0.68])
fig.canvas.draw()
for ax in axes:   # footnote at the height of the "Pearson R" label
    bb = ax.xaxis.label.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(-0.02, (bb.y0 + bb.y1) / 2, "*Rosetta based scores", transform=ax.transAxes, ha="right", va="center", fontsize=6)
fig.savefig(os.path.join(OUT, "Fig5cd.pdf"))
plt.close(fig)
