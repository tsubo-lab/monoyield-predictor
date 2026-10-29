"""Fig 5c / 5d: Pearson R (with bootstrap 95% CI) of nanobody features with monomer yield (5c) and monomer/dimer ratio (5d).

Tables: S07 (yields), S25 (full feature table), S25b (pooled representative features), S08 (AF3/ESMFold2 ipTM/pLDDT, ESM3dG).
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
    "cdr3_len": "CDR3 length", "total_score": "Rosetta total score",
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
    "iptm": "AF3 iptm", "plddt": "AF3 pLDDT", "esm_iptm": "ESMfold2 iptm", "esm_plddt": "ESMfold2 pLDDT",
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
top = load("S25b")
top["representative"] = top["representative"].str.replace("freq_CA", "cdr3_len")
learncols = {t: list(top[(top.group == "pooled") & (top.target == t)]["representative"]) + EXTRA
             for t in TARGETS.values()}

y = load("S07")
y = y[~y["is_spike"]].copy()
y["key"] = y["name"].str.lower()
y[["monomer_log_mgL", "dimer_log_mgL"]] = y[["monomer_log_mgL", "dimer_log_mgL"]].clip(lower=CLIP_LOWER)
y["monomer_dimer_ratio_log"] = y["monomer_log_mgL"] - y["dimer_log_mgL"]

feat_cols = sorted({c for v in learncols.values() for c in v} - set(EXTRA))
feat = pd.read_csv(path("S25"), usecols=["name"] + feat_cols)
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


def draw(plot_df, t, pos=None):
    fig = plt.figure(figsize=(6 / 2.54, 4 / 2.54), dpi=300, tight_layout=(pos is None))
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(data=plot_df, x="pearsonR", y="feature", orient="h",
                palette=[feature_color(f) for f in plot_df["feature"]], hue="feature", legend=False, ax=ax)
    ax.errorbar(plot_df["pearsonR"].values, np.arange(len(plot_df)),
                xerr=np.vstack([plot_df["err_low"].values, plot_df["err_high"].values]),
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8)
    labels = [FEATURE_NAME.get(l.get_text(), l.get_text()) for l in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels, linespacing=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson R")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    ax.set_title(YLABEL[t].split("\n")[0])
    ax.tick_params(axis="both", which="major", pad=1)
    if pos is not None:
        ax.set_position(pos)
    return fig, ax


# pass 1: measure each panel's axes box; pass 2: redraw both panels with the shared (most constrained) box
positions = {}
for t, p in plot_dfs.items():
    fig, ax = draw(p, t)
    fig.canvas.draw()
    positions[t] = ax.get_position()
    plt.close(fig)
common_pos = max(positions.values(), key=lambda b: b.x0)
for name, t in TARGETS.items():
    fig, _ = draw(plot_dfs[t], t, pos=common_pos)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
