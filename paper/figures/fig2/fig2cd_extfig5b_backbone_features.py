"""Fig 2c (|Pearson R| of backbone features vs mean yield, bootstrap CI), Fig 2d (yield vs Rg / contact order)
and ExtFig 5b (yield vs core fraction / SS fractions). Per-backbone mean yield over designs (wt entries excluded).
Tables: S04, S13, S17."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from _fig2_data import load_designs, drop_natural, load_backbones, outdir, set_style, TSM

set_style()
OUT = outdir("fig2")
FEATURES = ["p_loop", "p_alpha", "p_beta", "rg", "CO", "frac_core"]
NAMES = {"p_loop": "Fraction of loop", "p_alpha": "Fraction of alpha helix", "p_beta": "Fraction of beta strand",
         "rg": "Radius of gyration", "CO": "Contact order", "frac_core": "Fraction of core AA"}
N_BOOT, CI, SEED = 1000, 95, 0

bb = load_backbones(drop_natural(load_designs()))


def bootstrap_pearsonr(x, y, n_boot=N_BOOT, ci=CI, random_state=SEED):
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        rs.append(stats.pearsonr(xb, yb)[0])
    lo, hi = np.percentile(rs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


rows = []
for f in FEATURES:
    r = stats.linregress(bb[f], bb["monomer_log_mgL"])[2]
    lo, hi = bootstrap_pearsonr(bb[f], bb["monomer_log_mgL"])
    rows.append(dict(feature=f, feature_name=NAMES[f], pearsonR=r, ci_low=lo, ci_high=hi))
res = pd.DataFrame(rows)
print(f"n backbones = {len(bb)}")
print(res.round(3).to_string(index=False))

# ---- Fig 2c: |R| with CI mapped to absolute scale
d = res.copy()
d["abs_r"] = d["pearsonR"].abs()
d["abs_lo"] = np.where(d["ci_low"] * d["ci_high"] <= 0, 0, np.minimum(d["ci_low"].abs(), d["ci_high"].abs()))
d["abs_hi"] = np.maximum(d["ci_low"].abs(), d["ci_high"].abs())
d = d.sort_values("abs_r", ascending=False)
fig = plt.figure(figsize=(4.5 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.barplot(data=d, x="abs_r", y="feature_name", dodge=False, legend=False, color=TSM[2], ax=ax)
ax.errorbar(x=d["abs_r"], y=np.arange(len(d)), xerr=[d["abs_r"] - d["abs_lo"], d["abs_hi"] - d["abs_r"]],
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8)
ticks = [round(x, 1) for x in np.arange(0, 1.01, 0.2)]
ax.set_ylabel("")
ax.set_xlabel("|Pearson R| to\n" + r"log$_{10}$(monomeric yield) (mg/L)" + "\nper backbone", labelpad=2)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
ax.set_xlim(0, 0.9)
ax.grid(axis="x", linestyle="--", linewidth=0.5)
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(OUT, "Fig2c.pdf"))

# ---- Fig 2d (rg, CO) and ExtFig 5b (frac_core, p_alpha, p_beta, p_loop): yield vs backbone feature
PANELS = {"rg": "Fig2d-Rg", "CO": "Fig2d-ContactOrder", "frac_core": "ExtFig5b-frac_core",
          "p_alpha": "ExtFig5b-p_alpha", "p_beta": "ExtFig5b-p_beta", "p_loop": "ExtFig5b-p_loop"}
for f, name in PANELS.items():
    fig = plt.figure(tight_layout=True, figsize=(3.5 / 2.54, 3 / 2.54), dpi=300)
    ax = fig.add_subplot(111)
    sns.scatterplot(bb, x=f, y="monomer_log_mgL", s=6, alpha=0.5, color=TSM[2], ax=ax)
    ax.set_ylabel("Mean per backbone\n" + r"log$_{10}$(monomeric yield)" + "\n(mg/L)", labelpad=2)
    ax.set_xlabel(NAMES[f], labelpad=2)
    ax.tick_params(axis="both", which="major", pad=1)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
