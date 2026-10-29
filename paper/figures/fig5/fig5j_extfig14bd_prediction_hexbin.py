"""Fig 5j, ExtFig 14b, ExtFig 14d: predicted vs observed nanobody monomer yield / monomer-dimer ratio (hexbin, test-fold
predictions of the 5-fold cross-validation).
  Fig5j-yield / Fig5j-ratio : ESMC-Nb-MonoYield / ESMC-Nb-MDRatio
  ExtFig14b                 : ESMC-MonoYield (trained on de novo + DMS only) on nanobodies
  ExtFig14d-expression/-ratio : omnilib-ml CNN / LR scores vs observed yield / ratio

Tables: S16a (yield), S16b (ratio), S07 (unclipped observed monomer yield for ExtFig14d-expression).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig5")
dark_blues = mcolors.LinearSegmentedColormap.from_list("dark_blues", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))

yld = load("S16a")
rat = load("S16b")
print("n nanobodies:", len(yld), len(rat))
# ExtFig14d-expression uses the unclipped observed yield (S16a's observed value is clipped at -1.5)
s07 = load("S07")
s07 = s07[~s07["is_spike"]].assign(key=lambda d: d["name"].str.lower())
yld_unclipped = yld.assign(key=yld["name"].str.lower()).drop(columns="true_monomer_log_mgL").merge(
    s07[["key", "monomer_log_mgL"]].rename(columns={"monomer_log_mgL": "true_monomer_log_mgL"}), on="key")


def hexbin_panel(df, pred_col, true_col, title, name, ratio):
    y_pred, y_true = df[pred_col].values, df[true_col].values
    r = stats.pearsonr(y_pred, y_true)[0]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: Pearson r={r:.3f} R2={r2:.3f} RMSE={rmse:.3f} n={len(df)}")
    fig = plt.figure(figsize=(3.5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.hexbin(y_pred, y_true, gridsize=100, mincnt=1, bins="log", cmap=dark_blues, alpha=1.0)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", lw=0.5)
    ax.text(0.05, 0.95, f"Pearson r = {r:.2f}\nR² = {r2:.2f}\nRMSE = {rmse:.2f}\nn = {len(df)}",
            transform=ax.transAxes, va="top", fontsize=6)
    ticks = np.arange(-3, 11, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title(title)
    if ratio:
        ax.set_xlabel("Predicted ratio"); ax.set_ylabel("Observed ratio")
        ax.set_xlim(-2.5, 2); ax.set_ylim(-2.5, 2)
    else:
        ax.set_xlabel("Predicted yield"); ax.set_ylabel("Observed yield")
        ax.set_xlim(-2, 1.5); ax.set_ylim(-2, 1.5)
    ax.tick_params(labelsize=6, pad=1)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)


hexbin_panel(yld, "ESMC-Nb-MonoYield", "true_monomer_log_mgL", "ESMC-Nb-MonoYield", "Fig5j-yield", False)
hexbin_panel(rat, "ESMC-Nb-MDRatio", "true_monomer_dimer_ratio_log", "ESMC-Nb-MDRatio", "Fig5j-ratio", True)
hexbin_panel(yld, "ESMC-MonoYield", "true_monomer_log_mgL", "ESMC-MonoYield", "ExtFig14b", False)


def omnilib_panel(df, true_col, ylabel, name):
    df = df.dropna(subset=["omnilib-ml CNN", "omnilib-ml LR"])
    fig = plt.figure(figsize=(8 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
    for i, col in enumerate(["omnilib-ml CNN", "omnilib-ml LR"], start=1):
        r = stats.pearsonr(df[col].values, df[true_col].values)[0]
        print(f"{name} {col}: Pearson r={r:.3f} n={len(df)}")
        ax = fig.add_subplot(1, 2, i)
        ax.hexbin(df[col].values, df[true_col].values, gridsize=100, mincnt=1, bins="log", cmap=dark_blues, alpha=1.0)
        ax.text(0.05, 0.95, f"Pearson r = {r:.2f}\nn = {len(df)}", transform=ax.transAxes, va="top", fontsize=6)
        ax.set_title(col, fontsize=6)
        ax.set_xlabel("omnilib-ml score", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="both", which="major", pad=1)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)


omnilib_panel(yld_unclipped, "true_monomer_log_mgL", "Observed expression", "ExtFig14d-expression")
omnilib_panel(rat, "true_monomer_dimer_ratio_log", "Observed monomer/dimer ratio", "ExtFig14d-ratio")
