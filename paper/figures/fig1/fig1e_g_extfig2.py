"""Fig1e / ExtFig2a: per-sublibrary conversion, log10(MS intensity) of the 24 spike v1 proteins vs their reference log10(ng/barcode)
with the sublibrary regression line (Fig1e = dms-3; ExtFig2a = all 15 sublibraries).
Fig1g / ExtFig2b: per-sublibrary calibrated yield (log10 mg/L) of the 24 control proteins vs their mean over the 15 sublibraries
(Fig1g = denovo-3; ExtFig2b = all 15 sublibraries).
Tables: S02 (per-sublibrary slope/intercept), S10b (spike v1 mean MS intensity, log10 intensity, inferred ng per sublibrary), S11b (spike v1 reference ng), S09b (controls)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from common.tables import load, outdir
from common.style import set_style

set_style()
SUBLIBS = ["codon-1"] + [f"denovo-{i}" for i in range(1, 9)] + [f"dms-{i}" for i in range(1, 6)] + ["muts-1"]
FIG1E_LIB, FIG1G_LIB = "dms-3", "denovo-3"
out = outdir("fig1")

s02 = load("S02").set_index("sublib")
s10b = load("S10b").set_index("name")
ref_ng = load("S11b").assign(name=lambda d: d["name"].str.lower()).set_index("name")["log10_ng_per_barcode"]
ctrl = load("S09b").set_index("name")


def conversion_panel(lib):
    slope, intercept = s02.loc[lib, ["calibration_slope", "calibration_intercept"]]
    x = s10b[f"log_monomer__{lib}"]          # log10(mean spike v1 intensity + 1e4) of the sublibrary
    y = ref_ng.loc[x.index]
    res = stats.linregress(x, y)
    assert abs(res.slope - slope) < 1e-6 and abs(res.intercept - intercept) < 1e-5, (lib, res.slope, slope)
    print(f"{lib}: spike v1 n={len(x)} slope={res.slope:.3f} intercept={res.intercept:.2f} r={res.rvalue:.3f}")
    fig = plt.figure(figsize=(3.3 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111)
    plt.title(lib)
    sns.scatterplot(x=x, y=y, s=7, ax=ax)
    sns.regplot(x=x, y=y, scatter=False, line_kws={"lw": 1}, ax=ax, seed=0)
    ax.text(0.05, 0.95, f"slope = {res.slope:.2f}\nintercept = {res.intercept:.1f}", transform=ax.transAxes, va="top", ha="left")
    ticks = np.arange(-3, 11, 1)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
    ax.set_xlim(3, 9); ax.set_ylim(-3, 1)
    ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\log_{10}$(MS intensity)")
    ax.set_ylabel(r"$\log_{10}$(ng/barcode)")
    return fig


def control_panel(lib):
    mean = ctrl[[f"monomer_log_mgL__{s}" for s in SUBLIBS]].mean(axis=1)
    data = ctrl[[f"monomer_log_mgL__{lib}"]].assign(mean=mean).dropna()
    xv, yv = data["mean"].values, data[f"monomer_log_mgL__{lib}"].values
    r = stats.pearsonr(xv, yv)[0]
    rmse = np.sqrt(mean_squared_error(yv, xv))
    print(f"{lib}: controls n={len(data)} r={r:.2f} RMSE={rmse:.2f}")
    fig = plt.figure(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.text(0.05, 1.1, f"Pearson r = {r:.2f}\nRMSE = {rmse:.2f}\nn = {len(data)}", transform=ax.transAxes, va="top", fontsize=6)
    sns.scatterplot(x=xv, y=yv, alpha=0.7, s=6, ax=ax)
    plt.plot([-3, 5], [-3, 5], lw=0.5, ls="--")
    ticks = [-2, -1, 0, 1, 2, 3, 4]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim(-3, 2); ax.set_ylim(-3, 2)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.set_xlabel("mean"); ax.set_ylabel(lib)
    return fig


for lib in SUBLIBS:
    fig = conversion_panel(lib)
    fig.savefig(os.path.join(out, f"ExtFig2a_{lib}.pdf"))
    if lib == FIG1E_LIB:
        fig.savefig(os.path.join(out, "Fig1e.pdf"))
    plt.close(fig)
    fig = control_panel(lib)
    fig.savefig(os.path.join(out, f"ExtFig2b_{lib}.pdf"), bbox_inches="tight")
    if lib == FIG1G_LIB:
        fig.savefig(os.path.join(out, "Fig1g.pdf"), bbox_inches="tight")
    plt.close(fig)
