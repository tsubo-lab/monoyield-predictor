"""Fig1e / ExtFig1c: per-sublibrary conversion, log10(MS intensity) of the 24 spike v1 proteins vs their reference log10(ng/barcode)
with the sublibrary regression line (Fig1e = dms-3; ExtFig1c = all 15 sublibraries in one 3x5 figure).
Fig1g / ExtFig2: per-sublibrary calibrated yield (log10 mg/L) of the 24 control proteins vs their mean over the 15 sublibraries
(Fig1g = denovo-3; ExtFig2 = all 15 sublibraries in one 3x5 figure).
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

set_style(arial_math=True)
SUBLIBS = ["codon-1"] + [f"denovo-{i}" for i in range(1, 9)] + [f"dms-{i}" for i in range(1, 6)] + ["muts-1"]
GRID_ORDER = [f"denovo-{i}" for i in range(1, 9)] + [f"dms-{i}" for i in range(1, 6)] + ["muts-1", "codon-1"]   # 3 x 5 panel order
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


fig = conversion_panel(FIG1E_LIB); fig.savefig(os.path.join(out, "Fig1e.pdf")); plt.close(fig)
fig = control_panel(FIG1G_LIB); fig.savefig(os.path.join(out, "Fig1g.pdf"), bbox_inches="tight"); plt.close(fig)
for lib in SUBLIBS:      # printed for the record (slope / intercept / r / RMSE of every sublibrary)
    conversion_panel(lib); plt.close("all"); control_panel(lib); plt.close("all")


def grid():
    fig, axes = plt.subplots(3, 5, figsize=(16 / 2.54, 10 / 2.54), dpi=300, sharex=True, sharey=True)
    return fig, axes


def finish(fig, ax, i, ncols=5):
    row, col = divmod(i, ncols)
    ax.tick_params(axis="x", bottom=True, labelbottom=(row == 2), labelsize=6, pad=1, length=2)
    ax.tick_params(axis="y", left=True, labelleft=(col == 0), labelsize=6, pad=1, length=2)
    ax.minorticks_off(); ax.spines[["top", "right"]].set_visible(False)


# ---- ExtFig1c: conversion of all 15 sublibraries ----
fig, axes = grid()
for i, (ax, lib) in enumerate(zip(axes.flat, GRID_ORDER)):
    x = s10b[f"log_monomer__{lib}"]; y = ref_ng.loc[x.index]
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    sns.regplot(x=x, y=y, color="#2682BC", scatter_kws={"s": 7, "alpha": 0.8, "linewidths": 0}, line_kws={"lw": 0.8}, ax=ax, seed=0)
    res = stats.linregress(x, y)
    ax.text(0.05, 0.95, f"slope = {res.slope:.2f}\nintercept = {res.intercept:.1f}", transform=ax.transAxes, ha="left", va="top", fontsize=6, linespacing=1.1)
    ax.set_title(lib, fontsize=6, pad=4)
    ax.set_xlim(3, 9); ax.set_ylim(-3, 1); ax.set_xticks(np.arange(3, 10, 1)); ax.set_yticks(np.arange(-3, 2, 1))
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_axisbelow(True); ax.grid(color="#dddddd", linewidth=0.5, alpha=0.6)
    finish(fig, ax, i)
fig.supxlabel(r"$\log_{10}$(mean (MS intensity + $10^{4}$))", fontsize=6, y=0.015)
fig.supylabel(r"$\log_{10}$(ng/barcode)", fontsize=6, x=0.015)
fig.tight_layout(rect=(0.035, 0.04, 1, 1), pad=0.4, w_pad=0.6, h_pad=1.0)
fig.savefig(os.path.join(out, "ExtFig1c.pdf"), bbox_inches="tight"); plt.close(fig)

# ---- ExtFig2: calibrated yield of the controls in every sublibrary vs their mean ----
mean = ctrl[[f"monomer_log_mgL__{s}" for s in SUBLIBS]].mean(axis=1)
fig, axes = grid()
for i, (ax, lib) in enumerate(zip(axes.flat, GRID_ORDER)):
    d = ctrl[[f"monomer_log_mgL__{lib}"]].assign(mean=mean).replace([np.inf, -np.inf], np.nan).dropna()
    x, y = d["mean"].to_numpy(), d[f"monomer_log_mgL__{lib}"].to_numpy()
    r = stats.pearsonr(x, y)[0] if len(d) >= 2 else np.nan
    rmse = np.sqrt(mean_squared_error(y, x)) if len(d) else np.nan
    sns.scatterplot(x=x, y=y, alpha=0.7, s=6, linewidth=0, ax=ax)
    ax.plot([-3, 2], [-3, 2], lw=0.5, ls="--", color="grey", zorder=1)
    ax.text(0.05, 0.97, f"Pearson r = {r:.2f}\nRMSE = {rmse:.2f}\nn = {len(d)}", transform=ax.transAxes, ha="left", va="top", fontsize=6, linespacing=1.1)
    ax.set_xlim(-3, 2); ax.set_ylim(-3, 2); ax.set_xticks(np.arange(-3, 3, 1)); ax.set_yticks(np.arange(-3, 3, 1))
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel(""); ax.set_ylabel(lib, fontsize=6, labelpad=1)
    ax.yaxis.label.set_visible(True)   # sublibrary name on every panel (sharey would hide it)
    finish(fig, ax, i)
fig.supxlabel(r"Mean monomeric yield (log$_{10}$ mg/L)", fontsize=6, y=0.015)
fig.supylabel(r"Sublibrary monomeric yield (log$_{10}$ mg/L)", fontsize=6, x=0.015)
fig.tight_layout(rect=(0.035, 0.04, 1, 1), pad=0.4, w_pad=0.6, h_pad=1.0)
fig.savefig(os.path.join(out, "ExtFig2.pdf"), bbox_inches="tight"); plt.close(fig)
