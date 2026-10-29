"""Fig1d: left = spike v2 calibration (mean +- SD of the four calibration conditions per known amount, common regression line);
right = spike v1 reference proteins placed on the same line (reference amounts inferred from the line, SD over valid channels).
Tables: S12c (calibration points and line), S11b (spike v1 channels, MS_log, inferred log10 ng/barcode)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
from common.tables import load, outdir
from common.style import set_style

set_style()
cal = load("S12c")
v1 = load("S11b")
channels = [c for c in v1.columns if c.startswith("S") and "_mean_add4_log_rep" in c]

slope, intercept, r2 = cal[["common_calibration_slope", "common_calibration_intercept", "common_calibration_r2"]].iloc[0]
print(f"calibration: slope={slope:.4f} intercept={intercept:.4f} R2={r2:.4f} n_points={len(cal)}")

summary = (cal.groupby("log10_ng_per_barcode", as_index=False)
              .agg(log10_ms_mean=("log10_ms_intensity", "mean"), log10_ms_sd=("log10_ms_intensity", "std")))
v1["log10_ms_sd"] = v1[channels].std(axis=1, skipna=True)
print(f"spike v1 references: n={len(v1)}")

x_all = np.concatenate([cal["log10_ms_intensity"].to_numpy(), v1[channels].to_numpy().ravel()])
x_all = x_all[np.isfinite(x_all)]
x_line = np.linspace(x_all.min() - 0.2, x_all.max() + 0.2, 200)
y_line = slope * x_line + intercept


def finish(ax):
    ax.set_xlabel(r"log$_{10}$(MS intensity)")
    ax.set_ylabel(r"log$_{10}$(ng/barcode)")
    ax.tick_params(axis="both", which="major", pad=1)


fig, ax = plt.subplots(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
ax.errorbar(summary["log10_ms_mean"], summary["log10_ng_per_barcode"], xerr=summary["log10_ms_sd"], fmt="o",
            markersize=2, color="orange", capsize=1, elinewidth=0.5, capthick=0.5)
ax.plot(x_line, y_line, linewidth=0.8, color="orange")
finish(ax)
ax.grid(axis="both", linewidth=0.5, alpha=1)
fig.tight_layout()
fig.savefig(os.path.join(outdir("fig1"), "Fig1d-left.pdf"), bbox_inches="tight")

fig, ax = plt.subplots(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
ax.errorbar(v1["MS_log"], v1["log10_ng_per_barcode"], xerr=v1["log10_ms_sd"], fmt="s", markersize=2, capsize=1,
            elinewidth=0.5, capthick=0.5, label="Spike v1")
ax.plot(x_line, y_line, linewidth=0.8, color="orange")
ax.text(0.05, 0.95, f"n = {len(v1)}", transform=ax.transAxes, ha="left", va="top", fontsize=5)
finish(ax)
ax.grid(axis="both", linewidth=0.5, alpha=0.8)
fig.tight_layout()
fig.savefig(os.path.join(outdir("fig1"), "Fig1d-right.pdf"), bbox_inches="tight")
