"""Fig1c: fold error of the mean spike-v2 intensity vs the number of barcodes per set (bootstrap).
For each n barcodes (60 ... 5) and 50 trials, n barcodes per set are subsampled (without replacement; the same barcodes are used
for both loading levels of a replicate) and the mean log10(intensity + 1e4) is compared with the value from all 60 barcodes
(averaged over the 4 conditions). Orange = individual trials, triangles = clipped values; blue dashed = 2-fold; red = 15 barcodes.
Tables: S12a (n1 = S2 rep1, n2 = S2 rep2, n3 = S3 rep1, n4 = S3 rep2 raw intensities; input ng per barcode)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.tables import load, outdir
from common.style import set_style

set_style()
PSEUDOCOUNT = 10_000
N_BOOTSTRAP = 50
N_BC_GRID = list(range(60, 0, -5))
RNG_SEED = 0
# (condition, replicate) of the four calibration conditions in the order of the source notebook; columns of S12a
CAL_CONDITIONS = [("S3", "rep1", "n3"), ("S2", "rep1", "n1"), ("S3", "rep2", "n4"), ("S2", "rep2", "n2")]
REP_COLS = {"rep1": ["n1", "n3"], "rep2": ["n2", "n4"]}
LO, HI = 10 ** -0.7, 10 ** 0.7  # clip limits

s12a = load("S12a")
groups = {name: g.reset_index(drop=True) for name, g in s12a.groupby("name", sort=False)}
n_per_set = {len(g) for g in groups.values()}
assert n_per_set == {60}, n_per_set
true_ng = {name: g["input_ng_per_barcode_n1"].iloc[0] for name, g in groups.items()}  # ng per barcode of the S2 loading
set_names_by_conc = sorted(groups, key=true_ng.get)


def mean_log_intensity(group, col, idx):
    return np.log10(group.loc[idx, f"raw_{col}"].fillna(0).add(PSEUDOCOUNT).mean())


def aggregate(rep, n_bc, rng):
    """Subsample n_bc barcodes per set; return {set: {col: log10 mean intensity}} for the columns of this replicate."""
    out = {}
    for name, g in groups.items():
        idx = rng.choice(len(g), size=n_bc, replace=False) if n_bc < len(g) else np.arange(len(g))
        out[name] = {col: mean_log_intensity(g, col, idx) for col in REP_COLS[rep]}
    return out


# truth: mean over the 4 conditions of the log10 intensity from all barcodes
full = {rep: aggregate(rep, 60, None) for rep in REP_COLS}
x_true = {name: np.mean([full[rep][name][col] for _, rep, col in CAL_CONDITIONS]) for name in groups}

records = []
for n_bc in N_BC_GRID:
    for trial in range(N_BOOTSTRAP):
        rng = np.random.default_rng(RNG_SEED * 100_000 + n_bc * 1_000 + trial)
        sig = {rep: aggregate(rep, n_bc, rng) for rep in ("rep1", "rep2")}
        for sample, rep, col in CAL_CONDITIONS:
            for name in groups:
                records.append((n_bc, trial, sample, rep, name, 10 ** (sig[rep][name][col] - x_true[name])))
res = pd.DataFrame(records, columns=["n_bc", "trial", "sample", "rep", "set", "ratio"])
res["fold"] = 10 ** res["ratio"].apply(np.log10).abs()
print("records:", len(res))
for n in (60, 30, 15, 10):
    f = res.loc[res.n_bc == n, "fold"]
    print(f"n_bc={n}: median fold {f.median():.2f}  within 2x {100 * (f <= 2).mean():.1f}%  95th pct {f.quantile(.95):.2f}")

fig, axes = plt.subplots(1, 5, figsize=(13 / 2.54, 4 / 2.54), dpi=300, sharey=True)
rng_jitter = np.random.default_rng(2)
for ax, name in zip(axes, set_names_by_conc):
    sub = res[res["set"] == name]
    x = sub["n_bc"].to_numpy() + rng_jitter.uniform(-1.2, 1.2, size=len(sub))
    y = sub["ratio"].to_numpy()
    inside, over, under = (y >= LO) & (y <= HI), y > HI, y < LO
    ax.scatter(x[inside], y[inside], s=1.5, alpha=0.25, color="tab:orange", linewidths=0)
    ax.scatter(x[over], np.full(over.sum(), HI), s=6, marker="^", alpha=0.6, color="tab:orange", linewidths=0, clip_on=False)
    ax.scatter(x[under], np.full(under.sum(), LO), s=6, marker="v", alpha=0.6, color="tab:orange", linewidths=0, clip_on=False)
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    for ref in (2.0, 0.5):
        ax.axhline(ref, color="tab:blue", linewidth=0.6, linestyle="--")
    ax.axvline(15, color="tab:red", linewidth=0.6, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(LO * 0.8, HI * 1.25)
    ax.set_title(f"{true_ng[name]:.3g} ng/barcode", fontsize=6)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xticks([60, 50, 40, 30, 20, 10])
    ax.grid(axis="both", which="both", linewidth=0.4, alpha=0.4)
axes[0].set_yticks([LO, 0.5, 1, 2, HI])
axes[0].set_yticklabels([r"$\leq 0.2$", "0.5", r"$1$", "2", r"$\geq 5$"])
axes[0].minorticks_off()
axes[0].set_ylabel("Fold error\nof mean intensity", fontsize=6, labelpad=2)
fig.supxlabel("Number of barcodes", fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(outdir("fig1"), "Fig1c.pdf"), bbox_inches="tight")
