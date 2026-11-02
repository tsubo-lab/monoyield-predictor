"""ExtFig1a: replicate reproducibility of the lib77 sub16/17 Astral runs (rep1 = barcode-test-1, rep2 = barcode-test-2), 1,038 proteins.
left  = mean log10 intensity of all 40 barcodes per protein, rep1 vs rep2 (monomeric protein);
mid   = Pearson R between the replicates vs number of barcodes sampled per design (25 random seeds, mean +- SD);
right = rep1 vs rep2 using 15 random barcodes per design (monomer).
Tables: S22c (protein level, left), S22b (barcode level, mid and right)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import load, outdir
from common.style import set_style

set_style()
COLOR_SUP, COLOR_MONO = "#087295", "#0493C2"
N_BARCODES_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
N_SEEDS = 25
PSEUDOCOUNT = 10_000
out = outdir("fig1")


def finish_scatter(ax, title=None):
    ax.plot([-10, 10], [-10, 10], c="black", ls=":", linewidth=1, zorder=1, alpha=0.7)
    ax.set_aspect("equal", "box")
    ticks = np.arange(0, 11, 2)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
    ax.set_xlim(4, 10); ax.set_ylim(4, 10)
    ax.tick_params(axis="both", which="major", pad=1)
    if title:
        ax.set_title(title)


def annotate(ax, x, y, digits):
    ax.text(x=0.05, y=0.85, s="n = " + str(len(x)), transform=ax.transAxes)
    ax.text(x=0.05, y=0.95, s="r = " + str(round(stats.pearsonr(x, y)[0], digits)), transform=ax.transAxes)


# ---- left: all barcodes, protein level --------------------------------------------------------------------
prot = load("S22c")
prot = prot[~prot["name"].str.contains("spike", case=False)]
fig = plt.figure(figsize=(8 / 2.54, 4 / 2.54), tight_layout=True, dpi=300)
for pos, frac, color, title in [(2, "monomer", COLOR_MONO, "Monomeric protein")]:   # drawn in the right half of the 1x2 grid, as in the original notebook figure
    ax = fig.add_subplot(1, 2, pos)
    x, y = prot[f"log_{frac}_rep1"], prot[f"log_{frac}_rep2"]
    sns.scatterplot(x=x, y=y, alpha=0.25, s=6, ax=ax, color=color)
    annotate(ax, x, y, 2)
    ax.set_xlabel("log10(mean intensity)\nRelpicate 1")
    ax.set_ylabel("log10(mean intensity)\nRelpicate 2")
    finish_scatter(ax, title)
    print(f"left {frac}: n={len(x)} r={stats.pearsonr(x, y)[0]:.3f}")
fig.savefig(os.path.join(out, "ExtFig1a-left.pdf"))

# ---- barcode level: sample j barcodes per (protein, sublibrary) -------------------------------------------
bc = load("S22b")
bc["pos"] = bc.groupby(["name", "sublib"]).cumcount()
assert bc["pos"].max() == 39 and bc.groupby(["name", "sublib"]).size().eq(40).all()
mono = bc.assign(add=bc["monomer"].fillna(0) + PSEUDOCOUNT)
wide = mono.pivot_table(index=["sublib", "name"], columns="pos", values="add")  # 40 barcodes per row, original order


def replicate_logs(n_bc, seed):
    """log10 mean intensity per protein for both replicates; every group has 40 barcodes, so the same positions are drawn for all."""
    idx = np.random.RandomState(seed).choice(40, n_bc, replace=False)  # identical to DataFrame.sample(n_bc, random_state=seed)
    m = np.log10(wide[idx].mean(axis=1))
    reps = [m.loc[s] for s in sorted(m.index.get_level_values("sublib").unique())]
    return reps[0].to_frame("rep1").join(reps[1].to_frame("rep2"))


# mid
rows = []
for n_bc in N_BARCODES_GRID:
    for seed in range(N_SEEDS):
        d = replicate_logs(n_bc, seed)
        rows.append({"barcodes": n_bc, "seed": seed, "monomer_R": round(stats.pearsonr(d["rep1"], d["rep2"])[0], 3)})
boot = pd.DataFrame(rows)
print(boot.groupby("barcodes")["monomer_R"].agg(["mean", "std"]).round(3))
fig = plt.figure(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
ax = fig.add_subplot(111)
sns.lineplot(data=boot, x="barcodes", y="monomer_R", errorbar=("sd", 1), err_style="bars", lw=1, err_kws={"lw": 1},
             color=COLOR_MONO, marker="o", markersize=3)
ax.set_xlabel("Number of barcodes per design")
ax.set_ylabel("Pearson R between\nindependent barcodes sets")
ax.set_xticks(np.arange(5, 41, 10)); ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.set_xlim(2.5, 45); ax.set_ylim(0.5, 1)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.axvline(15, linestyle="--", linewidth=0.5, color="grey")
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(out, "ExtFig1a-mid.pdf"), bbox_inches="tight")

# right
d = replicate_logs(15, 42)
fig = plt.figure(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
ax = fig.add_subplot(1, 1, 1)
sns.scatterplot(x=d["rep1"], y=d["rep2"], alpha=0.25, s=6, ax=ax, color=COLOR_MONO)
annotate(ax, d["rep1"], d["rep2"], 3)
ax.set_xlabel("log10 (mean intensity)")
ax.set_ylabel("log10 (mean intensity)\nRelpicate 2")
finish_scatter(ax)
print(f"right (15 barcodes): n={len(d)} r={stats.pearsonr(d['rep1'], d['rep2'])[0]:.3f}")
fig.savefig(os.path.join(out, "ExtFig1a-right.pdf"), bbox_inches="tight")
