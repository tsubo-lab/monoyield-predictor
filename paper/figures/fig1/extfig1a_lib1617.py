"""ExtFig1a: replicate reproducibility of the lib77 sub16/17 Astral runs (rep1 = barcode-test-1, rep2 = barcode-test-2), 1,038 proteins; monomer only.
left  = mean log10 intensity of all 40 barcodes per protein, rep1 vs rep2;
mid   = rep1 vs rep2 using 15 random barcodes per design;
right = Pearson R between the replicates vs number of barcodes sampled per design (25 random seeds, mean +- SD).
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
COLOR_MONO = "#0493C2"
N_BARCODES_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
N_SEEDS = 25
PSEUDOCOUNT = 10_000
out = outdir("fig1")


prot = load("S22c")
prot = prot[~prot["name"].str.contains("spike", case=False)]

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


# bootstrap over the number of barcodes
rows = []
for n_bc in N_BARCODES_GRID:
    for seed in range(N_SEEDS):
        d = replicate_logs(n_bc, seed)
        rows.append({"barcodes": n_bc, "seed": seed, "monomer_R": round(stats.pearsonr(d["rep1"], d["rep2"])[0], 3)})
boot = pd.DataFrame(rows)
print(boot.groupby("barcodes")["monomer_R"].agg(["mean", "std"]).round(3))
b15 = replicate_logs(15, 42)


def rep_scatter(ax, x, y, title):
    sns.scatterplot(x=x, y=y, alpha=0.25, s=6, ax=ax, color=COLOR_MONO, linewidth=0)
    ax.plot([-10, 10], [-10, 10], c="black", ls=":", linewidth=1, zorder=1, alpha=0.7)
    r = stats.pearsonr(x, y)[0]
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, va="top")
    ax.text(0.05, 0.85, f"n = {len(x)}", transform=ax.transAxes, va="top")
    ax.set_title(title)
    ax.set_xlabel("log$_{10}$(mean intensity)\nReplicate 1", labelpad=1)
    ax.set_ylabel("log$_{10}$(mean intensity)\nReplicate 2", labelpad=1)
    ax.set_box_aspect(1)
    ticks = np.arange(4, 11, 2)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim(4, 10); ax.set_ylim(4, 10)
    ax.tick_params(axis="both", which="major", pad=1)
    print(f"{title}: n={len(x)} r={r:.3f}")


fig = plt.figure(figsize=(12 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
rep_scatter(fig.add_subplot(gs[0]), prot["log_monomer_rep1"], prot["log_monomer_rep2"], "Each protein 40 barcodes")
rep_scatter(fig.add_subplot(gs[1]), b15["rep1"], b15["rep2"], "Each protein 15 barcodes")
ax = fig.add_subplot(gs[2])
ax.set_box_aspect(1)
sns.lineplot(data=boot, x="barcodes", y="monomer_R", errorbar=("sd", 1), err_style="bars", lw=1, err_kws={"lw": 1},
             color=COLOR_MONO, marker="o", markersize=3, ax=ax)
ax.axvline(15, linestyle="--", linewidth=0.5, color="grey")
ax.set_xlabel("Number of barcodes\nper design")
ax.set_ylabel("Pearson R between\nindependent barcodes sets")
ax.set_xticks(np.arange(5, 41, 10)); ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.set_xlim(2.5, 45); ax.set_ylim(0.5, 1)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(out, "ExtFig1a.pdf"), bbox_inches="tight")
