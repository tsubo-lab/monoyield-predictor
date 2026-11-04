"""ExtFig 13a: monomer yield / monomer-dimer ratio of BoltzGen vs RFantibody nanobodies (boxplots).
ExtFig 13b: monomer yield vs dimer yield (13b-1) and monomer yield vs monomer/dimer ratio (13b-2).

Tables: S07 (yields, method; spikes removed).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import matplotlib.pyplot as plt
import seaborn as sns
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig5")
CLIP_LOWER = -1.5
TSM2 = "#0493C2"
YLABEL = {"monomer_log_mgL": "Monomer yield\n(log$_{10}$ mg/L)", "dimer_log_mgL": "Dimer yield\n(log$_{10}$ mg/L)",
          "monomer_dimer_ratio_log": "Monomer / Dimer\n(log$_{10}$ ratio)"}

df = load("S07")
df = df[~df["is_spike"]].copy()
# yields are clipped at CLIP_LOWER; the S07 ratio (unclipped monomer - dimer) is used as is
df[["monomer_log_mgL", "dimer_log_mgL"]] = df[["monomer_log_mgL", "dimer_log_mgL"]].clip(lower=CLIP_LOWER)
print("n nanobodies:", len(df), df["method"].value_counts().to_dict())

for name, ycol in [("ExtFig13a-yield", "monomer_log_mgL"), ("ExtFig13a-ratio", "monomer_dimer_ratio_log")]:
    print(name, df.groupby("method")[ycol].median().round(3).to_dict())
    fig, ax = plt.subplots(figsize=(2.5 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
    sns.boxplot(data=df, x="method", y=ycol, order=["bg", "rfab"], gap=0.2, linewidth=0.5, fliersize=1, ax=ax)
    ax.set_xticks(ax.get_xticks(), labels=["BoltzGen", "RFantibody"], rotation=45, ha="right")
    ax.set_xlabel("")
    ax.tick_params(axis="both", which="major", pad=1)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel(YLABEL[ycol], labelpad=0.1)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

for name, xcol in [("ExtFig13b-1", "dimer_log_mgL"), ("ExtFig13b-2", "monomer_dimer_ratio_log")]:
    fig = plt.figure(figsize=(3.5 / 2.54, 3.5 / 2.54), tight_layout=True, dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    sns.scatterplot(df, y="monomer_log_mgL", x=xcol, alpha=0.2, s=6, ax=ax, color=TSM2)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.set_xlabel(YLABEL[xcol], labelpad=0.1)
    ax.set_ylabel(YLABEL["monomer_log_mgL"], labelpad=0.1)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
