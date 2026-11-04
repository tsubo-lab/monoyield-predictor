"""ExtFig 14a: cDNA-display log2 enrichment (2nd round, PD-L1 and PD-1 selection) vs measured monomer yield (top row)
and monomer/dimer ratio (bottom row), one 2 x 2 figure (columns: PD-L1, PD-1 selection).

Tables: S19 (enrichment with matched yields; same nanobodies as S07). Yield clipped at -1.5, ratio not clipped.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import matplotlib.pyplot as plt
import seaborn as sns
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig5")
CONDITIONS = ["2nd_PDL1", "2nd_PD1"]
FS = 6  # all text 6 pt
LABEL_PAD = 0.012  # figure fraction between tick labels and axis label

df = load("S19")
df["monomer_log_mgL"] = df["monomer_log_mgL"].clip(lower=-1.5)  # yield clipped, ratio not
print("n nanobodies:", len(df))
ROWS = [("yield", "monomer_log_mgL", "log$_{10}$ (monomeric yield)\n(mg/L)"),
        ("ratio", "monomer_dimer_ratio_log", "log$_{10}$ (monomeric/dimeric yield)")]
TITLES = {"2nd_PDL1": "Against PD-L1", "2nd_PD1": "Against PD-1"}

fig, axes = plt.subplots(2, 2, figsize=(6.5 / 2.54, 6.5 / 2.54), dpi=300)
fig.subplots_adjust(left=0.2, right=0.97, top=0.92, bottom=0.14, hspace=0.8, wspace=0.3)
for r, (kind, xcol, _) in enumerate(ROWS):
    for c, cond in enumerate(CONDITIONS):
        ycol = cond + "_log2_enrich_finite"
        d = df[[xcol, ycol]].dropna()
        print(f"ExtFig14a-{kind} {cond}: n={len(d)} Pearson r={d[xcol].corr(d[ycol]):.3f}")
        ax = axes[r, c]
        sns.scatterplot(df, x=xcol, y=ycol, alpha=0.2, s=5, ax=ax)
        ax.set_ylim(-3, 5)
        ax.set_title(TITLES[cond], fontsize=FS)
        ax.tick_params(labelsize=FS, pad=1)
        ax.set_xlabel("")
        ax.set_ylabel("")
# shared labels: one x label centred under each row, one y label centred on the left, placed just outside the tick labels
fig.canvas.draw()
inv = fig.transFigure.inverted()
for r, (_, _, xlabel) in enumerate(ROWS):
    l = axes[r, 0].get_position().x0
    right = axes[r, 1].get_position().x1
    ymin = min(inv.transform_bbox(t.get_window_extent()).y0 for ax in axes[r] for t in ax.get_xticklabels() if t.get_text())
    fig.text((l + right) / 2, ymin - LABEL_PAD, xlabel, ha="center", va="top", fontsize=FS)
top, bottom = axes[0, 0].get_position().y1, axes[1, 0].get_position().y0
xmin = min(inv.transform_bbox(t.get_window_extent()).x0 for ax in axes[:, 0] for t in ax.get_yticklabels() if t.get_text())
fig.text(xmin - LABEL_PAD, (top + bottom) / 2, "log$_{2}$(enrichment + 0.1)", rotation=90, ha="right", va="center", fontsize=FS)
fig.savefig(os.path.join(OUT, "ExtFig14a.pdf"))  # canvas is exactly 6.5 x 6.5 cm
plt.close(fig)
