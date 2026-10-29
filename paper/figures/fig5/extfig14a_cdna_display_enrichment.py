"""ExtFig 14a: cDNA-display log2 enrichment (2nd round, PD-L1 and PD-1 selection) vs measured monomer yield (14a-yield)
and monomer/dimer ratio (14a-ratio).

Tables: S19 (enrichment with matched yields; same nanobodies as S07).
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

df = load("S19")
print("n nanobodies:", len(df))
for name, xcol in [("ExtFig14a-yield", "monomer_log_mgL"), ("ExtFig14a-ratio", "monomer_dimer_ratio_log")]:
    fig = plt.figure(figsize=(6 / 2.45, 3 / 2.45), tight_layout=True, dpi=300)
    for i, cond in enumerate(CONDITIONS, start=1):
        ycol = cond + "_log2_enrich_finite"
        d = df[[xcol, ycol]].dropna()
        print(f"{name} {cond}: n={len(d)} Pearson r={d[xcol].corr(d[ycol]):.3f}")
        ax = fig.add_subplot(1, 2, i)
        sns.scatterplot(df, x=xcol, y=ycol, alpha=0.2, s=5, ax=ax)
        ax.set_ylim(-3, 5)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
