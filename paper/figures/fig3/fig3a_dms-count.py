"""Fig3a: number of proteins with a full DMS vs proteins with ~50 mutations each.
Tables: S04 (wild-type entries per sublibrary)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import load, outdir
from common.style import set_style
import dms_data
import dms_plots

OUT = outdir("fig3")
set_style()

_, df_wt, _ = dms_data.fit_view()
is_full = df_wt["sublib"].str.startswith("dms")           # dms-1..5: complete saturation libraries
counts = {"full": int(is_full.sum()), "partial": int((~is_full).sum())}   # muts-1 / denovo-8: ~50 mutations each
print("proteins:", counts)

fig, ax = plt.subplots(figsize=(3.5 / 2.54, 2 / 2.54), dpi=300, tight_layout=True)
sns.barplot(x=[counts["full"], counts["partial"]], y=["full", "partial"], orient="h", color=dms_plots.TSM[2], ax=ax)
ax.set_yticks([0, 1], labels=["Full DMS", "50 mutations\neach"])
ax.set_xticks([0, 25, 50])
ax.set_xlabel("n")
ax.set_ylabel("")
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(OUT, "Fig3a.pdf"))
