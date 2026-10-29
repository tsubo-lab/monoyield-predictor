"""Fig3e (1PGA) and Fig3f (1AOY): dG-vs-yield scatter with linear fit, dG heatmap and monomeric-yield heatmap of the full DMS.
The structure renders of Fig3e/f are out of scope. Tables: S04 (yield, dG, dG CI, aa_seq, variant names)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from common.tables import outdir
from common.style import set_style
import dms_data
import dms_plots

OUT = outdir("fig3")
set_style()

PANELS = {"Fig3e": "1PGA", "Fig3f": "1AOY"}
_, df_dms, df_wt = dms_data.heatmap_view()

for panel, protein in PANELS.items():
    data = df_dms[df_dms["name"].str.contains(protein, na=False)].copy()

    fig = plt.figure(figsize=(4 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
    n, r, slope = dms_plots.scatter_with_fit(fig.add_subplot(1, 1, 1), data, title=protein)
    print(f"{panel} {protein}: n = {n}, r = {r:.3f}, slope = {slope:.3f}")
    fig.savefig(os.path.join(OUT, f"{panel}-{protein}-scatter.pdf"), bbox_inches="tight")

    for abundance, vmin, vmax, hatch, name in [("deltaG", -2, 5, True, "dG-heatmap"), ("monomer_log_mgL", -1.5, 1.5, False, "yield-heatmap")]:
        fig = plt.figure(figsize=(11 / 2.54, 5 / 2.54), dpi=300, tight_layout=True)
        dms_plots.heatmap(fig.add_subplot(1, 1, 1), data, df_wt, protein, abundance, vmin, vmax, show_ci_hatch=hatch)
        fig.savefig(os.path.join(OUT, f"{panel}-{protein}-{name}.pdf"), bbox_inches="tight")
