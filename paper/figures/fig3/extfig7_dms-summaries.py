"""ExtFig7 A-E: per-protein DMS summary (scatter with fit, yield heatmap, dG heatmap) for 9 natural proteins.
The structure image (top left of each original panel) is a rendered PDB and is left blank. Tables: S04."""
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

PANELS = {"A": ["1PGA", "1AOY"], "B": ["3L1X", "1R69"], "C": ["2KCM", "2BTT"], "D": ["6M3N", "2LGW"], "E": ["1UZC"]}
_, df_dms, df_wt = dms_data.heatmap_view()

for panel, proteins in PANELS.items():
    for protein in proteins:
        data = df_dms[df_dms["name"].str.contains(protein, na=False, regex=True)].copy()
        fig = plt.figure(figsize=(18 / 2.54, 10 / 2.54), dpi=300, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.0, 3.6], height_ratios=[1, 1])
        ax_struct, ax_scatter = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        ax_mono, ax_dg = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        ax_struct.set_axis_off()                       # structure render: out of scope
        n, r, slope = dms_plots.scatter_with_fit(ax_scatter, data, text_size=7, title=protein)
        ax_scatter.title.set_fontsize(8)
        ax_scatter.tick_params(axis="both", which="major", pad=1, labelsize=7)
        dms_plots.heatmap(ax_mono, data, df_wt, protein, "monomer_log_mgL", -1.5, 1.5, show_ci_hatch=False)
        ax_mono.set_title(r"log$_{10}$ (monomeric yield) (mg/L)", fontsize=8)
        dms_plots.heatmap(ax_dg, data, df_wt, protein, "deltaG", -2, 5)
        ax_dg.set_title(r"$\Delta G$ (kcal/mol)", fontsize=8)
        fig.suptitle(protein, fontsize=11)
        print(f"ExtFig7{panel} {protein}: n = {n}, r = {r:.3f}, slope = {slope:.3f}")
        fig.savefig(os.path.join(OUT, f"ExtFig7{panel}-{protein}.pdf"), bbox_inches="tight")
        plt.close(fig)
