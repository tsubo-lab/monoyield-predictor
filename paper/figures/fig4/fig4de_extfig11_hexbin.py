"""Fig4d (hexbin, all data: ESMC-MonoYield, SaProt LoRA, AGGRESCAN3D), Fig4e (de novo: ESMC-MonoYield, SaProt LoRA)
and ExtFig11A-C (hexbin grids of all models and solubility tools; all data / de novo / mutational effect).

Tables: S15a, S15b.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_common import *

set_style_pred()
OUT = outdir("fig4")
a, den, b = load_pred()
SINGLE = {'ESMC 6B': 'ESMC-MonoYield', 'Saprot 1.3B': 'SaProt-LoRA', 'AGGRESCAN3D': 'AGGRESCAN3D'}

def single(model, dt, fname):
    fig = plt.figure(figsize=(3.5 / 2.54, 3.5 / 2.54), dpi=300, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    is_tool = model in TOOLS
    yp, yt = condition_data(model, dt, a, den, b)
    r = hexbin_panel(ax, yp, yt, MODEL_LABEL.get(model, model), same_scale=not is_tool, cmap=hexbin_cmap_for(model))
    if not is_tool: ax.set_xlim(-2.5, 2.0)
    print(f'{fname}: n={len(yp)} r={r:.3f}')
    plt.savefig(os.path.join(OUT, fname), bbox_inches='tight')
    plt.close(fig)

for m, lab in SINGLE.items():
    single(m, 'all_data', f'Fig4d_{lab}.pdf')
for m in ['ESMC 6B', 'Saprot 1.3B']:
    single(m, 'denovo', f'Fig4e_{SINGLE[m]}.pdf')

# ---- ExtFig11A-C: 8 models per page, all data / de novo / mutational effect ----
order = ['ESM3', 'ESMC 600M', 'ESMC 6B', 'Saprot 1.3B'] + KEYS + TOOLS
pages = [order[i:i + 8] for i in range(0, len(order), 8)]
for pi, page in zip('ABC', pages):
    fig = plt.figure(figsize=(17 / 2.54, 17 / 2.54), dpi=300)
    outer = fig.add_gridspec(4, 2, left=0.065, right=0.99, bottom=0.06, top=0.95, wspace=0.16, hspace=0.26)
    for i, model in enumerate(page):
        row, col = divmod(i, 2)
        block = outer[row, col].subgridspec(2, 1, height_ratios=[0.12, 1], hspace=0.02)
        heading = fig.add_subplot(block[0]); heading.axis('off')
        heading.text(0.5, 0.5, MODEL_LABEL.get(model, model), ha='center', va='center', fontsize=6)
        panels = block[1].subgridspec(1, 3, wspace=0.45)
        for j, dt in enumerate(['all_data', 'denovo', 'dms']):
            ax = fig.add_subplot(panels[0, j])
            yp, yt = condition_data(model, dt, a, den, b)
            hexbin_panel(ax, yp, yt, dt, same_scale=model not in TOOLS, cmap=hexbin_cmap_for(model))
            if dt in ('all_data', 'denovo'): ax.set_ylim(-1.6, 2.2)
            ax.set_box_aspect(1)
            ax.set_title({'all_data': 'All data', 'denovo': 'De novo', 'dms': 'Mutational effect'}[dt], fontsize=6, pad=10)
            ax.set_xlabel('Predicted' if dt != 'dms' else 'Predicted\nmutational effect', fontsize=6, labelpad=1)
            ax.set_ylabel('Observed' if j == 0 else '', fontsize=6, labelpad=1)
            ax.tick_params(axis='both', labelsize=6, pad=1)
            ax.locator_params(axis='both', nbins=3)
    fig.savefig(os.path.join(OUT, f'ExtFig11{pi}.pdf'))
    plt.close(fig)
