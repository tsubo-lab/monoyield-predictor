"""Fig4c (Pearson R) and ExtFig10c (Spearman R): test-fold predictions of solubility tools, XGBoost and LoRA models
vs observed monomer yield (All data, De novo) and vs the mutational effect (mutant - WT, DMS).

Tables: S15a (observed + predictions, all/de novo), S15b (mutational effect).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_common import *

set_style_pred()
OUT = outdir("fig4")
a, den, b = load_pred()

# ---- statistics per model and data type ----
records = []
for key, col in NB_KEY.items():
    for dt in ['all_data', 'denovo', 'dms']:
        yp, yt = condition_data(key, dt, a, den, b)
        if len(yp) < 2 or np.abs(yp).sum() == 0: continue
        records.append(dict(model=key, data_type=dt, n=len(yp), r=stats.pearsonr(yp, yt)[0], rho=stats.spearmanr(yp, yt)[0]))
stat = pd.DataFrame(records)
print(stat[stat.model.isin(['ESMC 6B', 'Saprot 1.3B', 'Aggrescan3D', 'esm3_d'])].round(3).to_string(index=False))

# ---- layout (groups, order, labels, colours) ----
COMPARISON_GROUPS = [
    ('Solubility tools', TOOLS),
    ('XGBoost: base', ['feature', 'saprot_structure', 'esm2', 'esm3']),
    ('XGBoost: + features', ['saprot_structure_f', 'esm2_f', 'esm3_f']),
    ('XGBoost: + ΔG', ['feature_d', 'saprot_structure_d', 'esm2_d', 'esm3_d']),
    ('LoRA: linear head', ['ESM3 linear', 'ESMC 600M', 'Saprot 1.3B linear', 'ESMC 6B linear']),
    ('LoRA: MLP head', ['ESM3', 'Saprot 1.3B', 'ESMC 6B']),
]
LABELS = {'feature': 'Features', 'feature_d': 'Features + ΔG', 'saprot_structure': 'SaProt 1.3B', 'esm2': 'ESM2 3B',
          'esm3': 'ESMC 6B', 'saprot_structure_f': 'SaProt 1.3B + features', 'esm2_f': 'ESM2 3B + features',
          'esm3_f': 'ESMC 6B + features', 'saprot_structure_d': 'SaProt 1.3B + ΔG', 'esm2_d': 'ESM2 3B + ΔG',
          'esm3_d': 'ESMC 6B + ΔG', 'ESM3 linear': 'ESM3-open 1.4B', 'ESMC 600M': 'ESMC 600M',
          'Saprot 1.3B linear': 'SaProt 1.3B', 'ESMC 6B linear': 'ESMC 6B', 'ESM3': 'ESM3-open 1.4B',
          'Saprot 1.3B': 'SaProt 1.3B', 'ESMC 6B': 'ESMC 6B \n(ESMC-MonoYield)'}
group_cmaps = [gc_, bc_, bc_, bc_, pc_, pc_]
order, ypos, colors, bounds = [], [], {}, []
y = 0
for gi, (glabel, models) in enumerate(COMPARISON_GROUPS):
    start = y
    for m, shade in zip(models, np.linspace(0.3, 0.8, len(models))):
        order.append(m); ypos.append(y); colors[m] = group_cmaps[gi](shade); y += 1
    bounds.append((glabel, start - 0.4, y - 0.6))
    y += 0.9
colors['ESMC 6B linear'] = RED_SHADES[2]
colors['ESMC 6B'] = RED_SHADES[3]

def bars(metric, name, xlabel):
    set_style_pred()
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(20 / 2.54, 11.5 / 2.54), dpi=300)
    fig.subplots_adjust(left=0.38, right=0.99, bottom=0.08, top=0.95, wspace=0.2)
    allv = stat[metric].to_numpy()
    xleft, xright = min(0, float(np.nanmin(allv)) - 0.04), max(0.8, float(np.nanmax(allv)) + 0.1)
    for col, dt in enumerate(['all_data', 'denovo', 'dms']):
        ax = axes[col]
        values = stat[stat['data_type'] == dt].set_index('model').loc[order, metric]
        ax.barh(ypos, values, height=0.72, color=[colors[m] for m in order], edgecolor='none')
        for yp, v in zip(ypos, values):
            ax.text(max(0, v) + 0.015, yp, f'{v:.2f}', ha='left', va='center', fontsize=6)
        for prev, nxt in zip(bounds[:-1], bounds[1:]):
            ax.axhline((prev[2] + nxt[1]) / 2, color='grey', linewidth=0.5, linestyle=(0, (1, 3)))
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.set_xlim(xleft, xright)
        ax.set_ylim(ypos[-1] + 0.8, -0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABELS.get(m, m) for m in order], fontsize=6)
        ax.tick_params(axis='y', length=0, pad=3, labelleft=(col == 0))
        if col == 0:
            for tl, m in zip(ax.get_yticklabels(), order):
                if m == 'ESMC 6B':
                    tl.set_transform(tl.get_transform() + mpl.transforms.ScaledTranslation(0, -2 / 72, fig.dpi_scale_trans))
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_xticks([0, 0.35, 0.7])
        ax.set_title({'all_data': 'All data', 'denovo': 'De novo', 'dms': 'Mutational effect\n(Mut yield - WT yield)'}[dt], fontsize=6)
        ax.spines['left'].set_visible(False)
        if col == 0:
            for glabel, start, end in bounds:
                trans = mpl.transforms.blended_transform_factory(fig.transFigure, ax.transData)
                ax.plot([0.271, 0.261, 0.261, 0.271], [start, start, end, end], transform=trans, color='black', linewidth=0.5, clip_on=False)
                ax.text(0.257, (start + end) / 2, glabel.replace(': ', ':\n'), transform=trans, va='center', ha='right', fontsize=6, clip_on=False)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)

bars('r', 'Fig4c.pdf', 'Pearson R')
bars('rho', 'ExtFig10c.pdf', 'Spearman R')
