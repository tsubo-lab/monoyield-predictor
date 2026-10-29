"""Fig4f (positive rate vs fraction selected, de novo), ExtFig12a (cumulative observed yield of the top 1/5/10 % predictions, de novo),
ExtFig12d left (mutational-effect positive rate, >= 5-fold vs WT) and ExtFig12d right (cumulative observed effect of the top 1/5/10 %).

Tables: S15a (de novo observed + predictions), S15b (mutational effect observed + predictions).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_common import *
import seaborn as sns
from matplotlib.lines import Line2D

set_style_pred()
OUT = outdir("fig4")
a, den, b = load_pred()
FRACS = (0.01, 0.05, 0.10, 0.15, 0.20)
PLOT_MODELS = ['ESMC 6B', 'Saprot 1.3B', 'esm3_d', 'Aggrescan3D', 'NetSolP']      # notebook filter '1.3B|6B|esm3_d|Aggrescan|NetSol'
ALL_MODELS = list(LORA) + KEYS + TOOLS                                              # models of the precision table
LABEL = lambda m: MODEL_LABEL.get(m, m)
COLOR = lambda m: GROUP_PALETTE[classify_group(m)]

def table(dt):
    return {'denovo': (den, 'y_true'), 'dms': (b, 'y_delta_true')}[dt]

def model_frame(m, dt):
    d, tc = table(dt)
    return d[['name', NB_KEY[m], tc]].rename(columns={NB_KEY[m]: 'pred', tc: 'truth'}).dropna()

def precision_table(dt, threshold, models):
    rows = []
    for m in models:
        d = model_frame(m, dt)
        rows.append(precision_at(d['truth'].to_numpy(), d['pred'].to_numpy(), threshold, FRACS).assign(model=m))
    return pd.concat(rows, ignore_index=True)

def positive_rate_panel(prec, base, ylabel, title, name, ymax, ymin, yticks=None):
    sub = prec[prec['model'].isin(PLOT_MODELS)].copy()
    sub['ci_low'], sub['ci_high'] = wilson(sub['n_pos_top'].to_numpy(float), sub['k'].to_numpy(float))
    fig = plt.figure(figsize=(6.2 / 2.54, 4 / 2.54), dpi=300)
    ax = fig.add_axes([0.19, 0.19, 0.25, 0.73])
    ax.set_ylim(ymin, ymax); ax.set_xlim(0, 21); ax.set_xticks([0, 10, 20])
    if yticks is None: ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, integer=True))
    else: ax.set_yticks(yticks)
    ends = sub.sort_values('top_percent').groupby('model').tail(1).sort_values('precision')
    gap = 7 / 72 / (4 / 2.54 * 0.73) * (ymax - ymin)
    label_y, prev = {}, -np.inf
    for _, row in ends.iterrows():
        label_y[row['model']] = max(row['precision'] * 100, prev + gap); prev = label_y[row['model']]
    for m, g in sub.groupby('model'):
        g = g.sort_values('top_percent')
        ax.errorbar(g['top_percent'], g['precision'] * 100,
                    yerr=np.maximum(0, np.vstack([g['precision'] - g['ci_low'], g['ci_high'] - g['precision']]) * 100),
                    marker='o', color=COLOR(m), ms=3, linewidth=1, capsize=1.4, elinewidth=0.45, capthick=0.45, zorder=3)
        ax.annotate(LABEL(m), xy=(20, label_y[m]), xytext=(4, 0), textcoords='offset points', va='center', ha='left',
                    fontsize=6, color=COLOR(m), annotation_clip=False)
    ax.axhline(base * 100, color='grey', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_xlabel('Fraction of selection (%)', fontsize=6, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=6, labelpad=2)
    ax.set_title(title, fontsize=6, pad=2)
    ax.tick_params(axis='both', labelsize=6, width=0.45, length=2, pad=1)
    for side in ['left', 'bottom']: ax.spines[side].set_linewidth(0.45)
    for side in ['top', 'right']: ax.spines[side].set_visible(False)
    ax.grid(False)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

# ---- Fig4f: de novo, positive = yield >= 10 mg/L ----
prec = precision_table('denovo', np.log10(10), ALL_MODELS)
base = prec['base'].mean()
sub = prec[prec['model'].isin(PLOT_MODELS)]
print('Fig4f baseline positive rate %.2f%%; precision of ESMC-MonoYield at 1/5/10/15/20%%:' % (base * 100),
      sub[sub.model == 'ESMC 6B']['precision'].round(3).tolist())
ymax = min(100, np.ceil((wilson(sub['n_pos_top'].to_numpy(float), sub['k'].to_numpy(float))[1].max() * 100 + 2) / 10) * 10)
positive_rate_panel(prec, base, 'Positive rate after selection\n(yield ≥ 10 mg/L)', 'De novo', 'Fig4f.pdf',
                    ymax, -1, np.arange(0, ymax + 1, 10))

# ---- ExtFig12d left: mutational effect, positive = >= 5-fold increase vs WT ----
prec = precision_table('dms', np.log10(5), PLOT_MODELS)
sub = prec.copy(); sub['ci_high'] = wilson(sub['n_pos_top'].to_numpy(float), sub['k'].to_numpy(float))[1]
ymax = max(5, min(100, np.ceil(sub['ci_high'].max() * 100 / 5) * 5))
print('ExtFig12d left: %d / %d mutants >= 5x (%.2f%%)' % (prec['n_positive'].iloc[0], prec['n_total'].iloc[0], prec['base'].iloc[0] * 100))
positive_rate_panel(prec, prec['base'].mean(), 'Positive rate after selection\n(≥ 5-fold yield increase vs WT)',
                    'Mutational effect\n(fold change ≥ 5x vs WT)', 'ExtFig12d_left.pdf', ymax, -ymax * 0.015)

# ---- ECDF of observed values among the top-k predictions (ExtFig12a, ExtFig12d right) ----
ECDF_MODELS, PCTS = ['ESMC 6B', 'Saprot 1.3B', 'Aggrescan3D', 'NetSolP'], [1, 5, 10]
colors = {m: COLOR(m) for m in ECDF_MODELS}
if colors['NetSolP'] == colors['Aggrescan3D']: colors['NetSolP'] = '#4C9F70'

def ecdf_figure(dt, name):
    d_all = model_frame('ESMC 6B', dt)
    baseline = d_all['truth'].to_numpy()
    ann_x = np.log10(5.0) if dt == 'dms' else np.log10(10.0)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12 / 2.54, 5.8 / 2.54), dpi=300)
    for ax, pct in zip(axes, PCTS):
        sns.ecdfplot(x=baseline, ax=ax, stat='proportion', color='0.45', linestyle='--', linewidth=0.8, label='No selection', zorder=1)
        anns = []
        for m in ECDF_MODELS:
            d = model_frame(m, dt)
            d = d.sort_values('name', kind='mergesort').sort_values('pred', ascending=False, kind='mergesort')
            sel = d.iloc[:max(1, int(np.ceil(len(d) * pct / 100)))]['truth'].to_numpy()
            sns.ecdfplot(x=sel, ax=ax, stat='proportion', color=colors[m], linewidth=0.8, label=LABEL(m))
            frac = np.mean(sel <= ann_x)
            anns.append((frac, m))
            ax.plot(ann_x, frac, 'o', color=colors[m], markersize=2, zorder=4)
            if pct == 10: print(f'{name}: {m} top10% n={len(sel)} share above threshold {1 - frac:.0%}')
        anns.sort()
        label_y = np.array([v for v, m in anns])
        for i in range(1, len(label_y)): label_y[i] = max(label_y[i], label_y[i - 1] + 0.085)
        if len(label_y) and label_y[-1] > 1.04: label_y -= label_y[-1] - 1.04
        for (frac, m), yy in zip(anns, label_y):
            ax.annotate(f'{1 - frac:.0%}', xy=(ann_x, frac), xytext=(ann_x - 0.18, yy), ha='right', va='center', fontsize=6,
                        color=colors[m], bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.85),
                        arrowprops=dict(arrowstyle='-', color=colors[m], lw=0.4), zorder=5)
        ax.set_title(f'Top {pct}%', fontsize=6, pad=2)
        ax.set_ylim(0, 1.1); ax.margins(x=0.03)
        ax.set_yticks([0, 0.5, 1]); ax.locator_params(axis='x', nbins=3)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=6, pad=1, length=2)
        ax.axvline(ann_x, color='grey', linestyle='--', linewidth=0.5)
        ax.grid(axis='y', linewidth=0.3, alpha=0.3)
    axes[0].set_ylabel('Cumulative proportion', fontsize=6, labelpad=1)
    fig.supxlabel('Observed Δlog$_{10}$ yield (mutant − WT)' if dt == 'dms' else 'Observed log$_{10}$ monomer yield (mg/L)', fontsize=6, y=0.1)
    fig.suptitle({'denovo': 'De novo', 'dms': 'Mutational effect'}[dt], fontsize=6, y=0.94)
    handles = [Line2D([0], [0], color=colors[m], linewidth=0.8, label=LABEL(m)) for m in ECDF_MODELS]
    handles.append(Line2D([0], [0], color='0.45', linestyle='--', linewidth=0.8, label='No selection'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False, fontsize=6, handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.79), pad=0.4, w_pad=0.8)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

ecdf_figure('denovo', 'ExtFig12a.pdf')
ecdf_figure('dms', 'ExtFig12d_right.pdf')
