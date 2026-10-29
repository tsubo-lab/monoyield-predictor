"""Plot helpers shared by Fig3e/f and ExtFig7: per-protein dG-vs-yield scatter and the 22 x L substitution heatmaps."""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

TSM = ('#0B5269', '#087295', '#0493C2', '#6CD8FD')   # blue scale used for the scatter points
MUTS = list('QENHDRKTSAGMCLVIWYFP') + ['DEL', 'INS']
LABELS = {'deltaG': 'ΔG (kcal/mol)', 'monomer_log_mgL': r"log$_{10}$ (monomeric yield) (mg/L)"}
YIELD_LABEL = LABELS['monomer_log_mgL']


def scatter_with_fit(ax, data, text_size=None, title=None):
    """dG vs monomeric yield of one protein with the linear fit and n / r / slope (returns n, r, slope)."""
    sub = data[['deltaG', 'monomer_log_mgL']].dropna()
    sns.scatterplot(data=sub, x='deltaG', y='monomer_log_mgL', alpha=0.25, color=TSM[2], s=6, ax=ax)
    r = slope = np.nan
    if len(sub) >= 2 and sub['deltaG'].nunique() >= 2:
        reg = stats.linregress(sub['deltaG'], sub['monomer_log_mgL'])
        r, slope = reg.rvalue, reg.slope
        sns.regplot(data=sub, x='deltaG', y='monomer_log_mgL', scatter=False, ci=None,
                    line_kws={'color': 'grey', 'linewidth': 1, 'linestyle': '--'}, ax=ax)
        txt = f'n = {len(sub)}\nr = {r:.3f}\nslope = {slope:.3f}'
    else:
        txt = f'n = {len(sub)}\nr = NA\nslope = NA'
    kw = {} if text_size is None else {'fontsize': text_size}
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha='left', va='top', **kw)
    ax.set_xlim(-2.5, 7)
    ax.set_ylim(-1.6, 2)
    ax.set_xticks(np.arange(-2, 8, 2))
    ax.set_yticks(np.arange(-1, 3, 1))
    ax.set_title(title)
    ax.set_xlabel("ΔG (kcal/mol)")
    ax.set_ylabel(YIELD_LABEL)
    ax.tick_params(axis="both", which="major", pad=1)
    return len(sub), r, slope


def heatmap(ax, df, df_wt, protein, abundance, vmin, vmax, xmin=None, xmax=None,
            ci_col='deltaG_95CI', ci_threshold=0.5, show_ci_hatch=True):
    """Substitution matrix (rows: 20 aa + DEL + INS; columns: sequence positions) of `abundance`.

    WT residues are black dots, missing variants a red slash, variants with CI >= ci_threshold a grey slash."""
    df_idx = df.set_index('name')
    aa_seq = df_wt.loc[df_wt['name'] == protein, 'aa_seq'].values[0]
    n = len(aa_seq)
    mat = pd.DataFrame(index=MUTS, columns=range(1, n + 1), dtype=float)
    ci = pd.DataFrame(index=MUTS, columns=range(1, n + 1), dtype=float)

    wt_mask = (df['name'] == protein) | df['name'].str.contains('_WT', na=False)
    score_wt = df.loc[wt_mask, abundance].mean()
    ci_wt = df.loc[wt_mask, ci_col].mean() if ci_col in df.columns else np.nan
    wt_num = [MUTS.index(aa) for aa in aa_seq]

    for pos in mat.columns:
        wt_aa = aa_seq[pos - 1]
        for idx in mat.index:
            if idx == 'INS':
                key = f"{protein}_{idx}A{pos}"
            elif idx == 'DEL':
                key = f"{protein}_{idx}{wt_aa}{pos}"
            elif wt_aa == idx:
                mat.at[idx, pos], ci.at[idx, pos] = score_wt, ci_wt
                continue
            else:
                key = f"{protein}_{wt_aa}{pos}{idx}"
            if key in df_idx.index:
                mat.at[idx, pos] = df_idx.at[key, abundance]
                if ci_col in df_idx.columns:
                    ci.at[idx, pos] = df_idx.at[key, ci_col]

    na_pos = mat.isna().stack()
    na_pos = list(na_pos[na_pos].index)
    na_x = [p[1] - 1 for p in na_pos]
    na_y = [MUTS.index(p[0]) for p in na_pos]

    mat.columns = list(aa_seq)
    hm = sns.heatmap(mat, cmap='bwr', vmin=vmin, vmax=vmax, center=score_wt, cbar_kws={"pad": 0.01, 'label': ''}, ax=ax)
    if hm.collections:
        hm.collections[0].colorbar.ax.tick_params(axis='both', which='major', pad=1, labelsize=6)

    if show_ci_hatch:
        for y, aa in enumerate(MUTS):
            for x, pos in enumerate(range(1, n + 1)):
                c = ci.at[aa, pos]
                if pd.isna(mat.iloc[y, x]):
                    continue
                if pd.notna(c) and c >= ci_threshold:
                    ax.plot([x + 0.7, x + 1.0], [y + 0.3, y], color='grey', lw=0.6, zorder=5)
    for x, y in zip(na_x, na_y):
        ax.plot([x, x + 1], [y + 1, y], color='red', lw=0.5, zorder=6)
    ax.scatter(np.arange(n) + 0.5, np.array(wt_num) + 0.5, color="black", s=2, zorder=7)
    for y in [5, 7, 11, 16, 19]:
        ax.plot([0, n], [y, y], color="black", lw=0.25)
    ax.plot([0, n], [20, 20], color="grey", lw=1)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(len(MUTS)) + 0.5)
    ax.set_xticklabels([f'{a}{i + 1}' for i, a in enumerate(aa_seq)], rotation=90, fontsize=5)
    ax.set_yticklabels(MUTS, rotation=0, fontsize=5)
    ax.tick_params(axis='both', which='major', pad=1)
    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(len(MUTS), 0)
    ax.set_title(LABELS[abundance], fontsize=6)
    return ax

