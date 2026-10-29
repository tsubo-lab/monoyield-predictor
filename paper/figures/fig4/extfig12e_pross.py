"""ExtFig12e: PROSS designs (dms-5; 1aoy excluded) vs their wild type: monomeric yield (log10 mg/L), dG, and delta-yield vs delta-dG.

Tables: S04 (monomer_log_mgL, measured dG; sublib dms-5).
Outputs: ExtFig12e_1_yield.pdf, ExtFig12e_2_dG.pdf, ExtFig12e_3_ddG_vs_dyield.pdf
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig4")
THRESHOLD = {'monomer_log_mgL': np.log10(2), 'deltaG': 1.0}     # >= 2-fold yield / >= 1 kcal/mol change counts as improved / decreased
OUTNAME = {'monomer_log_mgL': 'ExtFig12e_1_yield.pdf', 'deltaG': 'ExtFig12e_2_dG.pdf'}

s4 = load("S04")
d = s4[(s4["sublib"] == "dms-5") & ~s4["name"].str.contains("spike") & ~s4["name"].str.contains("control")].copy()
d["target"] = d["name"].str.split("_").str[0]
d = d[d["target"] != "1aoy"].rename(columns={"deltaG_measured_kcal_mol": "deltaG"})
wt = d[d["target"] == d["name"]][["target", "monomer_log_mgL", "deltaG"]]
df = d.merge(wt, on="target", suffixes=["", "_ori"])
print(f"dms-5 designs (1aoy excluded): {len(df)} rows, {df['target'].nunique()} parent proteins")

for col in ['monomer_log_mgL', 'deltaG']:
    fig = plt.figure(figsize=(4 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    x, y, th = col + '_ori', col, THRESHOLD[col]
    valid = df[[x, y]].notna().all(axis=1)
    improved = valid & (df[y] > df[x] + th)
    decreased = valid & (df[y] < df[x] - th)
    n_valid = valid.sum()
    ir, dr = improved.sum() / n_valid, decreased.sum() / n_valid
    print(f"ExtFig12e {col}: n={n_valid} improved={ir:.3f} decreased={dr:.3f}")
    sns.scatterplot(data=df[valid], x=x, y=y, alpha=0.2, s=5, ax=ax, color='grey')
    sns.scatterplot(data=df[improved], x=x, y=y, alpha=0.5, s=5, ax=ax, color='tab:blue')
    sns.scatterplot(data=df[decreased], x=x, y=y, alpha=0.5, s=5, ax=ax, color='tab:red')
    ax.text(x=0.05, y=0.95, s=f'Improved: {ir:.3f}\nDecreased: {dr:.3f}', transform=ax.transAxes, va='top')
    mn, mx = min(df.loc[valid, x].min(), df.loc[valid, y].min()), max(df.loc[valid, x].max(), df.loc[valid, y].max())
    ax.plot([mn, mx], [mn, mx], c='grey', linestyle=':', linewidth=1)
    ax.plot([mn, mx], [mn + th, mx + th], c='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.plot([mn, mx], [mn - th, mx - th], c='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    if col == 'deltaG':
        ticks = np.arange(-2, 11, 2)
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_xlim(-2, 8); ax.set_ylim(-2, 8)
    else:
        ticks = np.arange(-2, 11, 1)
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_xlim(-1.5, mx + 0.2); ax.set_ylim(-1.5, mx + 0.2)
    plt.savefig(os.path.join(OUT, OUTNAME[col]))
    plt.close(fig)

# delta (mutant - WT) of yield vs dG
df["delta_deltaG"] = df["deltaG"] - df["deltaG_ori"]
df["delta_monomer_log_mgL"] = df["monomer_log_mgL"] - df["monomer_log_mgL_ori"]
fig = plt.figure(figsize=(4 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
sns.scatterplot(df, x='delta_deltaG', y='delta_monomer_log_mgL', alpha=0.2, s=5, ax=ax)
ax.axhline(0, color='grey', lw=0.5, linestyle='--')
ax.axvline(0, color='grey', lw=0.5, linestyle='--')
plt.savefig(os.path.join(OUT, "ExtFig12e_3_ddG_vs_dyield.pdf"))
