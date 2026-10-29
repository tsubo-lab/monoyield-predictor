"""ExtFig 13c-e: CDR2 / CDR3 amino-acid frequency heatmap (13c) and median monomer yield (13d) / median monomer-dimer
ratio (13e) heatmaps per CDR position (CDR2 from the N-terminus, CDR3 from the N- and the C-terminus).

Tables: S07 (yields), S18 (CDR2 / CDR3 sequences).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.tables import outdir
from common.style import set_style
from fig5i_rule_score import load_nanobodies

set_style()
OUT = outdir("fig5")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
CDR2_LEN = 6
MIN_N = 10
XLABELS = ["CDR2 position", "CDR3 position from N-term", "CDR3 position from C-term"]

df = load_nanobodies()
print("n nanobodies:", len(df))
MAXLEN = df["cdr3_seq"].str.len().max()


def position_long(seqs, values, maxlen, align):
    recs = []
    for s, v in zip(seqs, values):
        it = enumerate(s, start=1) if align == "nterm" else enumerate(reversed(s), start=1)
        for i, aa in it:
            if i > maxlen:
                break
            if aa in AA20:
                recs.append((i, aa, v))
    return pd.DataFrame(recs, columns=["position", "aa", "value"])


def freq_matrix(seqs, maxlen, align):
    long = position_long(seqs, np.zeros(len(seqs)), maxlen, align)
    cnt = long.groupby(["position", "aa"]).size().unstack("aa").reindex(index=range(1, maxlen + 1), columns=AA20).fillna(0)
    cnt.index.name = cnt.columns.name = None
    return cnt.div(cnt.sum(axis=1), axis=0)


def median_matrix(seqs, values, maxlen, align):
    stat = position_long(seqs, values, maxlen, align).groupby(["position", "aa"])["value"].agg(["median", "count"])
    stat.loc[stat["count"] <= MIN_N, "median"] = np.nan
    mat = stat["median"].unstack("aa").reindex(index=range(1, maxlen + 1), columns=AA20)
    mat.index.name = mat.columns.name = None
    return mat



def draw(mats, cmap, vmin, vmax, cbar_label, name):
    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[m.shape[0] for m in mats] + [0.8], wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for i, (m, ax, xl) in enumerate(zip(mats, axes, XLABELS)):
        sns.heatmap(m.T, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, cbar=(i == 2), cbar_ax=cax if i == 2 else None,
                    cbar_kws={"label": cbar_label} if i == 2 else None, xticklabels=True, yticklabels=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.tick_params(axis="both", which="major", pad=1)
        ax.set_xlabel(xl)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# ExtFig13c: frequency
fmats = [freq_matrix(df["cdr2_seq"], CDR2_LEN, "nterm"), freq_matrix(df["cdr3_seq"], MAXLEN, "nterm"),
         freq_matrix(df["cdr3_seq"], MAXLEN, "cterm")]
draw(fmats, "viridis", 0, 0.5, "Frequency", "ExtFig13c.pdf")

# ExtFig13d / e: median of the measurement per position x amino acid (cells with n <= 10 left empty)
for name, col, label in [("ExtFig13d", "monomer_log_mgL", "Median monomer yield\n(log$_{10}$ mg/L)"),
                         ("ExtFig13e", "monomer_dimer_ratio_log", "Median monomer/dimer\n(log$_{10}$ ratio)")]:
    mats = [median_matrix(df["cdr2_seq"], df[col], CDR2_LEN, "nterm"), median_matrix(df["cdr3_seq"], df[col], MAXLEN, "nterm"),
            median_matrix(df["cdr3_seq"], df[col], MAXLEN, "cterm")]
    allv = np.concatenate([m.values.astype(float).ravel() for m in mats])
    vmin, vmax = np.nanmin(allv) * 0.9, np.nanmax(allv) * 0.9
    print(name, "color range", round(vmin, 3), round(vmax, 3))
    draw(mats, "cool", vmin, vmax, label, f"{name}.pdf")
