"""Fig 5i: nanobody monomer yield / monomer-dimer ratio vs the number of satisfied CDR design rules
(McMahon 2018 rules, 0-6, and the extended 9-rule score). Numbers above the boxes: % of nanobodies above the threshold.

Tables: S07 (yields), S18 (CDR2 / CDR3 sequences).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
from common.tables import load, outdir
from common.style import set_style

set_style()
OUT = outdir("fig5")
CLIP_LOWER = -1.5
TWO = ("#0072BC", "#8FC1E1")
RULES = {
    "cdr2_pos1": set("ADGNST"), "cdr2_pos4": set("G"), "cdr2_pos5": set("AGST"), "cdr2_pos6": set("INST"),
    "cdr3_Cpos1": set("Y"), "cdr3_Cpos3": set("FHLY"),
}
COLS = {"monomer_log_mgL": r"log$_{10}$ (monomeric yield)" + "\n(mg/L)",
        "monomer_dimer_ratio_log": r"log$_{10}$" + "\n(monomeric/dimeric yield)"}
THRESHOLDS = {"monomer_log_mgL": 0.5, "monomer_dimer_ratio_log": 1.0}
SCORE_COLS = {"rule_score_v1": "McMahon 2018", "rule_score_v2": "+Additional rules"}
LABEL_SCORES = {"rule_score_v1": {5, 6}, "rule_score_v2": {8, 9}}


def load_nanobodies():
    """S07 yields (spikes removed, yields clipped; ratio not clipped) + S18 CDR sequences, matched by lower-case name."""
    y = load("S07")
    y = y[~y["is_spike"]].copy()
    y["key"] = y["name"].str.lower()
    s18 = load("S18")
    s18["key"] = s18["name"].str.lower()
    df = y[["key", "monomer_log_mgL", "dimer_log_mgL", "monomer_dimer_ratio_log"]].merge(
        s18[["key", "aa_seq", "cdr1_seq", "cdr2_seq", "cdr3_seq", "method", "target"]], on="key")
    df[["monomer_log_mgL", "dimer_log_mgL"]] = df[["monomer_log_mgL", "dimer_log_mgL"]].clip(lower=CLIP_LOWER)
    return df


def rule_scores(df):
    """rule_score_v1: 6 McMahon rules; rule_score_v2: Cpos1 = Y/W plus 3 additional rules (0-9)."""
    c2, c3 = df["cdr2_seq"], df["cdr3_seq"]
    v1 = ((c2.str[0].isin(RULES["cdr2_pos1"])).astype(int) + (c2.str[3].isin(RULES["cdr2_pos4"])).astype(int)
          + (c2.str[4].isin(RULES["cdr2_pos5"])).astype(int) + (c2.str[5].isin(RULES["cdr2_pos6"])).astype(int)
          + (c3.str[-1].isin(RULES["cdr3_Cpos1"])).astype(int) + (c3.str[-3].isin(RULES["cdr3_Cpos3"])).astype(int))
    n1 = ~c3.str[0].isin(list("YWVFLI"))
    n2 = c3.str[1].isin(["L", "T"])
    cpos2 = c3.apply(lambda s: s[-2] if len(s) > 1 else np.nan)
    c2rule = ~cpos2.isin(["F", "L", "Y"])
    old = c3.str[-1].eq("Y")
    new = c3.str[-1].isin(["Y", "W"])
    v2 = v1 - old.astype(int) + new.astype(int) + n1.astype(int) + n2.astype(int) + c2rule.astype(int)
    return v1, v2


if __name__ == "__main__":
    df = load_nanobodies()
    df["rule_score_v1"], df["rule_score_v2"] = rule_scores(df)
    print("n nanobodies:", len(df))
    print(df["rule_score_v1"].value_counts().sort_index().to_dict())
    print(df["rule_score_v2"].value_counts().sort_index().to_dict())
    medians = {"monomer_log_mgL": 0.5, "monomer_dimer_ratio_log": 1}

    fig, axes = plt.subplots(2, 2, figsize=(7 / 2.54, 5.5 / 2.54), dpi=300, sharey="row",
                             gridspec_kw={"width_ratios": [7, 10]})
    for r, col in enumerate(COLS):
        for c, score_col in enumerate(SCORE_COLS):
            ax = axes[r, c]
            sns.boxplot(df, x=score_col, y=col, linewidth=0.5, fliersize=0, width=0.8,
                        boxprops=dict(facecolor="none"), ax=ax)
            sns.stripplot(df, x=score_col, y=col, size=1, alpha=0.2, color=TWO[c], ax=ax)
            ax.tick_params(axis="both", pad=1)
            ax.axhline(y=medians[col], color="grey", linestyle="--", linewidth=0.6, zorder=0)
            categories = sorted(df[score_col].dropna().unique())
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            for xi, val in enumerate(categories):
                if val not in LABEL_SCORES[score_col]:
                    continue
                pct = (df.loc[df[score_col] == val, col] >= THRESHOLDS[col]).mean() * 100
                print(f"{col} {score_col}={val}: {pct:.1f}% above threshold")
                ax.text(xi, 1.0, f"{pct:.0f}", transform=trans, ha="center", va="bottom", fontsize=6)
            if c == len(SCORE_COLS) - 1:
                ax.annotate("(%)", xy=(len(categories) - 1, 1.0), xycoords=trans, xytext=(7, 0),
                            textcoords="offset points", ha="left", va="bottom", fontsize=6, annotation_clip=False)
            ax.set_title("" if col == "monomer_dimer_ratio_log" else SCORE_COLS[score_col])
            ax.set_xlabel("", labelpad=1)
            ax.set_ylabel(COLS[col], labelpad=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig5i.pdf"), bbox_inches="tight")
