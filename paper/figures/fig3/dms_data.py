"""Shared data preparation for the Fig. 3 family (DMS panels), built only from Table S04 (+ S23a/S23b for the Rosetta WT features).

Two views of the same table are used by the original notebooks:
  * `heatmap_view()`  (260701_dms_heatmap.ipynb): all DMS rows, dG without CI filter -> Fig3b/c/e/f, ExtFig6a, ExtFig7.
  * `fit_view()`      (260701_dms.ipynb):         dG only where deltaG_95CI <= 0.5, per-protein regression -> ExtFig8, Fig3g/h, ExtFig9.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common.tables import load  # noqa: E402

YIELD_FLOOR = -1.5      # monomeric yield (log10 mg/L) is clipped at this floor
CI_MAX = 0.5            # deltaG_95CI threshold used for the regression view
MIN_MUTS = 20           # proteins need more than this many valid mutants for Fig3g/h, ExtFig9


def _s04():
    """S04 with the notebook column names; DMS + de novo rows, '.pdb' structure entries removed, yield clipped."""
    df = load("S04").rename(columns={"deltaG_measured_kcal_mol": "deltaG", "source_dms_denovo": "source"})
    df["monomer_log_mgL"] = df["monomer_log_mgL"].clip(lower=YIELD_FLOOR)
    df = df[df["source"].isin(["dms", "denovo"]) & ~df["name"].str.contains(".pdb")]
    return df.reset_index(drop=True)


def heatmap_view():
    """Returns (df_gv, df_dms, df_wt).

    df_gv : all DMS/de novo rows (source, name, sublib, deltaG, deltaG_95CI, monomer_log_mgL, target)
    df_dms: DMS libraries dms-1..5 with sequence, upper-case names, one row per name
    df_wt : the wild-type row of every protein in df_dms
    """
    df_gv = _s04()
    df_gv["target"] = df_gv["name"].str.split("_").str[0]
    df_dms = df_gv[df_gv["sublib"].str.contains("dms")].copy()
    df_dms["name"] = df_dms["name"].str.upper()
    df_dms["target"] = df_dms["name"].str.split("_").str[0]
    df_dms = df_dms.drop_duplicates(subset="name")
    df_wt = df_dms[df_dms["name"] == df_dms["target"]].reset_index(drop=True)
    return df_gv, df_dms, df_wt


def mutation_deltas(df_gv):
    """Mutant - mean(WT) differences of yield and dG per DMS row, with the substitution class ('mut') and the original residue ('ori')."""
    d = df_gv[df_gv["source"] == "dms"].copy()
    wts = d[(d["name"] == d["target"]) | d["name"].str.contains("wt")]
    wts = wts.groupby("target")[["monomer_log_mgL", "deltaG"]].mean().add_suffix("_wt").reset_index()
    d = d.merge(wts, on="target")
    mut, ori = [], []
    for n in d["name"]:
        parts = n.split("_")
        if len(parts) == 1 or len(n.split("_wt")) == 2:
            mut.append("wt"); ori.append("wt")
        elif len(n.split("ins")) == 2:
            mut.append("ins"); ori.append("ins")
        elif len(n.split("del")) == 2:
            mut.append("del"); ori.append("del")
        elif len(parts) == 2:
            mut.append(n[-1]); ori.append(parts[-1][0])
    d["mut"], d["ori"] = mut, ori
    for c in ["monomer_log_mgL", "deltaG"]:
        d[f"delta_{c}"] = d[c] - d[f"{c}_wt"]
    return d


def fit_view():
    """Returns (df_dms, df_wt, df_pr) of 260701_dms.ipynb.

    df_dms: DMS-source rows (dms-1..5, muts-1, denovo-8), dG kept only if deltaG_95CI <= CI_MAX, one row per name
    df_wt : wild-type rows
    df_pr : per-protein linear fit of yield on dG over the valid mutants (monomer yield > floor and dG > -2), sorted by Pearson r
    """
    df = _s04()
    df = df[df["source"] == "dms"].copy()
    df.loc[~(df["deltaG_95CI"] <= CI_MAX), "deltaG"] = np.nan
    df["target"] = df["name"].str.split("_").str[0]
    df = df.drop_duplicates(subset="name")
    df_wt = df[df["name"] == df["target"]]
    valid = df[(df["monomer_log_mgL"] > YIELD_FLOOR) & (df["deltaG"] > -2)]
    rows = []
    for protein, g in valid.groupby("target"):
        if len(g) < 2:
            continue
        sub = g[["deltaG", "monomer_log_mgL"]].dropna()
        fit = stats.linregress(sub["deltaG"], sub["monomer_log_mgL"])
        rows.append({"name": protein, "pearsonR": round(stats.pearsonr(sub["deltaG"], sub["monomer_log_mgL"])[0], 3),
                     "slope": fit.slope, "intercept": fit.intercept, "p": fit.pvalue, "std_err": fit.stderr,
                     "valid_muts": len(sub)})
    df_pr = pd.DataFrame(rows).sort_values("pearsonR")
    return df, df_wt, df_pr


def rosetta_wt_features(df_wt, df_pr):
    """WT Rosetta score_monomeric features (S23a + S23b) merged with the per-protein fits (proteins with > MIN_MUTS valid mutants)."""
    sc = pd.concat([load("S23a"), load("S23b")])
    sc["name"] = sc.iloc[:, 0].apply(lambda x: x.split("/")[1][:-4].lower())
    sc = pd.merge(sc, df_wt["name"]).drop_duplicates(subset="name")
    sc = sc[sc.columns[sc.isna().sum(axis=0) == 0]]
    sc = sc[["name"] + list(sc.select_dtypes(include=["float64", "int64"]).columns)]
    m = pd.merge(df_pr[df_pr["valid_muts"] > MIN_MUTS], sc)
    keep = ~(m.columns.str.contains("_site_") | m.columns.str.contains("per_residue_") | m.columns.str.contains("_3d_contacts_"))
    return m[list(m.columns[keep])]
