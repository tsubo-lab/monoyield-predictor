"""Shared data assembly for the Fig 2 / Extended Data Fig 5 scripts (tables S04, S13, S17 only).

`load_designs(af)` returns one row per de novo design (S17, 19,506 designs) with
  - monomer_log_mgL   : measured monomer yield from S04 (clipped to [-1.5, 3.5] as in the analysis);
                        designs that are present in two sublibraries (sublib "a;b") have no yield
                        (the duplicated names were removed in the analysis),
  - mean_plddt, tmscore: structure-prediction metrics of the chosen predictor ('af3' = AF3-fast, 'af2' = AF2),
  - target (backbone), method (design method code), method_a (method label).
`load_backbones(df)` returns the 70-backbone table (S13) with the mean yield per backbone computed from `df`.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common.tables import load, outdir  # noqa: E402,F401
from common.style import set_style  # noqa: E402,F401

CLIP = (-1.5, 3.5)
METHOD_LABEL = {
    "campnnt01": "ProteinMPNN-Cα", "cdt04": "CarbonDesign", "ems3t03": "ESM3", "esmif1t03": "ESM-IF1",
    "f2seqt03": "Frame2seq", "hympnnt01": "HyperMPNN", "prmpnnt01": "ProteinMPNN-T0.1",
    "prmpnnt03": "ProteinMPNN-T0.3", "sompnnt01": "SolubleMPNN", "tdt014": "TIMED-Design",
    "boltzift01": "BoltzIF", "wt": "Natural/Rossetta",
}
# colours used throughout (from the analysis notebook)
TSM = ("#0B5269", "#087295", "#0493C2", "#6CD8FD")
TWO = ("#0072BC", "#8FC1E1")


def load_designs(af="af3"):
    suffix = {"af3": "_af3_fast", "af2": "_af2"}[af]
    s17 = load("S17")
    s04 = load("S04")[["sublib", "name", "monomer_log_mgL"]]
    df = s17.merge(s04, on=["sublib", "name"], how="left")
    df["monomer_log_mgL"] = df["monomer_log_mgL"].clip(*CLIP)
    df["mean_plddt"] = df["mean_plddt" + suffix]
    df["tmscore"] = df["tmscore" + suffix]
    df["target"] = df["name"].str.split("_").str[0]
    df["method"] = df["name"].str.split("_").str[1]
    df["method_a"] = df["method"].map(METHOD_LABEL)
    df["is_filtered"] = (df["mean_plddt"] > 90) & (df["tmscore"] > 0.80)
    return df


def drop_natural(df):
    """Remove the parent-sequence ('wt') entries, as done before the per-backbone analyses."""
    return df[df["method"] != "wt"].copy()


def load_backbones(df):
    """S13 backbone table (70 backbones) + mean/median yield per backbone computed from `df`."""
    bb = load("S13").rename(columns={
        "backbone": "target", "radius_of_gyration": "rg", "contact_order": "CO",
        "fraction_core_residues": "frac_core", "backbone_label_in_figure": "target1",
        "n_loop_residues": "loop", "n_helix_residues": "alpha", "n_strand_residues": "beta"})
    bb["target1"] = bb["target1"].str.replace("_", "-")
    bb["p_loop"] = bb["loop"] / (bb["loop"] + bb["alpha"] + bb["beta"])
    bb["p_alpha"] = bb["alpha"] / (bb["loop"] + bb["alpha"] + bb["beta"])
    bb["p_beta"] = bb["beta"] / (bb["loop"] + bb["alpha"] + bb["beta"])
    bb["ss_group"] = np.where(bb["alpha"] == 0, "all beta", np.where(bb["beta"] == 0, "all alpha", "mixed"))
    mean = df.groupby("target")["monomer_log_mgL"].mean().rename("monomer_log_mgL")
    # sorted by (label, backbone) like the original groupby, so that the seeded bootstrap is reproduced
    return bb.merge(mean, on="target").sort_values(["target1", "target"]).reset_index(drop=True)
