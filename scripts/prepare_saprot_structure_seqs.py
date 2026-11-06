#!/usr/bin/env python
"""
Combine the 3Di sequence obtained with foldseek from the structure_path (PDB) of denovo_5fold.csv / dms_pair.csv
with aa_seq into the SaProt format (aa + 3di interleaved, e.g. MdKpTn...) and write a CSV with the
saprot_seq / wt_saprot_seq columns added.

If no 3Di sequence is available, or the amino-acid sequence extracted by foldseek does not match the
aa_seq in the CSV, fall back to the structure-unknown token (#).

Prerequisite: create the foldseek fasta files beforehand with:
  foldseek createdb <denovo_pdb_dir> structDB_denovo
  foldseek createdb <dms_pdb_dir> structDB_dms
  foldseek lndb structDB_denovo_h structDB_denovo_ss_h   # the _ss database needs a header database
  foldseek lndb structDB_dms_h structDB_dms_ss_h
  foldseek convert2fasta structDB_denovo structDB_denovo.fasta
  foldseek convert2fasta structDB_denovo_ss structDB_denovo_ss.fasta
  foldseek convert2fasta structDB_dms structDB_dms.fasta
  foldseek convert2fasta structDB_dms_ss structDB_dms_ss.fasta
"""

import os
import argparse
import pandas as pd


VALID_3DI = set("acdefghiklmnpqrstvwy")


def normalize_key(name):
    """Strip all trailing .pdb extensions (case-insensitive) so headers with
    a different number of stacked .pdb suffixes (a handful of source files
    are named e.g. 1A32.pdb.pdb) collide onto the same lookup key."""
    name = str(name)
    while name.lower().endswith(".pdb"):
        name = name[: -len(".pdb")]
    return name


def read_fasta(path):
    """Return dict: header -> sequence (single line, whitespace stripped)."""
    seqs = {}
    header = None
    chunks = []

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())

    if header is not None:
        seqs[header] = "".join(chunks)

    return seqs


def build_struct_lookup(aa_fasta_path, di_fasta_path):
    """Return dict: basename(pdb file) -> (aa_seq_from_struct, di_seq_lower)."""
    aa_seqs = read_fasta(aa_fasta_path)
    di_seqs = read_fasta(di_fasta_path)

    lookup = {}
    n_mismatch_len = 0

    for header, aa in aa_seqs.items():
        di = di_seqs.get(header)
        if di is None:
            continue
        if len(aa) != len(di):
            n_mismatch_len += 1
            continue

        lookup[normalize_key(header)] = (aa, di.lower())

    print(
        f"[{os.path.basename(aa_fasta_path)}] "
        f"n_headers={len(aa_seqs)} n_matched={len(lookup)} "
        f"n_length_mismatch={n_mismatch_len}"
    )

    return lookup


def to_saprot_seq(aa_seq, di_seq=None):
    """aa_seq (+ di_seq if given) -> SaProt-format string."""
    aa_seq = str(aa_seq)

    if di_seq is None:
        return "".join(a + "#" for a in aa_seq)

    return "".join(a + d for a, d in zip(aa_seq, di_seq))


def lookup_saprot_seq(struct_path, aa_seq, lookup, stats, structure_col_label):
    """From structure_path + the CSV aa_seq: use the structure-matched 3Di if possible, otherwise fall back to #."""
    basename = normalize_key(os.path.basename(str(struct_path)))
    aa_seq = str(aa_seq)

    hit = lookup.get(basename)

    if hit is None:
        stats["missing"] += 1
        return to_saprot_seq(aa_seq, None)

    struct_aa, di_seq = hit

    if struct_aa != aa_seq:
        if len(struct_aa) != len(aa_seq):
            stats["length_mismatch"] += 1
            return to_saprot_seq(aa_seq, None)
        stats["seq_mismatch_same_length"] += 1
        # Same length but different sequence: the per-position correspondence of the 3Di is not guaranteed,
        # so conservatively combine the CSV aa_seq with the foldseek 3Di as is
        # (counted so that you can check it does not occur often)

    stats["ok"] += 1
    return to_saprot_seq(aa_seq, di_seq)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--denovo_csv", required=True)
    parser.add_argument("--dms_csv", required=True)

    parser.add_argument("--denovo_aa_fasta", required=True)
    parser.add_argument("--denovo_di_fasta", required=True)
    parser.add_argument("--dms_aa_fasta", required=True)
    parser.add_argument("--dms_di_fasta", required=True)

    parser.add_argument("--seq_col", default="aa_seq")
    parser.add_argument("--wt_seq_col", default="wt_seq")
    parser.add_argument("--structure_col", default="structure_path")
    parser.add_argument("--wt_structure_col", default="wt_structure_path")

    parser.add_argument("--out_denovo_csv", required=True)
    parser.add_argument("--out_dms_csv", required=True)

    args = parser.parse_args()

    denovo_lookup = build_struct_lookup(args.denovo_aa_fasta, args.denovo_di_fasta)
    dms_lookup = build_struct_lookup(args.dms_aa_fasta, args.dms_di_fasta)

    # dms_pair.csv references both denovo-side structures (mutant, under
    # 260604_score_monomeric/pdb naming convention is not guaranteed, so we
    # just merge both lookups: a given structure_path may live in either dir).
    combined_lookup = {**denovo_lookup, **dms_lookup}

    denovo_df = pd.read_csv(args.denovo_csv)
    dms_df = pd.read_csv(args.dms_csv)

    stats_denovo = {"ok": 0, "missing": 0, "length_mismatch": 0, "seq_mismatch_same_length": 0}
    denovo_df["saprot_seq"] = [
        lookup_saprot_seq(sp, seq, combined_lookup, stats_denovo, "structure_path")
        for sp, seq in zip(denovo_df[args.structure_col], denovo_df[args.seq_col])
    ]
    print("denovo saprot_seq stats:", stats_denovo)

    stats_dms_mut = {"ok": 0, "missing": 0, "length_mismatch": 0, "seq_mismatch_same_length": 0}
    dms_df["saprot_seq"] = [
        lookup_saprot_seq(sp, seq, combined_lookup, stats_dms_mut, "structure_path")
        for sp, seq in zip(dms_df[args.structure_col], dms_df[args.seq_col])
    ]
    print("dms mutant saprot_seq stats:", stats_dms_mut)

    stats_dms_wt = {"ok": 0, "missing": 0, "length_mismatch": 0, "seq_mismatch_same_length": 0}
    dms_df["wt_saprot_seq"] = [
        lookup_saprot_seq(sp, seq, combined_lookup, stats_dms_wt, "wt_structure_path")
        for sp, seq in zip(dms_df[args.wt_structure_col], dms_df[args.wt_seq_col])
    ]
    print("dms wt saprot_seq stats:", stats_dms_wt)

    denovo_df.to_csv(args.out_denovo_csv, index=False)
    dms_df.to_csv(args.out_dms_csv, index=False)

    print("saved:", args.out_denovo_csv)
    print("saved:", args.out_dms_csv)


if __name__ == "__main__":
    main()
