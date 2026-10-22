# Data
Training/evaluation data are the paper's source data tables (identifier = `sublib`, `name`):

| use | table |
|---|---|
| main library (de novo, DMS): sequence + monomer yield + dG | `TableS04_main_library_protein_level.csv` |
| nanobody yield (monomer, dimer, ratio) and sequences / CDRs / cluster / CV fold | `TableS07_nanobody_protein_level.csv`, `TableS18_nanobody_sequences.csv` |
| predictions of all models (test folds) | `TableS15a/b`, `TableS16a/b` (source data, not included here) |

The 5-fold split is by sequence cluster (mmseq2); fold column `group_1 ... group_5`.
The shell scripts in `slurm/` expect the csv files under `${DATA_ROOT}`; the exact file names are in the scripts.
