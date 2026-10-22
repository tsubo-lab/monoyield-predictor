# Data (`data/`)
All files contain sequences and the measured yields of the paper (the same values as the source-data tables; identifier = `name`).
5-fold assignment: column `split` (`group_1 ... group_5`) from sequence clustering (mmseq2); `rep` = cluster / backbone id used to keep similar sequences in the same fold.
Yields are log10 mg/L (`monomer_log_mgL`; nanobodies also `dimer_log_mgL` and `monomer_dimer_ratio_log`).

| file | rows | content | used by |
|---|---|---|---|
| `main_library_denovo_5fold.csv` | 19,582 | de novo designs: name, aa_seq, rep, split, yields | `run/train_esmc_monoyield.sh` |
| `main_library_dms_5fold.csv` | 15,393 | DMS variants paired with their wild type (`wt_name`, `wt_seq`, `wt_*_log_mgL`) | same |
| `main_library_*_3di.csv` | same | as above + `saprot_seq` (amino acid + Foldseek 3Di, from AF3 models) | `run/train_saprot_lora.sh` |
| `nanobody_5fold.csv` | 2,589 | nanobody sequences, CDRs, yields (monomer, dimer, ratio), cluster, split | `run/train_esmc_nb_*.sh` |
| `nanobody_designs.csv` | 9,284 | all designed nanobody sequences (prediction input) | `run/predict_nanobody.sh` |
| `cov_binders.csv`, `cov_binders_3di.csv` | 226 | SARS-CoV-2 binder designs (prediction input; `saprot_seq` for SaProt) | `run/predict_cov_binders.sh` |

The corresponding tables of the paper's source data are `TableS04`, `TableS07` / `TableS18`.
