# Supplementary tables

Identifier of a sequence: (`sublib`, `name`); names are lower case. Yields are log10 mg/L (`*_log_mgL`).
Main library = 15 sublibraries (codon-1, denovo-1..8, dms-1..5, muts-1); nanobody library = nanobody-4. Each sublibrary has its own MS run, spike-v1 proteins and calibration line.

* **raw MS intensity** (`raw_monomer`, ...): DIA-NN precursor intensity of the barcode peptide of the monomer fraction.
* **calibrated MS intensity**: raw intensity x `correction_scale` of the sublibrary (spike v1 / control ratio, see TableS02).
* **log ng, log mg/L** (protein level): mean over barcodes with NGS CPM >= 2 of (intensity + 1e4) -> log10 -> sublibrary calibration line (TableS02) -> mg/L.
* The main library tables contain the monomer fraction only; the nanobody tables contain monomer and dimer.

| table | content | rows |
|---|---|---|
| S01 | library definition, counts, barcode detection and protein measurement rates | 16 |
| S02 | calibration by sublibrary (spike v1 / control correction, regression, mg/L conversion) | 17 |
| S03 | main library, barcode level (csv.gz) | 578,310 |
| S04 | main library, protein level: yield, measured dG, sequence | 38,354 |
| S05 | main library features used by XGBoost / zero-shot (csv.gz) + `S05_feature_list` (paper labels) | 35,857 |
| S06, S07 | nanobody: barcode level, protein level | 39,000; 2,589 |
| S08 | nanobody features used by XGBoost + `S08_feature_list` | 2,589 |
| S09 | controls: barcode level (a), protein level (b), wide by sublibrary | |
| S10 | spike v1 in the main library: barcode level (a), protein level incl. mean MS intensity (b) | 24 |
| S11 | spike v1 in the conversion-equation experiment: barcode level (a), protein level (b) | |
| S12 | spike v2: barcode level (a), set level (b), common calibration points (c); n1-n4 = S2 rep1, S2 rep2, S3 rep1, S3 rep2 | |
| S13 | de novo / natural-protein backbones: structural features, per-backbone yield | 70 |
| S14 | HPLC: individual assay vs pooled MS (a), area -> mass conversion (b), cytochrome C calibration (c), SARS-CoV-2 binders (d) | |
| S15 | Fig. 4 test-fold predictions: yield (a), mutational effect (b) | 25,065; 5,735 |
| S16 | Fig. 5 test-fold predictions: nanobody yield (a), monomer/dimer ratio (b) | 2,589 |
| S17 | de novo AF2 and AF3 (fast) structure metrics | 19,506 |
| S18 | nanobody sequences, CDRs, method, target, cluster, CV fold | 2,600 |
| S19 | nanobody cDNA-display enrichment | 2,589 |
| S20 | codon-1 DNA sequences with CAI / tAI (a); CAI / tAI of de novo and DMS-WT constructs (b) | |
| S21 | index of the raw HPLC chromatogram files | |
| S22 | mGL barcode replicates (a, Fig. 1b); lib77 sub16/17 Astral run, barcode level (b) and protein level (c) (ExtFig. 1a) | |
| S23 | Rosetta score_monomeric of the WT structures (a) and of the feature-score set (b) | |
| S24 | full nanobody feature table (csv.gz) (a); top-correlated feature clusters (b) | 2,589 |
| S25 | ESMFold2 pLDDT / pTM, contact order, radius of gyration, core fraction per design | 35,798 |
| `hplc_chromatograms/` | raw HPLC chromatograms (R_Time + one column per sample, named by protein) | 7,201 |
