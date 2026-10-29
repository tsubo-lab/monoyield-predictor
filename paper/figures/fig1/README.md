# Figure 1 and Extended Data Figs 1-4 (family fig1)

Run any script as `python paper/figures/fig1/<script>.py`; PDFs go to `paper/outputs/fig1/`. All scripts read only `paper/tables/` through `common/`.
Agreement is measured against `reproduced_figures/Fig1/*_reproduced.pdf` (text tokens identical, fraction of pixels differing at 100 dpi).

| Panel | Script | Tables | Status |
|---|---|---|---|
| Fig1b (mGL barcode hexbin) | `fig1b_mGL.py` | S22a | done; text identical, 0.9 % pixels differ (see note 1) |
| Fig1c (spike v2 bootstrap fold error vs number of barcodes) | `fig1c_bootstrap.py` | S12a | done; identical text, 0.5 % pixels differ (bootstrap re-run with the original seeds) |
| Fig1d-left (spike v2 calibration) | `fig1d_spike_calibration.py` | S12c | done; pixel-identical |
| Fig1d-right (spike v1 reference QC) | `fig1d_spike_calibration.py` | S11b, S12c | done; pixel-identical |
| Fig1e (dms-3 conversion) | `fig1e_g_extfig2.py` | S02, S10b, S11b | done; pixel-identical (see note 2) |
| Fig1g (denovo-3 inferred vs mean of controls) | `fig1e_g_extfig2.py` | S09b | done; pixel-identical |
| Fig1h (pooled MS vs individual HPLC, n = 48) | `fig1h_hplc_vs_ms.py` | S14a | done; pixel-identical (r = 0.757, rho = 0.777) |
| ExtFig1a-left (rep1 vs rep2, 1,038 proteins) | `extfig1a_lib1617.py` | S22c | done; pixel-identical |
| ExtFig1a-mid (Pearson R vs number of barcodes) | `extfig1a_lib1617.py` | S22b | done; pixel-identical |
| ExtFig1a-right (15 barcodes) | `extfig1a_lib1617.py` | S22b | done; pixel-identical |
| ExtFig1b (library table) | - | S01 | manual / table, out of scope |
| ExtFig2a (15 libraries) | `fig1e_g_extfig2.py` -> `ExtFig2a_<lib>.pdf` | S02, S10b, S11b | done; 15/15 pixel-identical |
| ExtFig2b (15 libraries) | `fig1e_g_extfig2.py` -> `ExtFig2b_<lib>.pdf` | S09b | done; 15/15 pixel-identical |
| ExtFig3a chromatograms | `extfig3a_cytochrome.py` | hplc cytochromeC (wide csv) | done; pixel-identical |
| ExtFig3a AUC vs mass | `extfig3a_cytochrome.py` | S14c (S14b printed as check) | done; pixel-identical; 6th dilution is 0.03125 mg/mL (0.003125 mg), fit slope 0.9703 / intercept -7.2711 |
| ExtFig3b (48 chromatograms) | `extfig3b_hplc_individual.py` | hplc individual (wide csv), S14a | done; pixel-identical; panel order = row order of S14a |
| ExtFig4a (fold error vs protein median) | `extfig4_codon.py` | S04, S20a | done; pixel-identical |
| ExtFig4b (geometric SD) | `extfig4_codon.py` | S04, S20a | done; pixel-identical |
| ExtFig4c (yield vs CAI / tAI) | `extfig4_codon.py` | S04, S20a | done; pixel-identical (see note 3) |
| ExtFig4d (CAI / tAI density) | `extfig4_codon.py` | S20a, S20b | done; pixel-identical |
| Fig1a, Fig1f, Fig1i | - | - | schematic / manual, out of scope |

Notes
1. Fig1b: S22a carries the replicate-1/2 intensities (`e4_old`, `e8_old`), n = 70,116 barcodes; 37,632 are detected in both replicates, 36,978 are selected (r = 0.985 for the selected barcodes, r = 0.943 for all). A few isolated outlier hexagons in the reference (e.g. one barcode at replicate 1 = 0, replicate 2 = 8.4) are not present in the table values, and the table has another barcode (replicate 1 = 8.1, replicate 2 = 0) not in the reference; the global shape and all text are identical. No other pair of the eight `e*_old` columns matches the reference better. The original annotation text (n, r) is commented out in the notebook, so the counts could not be compared with a printed value.
2. Fig1e / ExtFig2a: the spike v1 intensities per sublibrary are tabulated in S10b (`monomer_mean_intensity_plus1e4__<sublib>` = mean intensity + 1e4 of the 24 spike v1 proteins, `log_monomer__<sublib>` = its log10); the script asserts that the refit reproduces the S02 slope and intercept.
3. ExtFig4c: the five representative proteins are the proteins whose median yield is closest to the 0, 25, 50, 75 and 100 % quantiles of all protein medians. The 50 % target lies exactly midway between two proteins; the tie is resolved towards the lower median (as in the published figure).
