# Supplementary Materials — Triangulated KG+HGT+MR ARIA Study

This package contains all supplementary data and figures for the KG-HGT-MR triangulation study of ARIA-H risk stratification with anti-amyloid-β therapy.

## Figures
- **FigS1_HGT_training_curves.png** — HGT training curves (train/val loss and per-task AUC over 50 epochs), generated from Table_S2.

## Tables
- **Table_S1_421_drugs_full_risk.csv** — Full 421-drug ARIA-H risk table (25 columns: pool, is_cns, targets, broad_class, GNN scores, FAERS ROR and n_report, MR genes, liability_grade, final_tier, paper_note).
- **Table_S2_HGT_training_curves.csv** — Per-epoch HGT training/validation loss and test AUC / AP across three tasks (50 epochs).
- **Table_S3_full_MR_results.csv** — Complete two-sample MR results across 24 genes × 9 outcomes (216 tests; 189 valid): nSNP, beta/se/p, OR and 95% CI, Egger intercept, MR-PRESSO, Q-stat, I-sq, mean F, Steiger, passes.
- **Table_S4_colocalization.csv** — Colocalization PP.H0–PP.H4 across six brain tissues + blood for every gene-outcome pair (1728 rows), with coloc_flag at PP.H4>=0.70.

## Excel bundle
- `Paper_Supplementary_Tables.xlsx` — All four Supplementary Tables as separate sheets in one workbook.

## Code
All reproducible code is provided in the GitHub repository `ad_antiab_kg` (`src/python/` and `src/r/`):
`01_download_faers.py`, `03_kg_build.py`, `04_gnn_risk.py`, `08_gnn_linkpred_hgt.py`, `08b_fusion_v3.py`, `02_signal_detection.R`, `05_mr_validation.R`, `06_coloc.R`, `07_coloc_brain.R`.
