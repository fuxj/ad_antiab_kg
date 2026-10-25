# Supplementary Materials — Triangulated KG+HGT+MR ARIA Study

This package contains all supplementary data and figures for the Pharmacological Research submission.

## Figures
- **FigS1_HGT_training_curves.png** — Full-resolution HGT training curves over 50 epochs (expanded version of manuscript Fig 5A,B).

## Tables
- **Table_S1_421_drugs_full_risk.csv** — Full 421-drug ARIA-H risk table (25 columns: pool, is_cns, targets, broad_class, GNN scores, FAERS ROR and n_report, MR genes, liability_grade, final_tier, paper_note).
- **Table_S2_HGT_training_curves.csv** — Per-epoch HGT training/validation loss and test AUC / AP across three tasks (50 epochs).
- **Table_S3_full_MR_results.csv** — Complete two-sample MR results across 24 genes × 9 outcomes (216 tests; 189 valid): nSNP, beta/se/p, OR and 95% CI, Egger intercept, MR-PRESSO, Q-stat, I-sq, mean F, Steiger, passes.
- **Table_S4_colocalization.csv** — Colocalization PP.H0–PP.H4 across six brain tissues + blood for every gene-outcome pair (1728 rows), with coloc_flag at PP.H4>=0.70.

## Excel bundle
- `Paper_Supplementary_Tables.xlsx` — All four Supplementary Tables as separate sheets in one workbook.

## Code
All reproducible code is provided in the GitHub repository `ad_antiab_kg`:
`03_kg_build.py`, `04*_faers_*.py`, `08_gnn_linkpred_hgt.py`, `05*_mr*.R`, `06_coloc.R`, `build_fusion.py`.
