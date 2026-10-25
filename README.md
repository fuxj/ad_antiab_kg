# Supplementary Materials — Triangulated KG+HGT+MR ARIA Study

This package contains all supplementary data and figures for the Pharmacological Research submission.

## Figures
No separate supplementary figure is provided. The HGT training and test performance curves (training/validation loss, per-task test AUC over epochs, and HGT versus matched-capacity MLP baseline) are presented once, as manuscript **Figure 5 (panels A–C)**.

## Tables
- **Table_S1_421_drugs_full_risk.csv** — Full 421-drug ARIA-H risk table (25 columns: rank, drug_id, drug, pool (KG-feasible vs Open Targets expansion), CNS label, targets, GNN per-task scores, FAERS ROR and n_reports, MR genes/direction, clinical bleed grade, liability score/tier, active-bleed flag, paper_note).
- **Table_S2_HGT_training_curves.csv** — Per-epoch HGT training/validation loss and test AUC / AP across three tasks (50 epochs; best ARIA-H risk AUC 0.888 @ epoch 48, synergy 0.833 @ epoch 48, repurposing 0.868 @ epoch 49).
- **Table_S3_full_MR_results.csv** — Complete two-sample MR results across 24 genes × 9 outcomes (216 tests; 189 valid): nSNP, beta/se/p, OR and 95% CI, Egger intercept, MR-PRESSO, Q-stat, I-sq, mean F, Steiger, passes.
- **Table_S4_colocalization.csv** — Colocalization PP.H0–PP.H4 across six brain tissues + blood for every gene-outcome pair (1728 rows), with coloc_flag at PP.H4>=0.70.

## Excel bundle
- `Paper_Supplementary_Tables_Revised.xlsx` — All four Supplementary Tables as separate sheets (S1_Full_Drug_Risk, S2_HGT_Training, S3_MR_Results, S4_Colocalization).

## Code
All reproducible code is provided in the GitHub repository `ad_antiab_kg`:
`03_kg_build.py`, `04*_faers_*.py`, `08_gnn_linkpred_hgt.py`, `05*_mr*.R`, `06_coloc.R`, `build_fusion.py`.
