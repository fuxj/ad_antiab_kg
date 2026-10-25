# ad_antiab_kg

**A triangulated multi-modal knowledge-graph, heterogeneous graph transformer (HGT) and Mendelian randomization framework for ARIA risk stratification and drug-repurposing screening in anti-amyloid-β immunotherapy for Alzheimer's disease**

Companion code, knowledge graph, fusion tables and supplementary materials for the manuscript submitted to *Pharmacological Research*.

---

## 1. Overview

This repository contains all analysis code, the constructed knowledge graph (KG node/edge tables), the intermediate data-fusion tables and the Supplementary Materials supporting the study. The pipeline integrates five open-access resources:

| Data source | Role in the study | Chapter |
|---|---|---|
| FDA FAERS (openFDA API) | Disproportionality signal detection (ROR / PRR / BCPNN / MGPS) for ARIA-related events | Ch. 1–2 |
| ClinicalTrials.gov | Anti-Aβ trial information and ARIA reporting | Ch. 1–2 |
| DrugBank / ChEMBL / GO / Reactome / KEGG | KG node and edge construction | Ch. 3 |
| OpenGWAS (IEU OpenGWAS) | Genetic instruments for 24 genes and 9 outcome GWAS | Ch. 5 |
| eQTL Catalogue (EMBL-EBI) | Brain and blood eQTL for colocalization | Ch. 5 |

The framework follows a **triangulation logic**: (i) pharmacovigilance signal detection, (ii) mechanistic KG + graph neural network (GNN/HGT) risk prediction, and (iii) causal inference by two-sample Mendelian randomization (MR) and Bayesian colocalization, finally fused into a single ranked list of repurposing candidates and combination partners.

---

## 2. Repository structure

```
ad_antiab_kg/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml                 # Python 3.10 (torch 2.3.1, torch-geometric 2.5.3, ...)
├── requirements_R.txt              # R package list (TwoSampleMR, coloc, ...)
├── src/
│   ├── py/
│   │   ├── 01_download_faers.py        # Ch.2  FAERS download + cleaning
│   │   ├── 03_kg_build.py              # Ch.3  multi-modal KG construction
│   │   ├── 04_gnn_risk.py              # Ch.4  GNN risk prediction
│   │   ├── 08_gnn_linkpred_hgt.py      # Ch.6  HGT link prediction (train / predict)
│   │   └── 08b_fusion_v3.py            # Ch.6  multi-evidence fusion & ranking
│   └── r/
│       ├── 02_signal_detection.R       # Ch.2  ROR / PRR / BCPNN / MGPS
│       ├── 05_mr_validation.R          # Ch.5  two-sample MR
│       ├── 06_coloc.R                  # Ch.5  colocalization (blood/generic)
│       └── 07_coloc_brain.R            # Ch.5  colocalization (brain eQTL)
├── data/
│   ├── kg/           nodes.parquet, edges.parquet        # constructed KG (core)
│   ├── kg_ext/       nodes.parquet, edges.parquet        # extended KG (HGT input)
│   └── (faers/, raw/, interim/  NOT included - large raw data, regenerate with src scripts)
├── results/
│   ├── tables/       signal tables, MR summary, fusion ranking, paper tables
│   └── figures/      generated figures
└── supplementary/    Table_S1-S4, FigS1, Excel compilation
```

> NOTE. All scripts were written for a portable layout. Original absolute paths
> (e.g. `/home/<user>/ad_antiab_kg`) must be adapted to your own environment;
> most Python scripts accept `--proj .` to define the project root.

---

## 3. Environment

### Python (3.10)

```bash
conda env create -f environment.yml
conda activate neuroai
```

Key packages: `torch==2.3.1`, `torch-geometric==2.5.3`, `pandas`, `numpy`, `scipy`, `networkx`, `pyarrow`, `lifelines`, `matplotlib`.

### R (4.4)

```r
# install.packages(c("remotes","BiocManager"))
install.packages("TwoSampleMR")
install.packages("coloc")     # v5.2.3
```

> Recommend installing R packages from conda-forge binaries if compiling from
> source fails, e.g. `conda install -c conda-forge r-coloc`.

---

## 4. Reproducing the analysis

```bash
# Ch. 2 - pharmacovigilance signals
python src/py/01_download_faers.py
Rscript src/r/02_signal_detection.R

# Ch. 3 - knowledge graph
python src/py/03_kg_build.py --proj .

# Ch. 4 - GNN drug-risk prediction
python src/py/04_gnn_risk.py --proj .

# Ch. 5 - causal inference
Rscript src/r/05_mr_validation.R
Rscript src/r/06_coloc.R
Rscript src/r/07_coloc_brain.R

# Ch. 6 - HGT link prediction and evidence fusion
python src/py/08_gnn_linkpred_hgt.py predict --proj . --device cuda
python src/py/08b_fusion_v3.py --proj .
```

Outputs are written to `results/tables/` (CSV) and `results/figures/` (PNG/SVG).
Long-running steps are idempotent and resumable.

---

## 5. Supplementary Materials

| File | Content |
|---|---|
| `Table_S1_421_drugs_full_risk.csv` | Full risk table for all 421 screened drugs (25 columns: GNN scores, FAERS ROR, MR evidence, liability grade, final tier, notes) |
| `Table_S2_HGT_training_curves.csv` | Per-epoch HGT training/validation metrics |
| `Table_S3_full_MR_results.csv` | Complete MR results (24 genes x 9 outcomes; nSNP, beta, SE, p, OR, 95% CI, Egger, MR-PRESSO, Q, I², F, Steiger) |
| `Table_S4_colocalization.csv` | Full colocalization results (6 brain regions + blood; PP.H0–H4) |
| `FigS1_HGT_training_curves.png` | HGT 50-epoch training curves |
| `Paper_Supplementary_Tables.xlsx` | Excel compilation of the above + main Tables 1–3 |

---

## 6. Data availability

All primary data are publicly available: FDA FAERS via the openFDA API, ClinicalTrials.gov API, OpenGWAS (IEU), eQTL Catalogue (EMBL-EBI), DrugBank, ChEMBL, GO, Reactome and KEGG. Code, constructed KG and derived tables generated in this study are provided in this repository.

---

## 7. Citation

If you use this code or data, please cite:

> [Author list]. [Title]. *Pharmacological Research* (2026). DOI: [to be assigned]

A machine-readable citation is available in `CITATION.cff`.

---

## 8. License

Code is released under the **MIT License** (see `LICENSE`).
Derived data tables are released under **CC BY 4.0**.

---

## 9. Contact

Corresponding author: [name], [institution], [email]
