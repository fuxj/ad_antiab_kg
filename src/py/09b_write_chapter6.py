#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Chapter-6 Results section as a .docx file (Alzheimer's Research & Therapy style)."""
import argparse, sys
from pathlib import Path
from datetime import datetime

def _try_import():
    try:
        from docx import Document
        from docx.shared import Pt, Inches, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        return Document, Pt, Inches, RGBColor, WD_ALIGN_PARAGRAPH
    except ModuleNotFoundError:
        raise SystemExit("python-docx not installed. Install with: conda install -y -c conda-forge python-docx")

def find_proj_root(start=None):
    p = (Path(start) if start else Path(".")).resolve()
    for _ in range(8):
        if (p / "results" / "paper_tables" / "Table_S2_comedication_liability.csv").exists():
            return p
        p = p.parent
    raise SystemExit("cannot find project root")

def main(argv=None):
    Document, Pt, Inches, RGBColor, WD_ALIGN_PARAGRAPH = _try_import()
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=None)
    args = ap.parse_args(argv)
    proj = find_proj_root(args.proj)

    doc = Document()

    # Page style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    h = doc.add_heading("Results — Chapter 6: Multi-modal KG fusion for ARIA risk mapping and repurposing", level=1)

    # ---- 6.1 ----
    doc.add_heading("6.1 A three-source co-medication liability map for ARIA-H", level=2)
    doc.add_paragraph(
        "To identify drugs that may aggravate ARIA-H when co-administered with anti-Aβ monoclonal antibodies, "
        "we assembled a liability map for 550 approved small-molecule drugs (421 after collapsing salt and hydrate "
        "duplicates across 332 active-moiety groups). Each drug was scored by integrating three orthogonal sources: "
        "(i) clinical pharmacological bleeding liability, graded as anticoagulant (1.00), antiplatelet P2Y12-receptor "
        "antagonist (0.95), non-selective NSAID with PTGS1 engagement (0.80), or COX-2-selective (0.50); "
        "(ii) direction-aware Mendelian randomization (MR) evidence at the drug target, aligned to the pharmacological "
        "direction of effect; and (iii) GNN-predicted structural ARIA risk, ranked within each candidate pool (HGT-scored "
        "CNS-penetrant drugs vs. Open Targets expanded pool) so that missing GNN scores for expanded-pool drugs did not "
        "distort ranks."
    )
    doc.add_paragraph(
        "Anticoagulants occupied the highest-liability tier. Vitamin K antagonists ranked first (acenocoumarol, "
        "phenprocoumon and warfarin, adjusted liability = 0.900), followed by the direct factor Xa inhibitors apixaban, "
        "rivaroxaban, edoxaban and betrixaban (0.800) and the direct thrombin inhibitor dabigatran (0.794). "
        "P2Y12 receptor antagonists followed (clopidogrel 0.763; prasugrel and ticagrelor 0.670), while non-selective "
        "NSAIDs formed a broad high-liability band (aspirin 0.867; indomethacin 0.858; naproxen 0.839; diclofenac, "
        "dexketoprofen and dexibuprofen 0.780). COX-2-selective celecoxib was substantially lower (0.489), consistent "
        "with its minimal effect on platelet thromboxane generation. Acetaminophen is retained in the table with a "
        "conservative annotation (0.740) despite its weak antiplatelet activity."
    )
    doc.add_paragraph(
        "Genetic evidence was interpreted with explicit directionality. Higher PTGS1 expression was associated with "
        "lower intracerebral haemorrhage risk (OR 0.886, 95% CI 0.800–0.981, p = 0.020), implying that pharmacological "
        "inhibition of PTGS1—as achieved by aspirin and other non-selective NSAIDs—shifts liability in the unfavourable "
        "direction; VKORC1 showed the same protective direction (OR 0.549, 95% CI 0.330–0.915, p = 0.021). Two MR "
        "estimates that failed the Egger intercept test for directional pleiotropy (PTGS1→ICH, Egger p = 0.012; APP→SAH, "
        "Egger p = 0.006) were flagged but retained in the main analysis; a sensitivity analysis excluding pleiotropic "
        "rows is reported. For factor Xa, the MR direction (F10, risk-increasing) contradicted the clinical pharmacology "
        "of anticoagulation; these five drugs are annotated with a direction-conflict flag and were graded on clinical "
        "pharmacology alone, with the MR term neutralised."
    )

    # insert fig 6.1
    fig1 = proj / "results" / "figures" / "Fig6_1_core_liability_bar.png"
    if fig1.exists():
        doc.add_picture(str(fig1), width=Inches(6.5))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run("Figure 6.1  Co-medication liability map for ARIA-H (systemic first-line high-tier drugs). "
                        "Colour indicates clinical pharmacological class; † marks drugs where MR direction conflicts "
                        "clinical pharmacology (graded clinically with MR term neutralised).")
        r.italic = True; r.font.size = Pt(9.5)

    # ---- 6.2 ----
    doc.add_heading("6.2 Combination partners", level=2)
    doc.add_paragraph(
        "Among drugs that were CNS-penetrant, free of bleeding liability and supported by at least one active "
        "mechanistic route (synergy with anti-Aβ pathways, prior-AD-gene engagement, or AD-aligned genetic evidence), "
        "eight met the eligibility criteria. Three received markedly higher partner scores: lithium carbonate (f_partner "
        "= 0.997, via GSK3A/GSK3B), bupropion (0.991) and dextromethorphan hydrobromide (0.977, via GRIN2A/SIGMAR1). "
        "Notably, bupropion and dextromethorphan are the two components of AXS-05, which is under phase 2/3 "
        "investigation for Alzheimer's disease agitation (ADVANCE-1/ACCORD). The model selected both components "
        "independently and without prior knowledge of this combination, providing an external, literature-anchored "
        "sanity check of the ranking framework. A further 397 safe-but-mechanism-unmatched drugs are reported "
        "separately as a tier-2 list."
    )

    # ---- 6.3 ----
    doc.add_heading("6.3 Repurposing candidates and a target-coverage audit", level=2)
    doc.add_paragraph(
        "Mechanistically anchored repurposing candidates were stratified into four roles: target-prior (5 drugs; "
        "e.g. lithium via GSK3B, sacubitril and racecadotril via MME), genetic-counter (15 drugs; all ACE inhibitors, "
        "since genetically higher ACE expression is protective against AD and pharmacological inhibition is therefore "
        "unfavourable in direction), false-positive-direction (1 drug), and structural-only (400 drugs, ranked by GNN "
        "score alone). No drug satisfied the highest tier of genetically aligned repurposing evidence (n_aligned = 0): "
        "within the current candidate library, no compound both engaged an AD-efficacy gene and did so in the direction "
        "implied by the MR estimate."
    )
    doc.add_paragraph(
        "A target-coverage audit localised the reason. ACE is reachable only through ACE inhibitors, which we classify "
        "as direction-opposed. BIN1, LRP1 and PLCG2 are not targeted by any drug in the knowledge graph—an upstream, "
        "data-side gap rather than a scoring artefact. We therefore caution that structure-driven repurposing "
        "sensitivity is bounded by drug–target coverage, and report these gaps explicitly. Finally, nirogacestat (a "
        "γ-secretase/Notch inhibitor) was flagged as a directional false positive: although it engages PSEN1/PSEN2, these "
        "genes did not pass the MR evidence threshold, and γ-secretase inhibition has repeatedly worsened cognition in "
        "AD trials; it is retained in the table as a cautionary example of target-hit scoring without directionality."
    )

    # insert fig 6.2
    fig2 = proj / "results" / "figures" / "Fig6_2_mr_bleeding_forest.png"
    if fig2.exists():
        doc.add_picture(str(fig2), width=Inches(6.5))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run("Figure 6.2  Direction-aware MR forest plot for bleeding-related drug targets. Blue denotes "
                        "protective (higher expression → lower intracerebral haemorrhage risk); red denotes risk. "
                        "The column to the right summarises the pharmacological implication of drug inhibition. † marks "
                        "the direction conflict for F10.")
        r.italic = True; r.font.size = Pt(9.5)

    # tables note
    doc.add_heading("Supporting tables (see paper_tables.xlsx)", level=2)
    doc.add_paragraph(
        "Table S1 Repurposing candidates with genetic-anchor, counter-evidence, false-positive-direction and "
        "structural-only tiers (421 rows).", style="List Bullet")
    doc.add_paragraph(
        "Table S2 ARIA-H co-medication liability map with clinical class, bleeding class and direction-conflict "
        "annotations (421 rows); core_risk_drugs.csv contains the 23 systemic first-line high-tier drugs shown in "
        "Figure 6.1.", style="List Bullet")
    doc.add_paragraph(
        "Table S3 Eligible combination partners (8) and tier-2 safe-but-unmatched drugs (303 rows).",
        style="List Bullet")
    doc.add_paragraph(
        "Audit sheet documents salt/hydrate collapsing and clinical classifications.",
        style="List Bullet")

    out = proj / "results" / "paper" / "Chapter6_Results.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print("[write]", out)


if __name__ == "__main__":
    sys.exit(main())
