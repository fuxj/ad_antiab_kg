#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08b_fusion_v3.py -- Chapter 6, step 2b
Direction-aware, outcome-stratified evidence fusion for
drug repurposing and combination therapy.

=============================================================================
WHY v3  (what was scientifically wrong with the v2 fusion)
=============================================================================
The v2 fusion collapsed every Mendelian-randomization (MR) hit of a drug's
target genes into ONE binary flag ("MR evidence"), ignoring BOTH the outcome
and the direction of the effect.  That produced two wrong boosts:

  * PTGS1 -> ICH  (p = 0.020, OR = 0.886)
        This is a BLEEDING signal: higher PTGS1 expression -> lower ICH risk,
        i.e. inhibiting PTGS1 (aspirin / ibuprofen / naproxen / indomethacin)
        would RAISE bleeding risk.  v2 nevertheless read it as positive
        "repurposing evidence" and pushed the NSAIDs to the top of the
        "novel AD drug" list.
  * HMGCR -> LDL  (p = 0.036, OR = 1.227)
        This is the pharmacology positive control (statins lower LDL), not AD
        efficacy evidence; v2 nevertheless boosted every statin.

The v2 mitigation score (1 - max_aria_risk) additionally rewarded drugs that
merely have no ARIA term in their adverse-event profile (psychotropics,
lithium, cannabidiol).  "Does not itself cause ARIA" is a TRUE-NEGATIVE
property, not protective activity, so it cannot by itself qualify a drug as
an ARIA-mitigating combination partner.

=============================================================================
WHAT v3 DOES
=============================================================================
1. Stratifies MR / colocalization results by OUTCOME and keeps the direction:
       AD efficacy   : AD, AD_repl1, AD_repl2
       bleeding      : ICH, ICH_fin, SAH
       pharmacology  : LDL, IS, SVD          (controls - reported, NOT scored)
   Only MR with p < 0.05 or coloc PP.H4 >= 0.80 is counted.

2. Replaces the single blended f_score with three task-specific,
   direction-aware scores:

       f_repurpose  (Task A) : GNN structural proximity to the AD indication
                               + synergy route + AD-efficacy genetic evidence
                               + CNS penetration
       liability    (Task B) : ARIA-H co-medication liability
                               = self ARIA risk + bleeding-gene evidence
                               + FAERS ARIA signal   (higher = more dangerous)
       f_partner    (Task C) : combination-partner desirability
                               = synergy route + low liability + CNS

3. Emits honest, clearly-named tables (no blended black-box f_score):
       repositioning_top_v3.csv       Task A  - repurposing candidates
       comedication_risk_v3.csv       Task B  - co-medication safety map
       combination_partners_v3.csv    Task C  - combination partners
       fusion_evidence_v3.csv         full merged evidence table
       fusion_v3_summary.json         machine-readable summary

=============================================================================
INPUTS  (all produced by 08_gnn_linkpred_hgt.py predict + chapter 5)
=============================================================================
  results/tables/repositioning_all_scored.csv      (all drugs, GNN score)
  results/tables/repositioning_known_recovery.csv  (known AD/dementia drugs)
  results/tables/combination_mitigation.csv        (Task B raw scores)
  results/tables/combination_synergy.csv           (Task C raw routes)
  results/tables/mr_summary.csv                    (chapter 5 MR)
  results/tables/coloc_brain_vs_blood.csv          (chapter 5 coloc)
  results/tables/faers_pt_level_signals.csv        (optional)
  data/processed/kg_ext/nodes.parquet              (node names)
  data/processed/kg_ext/edges.parquet              (drug -> target genes)
  data/interim/kg_ext/candidate_drugs.csv          (optional, is_cns)

USAGE
  python 08b_fusion_v3.py --proj .
  python 08b_fusion_v3.py --proj . --topk 30
  python 08b_fusion_v3.py --proj . --ad-pval 0.05 --coloc-thr 0.80

All column names are auto-detected (see *_COL_CANDIDATES below) because the
exact headers of mr_summary.csv / coloc_brain_vs_blood.csv depend on which
chapter-5 script version produced them; a missing column is reported LOUDLY
instead of silently degrading to an all-zero score.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import re

import numpy as np
import pandas as pd

VERSION = "3.0.1"

# ---------------------------------------------------------------------------
# outcome strata (compared case-insensitively)
# ---------------------------------------------------------------------------
AD_OUTCOMES = {"ad", "ad_repl1", "ad_repl2"}
BLEED_OUTCOMES = {"ich", "ich_fin", "sah"}
CONTROL_OUTCOMES = {"ldl", "is", "svd"}          # pharmacology / other - not scored

# curated AD GWAS / drug-target genes - only used as a transparent biological
# prior label on the output (never trained on, never affects the ranking)
AD_PRIOR_GENES = {
    "APOE", "BIN1", "CLU", "ABCA7", "CR1", "PICALM", "MS4A6A", "MS4A4E",
    "CD33", "TREM2", "PLCG2", "ABI3", "SORL1", "ADAM10", "ACE", "APP",
    "PSEN1", "PSEN2", "MAPT", "GSK3B", "INPP5D", "SPI1", "MS4A4A",
    "EPHA1", "CD2AP", "HLA-DRB1", "PTK2B", "SLC24A4", "ZCWPW1",
    "CASS4", "FERMT2", "ECHDC3", "MME", "IDE", "LRP1", "AGER",
}

# ---------------------------------------------------------------------------
# column-name candidates (first present wins)
# ---------------------------------------------------------------------------
MR_GENE_COLS = ["gene", "exposure", "exposure_gene", "gene_symbol",
                "target", "target_gene", "exposure_id"]
MR_OUTCOME_COLS = ["outcome_short", "outcome", "outcome_id", "outcome_name"]
MR_PVAL_COLS = ["pval", "pval_wald", "p_value", "p", "pval_mr", "pval_ivw",
               "p_wald", "p_ivw", "p_wald_ivw", "pvalue", "pval_ivw_random",
               "pval_mr_ivw"]
COLOC_GENE_COLS = MR_GENE_COLS
COLOC_OUTCOME_COLS = MR_OUTCOME_COLS
COLOC_H4_COLS = ["best_brain_PP_H4", "brain_PP_H4", "PP_H4", "pp_h4",
                 "PP.H4", "PP4", "H4", "blood_PP_H4", "best_PP_H4"]
SCORED_ID_COLS = ["drug_id", "drugid", "id", "chembl_id", "node_id"]
SCORED_NAME_COLS = ["drug", "drug_name", "name", "compound", "label"]
SCORED_SCORE_COLS = ["gnn_repurpose_score", "repurpose_score", "score",
                     "gnn_score", "pred_score"]
CAND_NAME_COLS = ["name", "drug", "drug_name", "label"]
CAND_CNS_COLS = ["is_cns", "cns", "bbb", "is_bbb", "cns_penetrant"]


# ===========================================================================
# 0. small utilities
# ===========================================================================
def log(fmt, *a):
    print("[%s] " % datetime.now().strftime("%H:%M:%S") + (fmt % a), flush=True)


def die(fmt, *a):
    print("[%s] [FATAL] " % datetime.now().strftime("%H:%M:%S") + (fmt % a),
          file=sys.stderr, flush=True)
    sys.exit(1)


def find_proj_root(start=None):
    p = Path(start or ".").resolve()
    for cand in [p] + list(p.parents):
        if (cand / "data").is_dir() and (cand / "results").is_dir():
            return cand
    return p


def pick_col(df, candidates, role, where, required=False, warn=True):
    """Return the first candidate column present in df (case-insensitive)."""
    if df is None:
        return None
    low = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    if required:
        die("column for '%s' not found in %s (tried %s); actual columns: %s",
            role, where, ", ".join(candidates), ", ".join(map(str, df.columns)))
    if warn:
        log("  [warn] no column for '%s' in %s (tried %s) -> evidence will be empty",
            role, where, ", ".join(candidates))
    return None


def pick_col_regex(df, pattern, role, where, warn=True):
    """Fallback: return the first column whose name matches `pattern`."""
    if df is None:
        return None
    rx = re.compile(pattern, re.I)
    for c in df.columns:
        if rx.search(str(c)):
            return c
    if warn:
        log("  [warn] no column matching /%s/ for '%s' in %s", pattern, role, where)
    return None


def _read_csv(path):
    if not path.exists():
        log("  [warn] missing %s", path.name)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:                                   # pragma: no cover
        log("  [warn] cannot read %s: %s", path.name, str(e)[:120])
        return None


def _read_parquet(path):
    if not path.exists():
        log("  [warn] missing %s", path.name)
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:                                   # pragma: no cover
        log("  [warn] cannot read %s: %s", path.name, str(e)[:120])
        return None


def pct_rank(s):
    """Percentile rank in (0,1]; robust to saturated / low-variance scores.
    Returns a constant 0.5 column when a score has no spread."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if len(x) <= 1 or x.nunique() <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    return x.rank(pct=True, method="average").astype(float)


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def norm_outcome(s):
    """Canonicalise outcome labels: lower-case, strip, spaces/hyphens -> _."""
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True))


# ===========================================================================
# 1. load every input
# ===========================================================================
def load_inputs(proj):
    t = proj / "results" / "tables"
    kg = proj / "data" / "processed" / "kg_ext"

    d = {}
    d["all_scored"] = _read_csv(t / "repositioning_all_scored.csv")
    d["known"] = _read_csv(t / "repositioning_known_recovery.csv")
    d["mitig"] = _read_csv(t / "combination_mitigation.csv")
    d["syn"] = _read_csv(t / "combination_synergy.csv")
    d["mr"] = _read_csv(t / "mr_summary.csv")
    d["coloc"] = _read_csv(t / "coloc_brain_vs_blood.csv")
    d["faers"] = _read_csv(t / "faers_pt_level_signals.csv")
    d["nodes"] = _read_parquet(kg / "nodes.parquet")
    d["edges"] = _read_parquet(kg / "edges.parquet")
    d["cand"] = _read_csv(proj / "data" / "interim" / "kg_ext" / "candidate_drugs.csv")
    return d


# ===========================================================================
# 2. stratify genetic evidence by outcome
# ===========================================================================
def stratify_genes(mr, coloc, p_thr=0.05, coloc_thr=0.80):
    """Return (ad_eff, bleed, control, bleed_dir).

    bleed_dir maps a bleeding-stratum gene to its MR effect direction:
      'protective' : OR < 1 - higher expression lowers bleeding risk, so
                     INHIBITING the gene raises bleeding risk
      'risk'       : OR > 1  - higher expression raises bleeding risk
      None         : direction unknown (entry came from coloc only)
    """
    ad_eff, bleed, control = set(), set(), set()
    bleed_dir = {}

    if mr is not None:
        gcol = pick_col(mr, MR_GENE_COLS, "gene", "mr_summary.csv", required=True)
        ocol = pick_col(mr, MR_OUTCOME_COLS, "outcome", "mr_summary.csv", required=True)
        pcol = pick_col(mr, MR_PVAL_COLS, "p-value", "mr_summary.csv", warn=False)
        if pcol is None:
            pcol = pick_col_regex(mr, r"^p(v(al)?)?(_|$)", "p-value",
                                  "mr_summary.csv", warn=False)
        if pcol is None:
            die("no p-value column in mr_summary.csv; columns: %s",
                ", ".join(map(str, mr.columns)))
        m = pd.DataFrame({
            "_g": mr[gcol].astype(str).str.upper(),
            "_o": norm_outcome(mr[ocol]),
            "_p": numeric(mr[pcol]),
        })
        # effect size: prefer an OR column, else exponentiate log-OR (b/beta)
        orcol = pick_col(mr, ["or", "OR", "odds_ratio", "or_ivw", "or_wald"],
                         "OR", "mr_summary.csv", warn=False)
        bcol = pick_col(mr, ["b", "beta", "b_ivw", "b_wald", "effect"],
                        "b(log-OR)", "mr_summary.csv", warn=False)
        if orcol is not None:
            m["_or"] = numeric(mr[orcol])
        elif bcol is not None:
            m["_or"] = np.exp(numeric(mr[bcol]))
        else:
            m["_or"] = np.nan
            log("  [warn] no OR/effect column in mr_summary.csv -> bleeding gene "
                "direction will be unknown")
        sig = m[m["_p"].notna() & (m["_p"] < p_thr)]
        ad_eff |= set(sig[sig["_o"].isin(AD_OUTCOMES)]["_g"])
        _bl = sig[sig["_o"].isin(BLEED_OUTCOMES)]
        bleed |= set(_bl["_g"])
        control |= set(sig[sig["_o"].isin(CONTROL_OUTCOMES)]["_g"])
        for g, o in zip(_bl["_g"], _bl["_or"]):
            if pd.isna(o):
                continue
            d = "protective" if float(o) < 1.0 else "risk"
            # a gene may have several bleeding outcomes: keep the strongest claim
            if g not in bleed_dir or bleed_dir[g] is None:
                bleed_dir[g] = d
        log("[G] MR rows=%d | nominal-significant (p<%.3g)=%d",
            len(m), p_thr, len(sig))
    else:
        log("  [warn] mr_summary.csv unavailable -> no MR genetic evidence")

    if coloc is not None:
        gcol = pick_col(coloc, COLOC_GENE_COLS, "gene", "coloc_brain_vs_blood.csv")
        ocol = pick_col(coloc, COLOC_OUTCOME_COLS, "outcome", "coloc_brain_vs_blood.csv")
        hcols = [c for c in COLOC_H4_COLS if c in coloc.columns]
        if gcol and ocol and hcols:
            ppmax = pd.Series(0.0, index=coloc.index)
            for c in hcols:
                m = numeric(coloc[c]).fillna(0.0)
                m = pd.Series(np.where(coloc[c].astype(str).str.lower()
                                       .isin(["nan", "na", "", "none"]), 0.0, m),
                              index=coloc.index)
                ppmax = np.maximum(ppmax, m)
            c = pd.DataFrame({
                "_g": coloc[gcol].astype(str).str.upper(),
                "_o": norm_outcome(coloc[ocol]),
                "_pp": ppmax,
            })
            hit = c[c["_pp"] >= coloc_thr]
            ad_eff |= set(hit[hit["_o"].isin(AD_OUTCOMES)]["_g"])
            for g in set(hit[hit["_o"].isin(BLEED_OUTCOMES)]["_g"]):
                bleed.add(g)
                bleed_dir.setdefault(g, None)
            control |= set(hit[hit["_o"].isin(CONTROL_OUTCOMES)]["_g"])
            log("[G] coloc rows=%d | PP.H4>=%.2f = %d", len(c), coloc_thr, len(hit))
        else:
            log("  [warn] coloc file present but gene/outcome/H4 columns not found "
                "(gene=%s outcome=%s H4=%s) -> coloc evidence skipped",
                gcol, ocol, hcols)
    else:
        log("  [warn] coloc_brain_vs_blood.csv unavailable -> no coloc evidence")

    for lo, hi in [(ad_eff, "AD-efficacy"), (bleed, "bleeding")]:
        if not lo:
            log("  [warn] genetic strata contain NO %s gene - check column mapping!", hi)
    return ad_eff, bleed, control, bleed_dir


# ===========================================================================
# 3. drug -> target genes, from the KG
# ===========================================================================
def drug_targets(edges, nodes):
    d2g = defaultdict(set)
    if edges is None or nodes is None:
        log("  [warn] nodes/edges parquet unavailable -> target mapping empty")
        return d2g
    ncol = pick_col(nodes, ["node_id", "id", "nodeid"], "node_id", "nodes.parquet")
    nname = pick_col(nodes, ["name", "node_name", "label"], "name", "nodes.parquet")
    if ncol is None or nname is None:
        log("  [warn] nodes.parquet lacks node_id/name -> target mapping empty")
        return d2g
    nm = dict(zip(nodes[ncol].astype(str), nodes[nname].astype(str)))
    # NB: do NOT put bare "source"/"target" early - edges.parquet carries a
    # provenance column named "source" (edge origin, constant string) which
    # would be mistaken for the source-node column.
    scol = pick_col(edges, ["src", "src_id", "head", "head_id", "from", "from_id",
                            "source_id", "source"], "src", "edges.parquet")
    dcol = pick_col(edges, ["dst", "dst_id", "tail", "tail_id", "to", "to_id",
                            "target_id", "target"], "dst", "edges.parquet")
    tcol = pick_col(edges, ["edge_type", "etype", "relation", "type"],
                    "edge_type", "edges.parquet")
    if scol is None or dcol is None or tcol is None:
        log("  [warn] edges.parquet lacks src/dst/edge_type -> target mapping empty")
        return d2g
    et = edges[tcol].astype(str).str.lower()
    dt = edges[et.str.contains("target", na=False)]
    if len(dt) == 0:                    # fall back to the whole edge table
        log("  [warn] no edge_type containing 'target'; using all edges for drug->gene")
        dt = edges
    for s, dd in zip(dt[scol].astype(str), dt[dcol].astype(str)):
        g = str(nm.get(dd, dd)).upper()
        if g and g not in {"NAN", "NONE", ""}:
            d2g[s].add(g)
    log("[G] drug->target map: %d drugs", len(d2g))
    return d2g


# ===========================================================================
# 4. assemble the evidence table
# ===========================================================================
def build_evidence(proj, args):
    d = load_inputs(proj)
    all_scored = d["all_scored"]
    if all_scored is None:
        die("repositioning_all_scored.csv missing or unreadable")

    idc = pick_col(all_scored, SCORED_ID_COLS, "drug_id",
                   "repositioning_all_scored.csv", required=True)
    ncol = pick_col(all_scored, SCORED_NAME_COLS, "drug name",
                    "repositioning_all_scored.csv", required=True)
    scol = pick_col(all_scored, SCORED_SCORE_COLS, "GNN score",
                    "repositioning_all_scored.csv")
    if scol is None:
        log("  [warn] no GNN repurpose score column; falling back to constant")
        all_scored = all_scored.copy()
        all_scored["_gnn_repurpose"] = 0.0
        scol = "_gnn_repurpose"

    ad_eff, bleed, control, bleed_dir = stratify_genes(d["mr"], d["coloc"],
                                                       args.ad_pval, args.coloc_thr)
    log("[G] genetic strata | AD-efficacy genes: %d | bleeding genes: %d | control genes: %d",
        len(ad_eff), len(bleed), len(control))
    if ad_eff:
        log("    AD-efficacy genes : %s", ", ".join(sorted(ad_eff)))
    if bleed:
        log("    bleeding genes    : %s", ", ".join(sorted(bleed)))
    if control:
        log("    control genes     : %s", ", ".join(sorted(control)))
    if bleed:
        log("    bleeding-gene direction: %s",
            ", ".join("%s=%s" % (g, bleed_dir.get(g) or "?") for g in sorted(bleed)))

    d2g = drug_targets(d["edges"], d["nodes"])

    # ---- novel candidate set = all scored minus known AD/dementia drugs -----
    known_ids = set()
    if d["known"] is not None:
        kcol = pick_col(d["known"], SCORED_ID_COLS, "drug_id",
                        "repositioning_known_recovery.csv", warn=False)
        if kcol:
            known_ids = set(d["known"][kcol].astype(str))
    df = pd.DataFrame({
        "drug_id": all_scored[idc].astype(str),
        "drug": all_scored[ncol].astype(str),
        "gnn_repurpose_score": numeric(all_scored[scol]),
    })
    df["known_indication"] = df["drug_id"].isin(known_ids).astype(int)
    df = df[df["known_indication"] == 0].drop(columns=["known_indication"])
    df = df.drop_duplicates(subset=["drug_id"]).reset_index(drop=True)

    # ---- merge raw task scores ---------------------------------------------
    for src, cols in [
        (d["mitig"], ["max_aria_risk", "mean_aria_risk",
                      "n_aria_concepts", "gnn_mitigation_score"]),
        (d["syn"], ["n_targets", "n_ad_prior_targets", "ad_prior_targets",
                    "n_synergy_pathways", "synergy_pathways",
                    "gnn_synergy_score"]),
    ]:
        if src is None:
            continue
        cid = pick_col(src, SCORED_ID_COLS, "drug_id", "task score table", warn=False)
        if cid is None:
            log("  [warn] task score table has no drug-id column -> skipped")
            continue
        keep = [cid] + [c for c in cols if c in src.columns]
        sub = src[keep].copy()
        sub["_id"] = sub[cid].astype(str)
        sub = sub.drop(columns=[cid])
        if "_id" in df.columns:
            sub = sub.rename(columns={"_id": "_id2"})
            df = df.merge(sub, left_on="drug_id", right_on="_id2", how="left",
                          suffixes=("", "_dup"))
            df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")
                                  or c == "_id2"], errors="ignore")
        else:
            df = df.merge(sub.rename(columns={"_id": "drug_id"}),
                          on="drug_id", how="left")

    for c in ["max_aria_risk", "mean_aria_risk", "gnn_mitigation_score",
              "gnn_synergy_score", "n_synergy_pathways", "n_targets",
              "n_ad_prior_targets"]:
        df[c] = numeric(df[c]) if c in df.columns else np.nan

    # ---- target genes + genetic evidence per drug ---------------------------
    tg = df["drug_id"].map(lambda x: d2g.get(x, set()))
    df["targets"] = tg.apply(lambda s: ",".join(sorted(s)))
    df["n_tg"] = tg.apply(len)
    df["ad_eff_genes"] = tg.apply(lambda s: ",".join(sorted(s & ad_eff)))
    df["bleed_genes"] = tg.apply(lambda s: ",".join(sorted(s & bleed)))
    df["ctrl_genes"] = tg.apply(lambda s: ",".join(sorted(s & control)))
    df["gene_prior"] = tg.apply(lambda s: ",".join(sorted(s & AD_PRIOR_GENES)))
    df["bleed_dir"] = tg.apply(
        lambda s: ",".join("%s(%s)" % (g, bleed_dir.get(g) or "?")
                           for g in sorted(s & bleed)))

    df["has_ad_eff"] = (df["ad_eff_genes"].str.len() > 0).astype(float)
    df["has_bleed"] = (df["bleed_genes"].str.len() > 0).astype(float)
    df["has_ctrl"] = (df["ctrl_genes"].str.len() > 0).astype(float)

    # ---- CNS + FAERS ARIA flags --------------------------------------------
    cns_map, faers_map = {}, {}
    if d["cand"] is not None:
        cn = pick_col(d["cand"], CAND_NAME_COLS, "name", "candidate_drugs.csv", warn=False)
        cc = pick_col(d["cand"], CAND_CNS_COLS, "is_cns", "candidate_drugs.csv", warn=False)
        if cn and cc:
            for n_, v in zip(d["cand"][cn].astype(str), numeric(d["cand"][cc])):
                cns_map[n_.upper()] = 0 if pd.isna(v) else int(v > 0)
        else:
            log("  [warn] candidate_drugs.csv lacks name/is_cns -> CNS flag all 0")
    if d["faers"] is not None:
        dcol = pick_col(d["faers"], SCORED_NAME_COLS, "drug", "faers_pt_level_signals.csv", warn=False)
        gcol = None
        for c in ("event_group", "aria_concept", "concept", "event"):
            if c in d["faers"].columns:
                gcol = c
                break
        if dcol and gcol:
            for n_, tag in zip(d["faers"][dcol].astype(str), d["faers"][gcol].astype(str)):
                if "aria" in tag.lower():
                    faers_map[n_.upper()] = 1
    du = df["drug"].astype(str).str.upper()
    df["is_cns"] = du.map(cns_map).fillna(0).astype(int)
    df["faers_aria_signal"] = du.map(faers_map).fillna(0).astype(int)

    # =======================================================================
    # 5. three task-specific, direction-aware composite scores
    # =======================================================================
    df["f_repurpose"] = (
        0.45 * pct_rank(df["gnn_repurpose_score"]) +
        0.25 * pct_rank(df["gnn_synergy_score"]) +
        0.20 * df["has_ad_eff"] +
        0.10 * df["is_cns"]
    )
    df["liability"] = (
        0.45 * pct_rank(df["max_aria_risk"]) +
        0.40 * df["has_bleed"] +
        0.15 * df["faers_aria_signal"].clip(0, 1)
    )
    # "active protection" gate: a combination partner must bring a POSITIVE
    # mechanism (a synergy route, an AD-anchored target, or AD-efficacy genetic
    # evidence).  Being merely "free of its own ARIA liability" is a
    # true-negative property and is NOT enough - such drugs are demoted.
    df["has_active_route"] = (
        (df["gnn_synergy_score"].fillna(0) > 0) |
        (df["n_synergy_pathways"].fillna(0) > 0) |
        (df["n_ad_prior_targets"].fillna(0) > 0) |
        (df["has_ad_eff"] > 0)
    ).astype(float)
    df["passive_only"] = 1.0 - df["has_active_route"]
    # A partner that itself carries bleeding liability (e.g. warfarin, NSAIDs
    # acting on the bleeding-protective PTGS1) is disqualified: raising
    # bleeding risk is the exact opposite of mitigating ARIA-H.
    df["bleed_excluded"] = (df["has_bleed"] > 0).astype(float)
    df["partner_eligible"] = ((df["has_active_route"] > 0) &
                              (df["has_bleed"] == 0)).astype(float)
    df["f_partner"] = df["partner_eligible"] * (
        0.50 * pct_rank(df["gnn_synergy_score"]) +
        0.30 * (1.0 - df["liability"]) +
        0.20 * df["is_cns"]
    )
    df["rank_repurpose"] = df["f_repurpose"].rank(ascending=False, method="min").astype(int)
    df["rank_liability"] = df["liability"].rank(ascending=False, method="min").astype(int)
    df["rank_partner"] = df["f_partner"].rank(ascending=False, method="min").astype(int)
    return df, ad_eff, bleed, control, bleed_dir


# ===========================================================================
# 6. write tables + console report
# ===========================================================================
def write_outputs(proj, df, ad_eff, bleed, control, bleed_dir, args):
    tdir = proj / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    common = ["drug_id", "drug", "is_cns", "targets", "n_tg",
              "gene_prior", "ad_eff_genes", "bleed_genes", "bleed_dir", "ctrl_genes",
              "gnn_repurpose_score", "gnn_synergy_score", "gnn_mitigation_score",
              "max_aria_risk", "faers_aria_signal"]
    common = [c for c in common if c in df.columns]

    a = df.sort_values("f_repurpose", ascending=False).reset_index(drop=True)
    a[common + ["f_repurpose", "has_ad_eff", "rank_repurpose"]].to_csv(
        tdir / "repositioning_top_v3.csv", index=False)

    b = df.sort_values("liability", ascending=False).reset_index(drop=True)
    b[common + ["liability", "has_bleed", "rank_liability"]].to_csv(
        tdir / "comedication_risk_v3.csv", index=False)

    c = df.sort_values("f_partner", ascending=False).reset_index(drop=True)
    c[common + ["f_partner", "liability", "has_active_route", "partner_eligible",
                "bleed_excluded", "passive_only", "rank_partner"]].to_csv(
        tdir / "combination_partners_v3.csv", index=False)

    df.sort_values("f_repurpose", ascending=False).to_csv(
        tdir / "fusion_evidence_v3.csv", index=False)

    def _s(v):
        try:
            return v if (isinstance(v, str) and v) else "-"
        except Exception:
            return "-"

    k = int(args.topk)
    log("--- Task A  TOP %d repurposing candidates (novel drugs only) ---", k)
    for _, r in a.head(k).iterrows():
        log("  #%-3d %-26s f_repurp=%.3f  ad_eff=%-10s CNS=%d  targets=%s",
            int(r["rank_repurpose"]), str(r["drug"])[:26], r["f_repurpose"],
            _s(r["ad_eff_genes"])[:10], int(r["is_cns"]), _s(r["targets"])[:34])

    log("--- Task B  TOP %d ARIA-H co-medication LIABILITY (avoid) ---", k)
    for _, r in b.head(k).iterrows():
        ar = r["max_aria_risk"]
        log("  #%-3d %-26s liability=%.3f  aria_risk=%s  bleed=%-12s FAERS_ARIA=%d",
            int(r["rank_liability"]), str(r["drug"])[:26], r["liability"],
            ("%.3f" % ar) if pd.notna(ar) else "NA",
            _s(r["bleed_genes"])[:12], int(r["faers_aria_signal"]))

    log("--- Task C  TOP %d combination partners (active route + safe + CNS) ---", k)
    for _, r in c.head(k).iterrows():
        log("  #%-3d %-26s f_partner=%.3f  liability=%.3f  CNS=%d  targets=%s",
            int(r["rank_partner"]), str(r["drug"])[:26], r["f_partner"],
            r["liability"], int(r["is_cns"]), _s(r["targets"])[:34])
    npas = int(((c["passive_only"] > 0) & (c["bleed_excluded"] == 0)).sum())
    if npas:
        log("  [note] %d drug(s) demoted: no active route, e.g. %s",
            npas, ", ".join(c[(c["passive_only"] > 0) &
                              (c["bleed_excluded"] == 0)]["drug"].astype(str).head(6)))
    nbx = int((c["bleed_excluded"] > 0).sum())
    if nbx:
        log("  [note] %d drug(s) EXCLUDED from partners: bleeding liability, e.g. %s",
            nbx, ", ".join(c[c["bleed_excluded"] > 0]["drug"].astype(str).head(8)))

    summary = {
        "version": VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_novel_candidates": int(len(df)),
        "outcome_strata": {
            "AD_efficacy": sorted(AD_OUTCOMES),
            "bleeding": sorted(BLEED_OUTCOMES),
            "control_not_scored": sorted(CONTROL_OUTCOMES),
        },
        "genetic_strata": {
            "ad_efficacy_genes": sorted(ad_eff),
            "bleeding_genes": sorted(bleed),
            "bleeding_gene_direction": {g: (bleed_dir.get(g) or "unknown")
                                        for g in sorted(bleed)},
            "control_genes": sorted(control),
        },
        "score_definition": {
            "f_repurpose": "0.45*pct(gnn_repurpose)+0.25*pct(gnn_synergy)+0.20*AD_eff_evidence+0.10*CNS",
            "liability": "0.45*pct(max_aria_risk)+0.40*bleeding_gene_evidence+0.15*FAERS_ARIA",
            "f_partner": "partner_eligible * [0.50*pct(gnn_synergy)+0.30*(1-liability)+0.20*CNS]",
            "has_active_route": "synergy_score>0 OR n_synergy_pathways>0 OR n_ad_prior_targets>0 OR AD-efficacy genetic evidence",
            "partner_eligible": "has_active_route==1 AND has_bleed==0",
        },
        "counts": {
            "with_AD_efficacy_evidence": int((df["has_ad_eff"] > 0).sum()),
            "with_bleeding_liability": int((df["has_bleed"] > 0).sum()),
            "CNS_penetrant": int((df["is_cns"] > 0).sum()),
            "passive_only_demoted": int(((df["passive_only"] > 0) &
                                         (df["bleed_excluded"] == 0)).sum()),
            "bleed_excluded_from_partners": int((df["bleed_excluded"] > 0).sum()),
        },
    }
    with open(tdir / "fusion_v3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for f in ("repositioning_top_v3.csv", "comedication_risk_v3.csv",
              "combination_partners_v3.csv", "fusion_evidence_v3.csv",
              "fusion_v3_summary.json"):
        log("[write] %s", str(tdir / f))
    return summary


# ===========================================================================
# 7. CLI
# ===========================================================================
def parse_args(argv):
    p = argparse.ArgumentParser(description="chapter-6 direction-aware fusion (v3)")
    p.add_argument("--proj", default=".", help="project root")
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--ad-pval", type=float, default=0.05,
                   help="nominal p for MR genetic evidence")
    p.add_argument("--coloc-thr", type=float, default=0.80,
                   help="PP.H4 threshold for coloc evidence")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    proj = find_proj_root(args.proj)
    log("project root: %s", proj)
    log("=== STAGE fusion v3 (direction-aware) ===")
    df, ad_eff, bleed, control, bleed_dir = build_evidence(proj, args)
    summary = write_outputs(proj, df, ad_eff, bleed, control, bleed_dir, args)
    log("[summary] novel=%d | AD-evidence=%d | bleeding-liability=%d | CNS=%d",
        summary["n_novel_candidates"],
        summary["counts"]["with_AD_efficacy_evidence"],
        summary["counts"]["with_bleeding_liability"],
        summary["counts"]["CNS_penetrant"])
    log("done.")


if __name__ == "__main__":
    main()
