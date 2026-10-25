#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08b_fusion_v41.py -- Chapter 6, step 2b (v4.1)

Direction-aware, outcome-stratified, coverage-audited evidence fusion for
  Task A  drug repurposing
  Task B  ARIA-H co-medication safety map
  Task C  combination-partner selection

=============================================================================
WHY v4.1  (what v4 still got wrong)
=============================================================================
1. v4 turned the pleiotropy filter into a BLIND auto-drop: any MR row with
   Egger intercept p < 0.05 was deleted before the genetic strata were built.
   That destroyed the single most important result of the chapter, because
   PTGS1 -> ICH (Egger p = 0.012) is exactly the row that anchors Task B.
   With PTGS1 gone the bleeding stratum collapsed to {ACE, F10, VKORC1},
   which no candidate drug targets, and Task B degenerated into a pure GNN
   ARIA-risk ordering topped by three progestogens.

   v4.1 therefore FLAGS pleiotropy instead of dropping it:
     * default  : the row is kept and its gene is tagged; the audit logs the
                  flagged rows and recommends the sensitivity run
     * --strict-pleiotropy : restores the v4 behaviour (drop) so the paper can
                  report a proper sensitivity analysis.

2. v4 could not tell the two very different reasons a stratum gene is
   "uncovered".  v4.1 audits BOTH levels and says which one applies:
     * "present in KG but reachable only via UNSCORED drug(s)" -> pool problem
       (this is perindopril: the KG has PERINDOPRIL->ACE, but perindopril was
        never scored by the GNN, so ACE looks unreachable)
     * "NO drug in the KG targets it at all" -> a genuine data-side gap

3. KEPT from v4: an AD-efficacy gene still needs a POWERED outcome
   (ad / ad_repl1) or >= 2 AD outcomes, so VKORC1 / APP stay out of the AD
   stratum; Task C still excludes bleeding-liable and AD-direction-opposed
   drugs. NEW in v4.1: a Tier-2 partner table keeps the safe-but-mechanism-
   unmatched drugs visible instead of hiding them in the full evidence dump.

2a. The BLEEDING side was direction-blind (has_bleed = the drug targets ANY
    bleeding gene).  v4 scores the DIRECTION of the causal effect:

        protective gene (OR < 1): inhibiting it RAISES bleeding -> penalty 1.0
        risk gene       (OR > 1): inhibiting it LOWERS bleeding -> penalty 0.0
        direction unknown                                      -> penalty 0.5

    Aspirin (PTGS1, protective) therefore stays a liability, while a drug
    acting on a "risk" gene is NOT wrongly flagged.

2b. The AD BENEFIT side had the same blind spot and v4 closes it too.  A drug
    that merely "engaged an AD gene" was credited whatever the direction.
    But chapter-5 MR says ACE -> AD OR = 0.523 (higher ACE expression LOWERS
    AD risk, consistent with the 2025 microglial-ACE work), so INHIBITING ACE
    is directionally unfavourable; whereas BIN1 -> AD OR = 1.106 and
    APP -> AD OR = 1.286 mean inhibiting those genes IS favourable.  v4
    scores AD evidence with the same direction logic:

        risk gene       (OR > 1): inhibiting lowers AD risk -> credit 1.0
        protective gene (OR < 1): inhibiting raises AD risk -> credit 0.0
        direction unknown                                   -> credit 0.5

    Assumption, stated explicitly: the drug's net engagement of its target is
    INHIBITORY.  That holds for the overwhelming majority of the small
    molecules in this pool (aspirin/ibuprofen/... inhibit PTGS1; perindopril
    inhibits ACE; statins inhibit HMGCR); documented agonists must be handled
    as manual exceptions.

3. Task A was a pure structural table: 0/107 candidates carried genetic
   evidence, so the ranking was driven by a near-constant GNN score and the
   psychiatric drugs floated to the top for no mechanistic reason.  v4 splits
   Task A into
       * genetic-anchored / genetic-counter / target-prior  -> repositioning_top_v4.csv
       * structural-only                                    -> repositioning_structural_v4.csv
   and adds a COVERAGE AUDIT that names every AD-efficacy / bleeding gene that
   no candidate drug can reach (the honest reason Task A is starved).

4. Task C's main table now holds eligible partners only; the route-less,
   score-zero drugs stay in the full evidence table rather than producing a
   14-way tie at the bottom of the ranking.

=============================================================================
INPUTS  (produced by 08_gnn_linkpred_hgt.py predict + chapter 5)
=============================================================================
  results/tables/repositioning_all_scored.csv
  results/tables/repositioning_known_recovery.csv
  results/tables/combination_mitigation.csv
  results/tables/combination_synergy.csv
  results/tables/mr_summary.csv
  results/tables/coloc_brain_vs_blood.csv
  results/tables/faers_pt_level_signals.csv        (optional)
  data/processed/kg_ext/nodes.parquet
  data/processed/kg_ext/edges.parquet
  data/interim/kg_ext/candidate_drugs.csv          (optional, is_cns)

OUTPUTS (results/tables/)
  repositioning_top_v4.csv          Task A  anchored / counter / prior
  repositioning_structural_v4.csv   Task A  structural-only (secondary)
  comedication_risk_v4.csv          Task B  co-medication safety map
  combination_partners_v4.csv       Task C  eligible partners
  fusion_evidence_v4.csv            full merged evidence table
  fusion_v4_summary.json            machine-readable summary + audit

USAGE
  python 08b_fusion_v4.py --proj . --topk 20
  python 08b_fusion_v4.py --proj . --drop-bleed-genes APP --drop-ad-genes VKORC1
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "4.3.0"

# ---------------------------------------------------------------------------
# outcome strata (compared case-insensitively)
#   AD_STRONG_OUTCOMES = the powered AD cohorts (Bellenguez 2022 + replication)
# ---------------------------------------------------------------------------
AD_OUTCOMES = {"ad", "ad_repl1", "ad_repl2"}
AD_STRONG_OUTCOMES = {"ad", "ad_repl1"}
BLEED_OUTCOMES = {"ich", "ich_fin", "sah"}
CONTROL_OUTCOMES = {"ldl", "is", "svd"}      # pharmacology / other - NOT scored

# heuristic pharmacology class labels (annotation only, never scored)
BLEED_CLASS_GENES = {
    "anticoagulant": {"F2", "F7", "F9", "F10", "F11", "VKORC1", "PROC", "PROS1",
                      "PLG", "SERPINC1", "SERPIND1"},
    "antiplatelet_nsaid": {"P2RY12", "ITGB3", "ITGA2B", "GP1BA", "GP6", "TBXA2R",
                           "PTGS1", "PTGS2"},
}

# curated AD GWAS / drug-target genes - used ONLY as a transparent biological
# prior label on the output (never trained on, never affects the MR strata)
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
        # fallback: try .csv counterpart (used in sandbox tests without pyarrow)
        csv_p = path.with_suffix(".csv")
        if csv_p.exists():
            try:
                return pd.read_csv(csv_p)
            except Exception:
                pass
        log("  [warn] missing %s", path.name)
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        csv_p = path.with_suffix(".csv")
        if csv_p.exists():
            try:
                return pd.read_csv(csv_p)
            except Exception:
                pass
        log("  [warn] cannot read %s: %s", path.name, str(e)[:120])
        return None


def pct_rank(s):
    """Percentile rank in (0,1]; constant 0.5 when a score has no spread."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if len(x) <= 1 or x.nunique() <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    return x.rank(pct=True, method="average").astype(float)


def pct_rank_pool(s, pool):
    """Percentile rank computed WITHIN each pool stratum (v4.3).

    Rows whose value is missing keep a neutral 0.5: with no HGT score there is
    no information, so the plan-B drugs are neither rewarded nor punished.
    Pooling HGT-scored and plan-B drugs into one percentile (the v4.2
    behaviour) collapsed every plan-B drug onto the bottom of the scale and
    produced large score ties.
    """
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(0.5, index=x.index, dtype=float)
    pool = pool.astype(str)
    for p in sorted(set(pool)):
        m = (pool == p) & x.notna()
        v = x[m]
        if len(v) > 1 and v.nunique() > 1:
            out[m] = v.rank(pct=True, method="average").astype(float)
    return out


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def norm_outcome(s):
    """Canonicalise outcome labels: lower-case, strip, spaces/hyphens -> _."""
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True))


def _dir_weight(direction):
    """Credit of INHIBITING a gene, given the causal direction of its effect.

    risk       (OR > 1): higher expression is harmful  -> inhibition helps -> 1.0
    protective (OR < 1): higher expression is helpful  -> inhibition hurts -> 0.0
    unknown                                            -> weak credit      -> 0.5
    """
    if direction == "risk":
        return 1.0
    if direction == "protective":
        return 0.0
    return 0.5


def _bleed_class(genes):
    hits = []
    for cls, gs in BLEED_CLASS_GENES.items():
        if genes & gs:
            hits.append(cls)
    return "+".join(sorted(set(hits))) if hits else "-"


# ===========================================================================
# 1. load every input
# ===========================================================================
def load_inputs(proj, args):
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
    cand_path = proj / "data" / "interim" / "kg_ext" / args.cand_pool
    d["cand"] = _read_csv(cand_path)
    d["cand_orig"] = _read_csv(proj / "data" / "interim" / "kg_ext" / "candidate_drugs.csv")
    if d["cand"] is None:
        die("candidate pool %s missing - run 08a_expand_pool.py first", cand_path)
    return d


# ===========================================================================
# 2. stratify genetic evidence by outcome (v4 rules + direction)
# ===========================================================================
def stratify_genes(mr, coloc, p_thr=0.05, coloc_thr=0.80,
                   drop_ad=(), drop_bleed=(), min_ad_outcomes=2,
                   pleio_thr=0.05, require_powered_ad=True, strict_pleio=False):
    """Return (ad_eff, bleed, control, bleed_dir, ad_dir, diag).

    ad_eff   : gene with AD efficacy evidence from a POWERED outcome, or >=2 outcomes
    bleed    : gene with non-pleiotropic bleeding-outcome evidence
    bleed_dir: 'protective' (OR<1, inhibiting RAISES bleeding) / 'risk' / None
    ad_dir   : 'risk' (OR>1, inhibiting LOWERS AD risk) / 'protective' / None
    diag     : audit dictionary (which rows were dropped and why)
    """
    drop_ad = {str(g).upper() for g in drop_ad}
    drop_bleed = {str(g).upper() for g in drop_bleed}

    ad_eff, bleed, control = set(), set(), set()
    bleed_dir, ad_dir = {}, {}
    ad_sig = defaultdict(set)          # gene -> {AD outcome}
    ad_ev = defaultdict(list)          # gene -> [(outcome, OR, p, pleio_flag, src)]
    bleed_ev = defaultdict(list)       # gene -> [(outcome, OR, p, pleio_flag, src)]

    diag = {
        "mr_rows": 0, "mr_significant": 0, "coloc_rows": 0, "coloc_hit": 0,
        "pleiotropy_column": None, "pleiotropy_flagged_rows": [],
        "pleiotropy_mode": "strict-drop" if strict_pleio else "flag-retain",
        "pleiotropy_dropped_rows": [],
        "ad_repl2_only_dropped": [], "ad_outcomes_by_gene": {},
        "bleed_outcomes_by_gene": {}, "ad_direction_by_gene": {},
        "bleed_direction_by_gene": {},
        "manual_drop_ad": sorted(drop_ad), "manual_drop_bleed": sorted(drop_bleed),
    }

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

        orcol = pick_col(mr, ["or", "OR", "odds_ratio", "or_ivw", "or_wald",
                              "OR_ivw", "or_ivw_random"], "OR", "mr_summary.csv",
                         warn=False)
        bcol = pick_col(mr, ["b", "beta", "b_ivw", "b_wald", "effect"],
                        "b(log-OR)", "mr_summary.csv", warn=False)

        pcol_pleio = None
        for pat in ("egger", "pleio", "intercept"):
            pcol_pleio = pick_col_regex(mr, pat, "pleiotropy-p", "mr_summary.csv",
                                        warn=False)
            if pcol_pleio is not None:
                break
        if pcol_pleio is not None:
            log("[G] pleiotropy column detected: %s (rows with p<%.3g are %s)",
                pcol_pleio, pleio_thr,
                "EXCLUDED from the genetic strata (--strict-pleiotropy)"
                if strict_pleio else
                "FLAGGED but RETAINED; re-run with --strict-pleiotropy for the "
                "sensitivity analysis")
        else:
            log("  [warn] no Egger/pleiotropy p column in mr_summary.csv -> "
                "pleiotropy-contaminated genes must be removed via --drop-bleed-genes")

        m = pd.DataFrame({
            "_g": mr[gcol].astype(str).str.upper(),
            "_o": norm_outcome(mr[ocol]),
            "_p": numeric(mr[pcol]),
        })
        if orcol is not None:
            m["_or"] = numeric(mr[orcol])
        elif bcol is not None:
            m["_or"] = np.exp(numeric(mr[bcol]))
        else:
            m["_or"] = np.nan
            log("  [warn] no OR/effect column in mr_summary.csv -> effect direction "
                "will be unknown on BOTH the AD and the bleeding side")
        m["_pleio"] = numeric(mr[pcol_pleio]) if pcol_pleio is not None else np.nan

        diag["mr_rows"] = int(len(m))
        diag["pleiotropy_column"] = pcol_pleio
        sig = m[m["_p"].notna() & (m["_p"] < p_thr)]
        diag["mr_significant"] = int(len(sig))
        for g, o, pp, orr, pl in zip(sig["_g"], sig["_o"], sig["_p"],
                                     sig["_or"], sig["_pleio"]):
            pleio_hit = bool(pd.notna(pl) and float(pl) < pleio_thr)
            if pleio_hit:
                diag["pleiotropy_flagged_rows"].append("%s->%s" % (g, o.upper()))
            # a flagged row is DISCARDED only under --strict-pleiotropy; by
            # default it is retained and merely flagged for the sensitivity run
            flagged = pleio_hit and strict_pleio
            if o in AD_OUTCOMES:
                ad_sig[g].add(o)
                ad_ev[g].append((o, orr, float(pp), flagged, "mr"))
            if o in BLEED_OUTCOMES:
                bleed_ev[g].append((o, orr, float(pp), flagged, "mr"))
            if o in CONTROL_OUTCOMES:
                control.add(g)
            if flagged:
                diag["pleiotropy_dropped_rows"].append("%s->%s" % (g, o.upper()))
        log("[G] MR rows=%d | nominal-significant (p<%.3g)=%d", len(m), p_thr, len(sig))
    else:
        log("  [warn] mr_summary.csv unavailable -> no MR genetic evidence")

    if coloc is not None:
        gcol = pick_col(coloc, COLOC_GENE_COLS, "gene", "coloc_brain_vs_blood.csv",
                        warn=False)
        ocol = pick_col(coloc, COLOC_OUTCOME_COLS, "outcome", "coloc_brain_vs_blood.csv",
                        warn=False)
        hcols = [c for c in COLOC_H4_COLS if c in coloc.columns]
        if gcol and ocol and hcols:
            ppmax = pd.Series(0.0, index=coloc.index)
            for c in hcols:
                v = numeric(coloc[c]).fillna(0.0)
                v = pd.Series(
                    np.where(coloc[c].astype(str).str.lower()
                             .isin(["nan", "na", "", "none"]), 0.0, v),
                    index=coloc.index)
                ppmax = np.maximum(ppmax, v)
            cc = pd.DataFrame({
                "_g": coloc[gcol].astype(str).str.upper(),
                "_o": norm_outcome(coloc[ocol]),
                "_pp": ppmax,
            })
            hit = cc[cc["_pp"] >= coloc_thr]
            diag["coloc_rows"] = int(len(cc))
            diag["coloc_hit"] = int(len(hit))
            for g, o in zip(hit["_g"], hit["_o"]):
                if o in AD_OUTCOMES:
                    ad_sig[g].add(o)
                    ad_ev[g].append((o, np.nan, 1.0, False, "coloc"))
                if o in BLEED_OUTCOMES:
                    bleed_ev[g].append((o, np.nan, 1.0, False, "coloc"))
                if o in CONTROL_OUTCOMES:
                    control.add(g)
            log("[G] coloc rows=%d | PP.H4>=%.2f = %d", len(cc), coloc_thr, len(hit))
        else:
            log("  [warn] coloc file present but gene/outcome/H4 columns not found "
                "(gene=%s outcome=%s H4=%s) -> coloc evidence skipped",
                gcol, ocol, hcols)
    else:
        log("  [warn] coloc_brain_vs_blood.csv unavailable -> no coloc evidence")

    # ---- decide AD-efficacy genes ------------------------------------------
    diag["ad_outcomes_by_gene"] = {g: sorted(v) for g, v in ad_sig.items()}
    for g, outs in ad_sig.items():
        if g in drop_ad:
            continue
        if outs & AD_STRONG_OUTCOMES:
            ad_eff.add(g)
        elif (not require_powered_ad) or len(outs) >= min_ad_outcomes:
            ad_eff.add(g)
        else:
            diag["ad_repl2_only_dropped"].append(g)
    ad_eff -= drop_ad

    # ---- AD direction (benefit side) ---------------------------------------
    for g, evs in ad_ev.items():
        ev_ok = [e for e in evs if not e[3]]
        if not ev_ok:
            ad_dir[g] = None
            continue
        best = min(ev_ok, key=lambda e: e[2])      # smallest p = strongest claim
        orr = best[1]
        ad_dir[g] = None if pd.isna(orr) else ("risk" if float(orr) > 1.0
                                               else "protective")
    diag["ad_direction_by_gene"] = {g: (ad_dir.get(g) or "unknown")
                                    for g in sorted(ad_eff)}

    # ---- decide bleeding genes + direction ---------------------------------
    for g, evs in bleed_ev.items():
        diag["bleed_outcomes_by_gene"].setdefault(g, sorted(set(e[0] for e in evs)))
        ev_ok = [e for e in evs if not e[3]]       # drop pleiotropy-flagged rows
        if not ev_ok or g in drop_bleed:
            continue
        bleed.add(g)
        best = min(ev_ok, key=lambda e: e[2])
        orr = best[1]
        bleed_dir[g] = None if pd.isna(orr) else ("protective" if float(orr) < 1.0
                                                  else "risk")
    bleed -= drop_bleed
    diag["bleed_direction_by_gene"] = {g: (bleed_dir.get(g) or "unknown")
                                       for g in sorted(bleed)}

    for lo, nm_ in [(ad_eff, "AD-efficacy"), (bleed, "bleeding")]:
        if not lo:
            log("  [warn] genetic strata contain NO %s gene - check column mapping!", nm_)
    return ad_eff, bleed, control, bleed_dir, ad_dir, diag


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
    if len(dt) == 0:
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
    d = load_inputs(proj, args)
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

    ad_eff, bleed, control, bleed_dir, ad_dir, gdiag = stratify_genes(
        d["mr"], d["coloc"], args.ad_pval, args.coloc_thr,
        drop_ad=args.drop_ad_genes, drop_bleed=args.drop_bleed_genes,
        min_ad_outcomes=args.min_ad_outcomes, pleio_thr=args.pleio_pval,
        require_powered_ad=not args.no_powered_ad_rule,
        strict_pleio=args.strict_pleiotropy)

    log("[G] genetic strata | AD-efficacy genes: %d | bleeding genes: %d | control genes: %d",
        len(ad_eff), len(bleed), len(control))
    if ad_eff:
        log("    AD-efficacy genes : %s", ", ".join(sorted(ad_eff)))
        log("    AD-efficacy direction: %s",
            ", ".join("%s=%s" % (g, ad_dir.get(g) or "?") for g in sorted(ad_eff)))
    if bleed:
        log("    bleeding genes    : %s", ", ".join(sorted(bleed)))
        log("    bleeding-gene direction: %s",
            ", ".join("%s=%s" % (g, bleed_dir.get(g) or "?") for g in sorted(bleed)))
    if gdiag["ad_repl2_only_dropped"]:
        log("    [drop] AD_repl2-only genes removed: %s",
            ", ".join(sorted(gdiag["ad_repl2_only_dropped"])))
    if gdiag.get("pleiotropy_flagged_rows"):
        if gdiag["pleiotropy_dropped_rows"]:
            log("    [drop] pleiotropic MR rows removed (--strict-pleiotropy): %s",
                ", ".join(sorted(gdiag["pleiotropy_dropped_rows"])))
        else:
            log("    [flag] pleiotropy-flagged MR rows RETAINED (sensitivity: "
                "--strict-pleiotropy): %s",
                ", ".join(sorted(gdiag["pleiotropy_flagged_rows"])))

    d2g = drug_targets(d["edges"], d["nodes"])

    # ---- novel candidate set = all scored minus known AD/dementia drugs -----
    known_ids = set()
    if d["known"] is not None:
        kcol = pick_col(d["known"], SCORED_ID_COLS, "drug_id",
                        "repositioning_known_recovery.csv", warn=False)
        if kcol:
            known_ids = set(d["known"][kcol].astype(str))
    # --- novel candidate set = union of GNN-scored + expanded OpenTargets pool ---
    scored_names = set(all_scored[ncol].astype(str).str.upper())
    extra_rows = []
    if d["cand"] is not None:
        cname_col = pick_col(d["cand"], CAND_NAME_COLS, "name", "candidate pool",
                             warn=False)
        if cname_col is None:
            cname_col = "name" if "name" in d["cand"].columns else d["cand"].columns[0]
        for _, cr in d["cand"].iterrows():
            nm = str(cr[cname_col]).strip()
            if not nm or nm.upper() in scored_names:
                continue
            # skip known AD drugs/antibodies in the expanded pool too
            skip = False
            for kn in {"LECANEMAB","DONANEMAB","ADUCANUMAB","GANTENERUMAB","SOLANEZUMAB",
                       "CRENEZUMAB","BAPINEUZUMAB","ELENESTUGENE AUTOTEMCEL"}:
                if kn in nm.upper():
                    skip = True; break
            if skip: continue
            extra_rows.append({
                "drug_id": "POOL:%s" % nm.upper().replace(" ","_"),
                "drug": nm.upper(),
                "gnn_repurpose_score": np.nan,  # no HGT score -> will be median-imputed
                "_pool": "expanded",
                "_ad_targets": str(cr.get("ad_targets","") or ""),
                "_chembl": str(cr.get("chembl_id","") or ""),
                "_drug_type": str(cr.get("drug_type","") or ""),
                "_cand_cns": cr.get("is_cns", None),
            })
    df_scored = pd.DataFrame({
        "drug_id": all_scored[idc].astype(str),
        "drug": all_scored[ncol].astype(str).str.upper(),
        "gnn_repurpose_score": numeric(all_scored[scol]),
        "_pool": "gnn",
        "_ad_targets": "",
        "_chembl": "",
        "_drug_type": "Small molecule",
        "_cand_cns": None,
    })
    if extra_rows:
        log("[G] expanded-pool drugs added to the scoring set: %d", len(extra_rows))
        df_extra = pd.DataFrame(extra_rows)
        df = pd.concat([df_scored, df_extra], ignore_index=True, sort=False)
    else:
        df = df_scored
    df["known_indication"] = df["drug_id"].isin(known_ids).astype(int)
    df = df[df["known_indication"] == 0].drop(columns=["known_indication"])
    df = df.drop_duplicates(subset=["drug"]).reset_index(drop=True)
    # GNN score for expanded drugs is median-imputed (0.5 percentile rank) -- neutral, not penalised
    df["gnn_repurpose_score"] = df["gnn_repurpose_score"].fillna(
        df["gnn_repurpose_score"].median() if df["gnn_repurpose_score"].notna().any() else 0.5)

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
    # drug targets = union of KG edges (for gnn-pool drugs) + ad_targets (for expanded-pool drugs)
    def _get_targets(row):
        s = set(d2g.get(row["drug_id"], set()))
        extra = str(row.get("_ad_targets","") or "")
        if extra and extra not in ("nan","-","NONE",""):
            for g in extra.split(";"):
                g = g.strip().upper()
                if g: s.add(g)
        return s
    tg = df.apply(_get_targets, axis=1)
    df["_tgt_set"] = tg
    df["targets"] = tg.apply(lambda s: ",".join(sorted(s)))
    df["n_tg"] = tg.apply(len)
    df["ad_eff_genes"] = tg.apply(lambda s: ",".join(sorted(s & ad_eff)))
    df["bleed_genes"] = tg.apply(lambda s: ",".join(sorted(s & bleed)))
    df["ctrl_genes"] = tg.apply(lambda s: ",".join(sorted(s & control)))
    df["gene_prior"] = tg.apply(lambda s: ",".join(sorted(s & AD_PRIOR_GENES)))
    df["n_prior_targets"] = tg.apply(lambda s: len(s & AD_PRIOR_GENES))

    df["has_ad_eff"] = (df["ad_eff_genes"].str.len() > 0).astype(float)
    df["has_ctrl"] = (df["ctrl_genes"].str.len() > 0).astype(float)

    # ---- DIRECTION-AWARE AD evidence ---------------------------------------
    df["ad_eff_dir"] = tg.apply(
        lambda s: ",".join("%s(%s)" % (g, ad_dir.get(g) or "?")
                           for g in sorted(s & ad_eff)))
    df["ad_eff_evidence"] = tg.apply(
        lambda s: float(sum(_dir_weight(ad_dir.get(g)) for g in sorted(s & ad_eff)))
    ).clip(0.0, 1.0)
    df["ad_eff_aligned"] = tg.apply(
        lambda s: float(any(ad_dir.get(g) == "risk" for g in sorted(s & ad_eff))))
    df["ad_eff_counter"] = tg.apply(
        lambda s: float(any(ad_dir.get(g) == "protective" for g in sorted(s & ad_eff))))

    # ---- DIRECTION-AWARE bleeding evidence ---------------------------------
    df["bleed_dir"] = tg.apply(
        lambda s: ",".join("%s(%s)" % (g, bleed_dir.get(g) or "?")
                           for g in sorted(s & bleed)))
    df["bleed_penalty"] = tg.apply(
        lambda s: float(sum(1.0 - _dir_weight(bleed_dir.get(g))
                            for g in sorted(s & bleed))))
    df["bleed_evidence"] = (df["bleed_penalty"] / 1.0).clip(0.0, 1.0)
    df["has_bleed"] = (df["bleed_penalty"] > 0).astype(float)
    df["bleed_class"] = tg.apply(_bleed_class)
    # ---- v4.3: GRADED, pool-independent clinical bleeding liability --------
    # The liability ordering must not depend on whether a drug happens to have
    # an HGT score, so the dominant term is a fixed clinical-pharmacology grade.
    #   anticoagulant (warfarin / DOAC / heparin)          -> 1.00
    #   antiplatelet receptor (clopidogrel / ticagrelor)   -> 0.95
    #   non-selective NSAID (aspirin / ibuprofen / ...)    -> 0.80
    #   COX-2 selective (celecoxib / rofecoxib)            -> 0.50
    CLINICAL_BLEED_GRADE_RULES = [
        (1.00, {"F2", "F7", "F9", "F10", "F11", "VKORC1", "PROC", "PROS1",
                "SERPINC1", "SERPIND1", "PLG"}),
        (0.95, {"P2RY12", "ITGB3", "ITGA2B", "GP1BA", "GP6", "TBXA2R"}),
        (0.80, {"PTGS1"}),
        (0.50, {"PTGS2"}),
    ]
    CLINICAL_BLEED_GENES = set().union(*[gs for _, gs in CLINICAL_BLEED_GRADE_RULES])

    def _clinical_grade(s):
        g = 0.0
        for w, gs in CLINICAL_BLEED_GRADE_RULES:
            if s & gs:
                g = max(g, w)
        return g

    df["clinical_bleed"] = tg.apply(_clinical_grade)
    df["has_bleed"] = ((df["has_bleed"] > 0) |
                       (df["clinical_bleed"] > 0)).astype(float)
    # evidence = stronger of (direction-aware MR) and (clinical grade)
    df["bleed_evidence"] = np.maximum(
        np.where(df["bleed_penalty"] > 0, df["bleed_evidence"], 0.0),
        df["clinical_bleed"])
    df["bleed_genes"] = df.apply(
        lambda r: ",".join(sorted((set(str(r["bleed_genes"]).split(",")) - {""})
                                  | (set(r["targets"].split(",")) & CLINICAL_BLEED_GENES))),
        axis=1)
    df["bleed_dir"] = df.apply(
        lambda r: ",".join(sorted(
            ["%s(%s)" % (g, bleed_dir.get(g) or "clinical")
             for g in sorted((set(str(r["bleed_genes"]).split(",")) - {""}))])),
        axis=1)
    df["bleed_class"] = df.apply(
        lambda r: "+".join(sorted(set(
            [cl for cl, gs in BLEED_CLASS_GENES.items()
             if (set(str(r["targets"]).split(",")) & gs)]))), axis=1).replace("", "-")
    # v4.3: flag an anticoagulant whose MR bleeding direction CONTRADICTS the
    # pharmacology (e.g. F10 = 'risk' => genetics says inhibiting F10 LOWERS
    # bleeding, which is the opposite of the DOAC/warfarin expectation).  For
    # these drugs the clinical grade governs and the conflict is reported.
    _CLIN_ANTICOAG = {"F2", "F7", "F9", "F10", "F11", "VKORC1", "PROC",
                      "PROS1", "SERPINC1", "SERPIND1"}

    def _dir_conflict(s):
        if not (s & _CLIN_ANTICOAG):
            return 0.0
        return 1.0 if any(bleed_dir.get(g) == "risk"
                          for g in sorted(s & bleed)) else 0.0

    df["bleed_dir_conflict"] = tg.apply(_dir_conflict)
    _nc = int((df["bleed_dir_conflict"] > 0).sum())
    if _nc:
        log("  [flag] %d anticoagulant drug(s): MR bleeding direction CONTRADICTS "
            "pharmacology -> clinical grade used, e.g. %s", _nc,
            ", ".join(df[df["bleed_dir_conflict"] > 0]["drug"].astype(str).head(6)))

    # ---- CNS + FAERS ARIA flags --------------------------------------------
    cns_map, faers_map = {}, {}
    # Build CNS map from BOTH the original cand file (has is_cns) AND the expanded pool.
    # Expanded-pool drugs are CNS-UNKNOWN (0) because OpenTargets does not provide BBB labels
    # and we cannot make a CNS call in either direction.
    for src_name, src_df, src_label in [
        ("candidate_drugs.csv", d["cand_orig"], "known"),
        ("candidate pool", d["cand"], "candidate"),
    ]:
        if src_df is None:
            continue
        cn = pick_col(src_df, CAND_NAME_COLS, "name", src_name, warn=False)
        cc = pick_col(src_df, CAND_CNS_COLS, "is_cns", src_name, warn=False)
        if cn and cc:
            for n_, v in zip(src_df[cn].astype(str), numeric(src_df[cc])):
                val = 0 if pd.isna(v) else int(v > 0)
                cns_map[n_.upper()] = val
    # Also pull is_cns directly from the inline _cand_cns column (expanded-pool rows)
    for _, r in df.iterrows():
        cv = r.get("_cand_cns")
        if pd.notna(cv):
            try:
                cns_map[str(r["drug"]).upper()] = int(float(cv) > 0)
            except Exception:
                pass
    if d["faers"] is not None:
        dcol = pick_col(d["faers"], SCORED_NAME_COLS, "drug",
                        "faers_pt_level_signals.csv", warn=False)
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
    # 3-class CNS label (transparent in tables) + pool provenance
    df["cns_label"] = np.where(
        df["_pool"].astype(str) == "gnn",
        np.where(df["is_cns"] > 0, "CNS-known", "non-CNS"),
        "CNS-unknown")
    df["pool"] = df["_pool"].astype(str).replace({"gnn": "HGT_scored", "expanded": "OT_expanded"})

    # =======================================================================
    # 5. three task-specific, direction-aware composite scores
    # =======================================================================
    # CNS bonus uses 3-class label: CNS-known=1.0, CNS-unknown=0.4, non-CNS=0.0
    cns_bonus = np.where(df["cns_label"]=="CNS-known", 1.0,
                         np.where(df["cns_label"]=="CNS-unknown", 0.4, 0.0))
    df["_pool"] = df["pool"] if "pool" in df.columns else "gnn"
    df["f_repurpose"] = (
        0.40 * pct_rank_pool(df["gnn_repurpose_score"], df["_pool"]) +
        0.25 * pct_rank_pool(df["gnn_synergy_score"], df["_pool"]) +
        0.25 * df["ad_eff_evidence"] +
        0.10 * cns_bonus
    )
    df["cns_bonus"] = cns_bonus
    # v4.3: bleed-first, pool-comparable liability
    #   0.60 * clinical pharmacology grade   (dominant, pool-independent)
    # + 0.20 * direction-aware MR evidence   (corroboration)
    # + 0.20 * pool-stratified GNN ARIA rank (structural, neutral if absent)
    df["gnn_aria_pct"] = pct_rank_pool(df["max_aria_risk"], df["pool"])
    df["liability"] = (
        0.60 * df["clinical_bleed"] +
        0.20 * df["bleed_penalty"].clip(0, 1) +
        0.20 * df["gnn_aria_pct"]
    )
    df["liability_tier"] = np.where(
        df["liability"] >= 0.60, "high",
        np.where(df["liability"] >= 0.35, "moderate", "low"))

    # "active protection" gate: a combination partner must bring a POSITIVE
    # mechanism (a synergy route, an AD-anchored target, or aligned AD evidence).
    df["has_active_route"] = (
        (df["gnn_synergy_score"].fillna(0) > 0) |
        (df["n_synergy_pathways"].fillna(0) > 0) |
        (df["n_prior_targets"].fillna(0) > 0) |
        (df["ad_eff_evidence"] > 0)
    ).astype(float)
    df["passive_only"] = 1.0 - df["has_active_route"]
    df["bleed_excluded"] = (df["has_bleed"] > 0).astype(float)
    # a partner must NOT be directionally opposed at the AD level either
    df["counter_only"] = ((df["ad_eff_counter"] > 0) &
                          (df["ad_eff_aligned"] == 0)).astype(float)
    df["partner_eligible"] = ((df["has_active_route"] > 0) &
                              (df["has_bleed"] == 0) &
                              (df["counter_only"] == 0)).astype(float)
    df["f_partner"] = df["partner_eligible"] * (
        0.50 * pct_rank_pool(df["gnn_synergy_score"], df["_pool"]) +
        0.30 * (1.0 - df["liability"]) +
        0.20 * df["is_cns"]
    )

    # ---- Task A evidence tier ---------------------------------------------
    df["evidence_tier"] = np.where(
        df["ad_eff_aligned"] > 0, "genetic-anchored",
        np.where(df["ad_eff_counter"] > 0, "genetic-counter",
                 np.where(df["n_prior_targets"] > 0, "target-prior",
                          "structural-only")))

    df["rank_repurpose"] = df["f_repurpose"].rank(ascending=False, method="min").astype(int)
    df["rank_liability"] = df["liability"].rank(ascending=False, method="min").astype(int)
    df["rank_partner"] = df["f_partner"].rank(ascending=False, method="min").astype(int)

    # =======================================================================
    # 6. COVERAGE AUDIT - which stratum gene can NO candidate drug reach?
    # =======================================================================
    ids = set(df["drug_id"])
    name_of = dict(zip(df["drug_id"], df["drug"]))
    # a drug-id -> name map covering the WHOLE KG (needed to name drugs that
    # exist in the KG but were never scored, e.g. perindopril)
    gid2name = {}
    if d["nodes"] is not None:
        _nid = pick_col(d["nodes"], ["node_id", "id", "nodeid"], "node_id",
                        "nodes.parquet", warn=False)
        _nnm = pick_col(d["nodes"], ["name", "node_name", "label"], "name",
                        "nodes.parquet", warn=False)
        if _nid is not None and _nnm is not None:
            gid2name = dict(zip(d["nodes"][_nid].astype(str),
                                d["nodes"][_nnm].astype(str)))

    # v4.3 FIX: the audit must use the SAME per-drug target set as the scoring
    # pipeline, i.e. KG edges UNION the plan-B "_ad_targets" column.  v4.2
    # audited d2g alone, so every plan-B drug was invisible here and the report
    # claimed "ACE unreachable" while Task A was simultaneously ranking 22 ACEIs.
    kg_covered = defaultdict(set)          # gene -> {drug_id}, KG + plan-B
    for did, genes in d2g.items():
        for g in genes:
            kg_covered[g].add(did)
    covered = defaultdict(set)             # gene -> {drug_id} over the SCORED pool
    ids = set(df["drug_id"].astype(str))
    for did, genes in zip(df["drug_id"].astype(str), df["_tgt_set"]):
        for g in genes:
            kg_covered[g].add(did)
            covered[g].add(did)

    def _audit(genes_set):
        hit, miss = {}, []
        for g in sorted(genes_set):
            ds = sorted({name_of[i] for i in covered.get(g, set())})
            if ds:
                hit[g] = ds
            else:
                miss.append(g)
        return hit, miss

    def _kg_audit(genes_set):
        """Per gene: who can reach it in the KG, marking unscored drugs."""
        out = {}
        for g in sorted(genes_set):
            names = []
            for did in sorted(kg_covered.get(g, set())):
                nm = name_of.get(did) or gid2name.get(did) or did
                names.append(nm if did in ids else "%s[unscored]" % nm)
            if names:
                out[g] = sorted(set(names))
        return out

    ad_cov, ad_miss = _audit(ad_eff)
    bl_cov, bl_miss = _audit(bleed)
    ad_kg, bl_kg = _kg_audit(ad_eff), _kg_audit(bleed)
    audit = {
        "ad_efficacy_covered": ad_cov, "ad_efficacy_uncovered": ad_miss,
        "bleeding_covered": bl_cov, "bleeding_uncovered": bl_miss,
        "ad_efficacy_kg_reachable": ad_kg, "bleeding_kg_reachable": bl_kg,
    }
    if ad_miss:
        log("[audit] AD-efficacy genes NO SCORED candidate drug can reach: %s",
            ", ".join(ad_miss))
    if ad_cov:
        log("[audit] AD-efficacy genes reachable: %s",
            "; ".join("%s<-%s" % (g, "/".join(v[:4])) for g, v in sorted(ad_cov.items())))
    if bl_miss:
        log("[audit] bleeding genes NO SCORED candidate drug can reach: %s",
            ", ".join(bl_miss))
    # split the two very different reasons a gene can be "uncovered"
    for label, genes, kg, miss in (("AD-efficacy", ad_eff, ad_kg, ad_miss),
                                   ("bleeding", bleed, bl_kg, bl_miss)):
        for g in miss:
            if g in kg:
                log("[audit] %s gene %s: present in KG but reachable ONLY via "
                    "UNSCORED drug(s): %s  -> expand the candidate pool "
                    "(plan B) to bring it into the ranking", label, g,
                    "/".join(kg[g][:4]))
            else:
                log("[audit] %s gene %s: NO drug in the KG targets it at all "
                    "(data-side gap, not a scoring bug)", label, g)

    return df, ad_eff, bleed, control, bleed_dir, ad_dir, gdiag, audit


# ===========================================================================
# 7. write tables + console report
# ===========================================================================
def write_outputs(proj, df, ad_eff, bleed, control, bleed_dir, ad_dir, gdiag,
                  audit, args):
    tdir = proj / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    common = ["drug_id", "drug", "pool", "cns_label", "is_cns", "targets", "n_tg", "gene_prior",
              "n_prior_targets", "ad_eff_genes", "ad_eff_dir", "ad_eff_evidence",
              "ad_eff_aligned", "ad_eff_counter", "bleed_genes", "bleed_dir",
              "bleed_class", "clinical_bleed", "bleed_dir_conflict", "ctrl_genes",
              "gnn_repurpose_score", "gnn_synergy_score", "gnn_mitigation_score",
              "max_aria_risk", "faers_aria_signal"]
    common = [c for c in common if c in df.columns]

    def _s(v):
        return v if (isinstance(v, str) and v) else "-"

    # ---- Task A -----------------------------------------------------------
    anchored = df[df["evidence_tier"] != "structural-only"].copy()
    anchored = anchored.sort_values(
        ["ad_eff_aligned", "n_prior_targets", "f_repurpose"],
        ascending=[False, False, False]).reset_index(drop=True)
    structural = df[df["evidence_tier"] == "structural-only"].copy()
    structural = structural.sort_values("f_repurpose", ascending=False).reset_index(drop=True)

    anchored[common + ["evidence_tier", "f_repurpose", "rank_repurpose"]].to_csv(
        tdir / "repositioning_top_v4.csv", index=False)
    structural[common + ["evidence_tier", "f_repurpose", "rank_repurpose"]].to_csv(
        tdir / "repositioning_structural_v4.csv", index=False)

    # ---- Task B -----------------------------------------------------------
    b = df.sort_values(["liability", "clinical_bleed", "has_bleed", "n_tg", "drug"],
                       ascending=[False, False, False, False, True]
                       ).reset_index(drop=True)
    b[common + ["liability", "bleed_evidence", "has_bleed",
                "liability_tier", "rank_liability"]].to_csv(
        tdir / "comedication_risk_v4.csv", index=False)

    # ---- Task C -----------------------------------------------------------
    c = df[df["partner_eligible"] > 0].sort_values(
        "f_partner", ascending=False).reset_index(drop=True)
    c[common + ["f_partner", "liability", "has_active_route", "partner_eligible",
                "bleed_excluded", "counter_only", "passive_only", "rank_partner"]].to_csv(
        tdir / "combination_partners_v4.csv", index=False)

    # Tier-2: drugs that are safe (no bleeding liability, not AD-direction
    # opposed) and carry at least one KG target, but whose mechanism did not
    # light up a synergy route.  Kept SEPARATE from the eligible main table so
    # the strict ranking is not diluted, yet still visible for discussion.
    tier2 = df[(df["partner_eligible"] == 0) &
               (df["bleed_excluded"] == 0) &
               (df["counter_only"] == 0) &
               (df["n_tg"] > 0)].copy()
    if len(tier2):
        tier2["f_tier2"] = (0.45 * pct_rank_pool(tier2["gnn_synergy_score"], tier2["pool"]) +
                            0.35 * (1.0 - tier2["liability"]) +
                            0.20 * pct_rank_pool(tier2["n_tg"], tier2["pool"]))
    else:
        tier2["f_tier2"] = pd.Series(dtype=float)
    tier2 = tier2.sort_values(["f_tier2", "liability", "drug"],
                              ascending=[False, True, True]).reset_index(drop=True)
    tier2[common + ["f_tier2", "liability", "n_tg", "has_active_route"]].to_csv(
        tdir / "combination_partners_tier2_v4.csv", index=False)

    df.sort_values("f_repurpose", ascending=False).to_csv(
        tdir / "fusion_evidence_v4.csv", index=False)

    # ---- console ----------------------------------------------------------
    log("--- Task A  mechanistically anchored candidates (%d) ---", len(anchored))
    for i, (_, r) in enumerate(anchored.iterrows(), 1):
        log("  #%-3d %-24s tier=%-16s ad_dir=%-26s prior=%-18s",
            i, str(r["drug"])[:24], str(r["evidence_tier"]),
            _s(r["ad_eff_dir"])[:26], _s(r["gene_prior"])[:18])
    nct = int((anchored["evidence_tier"] == "genetic-counter").sum())
    if nct:
        log("  [note] %d genetic-counter drug(s): they engage an AD gene but the MR "
            "direction implies inhibiting it is UNFAVOURABLE -> not repurposing leads",
            nct)
    k = int(args.topk)
    log("--- Task A  structural-only top %d (hypothesis-generating, NO genetic anchor) ---", k)
    for i, (_, r) in enumerate(structural.head(k).iterrows(), 1):
        log("  #%-3d %-26s f_repurp=%.3f  targets=%s",
            i, str(r["drug"])[:26], r["f_repurpose"], _s(r["targets"])[:34])

    log("--- Task B  TOP %d ARIA-H co-medication LIABILITY (avoid) ---", k)
    for _, r in b.head(k).iterrows():
        log("  #%-3d %-28s liability=%.3f tier=%-8s class=%-20s bleed=%-12s pool=%-11s cns=%s",
            int(r["rank_liability"]), str(r["drug"])[:28], r["liability"],
            str(r["liability_tier"]), _s(r["bleed_class"])[:20],
            _s(r["bleed_genes"])[:12], str(r.get("pool","-"))[:11],
            str(r.get("cns_label","-"))[:11])

    log("--- Task C  TOP %d eligible combination partners ---", k)
    for _, r in c.head(k).iterrows():
        log("  #%-3d %-24s f_partner=%.3f  liability=%.3f  CNS=%d  ad_dir=%s",
            int(r["rank_partner"]), str(r["drug"])[:24], r["f_partner"],
            r["liability"], int(r["is_cns"]), _s(r["ad_eff_dir"])[:18])
    ndem = int(((df["passive_only"] > 0) & (df["bleed_excluded"] == 0)).sum())
    if ndem:
        log("  [note] %d route-less drug(s) excluded from the Task C ranking "
            "(kept in fusion_evidence_v4.csv)", ndem)
    if len(tier2):
        log("  [note] %d safe-but-mechanism-unmatched drug(s) listed in "
            "combination_partners_tier2_v4.csv (top: %s)", len(tier2),
            ", ".join(tier2["drug"].astype(str).head(5)))
    nbx = int((df["bleed_excluded"] > 0).sum())
    if nbx:
        log("  [note] %d drug(s) EXCLUDED from partners: bleeding liability, e.g. %s",
            nbx, ", ".join(df[df["bleed_excluded"] > 0]["drug"].astype(str).head(8)))
    ncx = int((df["counter_only"] > 0).sum())
    if ncx:
        log("  [note] %d drug(s) EXCLUDED from partners: AD-direction opposed (counter-only), e.g. %s",
            ncx, ", ".join(df[df["counter_only"] > 0]["drug"].astype(str).head(8)))

    summary = {
        "version": VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_novel_candidates": int(len(df)),
        "outcome_strata": {
            "AD_efficacy": sorted(AD_OUTCOMES),
            "AD_efficacy_powered": sorted(AD_STRONG_OUTCOMES),
            "bleeding": sorted(BLEED_OUTCOMES),
            "control_not_scored": sorted(CONTROL_OUTCOMES),
        },
        "genetic_strata": {
            "ad_efficacy_genes": sorted(ad_eff),
            "ad_efficacy_direction": gdiag["ad_direction_by_gene"],
            "bleeding_genes": sorted(bleed),
            "bleeding_gene_direction": gdiag["bleed_direction_by_gene"],
            "control_genes": sorted(control),
        },
        "genetic_strata_rules": {
            "ad_efficacy": "powered AD outcome (ad / ad_repl1) OR >= %d AD outcomes"
                           % args.min_ad_outcomes,
            "bleeding": "nominal-significant bleeding outcome (MR p<%.3g) or coloc "
                        "PP.H4>=%.2f; pleiotropy (Egger p<%.3g) rows are %s"
                        % (args.ad_pval, args.coloc_thr, args.pleio_pval,
                           "DISCARDED" if args.strict_pleiotropy
                           else "FLAGGED but retained"),
            "direction_assumption": "a drug's net engagement of its target is "
                                    "INHIBITORY (documented agonists are manual exceptions)",
            "manual_drop_ad": sorted(gdiag["manual_drop_ad"]),
            "manual_drop_bleed": sorted(gdiag["manual_drop_bleed"]),
        },
        "genetic_audit": {
            "pleiotropy_column": gdiag["pleiotropy_column"],
            "pleiotropy_mode": gdiag.get("pleiotropy_mode"),
            "pleiotropy_flagged_rows": sorted(gdiag["pleiotropy_flagged_rows"]),
            "pleiotropy_dropped_rows": sorted(gdiag["pleiotropy_dropped_rows"]),
            "ad_repl2_only_dropped": sorted(gdiag["ad_repl2_only_dropped"]),
            "ad_outcomes_by_gene": gdiag["ad_outcomes_by_gene"],
            "bleed_outcomes_by_gene": gdiag["bleed_outcomes_by_gene"],
        },
        "coverage_audit": audit,
        "score_definition": {
            "f_repurpose": "0.40*pct_pool(gnn_repurpose)+0.25*pct_pool(gnn_synergy)"
                           "+0.25*ad_eff_evidence+0.10*CNS",
            "liability": "0.60*clinical_bleed_grade + 0.20*bleed_penalty_MR "
                         "+ 0.20*pct_pool(max_aria_risk)",
            "clinical_bleed_grade": "graded pharmacology (pool-independent): "
                                    "anticoagulant 1.00, antiplatelet receptor 0.95, "
                                    "non-selective NSAID (PTGS1) 0.80, COX-2 selective (PTGS2) 0.50",
            "pct_pool": "percentile rank computed WITHIN each pool stratum "
                        "(HGT_scored vs OT_expanded); missing value -> neutral 0.5",
            "ad_eff_evidence": "clip(sum over targeted AD genes of "
                               "{risk:1.0, protective:0.0, unknown:0.5}, 0, 1)",
            "bleed_evidence": "max( clinical_bleed_grade , clip(sum over targeted bleeding "
                              "genes of {protective:1.0, risk:0.0, unknown:0.5}, 0, 1) )",
            "f_partner": "partner_eligible * [0.50*pct_pool(gnn_synergy)"
                         "+0.30*(1-liability)+0.20*CNS]",
            "evidence_tier": "genetic-anchored (aligned) > genetic-counter > "
                             "target-prior (n_prior_targets>0) > structural-only",
        },
        "counts": {
            "with_AD_efficacy_gene": int((df["has_ad_eff"] > 0).sum()),
            "with_AD_aligned_evidence": int((df["ad_eff_aligned"] > 0).sum()),
            "with_AD_counter_evidence": int((df["ad_eff_counter"] > 0).sum()),
            "with_AD_prior_target": int((df["n_prior_targets"] > 0).sum()),
            "with_bleeding_liability": int((df["has_bleed"] > 0).sum()),
            "CNS_penetrant": int((df["is_cns"] > 0).sum()),
            "taskA_anchored": int(len(anchored)),
            "taskA_structural": int(len(structural)),
            "taskC_eligible": int(len(c)),
            "taskC_tier2": int(len(tier2)),
            "passive_only_demoted": ndem,
            "bleed_excluded_from_partners": nbx,
            "counter_excluded_from_partners": ncx,
            "from_HGT_pool": int((df["pool"]=="HGT_scored").sum()),
            "from_OT_expanded": int((df["pool"]=="OT_expanded").sum()),
            "CNS_known": int((df["cns_label"]=="CNS-known").sum()),
            "CNS_unknown": int((df["cns_label"]=="CNS-unknown").sum()),
            "non_CNS": int((df["cns_label"]=="non-CNS").sum()),
            "clinical_bleed_forced": int((df["clinical_bleed"]>0).sum()),
        },
    }
    with open(tdir / "fusion_v4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for f in ("repositioning_top_v4.csv", "repositioning_structural_v4.csv",
              "comedication_risk_v4.csv", "combination_partners_v4.csv",
              "combination_partners_tier2_v4.csv",
              "fusion_evidence_v4.csv", "fusion_v4_summary.json"):
        log("[write] %s", str(tdir / f))
    return summary


# ===========================================================================
# 8. CLI
# ===========================================================================
def _csv_list(s):
    return [x.strip().upper() for x in str(s).split(",") if x.strip()]


def parse_args(argv):
    p = argparse.ArgumentParser(description="chapter-6 direction-aware fusion (v4)")
    p.add_argument("--proj", default=".", help="project root")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--ad-pval", type=float, default=0.05,
                   help="nominal p for MR genetic evidence")
    p.add_argument("--coloc-thr", type=float, default=0.80,
                   help="PP.H4 threshold for coloc evidence")
    p.add_argument("--pleio-pval", type=float, default=0.05,
                   help="Egger/pleiotropy p below which an MR row is FLAGGED; it "
                        "is discarded only when --strict-pleiotropy is set")
    p.add_argument("--strict-pleiotropy", action="store_true",
                   help="discard pleiotropy-flagged MR rows (sensitivity analysis); "
                        "default keeps them and flags them instead")
    p.add_argument("--min-ad-outcomes", type=int, default=2,
                   help="min AD outcomes needed when no powered outcome supports a gene")
    p.add_argument("--no-powered-ad-rule", action="store_true",
                   help="disable the powered-outcome rule for AD-efficacy genes")
    p.add_argument("--drop-ad-genes", type=_csv_list, default=[],
                   help="comma-separated genes forced out of the AD-efficacy set")
    p.add_argument("--drop-bleed-genes", type=_csv_list, default=[],
                   help="comma-separated genes forced out of the bleeding set")
    p.add_argument("--cand-pool", default="candidate_drugs_expanded.csv",
                   help="candidate pool file (default: plan-B expanded pool)")
    p.add_argument("--force-clinical-bleed", action="store_true", default=True,
                   help="mark anticoagulants/antiplatelets/NSAIDs as high-liability "
                        "even if the MR direction logic would not (clinical prior)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    proj = find_proj_root(args.proj)
    log("project root: %s", proj)
    log("=== STAGE fusion v4.3 (pool-comparable + clinical-bleed-first) ===")
    df, ad_eff, bleed, control, bleed_dir, ad_dir, gdiag, audit = build_evidence(proj, args)
    summary = write_outputs(proj, df, ad_eff, bleed, control, bleed_dir, ad_dir,
                            gdiag, audit, args)
    log("[summary] novel=%d (HGT=%d + OT_exp=%d) | anchored=%d | AD-aligned=%d | AD-counter=%d | "
        "bleeding-liability=%d | partner-eligible=%d | partner-tier2=%d | CNS=%d(known %d / unknown %d / non %d)",
        summary["n_novel_candidates"],
        summary["counts"]["from_HGT_pool"], summary["counts"]["from_OT_expanded"],
        summary["counts"]["taskA_anchored"],
        summary["counts"]["with_AD_aligned_evidence"],
        summary["counts"]["with_AD_counter_evidence"],
        summary["counts"]["with_bleeding_liability"],
        summary["counts"]["taskC_eligible"],
        summary["counts"]["taskC_tier2"],
        summary["counts"]["CNS_penetrant"],
        summary["counts"]["CNS_known"], summary["counts"]["CNS_unknown"],
        summary["counts"]["non_CNS"])
    log("done.")


if __name__ == "__main__":
    main()
