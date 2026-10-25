#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08c_tables.py  --  chapter-6 publication-ready table builder (POST-PROCESSING ONLY)

This script does NOT touch the fusion logic.  It reads the v4 fusion outputs
and turns them into de-duplicated, clinically stratified, paper-ready tables:

    Table_S1_repositioning.csv           Task A - repositioning candidates
    Table_S2_comedication_liability.csv  Task B - ARIA-H co-medication liability
    Table_S3_combination_partners.csv    Task C - combination partners
    dedup_and_class_audit.csv            every salt-collapse decision (audit trail)
    table_build_summary.json             machine-readable build summary

Processing steps
  1. collapse salt / ester / hydrate / isomer duplicates onto the parent
     active moiety, so the paper lists one row per active substance
     (WARFARIN SODIUM -> WARFARIN, DICLOFENAC POTASSIUM -> DICLOFENAC, ...)
  2. attach a clinical-significance class so that topical / luminal /
     obsolete agents are labelled as such instead of silently competing with
     systemically acting drugs
  3. neutralise the MR term for drugs whose MR bleeding direction CONTRADICTS
     pharmacology (bleed_dir_conflict == 1): the lifetime-genetic-exposure
     estimate is not equivalent to acute pharmacological inhibition, so the
     clinical grade is allowed to lead.  Recorded as liability_adj.
  4. write the tables plus a full audit log

Reads (under <proj>/results/tables/):
    comedication_risk_v4.csv              fusion_evidence_v4.csv
    combination_partners_v4.csv           combination_partners_tier2_v4.csv
    repositioning_top_v4.csv              repositioning_structural_v4.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# salt / ester / hydrate suffixes that denote the SAME active moiety
# --------------------------------------------------------------------------- #
SALT_SUFFIXES = [
    "SODIUM", "DISODIUM", "POTASSIUM", "DIPOTASSIUM", "CALCIUM", "MAGNESIUM",
    "HYDROCHLORIDE", "DIHYDROCHLORIDE", "HYDROBROMIDE", "MALEATE", "MESYLATE",
    "MESILATE", "TOSYLATE", "TOSILATE", "BESYLATE", "BESILATE", "CITRATE",
    "TARTRATE", "BITARTRATE", "FUMARATE", "SUCCINATE", "MALATE", "OXALATE",
    "ACETATE", "SULFATE", "BISULFATE", "PHOSPHATE", "LACTATE", "GLUCONATE",
    "ARGININE", "ERBUMINE", "LYSINE", "TROMETHAMINE", "EPOLAMINE",
    "DIETHYLAMINE", "OLAMINE", "ANHYDROUS", "HYDRATE", "MONOHYDRATE",
    "DIHYDRATE", "TRIHYDRATE", "HEMIHYDRATE",
]

# --------------------------------------------------------------------------- #
# clinical significance
# --------------------------------------------------------------------------- #
# Agents that do NOT reach the systemic circulation in a meaningful way.
# Their bleeding liability for a patient on an anti-amyloid antibody is not
# the same as that of the oral / systemic form.
TOPICAL_OR_LUMINAL = {
    "BISMUTH SUBSALICYLATE",   # oral bismuth salt; acts in the gut lumen
    "BALSALAZIDE",             # colonic-release, acts locally in the gut
    "BENZYDAMINE",             # topical / mouthwash NSAID
    "IBUFENAC",                # topical NSAID
    "ESFLURBIPROFEN",          # topical NSAID
    "BROMFENAC",               # ophthalmic NSAID
    "GLAFENINE",               # now mainly topical
    "DICLOFENAC EPOLAMINE",    # topical formulation
    "KETOROLAC",               # frequently ophthalmic / topical
}
# Withdrawn, obsolete or of marginal contemporary use.
OBSOLETE_OR_MARGINAL = {
    "DICUMAROL", "PHENINDIONE", "DIPYRONE", "METAMIZOLE", "APAZONE",
    "AZAPROPRAZONE", "FENCLOFENAC", "ISOXICAM", "CLOMETACIN", "INDOPROFEN",
    "LOXOPROFEN", "MECLOFENAMATE",
}
# Widely used systemic agents - the ones that matter for a clinical warning.
FIRST_LINE_SYSTEMIC = {
    "WARFARIN", "ACENOCOUMAROL", "PHENPROCOUMON", "FLUINDIONE",
    "ASPIRIN", "IBUPROFEN", "NAPROXEN", "DICLOFENAC", "INDOMETHACIN",
    "CELECOXIB", "ETORICOXIB", "KETOPROFEN", "MELOXICAM", "PIROXICAM",
    "DEXIBUPROFEN", "DEXKETOPROFEN", "NABUMETONE", "TENOXICAM",
    "CLOPIDOGREL", "TICAGRELOR", "PRASUGREL", "CANGRELOR", "TICLOPIDINE",
    "DABIGATRAN", "APIXABAN", "RIVAROXABAN", "EDOXABAN", "BETRIXABAN",
    "NADROPARIN", "ENOXAPARIN", "DALTEPARIN", "TINZAPARIN", "FONDAPARINUX",
    "ACETAMINOPHEN",
}

# free-text notes surfaced as an extra column in the paper tables
MANUAL_NOTES = {
    "ACETAMINOPHEN": "known false positive: only weak antiplatelet activity; entered via PTGS1 target",
    "CELECOXIB": "COX-2 selective; lower GI bleeding risk than non-selective NSAIDs",
    "BISMUTH SUBSALICYLATE": "gut-lumen action, negligible systemic absorption",
    "BALSALAZIDE": "colonic-release, acts locally in the gut",
    "BENZYDAMINE": "topical / mouthwash route only",
    "BROMFENAC": "ophthalmic route only",
    "GLAFENINE": "now mainly topical",
    "IBUFENAC": "topical route only",
    "ESFLURBIPROFEN": "topical route only",
    "KETOROLAC": "frequently ophthalmic / topical",
    "DICUMAROL": "withdrawn / obsolete",
    "PHENINDIONE": "obsolete",
    "DIPYRONE": "withdrawn in many countries (agranulocytosis)",
    "APAZONE": "withdrawn",
    "FENCLOFENAC": "withdrawn (hepatotoxicity)",
}


# order in which the clinical groups are laid out in the paper table
GROUP_ORDER = [
    "anticoagulant", "antiplatelet", "nsaid_nonselective", "nsaid_cox2",
    "other_bleeding", "topical_or_luminal", "obsolete_or_marginal",
    "non_bleeding",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def clinical_group(r) -> str:
    """Coarse pharmacological group, used to lay the paper table out the way a
    clinician reads it (anticoagulants first, then antiplatelets, then NSAIDs),
    instead of letting a single blended score interleave them."""
    moi = str(r.get("active_moiety", "")).strip().upper()
    if moi in TOPICAL_OR_LUMINAL:
        return "topical_or_luminal"
    if moi in OBSOLETE_OR_MARGINAL:
        return "obsolete_or_marginal"
    cls = str(r.get("bleed_class", "")).strip().lower()
    tg = str(r.get("targets", ""))
    if cls == "anticoagulant":
        return "anticoagulant"
    if cls == "antiplatelet_nsaid":
        if ("P2RY12" in tg) or ("ITGB3" in tg):
            return "antiplatelet"
        if "PTGS1" in tg:
            return "nsaid_nonselective"
        if "PTGS2" in tg:
            return "nsaid_cox2"
        return "other_bleeding"
    if cls in ("", "-", "nan", "none"):
        return "non_bleeding"
    return "other_bleeding"


# --------------------------------------------------------------------------- #
def read_csv(p: Path):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        log(f"[warn] cannot read {p.name}: {e}")
        return None


def active_moiety(name) -> str:
    """Strip salt / ester / hydrate suffixes to recover the active moiety."""
    s = re.sub(r"\s+", " ", str(name).strip().upper())
    changed = True
    while changed:
        changed = False
        for suf in SALT_SUFFIXES:
            if s.endswith(" " + suf):
                s = s[: -(len(suf) + 1)].strip()
                changed = True
                break
    return s


def classify(moiety: str) -> str:
    m = str(moiety).strip().upper()
    if m in TOPICAL_OR_LUMINAL:
        return "topical_or_luminal"
    if m in OBSOLETE_OR_MARGINAL:
        return "obsolete_or_marginal"
    if m in FIRST_LINE_SYSTEMIC:
        return "systemic_first_line"
    return "systemic_other"


def pick_rep(g: pd.DataFrame, score_col: str) -> pd.Series:
    """Choose the representative row of a salt group: prefer the HGT-scored
    entry that actually carries a GNN score, then the highest score, then the
    shortest / alphabetically first name."""
    idx = g.index
    hgt = (g["pool"].astype(str) == "HGT_scored").astype(int) if "pool" in g.columns else pd.Series(0, index=idx)
    hasgnn = (pd.to_numeric(g["gnn_repurpose_score"], errors="coerce").notna().astype(int)
              if "gnn_repurpose_score" in g.columns else pd.Series(0, index=idx))
    score = (pd.to_numeric(g[score_col], errors="coerce").fillna(-1.0)
             if score_col in g.columns else pd.Series(-1.0, index=idx))
    nlen = g["drug"].astype(str).str.len()
    key = pd.DataFrame({
        "_hgt": hgt, "_hasgnn": hasgnn, "_score": score,
        "_nlen": nlen, "_name": g["drug"].astype(str),
    })
    key = key.sort_values(["_hgt", "_hasgnn", "_score", "_nlen", "_name"],
                          ascending=[False, False, False, True, True])
    return g.loc[key.index[0]]


def collapse(df: pd.DataFrame, score_col: str, table: str, log_rows: list) -> pd.DataFrame:
    df = df.copy()
    df["active_moiety"] = df["drug"].map(active_moiety)
    reps = []
    for moi, g in df.groupby("active_moiety", sort=False):
        rep = pick_rep(g, score_col)
        names = sorted(set(g["drug"].astype(str)))
        dropped = [x for x in names if x != str(rep["drug"])]
        if dropped:
            log_rows.append({
                "table": table,
                "active_moiety": moi,
                "kept_representative": rep["drug"],
                "dropped_duplicates": "; ".join(dropped),
                "n_rows_collapsed": int(len(g)),
            })
        reps.append(rep)
    return pd.DataFrame(reps).reset_index(drop=True)


def add_liability_adj(df: pd.DataFrame, gnn_pct: dict) -> pd.DataFrame:
    """Neutralise the MR term (set to 0.5) for drugs whose MR bleeding
    direction contradicts pharmacology; never lower an already-high clinical
    grade."""
    df = df.copy()
    liab = (pd.to_numeric(df["liability"], errors="coerce")
            if "liability" in df.columns else pd.Series(np.nan, index=df.index))
    clin = (pd.to_numeric(df["clinical_bleed"], errors="coerce")
            if "clinical_bleed" in df.columns else pd.Series(np.nan, index=df.index))
    flag = (pd.to_numeric(df["bleed_dir_conflict"], errors="coerce").fillna(0)
            if "bleed_dir_conflict" in df.columns else pd.Series(0, index=df.index))
    if "gnn_aria_pct" not in df.columns:
        if "drug_id" in df.columns and gnn_pct:
            df["gnn_aria_pct"] = pd.to_numeric(df["drug_id"].map(gnn_pct), errors="coerce")
        else:
            df["gnn_aria_pct"] = np.nan
    gp = pd.to_numeric(df["gnn_aria_pct"], errors="coerce")

    df["liability_adj"] = liab
    df["mr_term_neutralised"] = 0
    mask = (flag > 0) & clin.notna()
    if mask.any():
        neutral = 0.60 * clin[mask] + 0.20 * 0.50 + 0.20 * gp[mask].fillna(0.50)
        df.loc[mask, "liability_adj"] = np.maximum(
            liab[mask].fillna(0.0).to_numpy(), neutral.to_numpy())
        df.loc[mask, "mr_term_neutralised"] = 1
    return df


def _sort(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    asc = [c == "active_moiety" for c in cols]
    return df.sort_values(cols, ascending=asc).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="chapter-6 paper-ready table builder")
    ap.add_argument("--proj", default=".")
    ap.add_argument("--topk", type=int, default=40, help="rows printed per key-drug block")
    args = ap.parse_args(argv)

    proj = Path(args.proj).resolve()
    T = proj / "results" / "tables"
    OUT = proj / "results" / "paper_tables"
    OUT.mkdir(parents=True, exist_ok=True)

    log(f"project root: {proj}")
    log(f"=== STAGE paper tables (v{VERSION}) ===")

    log_rows: list = []
    summary: dict = {"version": VERSION, "tables": {}}

    # -- gnn aria percentile, for the MR-neutralisation --------------------- #
    gnn_pct: dict = {}
    fe = read_csv(T / "fusion_evidence_v4.csv")
    pct_col = None
    if fe is not None:
        for cand in ("gnn_aria_pct", "aria_pct", "gnn_pct", "max_aria_pct"):
            if cand in fe.columns:
                pct_col = cand
                break
    if fe is not None and pct_col and "drug_id" in fe.columns:
        gnn_pct = dict(zip(fe["drug_id"],
                           pd.to_numeric(fe[pct_col], errors="coerce")))
        log(f"[G] GNN ARIA percentile loaded from '{pct_col}' for {len(gnn_pct)} drugs")
    else:
        log("[warn] fusion_evidence_v4.csv or a GNN-percentile column missing "
            "-> neutral 0.5 used for the MR term (plan-B drugs are unaffected: "
            "their GNN term is neutral anyway)")

    # -- Task B : ARIA-H co-medication liability map ------------------------ #
    b = read_csv(T / "comedication_risk_v4.csv")
    if b is not None:
        n0 = len(b)
        b = add_liability_adj(b, gnn_pct)
        nconf = int((pd.to_numeric(b.get("bleed_dir_conflict"), errors="coerce").fillna(0) > 0).sum())
        b = collapse(b, "liability_adj", "TaskB", log_rows)
        b["clinical_class"] = b["active_moiety"].map(classify)
        b["clinical_group"] = b.apply(clinical_group, axis=1)
        b["paper_note"] = b["active_moiety"].map(MANUAL_NOTES).fillna("")
        b["rank_by_liability"] = b["liability_adj"].rank(ascending=False, method="min").astype(int)
        b["_gord"] = (b["clinical_group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
                      .fillna(99).astype(int))
        b = b.sort_values(["_gord", "liability_adj", "active_moiety"],
                          ascending=[True, False, True]).reset_index(drop=True)
        b["rank_paper"] = np.arange(1, len(b) + 1)
        b = b.drop(columns=["_gord"])
        b.to_csv(OUT / "Table_S2_comedication_liability.csv", index=False)
        log(f"[write] Table_S2_comedication_liability.csv | {n0} -> {len(b)} rows "
            f"(salt collapse) | {nconf} row(s) MR-neutralised")
        summary["tables"]["S2_comedication"] = {
            "rows_in": n0, "rows_out": int(len(b)), "mr_neutralised": nconf}

        log("--- key-drug positions in the adjusted liability map ---")
        keys = ["WARFARIN", "ACENOCOUMAROL", "PHENPROCOUMON", "DABIGATRAN",
                "APIXABAN", "RIVAROXABAN", "EDOXABAN", "CLOPIDOGREL",
                "TICAGRELOR", "PRASUGREL", "ASPIRIN", "IBUPROFEN", "NAPROXEN",
                "INDOMETHACIN", "DICLOFENAC", "CELECOXIB", "ACETAMINOPHEN"]
        for k in keys:
            h = b[b["active_moiety"] == k]
            if len(h):
                r = h.iloc[0]
                conf = pd.to_numeric(pd.Series([r.get("bleed_dir_conflict")]),
                                     errors="coerce").fillna(0).iloc[0]
                log(f"   {k:14s} rank={int(r['rank_paper']):3d} "
                    f"adj={float(r['liability_adj']):.3f} raw={float(r['liability']):.3f} "
                    f"cls={r['clinical_class']:20s} conf={conf:.0f}")
            else:
                log(f"   {k:14s} not present")
    else:
        log("[warn] comedication_risk_v4.csv not found -> Task B skipped")

    # -- Task A : repositioning --------------------------------------------- #
    a_frames = []
    for fn in ("repositioning_top_v4.csv", "repositioning_structural_v4.csv"):
        d = read_csv(T / fn)
        if d is not None:
            d["_src"] = fn.replace(".csv", "")
            a_frames.append(d)
    if a_frames:
        a = pd.concat(a_frames, ignore_index=True)
        n0 = len(a)
        a = collapse(a, "f_repurp", "TaskA", log_rows)
        a["clinical_class"] = a["active_moiety"].map(classify)
        a = _sort(a, ["f_repurp", "gnn_repurpose_score", "active_moiety"])
        a["rank_paper"] = np.arange(1, len(a) + 1)
        a.to_csv(OUT / "Table_S1_repositioning.csv", index=False)
        log(f"[write] Table_S1_repositioning.csv | {n0} -> {len(a)} rows")
        summary["tables"]["S1_repositioning"] = {"rows_in": n0, "rows_out": int(len(a))}

    # -- Task C : combination partners -------------------------------------- #
    c_frames = []
    d1 = read_csv(T / "combination_partners_v4.csv")
    if d1 is not None:
        d1["partner_tier"] = "eligible"
        c_frames.append(d1)
    d2 = read_csv(T / "combination_partners_tier2_v4.csv")
    if d2 is not None:
        d2["partner_tier"] = "tier2_safe_unmatched"
        c_frames.append(d2)
    if c_frames:
        c = pd.concat(c_frames, ignore_index=True)
        n0 = len(c)
        if "f_tier2" in c.columns:
            if "f_partner" not in c.columns:
                c["f_partner"] = np.nan
            c["f_partner"] = pd.to_numeric(c["f_partner"], errors="coerce").fillna(
                pd.to_numeric(c["f_tier2"], errors="coerce"))
        c = add_liability_adj(c, gnn_pct)
        c = collapse(c, "f_partner", "TaskC", log_rows)
        c["clinical_class"] = c["active_moiety"].map(classify)
        c["paper_note"] = c["active_moiety"].map(MANUAL_NOTES).fillna("")
        c["_tier_ord"] = (c["partner_tier"] == "eligible").astype(int)
        c = _sort(c, ["_tier_ord", "f_partner", "liability_adj", "active_moiety"])
        c = c.drop(columns=["_tier_ord"])
        c["rank_paper"] = np.arange(1, len(c) + 1)
        c.to_csv(OUT / "Table_S3_combination_partners.csv", index=False)
        log(f"[write] Table_S3_combination_partners.csv | {n0} -> {len(c)} rows "
            f"({int((c['partner_tier'] == 'eligible').sum())} eligible)")
        summary["tables"]["S3_partners"] = {
            "rows_in": n0, "rows_out": int(len(c)),
            "eligible": int((c["partner_tier"] == "eligible").sum())}

    # -- audit + summary ---------------------------------------------------- #
    pd.DataFrame(log_rows).to_csv(OUT / "dedup_and_class_audit.csv", index=False)
    summary["n_dedup_groups"] = len(log_rows)
    summary["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary["output_dir"] = str(OUT)
    (OUT / "table_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"[summary] dedup groups={len(log_rows)} | out={OUT}")
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
