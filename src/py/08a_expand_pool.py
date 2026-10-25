#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08a_expand_pool.py  --  Chapter 6 (plan B): expand the candidate drug pool.

WHY
---
08b_fusion_v4x.py can only rank the drugs that exist in the GNN prediction
tables, i.e. the 107 CNS-penetrant compounds of the original pool.  The
coverage audit exposed a structural gap: some chapter-5 panel genes (notably
ACE) ARE targeted by approved drugs (e.g. perindopril) that never entered that
pool, so Task A was starved of mechanistically anchored candidates.

This script pulls, from the Open Targets Platform GraphQL API, every APPROVED
small molecule that targets an Alzheimer-disease-associated gene, and merges
them with the existing candidates.  The fusion step can then bring those drugs
into the ranking.

DATA SOURCE  (public, no API key)
---------------------------------
https://api.platform.opentargets.org/api/v4/graphql

  * disease id          : MONDO_0004975  = Alzheimer disease
                          (verified by schema introspection: EFO_0000249
                           returns null on this endpoint)
  * disease -> targets  : disease(efoId:"MONDO_0004975"){
                              associatedTargets(page:{index:i,size:s}){
                                  count rows{ target{id approvedSymbol} score } } }
  * target  -> drugs    : target(ensemblId:"ENSG..."){
                              drugAndClinicalCandidates{
                                  rows{ drug{id name maximumClinicalStage drugType} } } }
  * maximumClinicalStage is an UPPER_SNAKE enum: APPROVAL, PHASE_2_3, ...

KNOWN LIMITATION
----------------
Open Targets does not expose the drug's action type (INHIBITOR / AGONIST) on
the drugAndClinicalCandidates edge.  The downstream fusion therefore treats
every small molecule as an INHIBITOR of its target, which is the dominant
mode of action for the genes of interest here (ACE, PTGS1, HMGCR, P2RY12,
F10).  This must be stated as a limitation in the manuscript.

OUTPUTS  (all under data/interim/kg_ext/)
-----------------------------------------
  ad_gene_set.csv                the AD-associated genes actually used
                                 ensembl, symbol, ot_score, is_mr_panel
  candidate_drugs_expanded.csv   original drugs + newly added molecules
                                 chembl_id, name, drug_type,
                                 maximum_clinical_stage, source,
                                 ad_targets, n_ad_targets,
                                 mr_panel_targets, n_mr_panel_targets,
                                 is_cns, n_targets, n_indications
  expand_summary.json            counts + provenance, for the paper appendix

CACHING / RESUME
----------------
  data/interim/expand_cache/ad_genes.json          disease->targets pages
  data/interim/expand_cache/drugs_<ensembl>.json   per-gene drug list
Re-running fetches only what is missing.  Use --offline to read cache only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
AD_DISEASE_ID = "MONDO_0004975"          # Alzheimer disease (verified)
DEFAULT_SLEEP = 0.25                     # seconds between calls (be polite)
MIN_FREE_MB = 500.0                      # abort if the disk is nearly full

_MR_GENE_COLS = ["gene", "exposure", "id_exposure", "exposure_gene"]
_MR_ENS_COLS = ["ensembl", "ensembl_id", "gene_id", "ensembl_gene_id"]
_CAND_NAME_COLS = ["name", "drug", "drug_name", "label"]
_CAND_CNS_COLS = ["is_cns", "cns", "bbb", "is_bbb", "cns_penetrant"]


# ===========================================================================
# 0. utilities
# ===========================================================================
def log(fmt, *a):
    print("[%s] " % datetime.now().strftime("%H:%M:%S") + (fmt % a), flush=True)


def die(fmt, *a):
    print("[%s] [FATAL] " % datetime.now().strftime("%H:%M:%S") + (fmt % a),
          file=sys.stderr, flush=True)
    sys.exit(1)


def pick_col(df, cands, role, where, required=False):
    """Return the first candidate column present in df (case-insensitive)."""
    if df is None:
        return None
    low = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    if required:
        die("column for '%s' not found in %s (tried %s); actual columns: %s",
            role, where, ", ".join(cands), ", ".join(map(str, df.columns)))
    return None


def find_proj_root(start=None):
    p = Path(start or ".").resolve()
    for cand in [p] + list(p.parents):
        if (cand / "data").is_dir() and (cand / "results").is_dir():
            return cand
    return p


def check_disk(path, min_mb=MIN_FREE_MB):
    try:
        u = shutil.disk_usage(str(path))
        free_mb = u.free / 1024.0 / 1024.0
        if free_mb < min_mb:
            die("free disk on %s is %.0f MB (< %.0f MB); aborting", path,
                free_mb, min_mb)
        log("[G] free disk: %.1f GB", free_mb / 1024.0)
        return free_mb
    except SystemExit:
        raise
    except Exception as e:
        log("  [warn] disk check failed: %s", e)
        return None


def read_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, str(p))


# ===========================================================================
# 1. Open Targets client
# ===========================================================================
class OT:
    def __init__(self, sleep=DEFAULT_SLEEP, offline=False, timeout=45,
                 retries=4):
        self.sleep = sleep
        self.offline = offline
        self.timeout = timeout
        self.retries = retries
        self.s = requests.Session()
        self.s.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "ad_antiab_kg-ch6-planB/1.0 (academic use)",
        })
        self.n_calls = 0

    def gq(self, query):
        if self.offline:
            raise RuntimeError("offline mode: network call blocked")
        last = None
        for k in range(self.retries):
            try:
                r = self.s.post(OT_URL, json={"query": query},
                                timeout=self.timeout)
                self.n_calls += 1
                if r.status_code == 200:
                    j = r.json()
                    if j.get("errors"):
                        raise RuntimeError("GraphQL: %s" % str(j["errors"])[:300])
                    time.sleep(self.sleep)
                    return j.get("data")
                last = "HTTP %d" % r.status_code
            except Exception as e:
                last = str(e)
            time.sleep(1.5 * (k + 1))
        raise RuntimeError("Open Targets query failed: %s" % last)


def fetch_ad_genes(ot, cache_path, n_total, page_size=200):
    """disease -> associatedTargets, paginated. Returns the raw row list."""
    cached = read_json(cache_path)
    if cached and len(cached) >= n_total:
        log("[G] AD genes: %d rows loaded from cache", len(cached))
        return cached[:n_total]
    rows = list(cached or [])
    idx = 0
    while len(rows) < n_total:
        size = min(page_size, n_total - len(rows))
        q = ('{ disease(efoId:"%s"){ id name associatedTargets('
             'page:{index:%d,size:%d}){ count rows{ target{ id approvedSymbol }'
             ' score } } } }' % (AD_DISEASE_ID, idx, size))
        d = ot.gq(q)
        dis = (d or {}).get("disease")
        if not dis:
            die("Open Targets returned no disease for %s", AD_DISEASE_ID)
        at = dis.get("associatedTargets") or {}
        got = at.get("rows") or []
        if not got:
            break
        rows.extend(got)
        idx += 1
        log("    page %d -> %d rows (cumulative %d, OT count %s)",
            idx, len(got), len(rows), at.get("count"))
        if len(rows) >= int(at.get("count") or 0):
            break
    rows = rows[:n_total]
    write_json(cache_path, rows)
    return rows


def fetch_gene_drugs(ot, ensembl, cache_dir):
    """target -> drugAndClinicalCandidates. Returns list of row dicts."""
    p = Path(cache_dir) / ("drugs_%s.json" % ensembl)
    cached = read_json(p)
    if cached is not None:
        return cached
    q = ('{ target(ensemblId:"%s"){ id approvedSymbol '
         'drugAndClinicalCandidates{ count rows{ drug{ id name '
         'maximumClinicalStage drugType } } } } }' % ensembl)
    d = ot.gq(q)
    tgt = (d or {}).get("target")
    out = []
    if tgt:
        out = (tgt.get("drugAndClinicalCandidates") or {}).get("rows") or []
    write_json(p, out)
    return out


def search_ensembl(ot, symbol, cache_path, cache):
    """Resolve a gene symbol to an Ensembl id via the Open Targets search API."""
    up = symbol.upper()
    if cache.get(up):
        return cache[up]
    q = ('{ search(queryString:"%s", entityNames:["target"], '
         'page:{index:0,size:5}){ hits{ id name entity } } }' % symbol)
    try:
        d = ot.gq(q)
    except Exception as e:
        log("    [warn] search '%s' failed: %s", symbol, e)
        return ""
    hits = ((d or {}).get("search") or {}).get("hits") or []
    ens = ""
    for h in hits:
        if str(h.get("name", "")).upper() == up:
            ens = h.get("id") or ""
            break
    if not ens and hits:
        ens = hits[0].get("id") or ""
    if ens:
        cache[up] = ens
        write_json(cache_path, cache)
    return ens


# ===========================================================================
# 2. main
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="chapter-6 plan B: expand the candidate drug pool")
    p.add_argument("--proj", default=".",
                   help="project root (folder containing data/ and results/)")
    p.add_argument("--n-ad-genes", type=int, default=200,
                   help="top-N Open Targets AD-associated genes (default 200)")
    p.add_argument("--all-ad-genes", action="store_true",
                   help="use ALL AD-associated genes (~13k; very slow)")
    p.add_argument("--stage", default="APPROVAL",
                   help="comma-separated maximumClinicalStage filter "
                        "(default APPROVAL; use '' to disable)")
    p.add_argument("--drug-types", default="Small molecule",
                   help="comma-separated drugType filter "
                        "(default 'Small molecule'; use '' to disable)")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                   help="seconds between API calls (default 0.25)")
    p.add_argument("--max-genes", type=int, default=0,
                   help="DEBUG: cap the number of genes queried (0 = no cap)")
    p.add_argument("--offline", action="store_true",
                   help="read cache only, never call the API")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    proj = find_proj_root(args.proj)
    log("project root: %s", proj)
    log("=== STAGE expand pool (plan B, Open Targets) ===")

    tdir = proj / "results" / "tables"
    ext = proj / "data" / "interim" / "kg_ext"
    cache_dir = proj / "data" / "interim" / "expand_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    check_disk(proj)

    stages = {s.strip().upper() for s in args.stage.split(",") if s.strip()}
    dtypes = {s.strip() for s in args.drug_types.split(",") if s.strip()}
    log("[G] filters | stage in %s | drug_type in %s",
        sorted(stages) or "ANY", sorted(dtypes) or "ANY")

    # ---- 1. chapter-5 panel genes (symbol -> ensembl) ----------------------
    panel = {}          # symbol -> ensembl or ""
    mr_path = tdir / "mr_summary.csv"
    if mr_path.exists():
        mr = pd.read_csv(mr_path)
        gcol = pick_col(mr, _MR_GENE_COLS, "gene", "mr_summary.csv")
        ecol = pick_col(mr, _MR_ENS_COLS, "ensembl", "mr_summary.csv")
        if gcol:
            for _, row in mr.iterrows():
                sym = str(row[gcol]).strip().upper()
                if not sym or sym == "NAN":
                    continue
                ens = ""
                if ecol and pd.notna(row[ecol]):
                    ens = str(row[ecol]).strip()
                if sym not in panel or (not panel[sym] and ens):
                    panel[sym] = ens
    else:
        log("  [warn] %s not found -> panel genes unavailable", mr_path)
    panel_genes = set(panel)
    log("[G] chapter-5 panel genes: %d | %s", len(panel_genes),
        ", ".join(sorted(panel_genes)))

    # ---- 2. AD-associated genes from Open Targets --------------------------
    ot = OT(sleep=args.sleep, offline=args.offline)

    # resolve any panel gene whose ensembl id is missing in mr_summary.csv
    emap_path = cache_dir / "ensembl_map.json"
    emap = read_json(emap_path, {}) or {}
    need = [s for s, e in panel.items() if not e]
    if need and not args.offline:
        log("[G] resolving %d panel gene ensembl id(s) via Open Targets search",
            len(need))
        for s in need:
            ens = search_ensembl(ot, s, emap_path, emap)
            if ens:
                panel[s] = ens
                log("    %s -> %s", s, ens)
            else:
                log("    [warn] %s: ensembl id could not be resolved", s)
    n_panel_ens = sum(1 for v in panel.values() if v)
    log("    panel genes with ensembl id: %d / %d", n_panel_ens, len(panel))

    n_target = 100000 if args.all_ad_genes else args.n_ad_genes
    ad_rows = fetch_ad_genes(ot, cache_dir / "ad_genes.json", n_target)

    genes = []
    seen_ens = set()
    for r in ad_rows:
        t = r.get("target") or {}
        ens = t.get("id")
        sym = (t.get("approvedSymbol") or "").strip().upper()
        if not ens or not sym or ens in seen_ens:
            continue
        seen_ens.add(ens)
        genes.append({"ensembl": ens, "symbol": sym,
                      "ot_score": r.get("score"),
                      "is_mr_panel": int(sym in panel_genes)})

    # force-include panel genes that are not in the OT top-N list
    forced = 0
    for sym, ens in sorted(panel.items()):
        if ens and ens not in seen_ens:
            genes.append({"ensembl": ens, "symbol": sym, "ot_score": None,
                          "is_mr_panel": 1})
            seen_ens.add(ens)
            forced += 1
    log("[G] AD-associated genes: %d (panel-matched %d, force-included %d)",
        len(genes), sum(g["is_mr_panel"] for g in genes), forced)

    # which panel genes could not be represented at all
    syms_in = {g["symbol"] for g in genes}
    lost = sorted(panel_genes - syms_in)
    if lost:
        log("  [warn] panel genes NOT represented (missing ensembl id): %s",
            ", ".join(lost))

    if args.max_genes and args.max_genes > 0:
        genes = genes[:args.max_genes]
        log("  [debug] --max-genes %d -> querying only %d genes",
            args.max_genes, len(genes))

    if args.dry_run:
        log("[dry-run] would query %d genes with n_ad_genes=%d; stop.",
            len(genes), n_target)
        log("  first 10: %s", ", ".join("%s(%s)" % (g["symbol"], g["ensembl"])
                                       for g in genes[:10]))
        return

    # ---- 3. per-gene drug pull --------------------------------------------
    drug2genes = {}      # chembl_id -> record with a set of gene symbols
    n_done = n_fail = 0
    for i, g in enumerate(genes, 1):
        ens = g["ensembl"]
        try:
            rows = fetch_gene_drugs(ot, ens, cache_dir)
        except Exception as e:
            n_fail += 1
            log("  [warn] %s (%s) failed: %s", g["symbol"], ens, e)
            continue
        n_done += 1
        for r in rows:
            dd = r.get("drug") or {}
            cid = dd.get("id")
            if not cid:
                continue
            rec = drug2genes.get(cid)
            if rec is None:
                rec = {
                    "chembl_id": cid,
                    "name": dd.get("name"),
                    "maximum_clinical_stage": dd.get("maximumClinicalStage"),
                    "drug_type": dd.get("drugType"),
                    "genes": set(),
                }
                drug2genes[cid] = rec
            rec["genes"].add(g["symbol"])
        if i % 25 == 0 or i == len(genes):
            log("    genes queried %d/%d | distinct drugs %d",
                i, len(genes), len(drug2genes))
    log("[G] per-gene pull done | genes ok %d, failed %d | distinct drugs %d "
        "| API calls %d", n_done, n_fail, len(drug2genes), ot.n_calls)

    if not drug2genes:
        die("no drug returned -- is the cache empty and the network blocked?")

    # ---- 4. filter to approved small molecules -----------------------------
    keep = []
    n_stage = n_type = 0
    for rec in drug2genes.values():
        st = (rec["maximum_clinical_stage"] or "").upper()
        ty = rec["drug_type"] or ""
        if stages and st not in stages:
            n_stage += 1
            continue
        if dtypes and ty not in dtypes:
            n_type += 1
            continue
        keep.append(rec)
    log("[G] filter | dropped by stage %d | dropped by type %d | kept %d",
        n_stage, n_type, len(keep))
    if not keep:
        die("no drug survived the filter -- check --stage / --drug-types")

    # ---- 5. assemble the expanded table ------------------------------------
    exp = pd.DataFrame([{
        "chembl_id": r["chembl_id"],
        "name": r["name"],
        "drug_type": r["drug_type"],
        "maximum_clinical_stage": r["maximum_clinical_stage"],
        "source": "pool_expanded",
        "ad_targets": ";".join(sorted(r["genes"])),
        "n_ad_targets": len(r["genes"]),
        "mr_panel_targets": ";".join(sorted(r["genes"] & panel_genes)),
        "n_mr_panel_targets": len(r["genes"] & panel_genes),
    } for r in keep])
    exp = exp.sort_values(["n_mr_panel_targets", "n_ad_targets", "name"],
                          ascending=[False, False, True]).reset_index(drop=True)
    exp["_key"] = exp["name"].astype(str).str.upper()

    # ---- 6. merge with the original candidate pool -------------------------
    old_path = ext / "candidate_drugs.csv"
    n_orig = 0
    old_keys = set()
    parts = []
    if old_path.exists():
        old = pd.read_csv(old_path)
        n_orig = len(old)
        o = old.copy()
        o["source"] = "gnn_scored"
        ncol = pick_col(o, _CAND_NAME_COLS, "name", "candidate_drugs.csv")
        o["_key"] = o[ncol].astype(str).str.upper()
        old_keys = set(o["_key"])
        lk = {k: (t, n, m, mn) for k, t, n, m, mn in zip(
            exp["_key"], exp["ad_targets"], exp["n_ad_targets"],
            exp["mr_panel_targets"], exp["n_mr_panel_targets"])}
        o["ad_targets"] = [lk.get(k, ("", 0, "", 0))[0] for k in o["_key"]]
        o["n_ad_targets"] = [lk.get(k, ("", 0, "", 0))[1] for k in o["_key"]]
        o["mr_panel_targets"] = [lk.get(k, ("", 0, "", 0))[2] for k in o["_key"]]
        o["n_mr_panel_targets"] = [lk.get(k, ("", 0, "", 0))[3] for k in o["_key"]]
        parts.append(o)
        log("[G] original candidate pool: %d rows (%s)", n_orig, old_path.name)

    exp_new = exp[~exp["_key"].isin(old_keys)].copy()
    exp_new["is_cns"] = np.nan
    exp_new["n_targets"] = exp_new["n_ad_targets"]
    exp_new["n_indications"] = np.nan
    exp_new["n_adverse_events"] = np.nan
    parts.append(exp_new)
    log("[G] new drugs added to the pool: %d", len(exp_new))

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged = merged.drop(columns=[c for c in ["_key"] if c in merged.columns])
    # stable column order
    front = ["chembl_id", "name", "drug_type", "maximum_clinical_stage",
             "source", "ad_targets", "n_ad_targets", "mr_panel_targets",
             "n_mr_panel_targets"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]
    merged.to_csv(ext / "candidate_drugs_expanded.csv", index=False)
    log("[write] %s", ext / "candidate_drugs_expanded.csv")

    pd.DataFrame(genes)[["ensembl", "symbol", "ot_score", "is_mr_panel"]] \
        .to_csv(ext / "ad_gene_set.csv", index=False)
    log("[write] %s", ext / "ad_gene_set.csv")

    # ---- 7. summary --------------------------------------------------------
    def _count_gene(g):
        n = 0
        for s in exp["mr_panel_targets"].fillna(""):
            if g in [x for x in str(s).split(";") if x]:
                n += 1
        return n

    anchored = exp[exp["n_mr_panel_targets"] > 0].sort_values(
        ["n_ad_targets", "name"])
    top = (anchored.drop_duplicates(subset=["mr_panel_targets"])
           .head(args.topk)[["name", "maximum_clinical_stage", "ad_targets",
                             "n_ad_targets", "mr_panel_targets",
                             "n_mr_panel_targets"]])
    summary = {
        "version": "1.0.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ad_disease_id": AD_DISEASE_ID,
        "n_ad_genes_used": len(genes),
        "n_panel_genes": len(panel_genes),
        "n_panel_genes_with_ensembl": n_panel_ens,
        "n_genes_queried_ok": n_done,
        "n_genes_failed": n_fail,
        "n_distinct_drugs_raw": len(drug2genes),
        "n_drugs_expanded_after_filter": int(len(exp)),
        "n_original_pool": n_orig,
        "n_new_drugs_added": int(len(exp_new)),
        "n_total_after_merge": int(len(merged)),
        "n_drugs_touching_panel_gene": int((exp["n_mr_panel_targets"] > 0).sum()),
        "stage_filter": sorted(stages),
        "drug_type_filter": sorted(dtypes),
        "api_calls": ot.n_calls,
        "panel_hit_drug_counts": {g: _count_gene(g) for g in sorted(panel_genes)},
        "top_expanded": top.to_dict("records"),
    }
    write_json(ext / "expand_summary.json", summary)
    log("[write] %s", ext / "expand_summary.json")

    log("--- plan-B pool expansion summary ---")
    log("  AD genes used        : %d (panel-matched %d)",
        len(genes), sum(g["is_mr_panel"] for g in genes))
    log("  approved drugs pulled: %d raw -> %d after filter",
        len(drug2genes), len(exp))
    log("  original pool %d + new %d = %d total", n_orig, len(exp_new),
        len(merged))
    log("  drugs touching a chapter-5 panel gene: %d",
        int((exp["n_mr_panel_targets"] > 0).sum()))
    log("  approved drugs per panel gene:")
    for g in sorted(panel_genes):
        c = _count_gene(g)
        if c:
            log("    %-8s <- %d drug(s)", g, c)
    if len(exp):
        log("  one representative per panel-target combination:")
        for _, r in top.iterrows():
            log("    %-34s stage=%s ad_targets=%s",
                str(r["name"])[:34], r["maximum_clinical_stage"],
                r["ad_targets"])
    log("done.")


if __name__ == "__main__":
    main()
