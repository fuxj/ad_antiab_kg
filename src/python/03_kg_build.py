#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_kg_build.py —— 多模态 AD 抗 Abeta 药物安全性知识图谱构建

数据源（全部开源、免注册）
  [1] Open Targets Platform GraphQL   https://api.platform.opentargets.org/api/v4/graphql
      药物 -> 靶点(mechanismsOfAction)、药物 -> 适应症(indications)、
      药物 -> 不良事件(adverseEvents)、靶点 -> 通路(targets.pathways)、
      疾病 -> 关联靶点(disease.associatedTargets，含遗传/文献/临床等多维证据打分)
  [2] 本地 FAERS 不成比例分析结果（第 2 章产出）
      results/tables/faers_pt_level_signals.csv

节点类型（5 类）
  Drug          药物
  Gene          基因/靶点
  Pathway       Reactome 通路
  Indication    适应症/疾病
  AdverseEvent  不良事件（跨源统一到"概念"粒度）

边类型（6 类）
  drug_target       药物 -> 靶点      （Open Targets MoA）
  drug_indication   药物 -> 适应症    （Open Targets，含最高临床阶段）
  drug_ae           药物 -> 不良事件  （FAERS 四法信号 + Open Targets 文献/FAERS 挖掘）
  gene_pathway      基因 -> 通路      （Open Targets / Reactome）
  gene_indication   基因 -> 适应症    （Open Targets 关联打分）
  gene_ae           基因 -> 不良事件  （派生：靶点经药物中介关联到事件）

产出（落到项目既有规范目录）
  data/interim/kg/ot_cache/*.json           Open Targets 原始响应缓存（断点续跑用）
  data/processed/kg/nodes.parquet           节点表
  data/processed/kg/edges.parquet           边表
  data/processed/kg/kg_edge_list.csv        人类可读边清单（可直接丢进 Gephi / Cytoscape）
  data/processed/kg/kg_stats.json           图统计量
  results/tables/kg_aria_cross_source.csv   ARIA 事件跨源一致性表
  results/figures/kg_overview.html          交互式图谱可视化（vis-network）

用法
  python 03_kg_build.py all                    # 抓取 + 建图 + 统计 + 可视化（联网）
  python 03_kg_build.py build --offline        # 完全离线，用脚本内置种子数据复现
  python 03_kg_build.py fetch --force          # 强制重抓 Open Targets
  python 03_kg_build.py stats                  # 只重算统计与可视化

常用参数
  --proj DIR            项目根目录（默认自动向上查找）
  --offline             不联网，使用内置种子数据
  --max-ad-genes N      AD 关联基因纳入上限（按综合证据分降序，默认 60）
  --min-a N             FAERS 事件纳入的最小报告数（默认 3）
  --ae-criterion STR    drug_ae 边取自 FAERS 的判据：robust(四法全阳) | any(任一法阳) | all(全部 PT)
  --min-free-gb N       剩余磁盘低于该值时中止（默认 20）
  --force               忽略缓存，强制重抓
"""

import argparse
import base64
import csv
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd

# =============================================================================
# 0. 常量
# =============================================================================

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
AD_EFO = "MONDO_0004975"          # Open Targets: Alzheimer disease

DRUG_REGISTRY = [
    {"key": "lecanemab",  "chembl": "CHEMBL3833321", "label": "Lecanemab"},
    {"key": "donanemab",  "chembl": "CHEMBL4297245", "label": "Donanemab"},
    {"key": "aducanumab", "chembl": "CHEMBL3039540", "label": "Aducanumab"},
]

NODE_TYPES = ["Drug", "Gene", "Pathway", "Indication", "AdverseEvent"]

EDGE_TYPES = [
    "drug_target", "drug_indication", "drug_ae",
    "gene_pathway", "gene_indication", "gene_ae",
]

# Open Targets 查询
Q_DRUG = """
query D($id:String!){ drug(chemblId:$id){
  id name drugType maximumClinicalStage
  mechanismsOfAction{ uniqueActionTypes rows{ actionType mechanismOfAction targetName
    targets{ id approvedSymbol approvedName } } }
  indications{ count rows{ id maxClinicalStage disease{ id name } } }
  adverseEvents{ count criticalValue rows{ meddraCode name count logLR } }
} }
"""

Q_DISEASE = """
query A($id:String!,$n:Int!){ disease(efoId:$id){ id name
  associatedTargets(page:{index:0,size:$n}){ count
    rows{ score target{ id approvedSymbol approvedName } datatypeScores{ id score } } } } }
"""

Q_TARGETS = """
query T($ids:[String!]!){ targets(ensemblIds:$ids){ id approvedSymbol approvedName
  pathways{ pathwayId pathway topLevelTerm } } }
"""

# 内置离线种子（zlib + base64），为真实 Open Targets 响应的压缩副本
EMBEDDED_SEED_B64 = "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"

# =============================================================================
# 1. 工具函数
# =============================================================================

_T0 = time.time()


def log(fmt, *a):
    ts = datetime.now().strftime("%H:%M:%S")
    msg = fmt % a if a else fmt
    print("[%s] %s" % (ts, msg), flush=True)


def die(fmt, *a):
    log("错误: " + (fmt % a if a else fmt))
    sys.exit(1)


def human(n):
    return "{:,}".format(int(n))


def check_disk(path, min_free_gb):
    try:
        total, used, free = shutil.disk_usage(str(path))
    except OSError:
        return
    free_gb = free / 1024 ** 3
    if free_gb < min_free_gb:
        die("磁盘剩余 %.1f GB 低于阈值 %d GB，已中止（避免写满盘）", free_gb, min_free_gb)
    log("磁盘: 剩余 %.1f GB（阈值 %d GB）", free_gb, min_free_gb)


def find_proj_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for cand in [p] + list(p.parents):
        if (cand / "src").is_dir() and (cand / "data").is_dir():
            return cand
    return p


def embedded_seed():
    raw = zlib.decompress(base64.b64decode(EMBEDDED_SEED_B64)).decode("utf-8")
    return json.loads(raw)


def norm_ae(name):
    """不良事件名称归一化：统一拼写差异 -> 便于跨源匹配。"""
    n = (name or "").lower()
    n = n.replace("oedema", "edema").replace("haemorrhage", "hemorrhage")
    n = n.replace("haemosiderin", "hemosiderin").replace("haematoma", "hematoma")
    n = n.replace("ischemic", "ischemic").replace("oedematous", "edematous")
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ARIA 概念归类：把 FAERS 的 MedDRA PT 与 Open Targets 的事件名对齐到同一"概念"
def aria_concept(norm_name):
    n = norm_name
    if "amyloid related imaging" in n or "aria" in n.split():
        if "edema" in n or "effusion" in n or "exudative" in n:
            return "ARIA-E"
        if ("microbleed" in n or "hemosiderin" in n or "iron deposition" in n
                or "microhemorrhage" in n or "hemorrhage" in n):
            return "ARIA-H"
        return "ARIA-NOS"
    if "superficial siderosis" in n:
        return "SUPERFICIAL_SIDEROSIS"
    if "cerebral microhemorrhage" in n or "cerebral microhemorrhage" in n:
        return "CEREBRAL_MICROHEMORRHAGE"
    if "cerebral hemorrhage" in n:
        return "CEREBRAL_HEMORRHAGE"
    if "brain edema" in n or "cerebral edema" in n or "vasogenic cerebral edema" in n:
        return "CEREBRAL_EDEMA"
    return None


CONCEPT_LABEL = {
    "ARIA-E": "ARIA-E (amyloid-related imaging abnormality, edema/effusion)",
    "ARIA-H": "ARIA-H (amyloid-related imaging abnormality, microbleeds/hemosiderin)",
    "ARIA-NOS": "ARIA, unspecified",
    "SUPERFICIAL_SIDEROSIS": "Superficial siderosis of the CNS",
    "CEREBRAL_MICROHEMORRHAGE": "Cerebral microhemorrhage",
    "CEREBRAL_HEMORRHAGE": "Cerebral hemorrhage",
    "CEREBRAL_EDEMA": "Cerebral edema",
}

STAGE_RANK = {
    "UNKNOWN": 0, "PRECLINICAL": 1, "PHASE_1": 2, "EARLY_PHASE_1": 2,
    "PHASE_2": 3, "PHASE_2_3": 4, "PHASE_3": 5, "APPROVAL": 6,
}

SS = {"drug": "opentargets", "ae": "opentargets"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


# =============================================================================
# 2. Open Targets GraphQL 客户端
# =============================================================================

def ot_query(query, variables=None, tries=4, timeout=90):
    payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                OT_URL, data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json",
                         "User-Agent": "ad-antiab-kg/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "errors" in data:
                raise RuntimeError("GraphQL: " + json.dumps(data["errors"])[:300])
            return data["data"]
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError("Open Targets 查询失败: %r" % (last,))


# =============================================================================
# 3. 抓取阶段
# =============================================================================

def cache_path(proj, name):
    d = proj / "data" / "interim" / "kg" / "ot_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / (name + ".json")


def fetch_stage(proj, args):
    """联网抓取 Open Targets 数据，逐项落盘缓存（幂等、可断点续跑）。"""
    cache = {"drugs": {}}
    for d in DRUG_REGISTRY:
        cp = cache_path(proj, "drug_" + d["key"])
        if cp.exists() and not args.force:
            log("[缓存] 药物 %s（%s）", d["key"], cp.name)
            cache["drugs"][d["key"]] = json.loads(cp.read_text(encoding="utf-8"))
            continue
        log("[抓取] 药物 %s（%s）...", d["key"], d["chembl"])
        try:
            drug = ot_query(Q_DRUG, {"id": d["chembl"]})["drug"]
        except Exception as e:  # noqa: BLE001
            die("抓取 %s 失败: %s", d["key"], e)
        cp.write_text(json.dumps(drug, ensure_ascii=False, indent=1), encoding="utf-8")
        cache["drugs"][d["key"]] = drug
        n_moa = len((drug.get("mechanismsOfAction") or {}).get("rows") or [])
        n_ind = len((drug.get("indications") or {}).get("rows") or [])
        n_ae = len((drug.get("adverseEvents") or {}).get("rows") or [])
        log("       靶点记录 %d | 适应症 %d | 不良事件 %d", n_moa, n_ind, n_ae)

    cp = cache_path(proj, "disease_ad")
    if cp.exists() and not args.force:
        log("[缓存] AD 关联基因（%s）", cp.name)
        dz = json.loads(cp.read_text(encoding="utf-8"))
    else:
        log("[抓取] Alzheimer disease 关联靶点（%s）...", AD_EFO)
        dz = ot_query(Q_DISEASE, {"id": AD_EFO, "n": args.max_ad_genes})["disease"]
        cp.write_text(json.dumps(dz, ensure_ascii=False, indent=1), encoding="utf-8")
    log("       关联靶点总数 %s，本次纳入 %d",
        human(dz["associatedTargets"]["count"]), len(dz["associatedTargets"]["rows"]))
    cache["disease"] = dz

    genes = sorted({r["target"]["id"] for r in dz["associatedTargets"]["rows"]})
    for d in DRUG_REGISTRY:
        for m in cache["drugs"][d["key"]]["mechanismsOfAction"]["rows"] or []:
            for t in (m.get("targets") or []):
                genes.append(t["id"])
    genes = sorted(set(genes))

    cp = cache_path(proj, "target_pathways")
    if cp.exists() and not args.force:
        log("[缓存] 靶点通路（%s）", cp.name)
        cache["target_pathways"] = json.loads(cp.read_text(encoding="utf-8"))
    else:
        log("[抓取] %d 个基因的 Reactome 通路（分批）...", len(genes))
        pw = {}
        for i in range(0, len(genes), 25):
            chunk = genes[i:i + 25]
            try:
                rows = ot_query(Q_TARGETS, {"ids": chunk})["targets"]
            except Exception as e:  # noqa: BLE001
                die("抓取通路失败（批次 %d）: %s", i // 25 + 1, e)
            for t in rows:
                pw[t["id"]] = {"approvedSymbol": t["approvedSymbol"],
                               "approvedName": t["approvedName"],
                               "pathways": t.get("pathways") or []}
            log("       批次 %d/%d：累计基因 %d，累计通路 %d",
                i // 25 + 1, (len(genes) + 24) // 25, len(pw),
                sum(len(v["pathways"]) for v in pw.values()))
            time.sleep(0.4)
        cp.write_text(json.dumps(pw, ensure_ascii=False, indent=1), encoding="utf-8")
        cache["target_pathways"] = pw

    return cache


def load_ot_data(proj, args):
    """联网优先；无网或 --offline 时回落到内置种子。"""
    if args.offline:
        log("[数据] --offline：使用脚本内置种子（真实 Open Targets 响应的压缩副本）")
        seed = embedded_seed()
        return seed, "embedded-seed"
    try:
        return fetch_stage(proj, args), "opentargets-live"
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log("[警告] 联网抓取失败（%s），回落到脚本内置种子数据", e)
        seed = embedded_seed()
        return seed, "embedded-seed(fallback)"


# =============================================================================
# 4. 建图阶段
# =============================================================================

def load_faers(proj, args):
    f = proj / "results" / "tables" / "faers_pt_level_signals.csv"
    if not f.exists():
        log("[警告] 未找到 %s，drug_ae 边将仅来自 Open Targets", f)
        return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]

    # 列名别名容错：R 脚本或不同版本可能用 drug_name / pt_name 等写法
    alias = {"drug": ["drug", "drug_name", "drugname", "faers_key", "drug_key", "name", "drugname_std"],
             "pt": ["pt", "pt_name", "ptname", "event", "ae", "term", "event_name", "reaction"]}
    for std, cands in alias.items():
        if std not in df.columns:
            hit = next((c for c in cands if c in df.columns), None)
            if hit:
                df = df.rename(columns={hit: std})
                log("[FAERS] 列名归一：%s -> %s", hit, std)

    need = {"drug", "pt", "a"}
    if not need.issubset(df.columns):
        log("[警告] %s 缺少列 %s，跳过 FAERS 侧（现有列：%s）",
            f.name, sorted(need - set(df.columns)), list(df.columns)[:14])
        return None
    df["drug"] = df["drug"].astype(str).str.lower().str.strip()
    df["pt"] = df["pt"].astype(str)
    keep = df["drug"].isin([d["key"] for d in DRUG_REGISTRY])
    df = df[keep].copy()
    log("[FAERS] 读入 %s 行（目标药 %d 种）", human(len(df)), df["drug"].nunique())

    n0 = len(df)
    if args.ae_criterion == "robust" and "robust" in df.columns:
        df = df[df["robust"].astype(bool)]
    elif args.ae_criterion == "any" and "n_sig" in df.columns:
        df = df[df["n_sig"].astype(float) > 0]
    elif args.ae_criterion == "all":
        pass
    df = df[df["a"].astype(float) >= args.min_a]
    log("[FAERS] 判据=%s 且 a>=%d 后保留 %s 行（剔除 %s）",
        args.ae_criterion, args.min_a, human(len(df)), human(n0 - len(df)))
    return df


def build_graph(ot, faers, args):
    nodes = {}   # node_id -> dict
    edges = []   # dict
    node_src = defaultdict(set)   # node_id -> {数据源标签}，用于统一计算 n_sources / sources

    def add_node(nid, ntype, name, **attrs):
        if attrs.get("source"):
            node_src[nid].add(attrs["source"])
        if nid in nodes:
            nodes[nid].update({k: v for k, v in attrs.items() if v not in (None, "", [])})
            return
        rec = {"node_id": nid, "node_type": ntype, "name": name, "label": name}
        rec.update(attrs)
        nodes[nid] = rec

    def add_edge(src, dst, etype, source, weight=1.0, **attrs):
        if src not in nodes or dst not in nodes:
            return
        rec = {"src": src, "dst": dst, "edge_type": etype, "source": source,
               "weight": float(weight)}
        rec.update(attrs)
        edges.append(rec)

    # ---- 4.1 药物节点 + drug_target + drug_indication ----
    ot_ae_by_drug = defaultdict(list)
    for d in DRUG_REGISTRY:
        drug = ot["drugs"][d["key"]]
        did = "DRUG:" + drug["id"]
        add_node(did, "Drug", drug["name"], chembl_id=drug["id"],
                 drug_type=drug.get("drugType"), max_stage=drug.get("maximumClinicalStage"),
                 faers_key=d["key"], source="opentargets")

        for m in (drug.get("mechanismsOfAction") or {}).get("rows") or []:
            for t in (m.get("targets") or []):
                gid = "GENE:" + t["id"]
                add_node(gid, "Gene", t.get("approvedSymbol") or t["id"],
                         ensembl_id=t["id"], full_name=t.get("approvedName"),
                         source="opentargets")
                add_edge(did, gid, "drug_target", "opentargets", 1.0,
                         action_type=m.get("actionType"),
                         mechanism=m.get("mechanismOfAction"),
                         target_name=m.get("targetName"))

        for ind in (drug.get("indications") or {}).get("rows") or []:
            dis = ind.get("disease") or {}
            iid = "IND:" + slug(dis.get("name"))
            add_node(iid, "Indication", dis.get("name"), ot_id=dis.get("id"),
                     source="opentargets")
            st = ind.get("maxClinicalStage")
            add_edge(did, iid, "drug_indication", "opentargets",
                     STAGE_RANK.get(st, 0) or 1.0, max_stage=st, ot_ind_id=ind.get("id"))

        for ae in (drug.get("adverseEvents") or {}).get("rows") or []:
            ot_ae_by_drug[d["key"]].append(ae)

    # ---- 4.2 AD 疾病节点 + 基因节点 + gene_indication ----
    dz = ot["disease"]
    ad_iid = "IND:" + slug(dz["name"])
    add_node(ad_iid, "Indication", dz["name"], ot_id=dz["id"], is_focus_disease=True,
             source="opentargets")

    for r in dz["associatedTargets"]["rows"]:
        t = r["target"]
        gid = "GENE:" + t["id"]
        add_node(gid, "Gene", t.get("approvedSymbol") or t["id"],
                 ensembl_id=t["id"], full_name=t.get("approvedName"),
                 source="opentargets")
        dts = {x["id"]: round(float(x["score"]), 4) for x in (r.get("datatypeScores") or [])}
        add_edge(gid, ad_iid, "gene_indication", "opentargets",
                 float(r["score"]), evidence=dts, disease=dz["name"])

    # ---- 4.3 gene_pathway ----
    n_gp = 0
    for gid_ens, rec in ot["target_pathways"].items():
        gid = "GENE:" + gid_ens
        for p in rec.get("pathways") or []:
            pid = "PATHWAY:" + p["pathwayId"]
            add_node(pid, "Pathway", (p.get("pathway") or "").strip(),
                     pathway_id=p["pathwayId"], top_level=p.get("topLevelTerm"),
                     source="reactome")
            add_edge(gid, pid, "gene_pathway", "opentargets", 1.0,
                     top_level=p.get("topLevelTerm"))
            n_gp += 1

    # ---- 4.4 drug_ae：FAERS 侧 ----
    ae_meta = defaultdict(lambda: {"sources": set(), "faers_pt": None, "ot_name": None,
                                   "faers_pts": set(), "ot_names": set(),
                                   "event_group": None, "aria_concept": None})
    n_faers_edge = 0
    if faers is not None:
        for row in faers.itertuples(index=False):
            dkey = getattr(row, "drug")
            reg = next((x for x in DRUG_REGISTRY if x["key"] == dkey), None)
            if reg is None:
                continue
            did = "DRUG:" + reg["chembl"]
            pt = getattr(row, "pt")
            n = norm_ae(pt)
            concept = aria_concept(n)
            aid = "AE:" + (concept if concept else "RAW:" + n)
            grp = getattr(row, "event_group", None)
            add_node(aid, "AdverseEvent", CONCEPT_LABEL.get(concept, pt),
                     faers_pt=pt, ot_name=None, event_group=grp, aria_concept=concept,
                     source="faers")
            m = ae_meta[aid]
            m["sources"].add("faers")
            m["faers_pts"].add(pt)
            m["faers_pt"] = pt
            if grp:
                m["event_group"] = grp
            if concept:
                m["aria_concept"] = concept
            attrs = {}
            for c in ["a", "b", "c", "d", "n_drug", "n_event", "ROR", "ROR025", "ROR975",
                      "PRR", "PRR025", "PRR975", "chi2", "IC", "IC025", "EBGM", "EB05",
                      "n_sig", "robust"]:
                if hasattr(row, c):
                    v = getattr(row, c)
                    attrs["faers_" + c] = v
            attrs["faers_pt"] = pt
            attrs["aria_concept"] = concept
            add_edge(did, aid, "drug_ae", "faers", float(getattr(row, "a", 1) or 1), **attrs)
            n_faers_edge += 1
    log("[建图] drug_ae（FAERS）：%s 条边", human(n_faers_edge))

    # ---- 4.5 drug_ae：Open Targets 侧 ----
    n_ot_edge = 0
    for d in DRUG_REGISTRY:
        did = "DRUG:" + ot["drugs"][d["key"]]["id"]
        for ae in ot_ae_by_drug.get(d["key"], []):
            nm = ae.get("name") or ""
            n = norm_ae(nm)
            concept = aria_concept(n)
            aid = "AE:" + (concept if concept else "RAW:" + n)
            add_node(aid, "AdverseEvent", CONCEPT_LABEL.get(concept, nm),
                     faers_pt=None, ot_name=nm, meddra_code=ae.get("meddraCode") or None,
                     aria_concept=concept, source="opentargets")
            m = ae_meta[aid]
            m["sources"].add("opentargets")
            m["ot_names"].add(nm)
            if not m["ot_name"]:
                m["ot_name"] = nm
            if concept:
                m["aria_concept"] = concept
            add_edge(did, aid, "drug_ae", "opentargets",
                     float(ae.get("count") or 1), ot_count=ae.get("count"),
                     ot_logLR=ae.get("logLR"), ot_name=nm, aria_concept=concept)
            n_ot_edge += 1
    log("[建图] drug_ae（Open Targets）：%s 条边", human(n_ot_edge))

    # ---- 4.5b 药物跨源标记：同时出现在 Open Targets 与 FAERS 的药物记为 2 源 ----
    if faers is not None and "drug" in faers.columns:
        present = set(faers["drug"].astype(str).str.lower().str.strip().unique())
        for d in DRUG_REGISTRY:
            if d["key"] in present:
                node_src["DRUG:" + ot["drugs"][d["key"]]["id"]].add("faers")

    # ---- 4.5c AE 概念节点记录折叠前的全部原始事件名（可追溯） ----
    for aid, m in ae_meta.items():
        if aid in nodes:
            if m["faers_pts"]:
                nodes[aid]["faers_pt"] = " | ".join(sorted(m["faers_pts"]))
                nodes[aid]["n_faers_pt"] = len(m["faers_pts"])
            if m["ot_names"]:
                nodes[aid]["ot_name"] = " | ".join(sorted(m["ot_names"]))

    # ---- 4.5d 所有节点统一补齐 n_sources / sources ----
    for nid in nodes:
        ss = node_src.get(nid) or set()
        if ss:
            nodes[nid]["n_sources"] = len(ss)
            nodes[nid]["sources"] = "+".join(sorted(ss))

    # ---- 4.6 gene_ae（派生：靶点经药物中介关联事件）----
    drug_targets = defaultdict(set)
    drug_aes = defaultdict(set)
    for e in edges:
        if e["edge_type"] == "drug_target":
            drug_targets[e["src"]].add(e["dst"])
        elif e["edge_type"] == "drug_ae":
            drug_aes[e["src"]].add(e["dst"])
    derived = defaultdict(set)
    for dr, genes in drug_targets.items():
        for g in genes:
            for a in drug_aes.get(dr, ()):
                derived[(g, a)].add(dr)
    for (g, a), drs in derived.items():
        labs = ",".join(sorted(nodes[x]["name"] for x in drs if x in nodes))
        add_edge(g, a, "gene_ae", "derived", 1.0, via_drugs=labs, n_via_drugs=len(drs))
    log("[建图] gene_ae（派生）：%s 条边", human(len(derived)))

    # ---- 4.7 输出 ----
    ndf = pd.DataFrame(list(nodes.values()))
    for c in ["n_sources", "sources", "aria_concept", "event_group", "faers_pt", "ot_name",
              "ensembl_id", "pathway_id", "top_level", "chembl_id", "drug_type", "max_stage",
              "faers_key", "ot_id", "full_name", "meddra_code", "is_focus_disease"]:
        if c not in ndf.columns:
            ndf[c] = None
    ndf["attrs_json"] = ndf.apply(
        lambda r: json.dumps({k: v for k, v in r.items()
                              if k not in ("node_id", "node_type", "name", "label")
                              and v is not None and not (isinstance(v, float) and pd.isna(v))
                              and str(v) != "nan"}, ensure_ascii=False, default=str), axis=1)
    ndf = ndf[["node_id", "node_type", "name", "label", "n_sources", "sources",
               "aria_concept", "event_group", "faers_pt", "ot_name", "n_faers_pt",
               "meddra_code", "attrs_json"]]

    edf = pd.DataFrame(edges)
    if edf.empty:
        die("图为空，请检查数据源")
    edf["attrs_json"] = edf.apply(
        lambda r: json.dumps({k: v for k, v in r.items()
                              if k not in ("src", "dst", "edge_type", "source", "weight")
                              and v is not None and not (isinstance(v, float) and pd.isna(v))
                              and str(v) != "nan"}, ensure_ascii=False, default=str), axis=1)
    edf = edf[["src", "dst", "edge_type", "source", "weight", "attrs_json"]]
    return ndf, edf, ae_meta


# =============================================================================
# 5. 统计阶段
# =============================================================================

def stats_stage(ndf, edf):
    G = nx.DiGraph()
    for r in ndf.itertuples(index=False):
        G.add_node(r.node_id, ntype=r.node_type)
    for r in edf.itertuples(index=False):
        if G.has_edge(r.src, r.dst):
            G[r.src][r.dst]["weight"] += r.weight
        else:
            G.add_edge(r.src, r.dst, weight=r.weight, etype=r.edge_type)

    ntype = Counter(ndf["node_type"])
    etype = Counter(edf["edge_type"])
    deg = pd.Series(dict(G.degree()), dtype=float)
    hubs = []
    for nid, d in deg.sort_values(ascending=False).head(15).items():
        hubs.append({"node_id": nid, "name": str(ndf.loc[ndf.node_id == nid, "name"].iloc[0]),
                     "node_type": G.nodes[nid]["ntype"], "degree": int(d)})
    comps = list(nx.weakly_connected_components(G))
    comp_sizes = sorted((len(c) for c in comps), reverse=True)

    cross = []
    if "AdverseEvent" in set(ndf["node_type"]):
        ae = ndf[(ndf["node_type"] == "AdverseEvent")].copy()
        ae["n_src"] = pd.to_numeric(ae["n_sources"], errors="coerce").fillna(0).astype(int)

        def _attrs(js):
            try:
                return json.loads(js) if isinstance(js, str) and js else {}
            except Exception:  # noqa: BLE001
                return {}

        for r in ae.itertuples(index=False):
            nm = "None" if r.aria_concept is None else str(r.aria_concept)
            if r.n_src >= 2 or nm != "None":
                a_ = _attrs(r.attrs_json)
                cross.append({"ae_node": r.node_id, "label": r.label,
                              "aria_concept": r.aria_concept,
                              "event_group": r.event_group,
                              "n_sources": int(r.n_src), "sources": r.sources,
                              "n_faers_pt": a_.get("n_faers_pt"),
                              "faers_pt": a_.get("faers_pt"), "ot_name": a_.get("ot_name"),
                              "cross_source_confirmed": bool(int(r.n_src) >= 2)})

    st = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_nodes": int(len(ndf)), "n_edges": int(len(edf)),
        "nodes_by_type": {k: int(v) for k, v in sorted(ntype.items())},
        "edges_by_type": {k: int(v) for k, v in sorted(etype.items())},
        "graph": {
            "is_dag": bool(nx.is_directed_acyclic_graph(G)),
            "n_weakly_connected_components": len(comps),
            "largest_component_size": comp_sizes[0] if comp_sizes else 0,
            "top_component_sizes": comp_sizes[:10],
            "avg_degree": round(float(deg.mean()), 4) if len(deg) else 0.0,
            "median_degree": float(deg.median()) if len(deg) else 0.0,
            "max_degree": int(deg.max()) if len(deg) else 0,
            "density": round(float(nx.density(G)), 6),
        },
        "top_hubs": hubs,
        "aria_cross_source": cross,
    }
    return st, cross


# =============================================================================
# 6. 可视化（vis-network，单文件 HTML）
# =============================================================================

COLORS = {"Drug": "#d62728", "Gene": "#1f77b4", "Pathway": "#2ca02c",
          "Indication": "#9467bd", "AdverseEvent": "#ff7f0e"}


def viz_stage(ndf, edf, out_html):
    ns = []
    for r in ndf.itertuples(index=False):
        n_src = 0
        try:
            n_src = int(float(r.n_sources))
        except Exception:  # noqa: BLE001
            n_src = 0
        sz = {"Drug": 34, "Gene": 16, "Pathway": 12, "Indication": 26, "AdverseEvent": 24}.get(
            r.node_type, 14)
        if r.node_type == "Gene":
            sz = 12 + min(22, n_src * 0)
        if r.node_type == "Drug":
            sz = 30
        if r.node_type == "AdverseEvent" and n_src >= 2:
            sz = 32
        title = "<b>%s</b><br>type: %s" % (r.label, r.node_type)
        if r.aria_concept and str(r.aria_concept) != "None":
            title += "<br>ARIA concept: %s" % r.aria_concept
        if n_src:
            title += "<br>数据源数: %d (%s)" % (n_src, r.sources)
        ns.append({"id": r.node_id, "label": str(r.name)[:38],
                   "color": COLORS.get(r.node_type, "#888888"),
                   "size": sz, "group": r.node_type, "title": title,
                   "shape": "box" if r.node_type in ("Drug", "Indication") else (
                       "dot" if r.node_type != "AdverseEvent" else "triangle")})
    es = [{"from": r.src, "to": r.dst,
           "color": {"color": "#cccccc" if r.source != "faers" else "#ff9a9a"},
           "title": r.edge_type + " [" + r.source + "]"} for r in edf.itertuples(index=False)]
    payload = json.dumps({"nodes": ns, "edges": es}, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>AD anti-Abeta drug safety KG</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
 #net{width:100vw;height:100vh;display:block}
 #legend{position:fixed;left:12px;top:12px;background:rgba(255,255,255,.92);border:1px solid #ddd;
   border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.7;box-shadow:0 2px 8px rgba(0,0,0,.08)}
 #legend b{font-size:13px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
 #stat{position:fixed;right:12px;top:12px;background:rgba(255,255,255,.92);border:1px solid #ddd;
   border-radius:8px;padding:10px 12px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
</style></head><body>
<div id="net"></div>
<div id="legend"><b>节点类型</b>（勾选即过滤）<br>
 <label><input type="checkbox" class="f" value="Drug" checked><span class="sw" style="background:#d62728"></span>Drug 药物</label><br>
 <label><input type="checkbox" class="f" value="Gene" checked><span class="sw" style="background:#1f77b4"></span>Gene 基因/靶点</label><br>
 <label><input type="checkbox" class="f" value="Pathway"><span class="sw" style="background:#2ca02c"></span>Pathway 通路</label><br>
 <label><input type="checkbox" class="f" value="Indication" checked><span class="sw" style="background:#9467bd"></span>Indication 适应症</label><br>
 <label><input type="checkbox" class="f" value="AdverseEvent" checked><span class="sw" style="background:#ff7f0e"></span>AdverseEvent 不良事件</label><br>
 <span style="color:#ff9a9a">粉红边</span>=FAERS 信号 &nbsp; <span style="color:#bbb">灰边</span>=Open Targets/Reactome
</div>
<div id="stat"></div>
<script>
var DATA = __PAYLOAD__;
var ALL_NODES = DATA.nodes.slice(), ALL_EDGES = DATA.edges.slice();
var nodes = new vis.DataSet([]);
var edges = new vis.DataSet([]);
var container = document.getElementById('net');
var network = new vis.Network(container, {nodes:nodes, edges:edges}, {
  physics:{stabilization:{iterations:300}, barnesHut:{gravitationalConstant:-9000, springLength:130}},
  interaction:{hover:true, tooltipDelay:120, navigationButtons:true, keyboard:true},
  edges:{smooth:{type:'continuous'}, width:1}
});
function applyFilter(){
  var on = {};
  document.querySelectorAll('input.f').forEach(function(cb){ on[cb.value] = cb.checked; });
  var keep = ALL_NODES.filter(function(n){ return on[n.group] !== false; });
  var ids = {};
  keep.forEach(function(n){ ids[n.id] = 1; });
  var es = ALL_EDGES.filter(function(e){ return ids[e.from] && ids[e.to]; });
  nodes.clear(); nodes.add(keep); edges.clear(); edges.add(es);
  document.getElementById('stat').innerHTML =
    '<b>Knowledge Graph</b><br>nodes: ' + keep.length + ' / ' + ALL_NODES.length +
    '<br>edges: ' + es.length + ' / ' + ALL_EDGES.length;
  network.fit();
}
document.querySelectorAll('input.f').forEach(function(cb){
  cb.addEventListener('change', applyFilter);
});
applyFilter();
</script></body></html>
"""
    html = html.replace("__PAYLOAD__", payload)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    log("[输出] %s（%.1f KB）", out_html, out_html.stat().st_size / 1024)


# =============================================================================
# 7. 参数与主流程
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="多模态 AD 抗 Abeta 药物安全性知识图谱构建",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["fetch", "build", "stats", "viz", "all"])
    ap.add_argument("--proj", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-ad-genes", type=int, default=60)
    ap.add_argument("--min-a", type=int, default=3)
    ap.add_argument("--ae-criterion", default="robust",
                    choices=["robust", "any", "all"])
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    return ap.parse_args()


def main():
    args = parse_args()
    proj = Path(args.proj).resolve() if args.proj else find_proj_root()
    kgdir = proj / "data" / "processed" / "kg"
    kgdir.mkdir(parents=True, exist_ok=True)
    tdir = proj / "results" / "tables"
    fdir = proj / "results" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    log("=" * 74)
    log("多模态 AD 知识图谱构建  stage=%s", args.stage)
    log("项目根: %s", proj)
    log("AD 关联基因上限=%d | FAERS 最小报告数=%d | 事件判据=%s | 离线=%s",
        args.max_ad_genes, args.min_a, args.ae_criterion, args.offline)
    log("=" * 74)
    check_disk(proj, args.min_free_gb)

    np_path = kgdir / "nodes.parquet"
    ep_path = kgdir / "edges.parquet"

    if args.stage in ("build", "stats", "viz", "all") and np_path.exists() \
            and ep_path.exists() and args.stage != "all" and not args.force:
        log("[缓存] 复用已有 nodes.parquet / edges.parquet")
        ndf = pd.read_parquet(np_path)
        edf = pd.read_parquet(ep_path)
    else:
        ot, prov = load_ot_data(proj, args)
        log("[数据] 来源 = %s", prov)
        faers = load_faers(proj, args)
        ndf, edf, ae_meta = build_graph(ot, faers, args)
        log("[建图] 节点 %s | 边 %s", human(len(ndf)), human(len(edf)))
        for t, c in sorted(Counter(ndf["node_type"]).items()):
            log("  节点 %-14s %s", t, human(c))
        for t, c in sorted(Counter(edf["edge_type"]).items()):
            log("  边   %-14s %s", t, human(c))

        if args.stage in ("build", "stats", "viz", "all"):
            ndf.to_parquet(np_path, index=False)
            edf.to_parquet(ep_path, index=False)
            log("[输出] %s（%s 行）", np_path, human(len(ndf)))
            log("[输出] %s（%s 行）", ep_path, human(len(edf)))

            el = kgdir / "kg_edge_list.csv"
            with open(el, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["source_id", "source_type", "source_name",
                            "target_id", "target_type", "target_name",
                            "edge_type", "data_source", "weight"])
                nmap = dict(zip(ndf["node_id"], zip(ndf["node_type"], ndf["name"])))
                for r in edf.itertuples(index=False):
                    st_, sn = nmap.get(r.src, ("", ""))
                    tt_, tn = nmap.get(r.dst, ("", ""))
                    w.writerow([r.src, st_, sn, r.dst, tt_, tn, r.edge_type, r.source, r.weight])
            log("[输出] %s", el)

    if args.stage in ("stats", "viz", "all"):
        st, cross = stats_stage(ndf, edf)
        sp = kgdir / "kg_stats.json"
        sp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        log("[输出] %s", sp)
        log("图统计: 节点 %s | 边 %s | 连通分量 %d | 最大分量 %d | 平均度 %.2f",
            human(st["n_nodes"]), human(st["n_edges"]),
            st["graph"]["n_weakly_connected_components"],
            st["graph"]["largest_component_size"], st["graph"]["avg_degree"])

        if cross:
            cp = tdir / "kg_aria_cross_source.csv"
            pd.DataFrame(cross).to_csv(cp, index=False, encoding="utf-8-sig")
            log("[输出] %s（%d 条 ARIA 相关事件）", cp, len(cross))
            n_conf = sum(1 for c in cross if c["cross_source_confirmed"])
            log("        其中被 >=2 个独立数据源确认：%d 条", n_conf)

        if args.stage in ("viz", "all"):
            viz_stage(ndf, edf, fdir / "kg_overview.html")

    log("=" * 74)
    log("完成，用时 %.1f 秒", time.time() - _T0)


if __name__ == "__main__":
    main()
