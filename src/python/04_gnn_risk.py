#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_gnn_risk.py —— 机制感知的可解释 GNN 药物-事件风险预测与老药候选排序

背景
    第 2 章用四法不成比例分析（ROR / PRR / BCPNN / MGPS）在 FAERS 中确认了
    抗 Abeta 单抗的 ARIA 信号；第 3 章把这些信号与 Open Targets 的
    药物-靶点-通路-适应症证据集成为一张多模态知识图谱（5 类节点 / 6 类边）。

本章要回答的临床问题
    "哪些已上市老药可能与抗 Abeta 抗体的 ARIA 风险发生相互作用？"
    这一问决定了联合用药方案的安全性，也是老药新用最贴近临床的切入点。

做法
    1) 候选池：Open Targets 中 disease(Alzheimer disease).drugAndClinicalCandidates
       里 maximumClinicalStage = APPROVAL 的药物（即已上市、且与 AD 有关联证据的老药）。
    2) 扩展图：把候选药及其靶点 / 适应症 / 不良事件并入第 3 章的图，得到"扩展知识图谱"。
       不良事件沿用与第 3 章完全一致的归一化与 ARIA 概念折叠规则，保证口径统一。
    3) 建模：在扩展图上训练关系图神经网络（R-GCN）做药物-不良事件链路预测。
       正样本 = 已知 drug_ae 边，负样本 = 同类型随机采样；
       drug_ae 边按 8:1:1 划分 train / val / test，且测试边不参与消息传递，
       严格避免信息泄漏（这是链路预测最容易被审稿人抓住的地方）。
    4) 排序：用训练好的模型对"候选药 x ARIA 概念"打分排序，得到老药候选清单。
    5) 解释：靶点级留一归因（leave-one-target-out），给出每个预测背后的机制，
       把"可解释 GNN"落成可写进论文的具体证据，而不是黑箱分数。

产出
    data/interim/kg_ext/ot_cache/candidates.json         候选池原始响应
    data/interim/kg_ext/ot_cache/cand_<CHEMBL>.json      各候选药原始响应
    data/processed/kg_ext/nodes.parquet                  扩展图节点表
    data/processed/kg_ext/edges.parquet                  扩展图边表
    data/processed/kg_ext/kg_ext_stats.json              扩展图统计
    results/tables/gnn_metrics.json                      训练与评估指标
    results/tables/gnn_candidate_aria_ranking.csv        候选药 x ARIA 风险排序
    results/tables/gnn_mechanism_explanations.csv        靶点级机制解释
    results/tables/gnn_test_predictions.csv              测试集逐边预测
    results/figures/gnn_training_curve.png               训练曲线
    results/figures/gnn_roc_pr.png                       ROC / PR 曲线
    results/figures/gnn_candidate_heatmap.png            候选药 x ARIA 热力图

用法
    python 04_gnn_risk.py all     --proj /home/hyzyai/ad_antiab_kg
    python 04_gnn_risk.py fetch   --max-drugs 60          # 小规模试跑
    python 04_gnn_risk.py build
    python 04_gnn_risk.py train   --epochs 300
    python 04_gnn_risk.py rank    --top-k 40
    python 04_gnn_risk.py explain --top-k 30

依赖
    torch(>=2.0)、pandas、numpy、pyarrow、scikit-learn、matplotlib
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 0. 常量
# =============================================================================

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
AD_EFO = "MONDO_0004975"          # Open Targets: Alzheimer disease

NODE_TYPES = ["Drug", "Gene", "Pathway", "Indication", "AdverseEvent"]
EDGE_TYPES = ["drug_target", "drug_indication", "drug_ae",
              "gene_pathway", "gene_indication", "gene_ae"]
EDGE2IDX = {e: i for i, e in enumerate(EDGE_TYPES)}

# 与第 3 章完全一致的 7 个 ARIA 相关概念
ARIA_CONCEPTS = ["ARIA-E", "ARIA-H", "ARIA-NOS", "SUPERFICIAL_SIDEROSIS",
                 "CEREBRAL_MICROHEMORRHAGE", "CEREBRAL_HEMORRHAGE", "CEREBRAL_EDEMA"]

CONCEPT_LABEL = {
    "ARIA-E": "ARIA-E (amyloid-related imaging abnormality, edema/effusion)",
    "ARIA-H": "ARIA-H (amyloid-related imaging abnormality, microbleeds/hemosiderin)",
    "ARIA-NOS": "ARIA, unspecified",
    "SUPERFICIAL_SIDEROSIS": "Superficial siderosis of the CNS",
    "CEREBRAL_MICROHEMORRHAGE": "Cerebral microhemorrhage",
    "CEREBRAL_HEMORRHAGE": "Cerebral hemorrhage",
    "CEREBRAL_EDEMA": "Cerebral edema",
}

STAGE_RANK = {"UNKNOWN": 0, "PRECLINICAL": 1, "PHASE_1": 2, "EARLY_PHASE_1": 2,
              "PHASE_1_2": 3, "PHASE_2": 3, "PHASE_2_3": 4, "PHASE_3": 5, "APPROVAL": 6}

# CNS 关键词：仅用于给候选药打分层标签，不参与 GNN 训练
CNS_HINTS = ["alzheimer", "dementia", "parkinson", "epilep", "seizure", "depress",
             "schizophren", "bipolar", "anxiety", "migraine", "multiple sclerosis",
             "neuropath", "neurodegener", "huntington", "amyotrophic", "stroke",
             "cerebr", "brain", "cognit", "insomnia", "narcolep",
             "attention deficit", "autism", "ataxia", "dystonia", "neuralgia"]

Q_CANDIDATES = """
query { disease(efoId:"%s"){ drugAndClinicalCandidates{ count
  rows{ maxClinicalStage drug{ id name maximumClinicalStage drugType } } } } }
""" % AD_EFO

Q_DRUG = """
query D($id:String!){ drug(chemblId:$id){
  id name drugType maximumClinicalStage
  mechanismsOfAction{ rows{ actionType mechanismOfAction targetName
    targets{ id approvedSymbol approvedName } } }
  indications{ count rows{ id maxClinicalStage disease{ id name } } }
  adverseEvents{ count criticalValue rows{ meddraCode name count logLR } }
} }
"""


# =============================================================================
# 1. 基础工具（口径与第 3 章保持一致）
# =============================================================================

def log(fmt, *a):
    ts = datetime.now().strftime("%H:%M:%S")
    print(("[%s] " % ts) + (fmt % a if a else fmt), flush=True)


def die(fmt, *a):
    log("错误：" + (fmt % a if a else fmt))
    sys.exit(1)


def human(n):
    try:
        return "{:,}".format(int(n))
    except Exception:
        return str(n)


def slug(s):
    """与第 3 章 03_kg_build.py 的 slug 完全一致，保证节点 ID 可对齐。"""
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def norm_ae(name):
    """不良事件名称归一化：与第 3 章完全一致。"""
    n = (name or "").lower()
    n = n.replace("oedema", "edema").replace("haemorrhage", "hemorrhage")
    n = n.replace("haemosiderin", "hemosiderin").replace("haematoma", "hematoma")
    n = n.replace("oedematous", "edematous")
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def aria_concept(n):
    """ARIA 概念归类：与第 3 章 03_kg_build.py 完全一致。"""
    if "amyloid related imaging" in n or "aria" in n.split():
        if "edema" in n or "effusion" in n or "exudative" in n:
            return "ARIA-E"
        if ("microbleed" in n or "hemosiderin" in n or "iron deposition" in n
                or "microhemorrhage" in n or "hemorrhage" in n):
            return "ARIA-H"
        return "ARIA-NOS"
    if "superficial siderosis" in n:
        return "SUPERFICIAL_SIDEROSIS"
    if "cerebral microhemorrhage" in n:
        return "CEREBRAL_MICROHEMORRHAGE"
    if "cerebral hemorrhage" in n:
        return "CEREBRAL_HEMORRHAGE"
    if "brain edema" in n or "cerebral edema" in n or "vasogenic cerebral edema" in n:
        return "CEREBRAL_EDEMA"
    return None


def check_disk(path, min_free_gb):
    try:
        import shutil
        free = shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:
        return True
    if free < min_free_gb:
        die("剩余磁盘 %.1f GB < 阈值 %d GB，已中止", free, min_free_gb)
    return True


def find_proj_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for _ in range(8):
        if (p / "src").exists() or (p / "data").exists() or (p / "results").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(start or Path.cwd()).resolve()


def ot_query(query, variables=None, tries=6, timeout=90):
    """Open Targets GraphQL 查询。

    网络类异常（SSL EOF / 连接重置 / 读超时）做指数退避重试；
    业务错误（响应里带 errors 字段）不重试，直接抛出。
    """
    payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(
                OT_URL, data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                js = json.loads(resp.read().decode("utf-8"))
            if js.get("errors"):
                raise RuntimeError(str(js["errors"])[:300])
            return js
        except Exception as e:
            last = e
            time.sleep(min(2.0 ** k, 20.0))
    raise RuntimeError("Open Targets 查询失败（已重试 %d 次）：%s: %s"
                       % (tries, type(last).__name__, last))
# =============================================================================
# 2. fetch 阶段：候选药物池 + 逐药详情
# =============================================================================

def stage_fetch(proj, args):
    cdir = proj / "data" / "interim" / "kg_ext" / "ot_cache"
    cdir.mkdir(parents=True, exist_ok=True)

    # ---- 2.1 候选池 ----
    cp = cdir / "candidates.json"
    if args.force or not cp.exists():
        log("[抓取] AD 关联药物候选池 ...")
        js = ot_query(Q_CANDIDATES)
        cp.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    else:
        log("[缓存] 复用 %s", cp.name)
        js = json.loads(cp.read_text(encoding="utf-8"))

    rows = js["data"]["disease"]["drugAndClinicalCandidates"]["rows"]
    total = js["data"]["disease"]["drugAndClinicalCandidates"]["count"]
    log("[候选池] 与 AD 相关药物/候选共 %s 个", human(total))

    approved, seen = [], set()
    for r in rows:
        dr = r.get("drug") or {}
        if dr.get("maximumClinicalStage") != "APPROVAL":
            continue
        if dr.get("id") in seen:
            continue
        seen.add(dr.get("id"))
        approved.append(dr)
    approved.sort(key=lambda d: d.get("name") or "")
    if args.max_drugs:
        approved = approved[:args.max_drugs]
    log("[候选池] 已上市候选药 %s 个%s", human(len(approved)),
        "（--max-drugs 截断）" if args.max_drugs else "")
    if not approved:
        die("候选池为空，请检查 Open Targets 响应")

    # ---- 2.2 逐药详情（断点续跑）----
    ok, fail, skip = 0, 0, 0
    failed_ids = []
    for i, dr in enumerate(approved, 1):
        cid = dr["id"]
        p = cdir / ("cand_%s.json" % cid)
        if p.exists() and not args.force:
            # 缓存健全性检查：损坏 / 空 / 无 id 的缓存视为未抓取，重抓
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                c = None
            if isinstance(c, dict) and c.get("id"):
                skip += 1
                continue
            log("  [警告] %s 缓存损坏，改为重新抓取", cid)
        try:
            jj = ot_query(Q_DRUG, {"id": cid})
            drug = (jj.get("data") or {}).get("drug")
            if not drug or not drug.get("id"):
                fail += 1
                failed_ids.append(cid)
                continue
            p.write_text(json.dumps(drug, ensure_ascii=False), encoding="utf-8")
            ok += 1
        except Exception as e:
            fail += 1
            failed_ids.append(cid)
            log("  [警告] %s 抓取失败：%s", cid, str(e)[:120])
        if i % 25 == 0:
            log("  [抓取] 进度 %d/%d（成功 %d，失败 %d）", i, len(approved), ok, fail)
        time.sleep(0.05)

    (cdir.parent / "failed_drugs.json").write_text(
        json.dumps(failed_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[抓取] 完成：新抓 %s，复用 %s，失败 %s", human(ok), human(skip), human(fail))
    if failed_ids:
        log("[抓取] 有 %s 个药未取到 -> failed_drugs.json；重跑同一条命令会自动补抓",
            human(len(failed_ids)))

    # ---- 2.3 候选清单（含 CNS 标注）----
    recs = []
    for p in sorted(cdir.glob("cand_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        inds = (d.get("indications") or {}).get("rows") or []
        names = " ".join(((x.get("disease") or {}).get("name") or "").lower() for x in inds)
        cns = int(any(h in names for h in CNS_HINTS))
        recs.append({"chembl_id": d.get("id"), "name": d.get("name"),
                     "drug_type": d.get("drugType"),
                     "n_targets": len({t.get("approvedSymbol")
                                       for m in ((d.get("mechanismsOfAction") or {}).get("rows") or [])
                                       for t in (m.get("targets") or [])}),
                     "n_indications": (d.get("indications") or {}).get("count", 0),
                     "n_adverse_events": (d.get("adverseEvents") or {}).get("count", 0),
                     "is_cns": cns})
    base = proj / "data" / "interim" / "kg_ext"
    base.mkdir(parents=True, exist_ok=True)
    cdf = pd.DataFrame(recs).sort_values(["is_cns", "n_adverse_events"], ascending=False)
    out = base / "candidate_drugs.csv"
    cdf.to_csv(out, index=False)
    # 记录"本次选中"的药单，build 阶段据此建图，保证 --max-drugs 口径一致
    sel = [d["id"] for d in approved if (cdir / ("cand_%s.json" % d["id"])).exists()]
    (base / "selected_drugs.json").write_text(
        json.dumps(sel, ensure_ascii=False), encoding="utf-8")
    log("[清单] 候选药清单 -> %s（%s 个，其中 CNS 相关 %s 个）",
        out.name, human(len(cdf)), human(int(cdf["is_cns"].sum()) if len(cdf) else 0))
    return cdf


# =============================================================================
# 3. build 阶段：合并第 3 章 KG + 候选药 -> 扩展异质图
# =============================================================================

def _node_stub(nid, ntype, name, **attrs):
    rec = {"node_id": nid, "node_type": ntype, "name": name, "label": name}
    rec.update({k: v for k, v in attrs.items() if v is not None})
    return rec


def stage_build(proj, args):
    kg = proj / "data" / "processed" / "kg"
    npath, epath = kg / "nodes.parquet", kg / "edges.parquet"
    if not npath.exists() or not epath.exists():
        die("未找到第 3 章知识图谱（%s）。请先运行 03_kg_build.py all", kg)
    ndf = pd.read_parquet(npath)
    edf = pd.read_parquet(epath)
    log("[读入] 第3章 KG：节点 %s | 边 %s", human(len(ndf)), human(len(edf)))

    nodes = {r["node_id"]: dict(r) for r in ndf.to_dict("records")}
    edges = [dict(r) for r in edf.to_dict("records")]

    cdir = proj / "data" / "interim" / "kg_ext" / "ot_cache"
    cand_files = sorted(cdir.glob("cand_*.json"))
    if not cand_files:
        die("未找到候选药缓存（%s）。请先运行 fetch 阶段", cdir)
    # 严格按 fetch 阶段选定的药单建图（缓存里可能残留历次多抓的药）
    selp = proj / "data" / "interim" / "kg_ext" / "selected_drugs.json"
    if selp.exists():
        sel = {str(x) for x in json.loads(selp.read_text(encoding="utf-8"))}
        cand_files = [f for f in cand_files if f.name.replace("cand_", "").replace(".json", "") in sel]
        log("[建图] 按 selected_drugs.json 限定候选药 %s 个", human(len(cand_files)))
    if not cand_files:
        die("选中药单为空，请检查 fetch 阶段")

    stat = Counter()
    new_drugs = []
    for f in cand_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        cid = d.get("id")
        if not cid:
            continue
        did = "DRUG:" + cid
        if did in nodes:
            continue                                   # 3 个靶药已在图中
        nodes[did] = _node_stub(did, "Drug", d.get("name"),
                                chembl_id=cid, drug_type=d.get("drugType"),
                                max_stage=d.get("maximumClinicalStage"),
                                sources="opentargets", n_sources=1, candidate=True)
        stat["drug"] += 1
        new_drugs.append(d.get("name"))

        # --- 靶点 ---
        for m in ((d.get("mechanismsOfAction") or {}).get("rows") or []):
            for t in (m.get("targets") or []):
                gid = "GENE:" + t["id"]
                if gid not in nodes:
                    nodes[gid] = _node_stub(gid, "Gene",
                                            t.get("approvedSymbol") or t["id"],
                                            ensembl_id=t["id"], full_name=t.get("approvedName"),
                                            sources="opentargets", n_sources=1)
                    stat["gene"] += 1
                edges.append({"src": did, "dst": gid, "edge_type": "drug_target",
                              "source": "opentargets", "weight": 1.0,
                              "action_type": m.get("actionType"),
                              "mechanism": m.get("mechanismOfAction"),
                              "target_name": m.get("targetName")})
                stat["edge_drug_target"] += 1

        # --- 适应症（按临床阶段取前 N，避免图爆炸）---
        inds = ((d.get("indications") or {}).get("rows") or [])
        inds = sorted(inds, key=lambda x: STAGE_RANK.get(x.get("maxClinicalStage"), 0),
                      reverse=True)[:args.max_ind]
        for ind in inds:
            dis = ind.get("disease") or {}
            if not dis.get("name"):
                continue
            iid = "IND:" + slug(dis["name"])
            if iid not in nodes:
                nodes[iid] = _node_stub(iid, "Indication", dis["name"],
                                        ot_id=dis.get("id"), sources="opentargets", n_sources=1)
                stat["indication"] += 1
            st = ind.get("maxClinicalStage")
            edges.append({"src": did, "dst": iid, "edge_type": "drug_indication",
                          "source": "opentargets",
                          "weight": float(STAGE_RANK.get(st, 0) or 1.0),
                          "max_stage": st})
            stat["edge_drug_indication"] += 1

        # --- 不良事件（trial 事件的归一化 + ARIA 折叠，与第 3 章同规则）---
        aes = ((d.get("adverseEvents") or {}).get("rows") or [])
        aes = sorted(aes, key=lambda x: (x.get("count") or 0), reverse=True)[:args.max_ae]
        for ae in aes:
            nm = ae.get("name") or ""
            n = norm_ae(nm)
            concept = aria_concept(n)
            aid = "AE:" + (concept if concept else "RAW:" + n)
            if aid not in nodes:
                nodes[aid] = _node_stub(aid, "AdverseEvent", CONCEPT_LABEL.get(concept, nm),
                                        aria_concept=concept, meddra_code=ae.get("meddraCode"),
                                        ot_name=nm, sources="opentargets", n_sources=1)
                stat["adverse_event"] += 1
            edges.append({"src": did, "dst": aid, "edge_type": "drug_ae",
                          "source": "opentargets", "weight": float(ae.get("count") or 1),
                          "ot_count": ae.get("count"), "ot_logLR": ae.get("logLR"),
                          "ot_name": nm, "aria_concept": concept})
            stat["edge_drug_ae"] += 1

    # ---- 输出 ----
    ncols = ["node_id", "node_type", "name", "label", "n_sources", "sources",
             "aria_concept", "event_group", "faers_pt", "ot_name", "n_faers_pt",
             "meddra_code", "attrs_json"]
    ndf2 = pd.DataFrame(list(nodes.values()))
    for c in ncols:
        if c not in ndf2.columns:
            ndf2[c] = None
    ndf2["attrs_json"] = ndf2.apply(
        lambda r: json.dumps({k: v for k, v in r.items()
                              if k not in ("node_id", "node_type", "name", "label", "attrs_json")
                              and v is not None and str(v) != "nan"},
                             ensure_ascii=False, default=str), axis=1)
    ndf2 = ndf2[ncols]

    ecols = ["src", "dst", "edge_type", "source", "weight", "attrs_json"]
    edf2 = pd.DataFrame(edges)
    for c in ecols:
        if c not in edf2.columns:
            edf2[c] = None
    edf2["attrs_json"] = edf2.apply(
        lambda r: json.dumps({k: v for k, v in r.items()
                              if k not in ("src", "dst", "edge_type", "source", "weight", "attrs_json")
                              and v is not None and str(v) != "nan"},
                             ensure_ascii=False, default=str), axis=1)
    edf2 = edf2[ecols]

    # ---- 派生 gene_ae（保证与第 3 章同构，候选药带来的新靶点也参与）----
    d2t, d2a = defaultdict(set), defaultdict(set)
    for e in edges:
        if e["edge_type"] == "drug_target":
            d2t[e["src"]].add(e["dst"])
        elif e["edge_type"] == "drug_ae":
            d2a[e["src"]].add(e["dst"])
    derived = set()
    for dr, gs in d2t.items():
        for g in gs:
            for a in d2a.get(dr, ()):
                derived.add((g, a))
    have = {(e["src"], e["dst"]) for e in edges if e["edge_type"] == "gene_ae"}
    add_rows = []
    for (g, a) in derived:
        if (g, a) in have:
            continue
        add_rows.append({"src": g, "dst": a, "edge_type": "gene_ae",
                         "source": "derived", "weight": 1.0})
    if add_rows:
        edf2 = pd.concat([edf2, pd.DataFrame(add_rows)], ignore_index=True)
        for c in ecols:
            if c not in edf2.columns:
                edf2[c] = None
        edf2["attrs_json"] = edf2["attrs_json"].fillna(json.dumps({}, ensure_ascii=False))
        edf2 = edf2[ecols]
    stat["edge_gene_ae"] = len(add_rows)

    outdir = proj / "data" / "processed" / "kg_ext"
    outdir.mkdir(parents=True, exist_ok=True)
    ndf2.to_parquet(outdir / "nodes.parquet", index=False)
    edf2.to_parquet(outdir / "edges.parquet", index=False)

    ntc = Counter(ndf2["node_type"])
    etc = Counter(edf2["edge_type"])
    stats = {"nodes": int(len(ndf2)), "edges": int(len(edf2)),
             "node_type": dict(ntc), "edge_type": dict(etc),
             "added": {k: int(v) for k, v in stat.items()},
             "n_candidate_drugs": int(stat["drug"]),
             "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (outdir / "kg_ext_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    log("[建图] 新增候选药 %s 个；扩展图：节点 %s | 边 %s",
        human(stat["drug"]), human(len(ndf2)), human(len(edf2)))
    for t, c in sorted(ntc.items()):
        log("   节点 %-13s %s", t, human(c))
    for t, c in sorted(etc.items()):
        log("   边   %-16s %s", t, human(c))
    log("[输出] %s", outdir)
    return ndf2, edf2


# =============================================================================
# 4. train 阶段：R-GCN 链路预测
# =============================================================================

TARGET_DRUG_IDS = ("DRUG:CHEMBL3833321", "DRUG:CHEMBL4297245", "DRUG:CHEMBL3039540")

# 不良事件大类（用于构造与标签无关的节点特征）
FEAT_AE_GROUPS = ["HEMORRHAGE", "EDEMA", "NEURO", "GI", "CARDIO", "METAB",
                  "INFUSION", "OTHER", "ARIA", "UNK"]

# 与 drug_ae 无关的关系类型：只有这些关系能用来做节点特征，否则会引入标签泄漏
STRUCT_RELS = ("drug_target", "drug_indication", "gene_pathway", "gene_indication")


AE_GROUP_KEYWORDS = [
    ("HEMORRHAGE", ("haemorrhag", "hemorrhag", "haematoma", "hematoma", "bleed",
                    "haematuria", "hematuria", "epistaxis", "melena", "melaena",
                    "siderosis", "purpura", "petechi", "haemosider", "hemosider")),
    ("EDEMA", ("oedema", "edema", "effusion", "swelling")),
    ("NEURO", ("headache", "dizz", "confus", "somnol", "seizure", "paraesth",
               "paresth", "tremor", "ataxia", "insomnia", "syncope", "dysarthr",
               "aphasia", "neuropath", "encephal", "cognit", "memory", "fall",
               "lethargy", "depress", "anxiety", "somnolen")),
    ("GI", ("nausea", "vomit", "diarrh", "constipat", "dyspeps", "abdominal",
            "gastr", "colitis", "dysphag")),
    ("CARDIO", ("cardiac", "atrial", "bradycard", "tachycard", "hypertens",
                "hypotens", "palpitat", "arrhythm", "angina", "myocard")),
    ("METAB", ("hyponatraem", "hyponatrem", "hyperkalaem", "hyperkalem",
               "hyperglyc", "hypoglyc", "weight ", "dehydrat", "acidosis")),
    ("INFUSION", ("infusion", "pyrexia", "chill", "injection site", "rash",
                  "urticar", "prurit", "hypersensitiv")),
]


def _infer_ae_group(text, is_aria):
    """按事件名推断大类。候选药的 AE 节点没有 event_group 字段，
    若不推断就只能落进 UNK，节点特征会退化。"""
    if is_aria:
        return "ARIA"
    t = (text or "").lower()
    for grp, kws in AE_GROUP_KEYWORDS:
        for kw in kws:
            if kw in t:
                return grp
    return "OTHER"


def _build_features(ndf, edf, idx):
    """构造**不依赖 drug_ae / gene_ae** 的节点特征。

    为什么必须排除这两类边：drug_ae 就是被预测的标签本身，gene_ae 又是从
    drug_ae 派生而来（同一个药物的靶点与它的事件共现），二者都会把答案带进特征。
    这里只用药物-靶点、药物-适应症、基因-通路、基因-适应症四类关系，
    加上节点固有属性（类型、事件大类、ARIA 概念），这些在预测某条 drug_ae
    边时本来就是已知信息，因此不构成泄漏。
    """
    n = len(ndf)
    nid = list(ndf["node_id"])
    ntype = ndf["node_type"].tolist()

    deg = np.zeros(n, dtype=np.float64)
    n_tgt = np.zeros(n, dtype=np.float64)
    n_ind = np.zeros(n, dtype=np.float64)
    for s, d, t in zip(edf["src"], edf["dst"], edf["edge_type"]):
        if t not in STRUCT_RELS:
            continue
        si, di = idx.get(s), idx.get(d)
        if si is None or di is None:
            continue
        deg[si] += 1.0
        deg[di] += 1.0
        if t == "drug_target":
            n_tgt[si] += 1.0
        elif t == "drug_indication":
            n_ind[si] += 1.0

    cols = [np.log1p(deg), np.log1p(n_tgt), np.log1p(n_ind)]
    is_drug = np.array([t == "Drug" for t in ntype], dtype=np.float64)
    is_ae = np.array([t == "AdverseEvent" for t in ntype], dtype=np.float64)
    is_gene = np.array([t == "Gene" for t in ntype], dtype=np.float64)
    is_ind = np.array([t == "Indication" for t in ntype], dtype=np.float64)
    is_path = np.array([t == "Pathway" for t in ntype], dtype=np.float64)
    cols += [is_drug, is_ae, is_gene, is_ind, is_path]
    is_target_drug = np.array([x in TARGET_DRUG_IDS for x in nid], dtype=np.float64)
    is_cand = np.array([(t == "Drug") and (x not in TARGET_DRUG_IDS)
                        for t, x in zip(ntype, nid)], dtype=np.float64)
    cols += [is_target_drug, is_cand]

    grp = np.zeros((n, len(FEAT_AE_GROUPS)), dtype=np.float64)
    con = np.zeros((n, len(ARIA_CONCEPTS)), dtype=np.float64)
    gmap = {g: i for i, g in enumerate(FEAT_AE_GROUPS)}
    cmap = {c: i for i, c in enumerate(ARIA_CONCEPTS)}

    def _clean(v):
        return None if (v is None or str(v) == "nan" or str(v) == "None") else str(v)

    eg_col = ndf["event_group"].tolist() if "event_group" in ndf.columns else [None] * n
    ac_col = ndf["aria_concept"].tolist() if "aria_concept" in ndf.columns else [None] * n
    for i in range(n):
        if ntype[i] != "AdverseEvent":
            continue
        e = _clean(eg_col[i])
        c = _clean(ac_col[i])
        gi = gmap.get(e.upper()) if e else None
        if gi is None or not e:
            # 候选药的事件节点没有 event_group，就按事件名归类；仍归不出才落 UNK
            raw = ""
            for col in ("ot_name", "faers_pt", "name", "label"):
                if col in ndf.columns:
                    v = _clean(ndf[col].tolist()[i])
                    if v:
                        raw = v
                        break
            if not raw:
                raw = _clean(nid[i]) or ""
            gi = gmap.get(_infer_ae_group(raw, bool(c)), gmap["UNK"])
        grp[i, gi] = 1.0
        if c in cmap:
            con[i, cmap[c]] = 1.0
    for j in range(len(FEAT_AE_GROUPS)):
        cols.append(grp[:, j])
    for j in range(len(ARIA_CONCEPTS)):
        cols.append(con[:, j])

    feat = np.stack(cols, axis=1).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _build_tensors(ndf, edf, seed=42, split_mode="random", test_frac=0.1):
    """把节点/边表转成模型可用的索引与边表，并切分 drug_ae 边。"""
    ids = list(ndf["node_id"])
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    ntype = ndf["node_type"].tolist()

    src, dst, et = [], [], []
    for e in edf.to_dict("records"):
        s, d = idx.get(e["src"]), idx.get(e["dst"])
        if s is None or d is None:
            continue
        if e["edge_type"] not in EDGE2IDX:
            continue
        src.append(s); dst.append(d); et.append(EDGE2IDX[e["edge_type"]])
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    et = np.asarray(et, dtype=np.int64)

    m = et == EDGE2IDX["drug_ae"]
    da = np.unique(np.stack([src[m], dst[m]], axis=1), axis=0)

    rng = np.random.RandomState(seed)
    if split_mode == "drug":
        # 整药留出：测试集里的药在训练时完全没出现过 drug_ae 边，
        # 比随机边划分严格得多，用来检验"对新药也能预测"
        drugs_in = np.unique(da[:, 0]).tolist()
        rng.shuffle(drugs_in)
        n_te = max(1, int(round(0.2 * len(drugs_in))))
        n_va = max(1, int(round(0.1 * len(drugs_in))))
        te_d = set(drugs_in[:n_te])
        va_d = set(drugs_in[n_te:n_te + n_va])
        te = da[np.isin(da[:, 0], list(te_d))]
        va = da[np.isin(da[:, 0], list(va_d))]
        tr = da[~np.isin(da[:, 0], list(te_d | va_d))]
    else:
        perm = rng.permutation(len(da))
        n_va = int(test_frac * len(da))
        n_tr = len(da) - 2 * n_va
        tr = da[perm[:n_tr]]
        va = da[perm[n_tr:n_tr + n_va]]
        te = da[perm[n_tr + n_va:]]

    feat = _build_features(ndf, edf, idx)
    return dict(ids=ids, idx=idx, n=n, ntype=np.array(ntype),
                src=src, dst=dst, et=et, da=da, train=tr, val=va, test=te, feat=feat)


def _make_adj(src, dst, et, n_rel, drop_pairs=None):
    """构造每种关系的 (src, dst) 索引对（双向传播）。

    drop_pairs 对**任意关系**都生效：训练时用于屏蔽 val/test 的 drug_ae 边（防泄漏），
    归因时用于屏蔽某条 drug_target 边（看靶点贡献）。
    """
    adj = []
    drop = set()
    if drop_pairs is not None and len(drop_pairs):
        drop = {(int(a), int(b)) for a, b in drop_pairs}
    for r in range(n_rel):
        m = et == r
        pairs = [(int(a), int(b)) for a, b in zip(src[m], dst[m])]
        if drop:
            pairs = [pp for pp in pairs if pp not in drop]
        if pairs:
            # 双向传播：药物节点本身没有入边，若只做 src->dst，药物表示将完全不含其靶点信息
            s = [pp[0] for pp in pairs] + [pp[1] for pp in pairs]
            d = [pp[1] for pp in pairs] + [pp[0] for pp in pairs]
            s = np.array(s, dtype=np.int64); d = np.array(d, dtype=np.int64)
        else:
            s = np.zeros(0, dtype=np.int64)
            d = np.zeros(0, dtype=np.int64)
        adj.append((s, d))
    return adj


def _make_adj_train(G, allowed_da, drop_pairs=None):
    """v2 核心修复：构造不含测试信息的邻接表。

    第 3 章的 gene_ae 边是由**全量** drug_ae 派生的（"某药的靶点与该药的事件共现"），
    如果直接沿用，测试边会以 药物-基因-事件 的二跳路径泄漏进消息传递，
    让链路预测的评估结果失去意义。这里改为：gene_ae 只由 allowed_da 重新派生。
    """
    n_rel = len(EDGE_TYPES)
    drop = set()
    if drop_pairs is not None and len(drop_pairs):
        drop = {(int(a), int(b)) for a, b in drop_pairs}
    allowed = {(int(a), int(b)) for a, b in allowed_da}

    d2t = defaultdict(set)
    for s, d, t in zip(G["src"], G["dst"], G["et"]):
        if t == EDGE2IDX["drug_target"] and (int(s), int(d)) not in drop:
            d2t[int(s)].add(int(d))
    d2a = defaultdict(set)
    for dr, ae in allowed:
        d2a[dr].add(ae)
    gene_ae = set()
    for dr, gs in d2t.items():
        for g in gs:
            for a in d2a.get(dr, ()):
                gene_ae.add((g, a))

    adj = []
    for r, ename in enumerate(EDGE_TYPES):
        if ename == "drug_ae":
            pairs = sorted(allowed)
        elif ename == "gene_ae":
            pairs = sorted(gene_ae)
        else:
            mask = G["et"] == r
            pairs = [(int(a), int(b)) for a, b in zip(G["src"][mask], G["dst"][mask])]
            if drop:
                pairs = [pp for pp in pairs if pp not in drop]
        if pairs:
            s = [pp[0] for pp in pairs] + [pp[1] for pp in pairs]
            d = [pp[1] for pp in pairs] + [pp[0] for pp in pairs]
            adj.append((np.array(s, dtype=np.int64), np.array(d, dtype=np.int64)))
        else:
            adj.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
    return adj


def _sample_neg(n_neg, drug_glob, ae_glob, pos_set, rng):
    """同类型负采样：随机 (drug, ae) 全局索引对且不在正样本中。"""
    neg = []
    while len(neg) < n_neg:
        d = int(drug_glob[rng.randint(0, len(drug_glob))])
        a = int(ae_glob[rng.randint(0, len(ae_glob))])
        if (d, a) in pos_set:
            continue
        neg.append((d, a))
    return np.array(neg, dtype=np.int64)


def _boot_ci(y, sc, n_boot=1000, seed=0):
    """自助法给出 AUC-ROC / AUC-PR 的 95% 置信区间。"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    rs = np.random.RandomState(seed)
    aucs, aps = [], []
    for _ in range(n_boot):
        i = rs.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], sc[i]))
        aps.append(average_precision_score(y[i], sc[i]))
    if not aucs:
        nan = float("nan")
        return (nan, nan), (nan, nan)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))), \
           (float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5)))


def stage_train(proj, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from collections import Counter as _C
    from sklearn.metrics import roc_auc_score, average_precision_score

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    kgd = proj / "data" / "processed" / "kg_ext"
    npath, epath = kgd / "nodes.parquet", kgd / "edges.parquet"
    if not npath.exists() or not epath.exists():
        die("未找到扩展图（%s）。请先运行 build 阶段", kgd)
    ndf = pd.read_parquet(npath)
    edf = pd.read_parquet(epath)
    log("[读入] 扩展图：节点 %s | 边 %s", human(len(ndf)), human(len(edf)))

    G = _build_tensors(ndf, edf, seed=args.seed, split_mode=args.split)
    n, n_rel = G["n"], len(EDGE_TYPES)
    feat_dim = int(G["feat"].shape[1])
    log("[数据] drug_ae 边 %s 条 -> train %s / val %s / test %s（划分=%s）",
        human(len(G["da"])), human(len(G["train"])), human(len(G["val"])),
        human(len(G["test"])), args.split)
    if len(G["train"]) < 20:
        die("drug_ae 正样本过少（%d），无法训练；请增大候选药物池", len(G["train"]))

    drug_ids = np.where(G["ntype"] == "Drug")[0]
    ae_ids = np.where(G["ntype"] == "AdverseEvent")[0]
    log("[数据] 药物节点 %s | 不良事件节点 %s | 节点特征维 %d",
        human(len(drug_ids)), human(len(ae_ids)), feat_dim)

    tr_pairs = G["train"]; va_pairs = G["val"]; te_pairs = G["test"]
    pos_all = {(int(a), int(b)) for a, b in G["da"]}

    class RGCNLayer(nn.Module):
        def __init__(self, dim, n_rel, dropout):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_rel, dim, dim))
            nn.init.xavier_uniform_(self.W)
            self.loop = nn.Linear(dim, dim, bias=False)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, adj):
            out = self.loop(x)
            for r, (s, d) in enumerate(adj):
                if len(s) == 0:
                    continue
                st = torch.as_tensor(s, dtype=torch.long, device=x.device)
                dt = torch.as_tensor(d, dtype=torch.long, device=x.device)
                msg = x[st] @ self.W[r]
                deg = torch.zeros(x.size(0), device=x.device)
                deg.index_add_(0, dt, torch.ones_like(dt, dtype=x.dtype))
                deg = deg.clamp(min=1.0)
                msg = msg / deg[dt].unsqueeze(1)
                out.index_add_(0, dt, msg)
            return self.drop(F.relu(self.norm(out)))

    class RGCN(nn.Module):
        """R-GCN：节点表示 = 可学习嵌入 + 固有特征投影。

        v1 只有随机初始化嵌入，图上没有任何"标签无关"的信号可用，
        在小图上必然退化为记忆训练边；v2 把节点固有特征投进表示，
        让模型能利用靶点/事件类别这类可在预测时刻获得的信息。
        """

        def __init__(self, n_nodes, n_rel, dim, layers, dropout, feat):
            super().__init__()
            self.emb = nn.Embedding(n_nodes, dim)
            nn.init.xavier_uniform_(self.emb.weight)
            self.feat = nn.Parameter(torch.as_tensor(feat, dtype=torch.float32),
                                     requires_grad=False)
            self.fproj = nn.Linear(self.feat.size(1), dim, bias=False)
            nn.init.xavier_uniform_(self.fproj.weight)
            self.layers = nn.ModuleList([RGCNLayer(dim, n_rel, dropout) for _ in range(layers)])
            self.dec = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(dim, 1))

        def encode(self, adj):
            x = self.emb.weight + self.fproj(self.feat)
            for L in self.layers:
                x = L(x, adj)
            return x

        def score(self, z, pairs_global):
            g = torch.as_tensor(pairs_global, dtype=torch.long, device=z.device)
            return self.dec(torch.cat([z[g[:, 0]], z[g[:, 1]]], dim=-1)).squeeze(-1)

    torch.manual_seed(args.seed)
    model = RGCN(n, n_rel, args.dim, args.layers, args.dropout, G["feat"]).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-5)
    log("[模型] R-GCN 可训练参数 %s | dim=%d layers=%d device=%s",
        human(sum(p.numel() for p in trainable)), args.dim, args.layers, args.device)

    def _to_t(adj):
        return [(torch.as_tensor(s, dtype=torch.long, device=args.device),
                 torch.as_tensor(d, dtype=torch.long, device=args.device)) for s, d in adj]

    rng = np.random.RandomState(args.seed)
    best = {"auc": -1.0, "epoch": -1, "state": None}
    hist = []
    bad = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        adj = _to_t(_make_adj_train(G, tr_pairs))
        z = model.encode(adj)
        neg_g = _sample_neg(len(tr_pairs), drug_ids, ae_ids, pos_all, rng)
        logits = torch.cat([model.score(z, tr_pairs), model.score(z, neg_g)])
        y = torch.cat([torch.ones(len(tr_pairs)), torch.zeros(len(neg_g))]).to(args.device)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                zz = model.encode(adj)
                vn = _sample_neg(len(va_pairs), drug_ids, ae_ids, pos_all, rng)
                sc = torch.cat([model.score(zz, va_pairs), model.score(zz, vn)]).cpu().numpy()
                yy = np.concatenate([np.ones(len(va_pairs)), np.zeros(len(vn))])
                auc = roc_auc_score(yy, sc); ap = average_precision_score(yy, sc)
            hist.append({"epoch": ep, "loss": float(loss.item()),
                         "val_auc": float(auc), "val_ap": float(ap)})
            if auc > best["auc"] + 1e-6:
                best = {"auc": float(auc), "epoch": ep,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            if ep % (args.eval_every * 5) == 0 or ep == args.epochs:
                log("  epoch %4d | loss %.4f | val AUC %.4f | val AP %.4f",
                    ep, loss.item(), auc, ap)
            if bad >= args.patience:
                log("  [早停] 连续 %d 次评估无提升，于 epoch %d 停止", args.patience, ep)
                break

    # ---- 测试集评估：消息传递只含训练边（val/test 全部不可见）----
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        zz = model.encode(_to_t(_make_adj_train(G, tr_pairs)))
        rng2 = np.random.RandomState(args.seed + 1)
        tn = _sample_neg(len(te_pairs), drug_ids, ae_ids, pos_all, rng2)
        sc_pos = model.score(zz, te_pairs).cpu().numpy()
        sc_neg = model.score(zz, tn).cpu().numpy()

    y = np.concatenate([np.ones(len(sc_pos)), np.zeros(len(sc_neg))])
    sc = np.concatenate([sc_pos, sc_neg])
    test_auc = float(roc_auc_score(y, sc)); test_ap = float(average_precision_score(y, sc))
    (ci_lo, ci_hi), (p_lo, p_hi) = _boot_ci(y, sc, n_boot=args.n_boot, seed=args.seed)

    # ---- 流行度基线（只用训练边统计，不含任何测试信息）----
    c_ae = _C(int(b) for a, b in tr_pairs)
    c_dr = _C(int(a) for a, b in tr_pairs)
    pairs_te = np.concatenate([te_pairs, tn])
    b_ae = np.array([c_ae.get(int(p[1]), 0) for p in pairs_te], dtype=np.float64)
    b_dr = np.array([c_dr.get(int(p[0]), 0) for p in pairs_te], dtype=np.float64)
    b_both = b_ae / (b_ae.max() or 1.0) + b_dr / (b_dr.max() or 1.0)
    auc_ae = float(roc_auc_score(y, b_ae))
    auc_dr = float(roc_auc_score(y, b_dr))
    auc_both = float(roc_auc_score(y, b_both))
    prevalence = float(len(sc_pos) / len(y))

    log("[评估] test AUC-ROC = %.4f（95%%CI %.4f-%.4f）| AUC-PR = %.4f（95%%CI %.4f-%.4f）",
        test_auc, ci_lo, ci_hi, test_ap, p_lo, p_hi)
    log("[基线] 随机 0.5000 | 事件流行度 %.4f | 药物流行度 %.4f | 两者合并 %.4f | 阳性率 %.4f",
        auc_ae, auc_dr, auc_both, prevalence)

    # ---- 结构诊断：直接回答"这张图到底有没有可学的信号" ----
    cold = int(np.sum([c_ae.get(int(p[1]), 0) == 0 for p in te_pairs]))
    log("[诊断] 测试正样本中，事件在训练集里从未出现过的占 %d/%d（%.1f%%）——这类边"
        "在消息传递里是完全孤立的节点，结构上无法预测，是 AUC 的天花板",
        cold, len(te_pairs), 100.0 * cold / max(1, len(te_pairs)))
    n_drugs_with_pos = len(np.unique(G["da"][:, 0]))
    per_drug = len(tr_pairs) / max(1, n_drugs_with_pos)
    log("[诊断] 有 drug_ae 边的药物 %d 个，平均每药训练边 %.1f 条；"
        "候选药的事件画像越趋同，可学信号越弱", n_drugs_with_pos, per_drug)
    log("[评估] 最佳 epoch %d，val AUC %.4f", best["epoch"], best["auc"])
    best_base = max(auc_ae, auc_dr, auc_both)
    if test_auc < best_base + 0.02:
        log("[提醒] 模型未明显优于流行度基线：此规模下图信号不足以支撑链路预测结论。"
            "建议 ① 放开 --max-drugs 用全量候选药；② 加 --split drug 复核泛化。")

    # ---- 保存 ----
    outdir = proj / "models"; outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"n_nodes": n, "n_rel": n_rel, "dim": args.dim,
                           "layers": args.layers, "dropout": args.dropout,
                           "feat_dim": feat_dim},
                "ids": G["ids"]}, outdir / "gnn_rgcn.pt")

    tdir = proj / "results" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame({"src": [G["ids"][int(a)] for a, b in te_pairs],
                        "dst": [G["ids"][int(b)] for a, b in te_pairs],
                        "type": "positive", "score": sc_pos})
    tdf = pd.concat([tdf, pd.DataFrame({"src": [G["ids"][int(a)] for a, b in tn],
                                        "dst": [G["ids"][int(b)] for a, b in tn],
                                        "type": "negative", "score": sc_neg})],
                    ignore_index=True)
    tdf.to_csv(tdir / "gnn_test_predictions.csv", index=False)

    metrics = {"n_nodes": int(n), "n_edges": int(len(edf)),
               "n_drug_nodes": int(len(drug_ids)), "n_ae_nodes": int(len(ae_ids)),
               "n_drug_ae_pos": int(len(G["da"])),
               "n_train": int(len(tr_pairs)), "n_val": int(len(va_pairs)),
               "n_test": int(len(te_pairs)),
               "split": args.split, "feat_dim": int(feat_dim),
               "best_epoch": int(best["epoch"]), "best_val_auc": float(best["auc"]),
               "test_auc_roc": test_auc, "test_auc_pr": test_ap,
               "test_auc_roc_ci95": [ci_lo, ci_hi], "test_auc_pr_ci95": [p_lo, p_hi],
               "baseline_auc_ae_pop": auc_ae, "baseline_auc_drug_pop": auc_dr,
               "baseline_auc_both": auc_both, "prevalence": prevalence,
               "dim": args.dim, "layers": args.layers, "seed": args.seed,
               "history": hist, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (tdir / "gnn_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    log("[输出] %s / gnn_metrics.json", tdir)
    return metrics


# =============================================================================
# 5. rank 阶段：候选药 x ARIA 概念风险排序
# =============================================================================

def _load_model(proj, args):
    import torch
    ck = proj / "models" / "gnn_rgcn.pt"
    if not ck.exists():
        die("未找到模型 %s，请先运行 train 阶段", ck)

    class RGCNLayer(torch.nn.Module):
        def __init__(self, dim, n_rel, dropout):
            super().__init__()
            self.W = torch.nn.Parameter(torch.empty(n_rel, dim, dim))
            self.loop = torch.nn.Linear(dim, dim, bias=False)
            self.norm = torch.nn.LayerNorm(dim)
            self.drop = torch.nn.Dropout(dropout)

        def forward(self, x, adj):
            import torch.nn.functional as F
            out = self.loop(x)
            for r, (s, d) in enumerate(adj):
                if len(s) == 0:
                    continue
                st = torch.as_tensor(s, dtype=torch.long, device=x.device)
                dt = torch.as_tensor(d, dtype=torch.long, device=x.device)
                msg = x[st] @ self.W[r]
                deg = torch.zeros(x.size(0), device=x.device)
                deg.index_add_(0, dt, torch.ones_like(dt, dtype=x.dtype))
                deg = deg.clamp(min=1.0)
                msg = msg / deg[dt].unsqueeze(1)
                out.index_add_(0, dt, msg)
            return self.drop(F.relu(self.norm(out)))

    class RGCN(torch.nn.Module):
        def __init__(self, n_nodes, n_rel, dim, layers, dropout, feat_dim):
            super().__init__()
            self.emb = torch.nn.Embedding(n_nodes, dim)
            self.feat = torch.nn.Parameter(torch.zeros(n_nodes, feat_dim),
                                           requires_grad=False)
            self.fproj = torch.nn.Linear(feat_dim, dim, bias=False)
            self.layers = torch.nn.ModuleList([RGCNLayer(dim, n_rel, dropout)
                                               for _ in range(layers)])
            self.dec = torch.nn.Sequential(torch.nn.Linear(dim * 2, dim), torch.nn.ReLU(),
                                           torch.nn.Dropout(dropout), torch.nn.Linear(dim, 1))

        def encode(self, adj):
            x = self.emb.weight + self.fproj(self.feat)
            for L in self.layers:
                x = L(x, adj)
            return x

        def score(self, z, pairs_global):
            g = torch.as_tensor(pairs_global, dtype=torch.long, device=z.device)
            return self.dec(torch.cat([z[g[:, 0]], z[g[:, 1]]], dim=-1)).squeeze(-1)

    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = RGCN(cfg["n_nodes"], cfg["n_rel"], cfg["dim"], cfg["layers"],
                 cfg["dropout"], cfg.get("feat_dim", 0))
    model.load_state_dict(blob["state_dict"])
    model.to(args.device).eval()
    return model, blob["ids"]


def _recompute_z(model, G, args, drop_edges=None):
    """重算节点表示。邻接表用 _make_adj_train：gene_ae 由给定 drug_ae 重新派生，
    避免把被预测的边以二跳路径混进消息传递。"""
    import torch
    adj = _make_adj_train(G, G["da"], drop_pairs=drop_edges)
    with torch.no_grad():
        z = model.encode([(torch.as_tensor(s, dtype=torch.long, device=args.device),
                           torch.as_tensor(d, dtype=torch.long, device=args.device))
                          for s, d in adj])
    return z


def stage_rank(proj, args):
    import torch
    kgd = proj / "data" / "processed" / "kg_ext"
    ndf = pd.read_parquet(kgd / "nodes.parquet")
    edf = pd.read_parquet(kgd / "edges.parquet")
    G = _build_tensors(ndf, edf, seed=args.seed)
    model, ids = _load_model(proj, args)
    idx = {nid: i for i, nid in enumerate(ids)}

    drug_ids = [i for i, t in enumerate(G["ntype"]) if t == "Drug"]
    cand = ndf[(ndf["node_type"] == "Drug")].copy()
    cns_map = {}
    cinfo = proj / "data" / "interim" / "kg_ext" / "candidate_drugs.csv"
    if cinfo.exists():
        for r in pd.read_csv(cinfo).to_dict("records"):
            cns_map["DRUG:" + str(r["chembl_id"])] = int(r.get("is_cns") or 0)

    z = _recompute_z(model, G, args)
    rows = []
    for _, r in cand.iterrows():
        did = r["node_id"]
        if did not in idx:
            continue
        for con in ARIA_CONCEPTS:
            aid = "AE:" + con
            if aid not in idx:
                continue
            gp = np.array([[idx[did], idx[aid]]], dtype=np.int64)
            s = float(torch.sigmoid(model.score(z, gp)).detach().cpu().numpy()[0])
            rows.append({"drug": r["name"], "drug_id": did,
                         "chembl_id": r.get("chembl_id"),
                         "aria_concept": con, "concept_label": CONCEPT_LABEL.get(con, con),
                         "risk_score": round(s, 6),
                         "is_cns": cns_map.get(did, 0),
                         "is_target_drug": int(did in ("DRUG:CHEMBL3833321", "DRUG:CHEMBL4297245",
                                                       "DRUG:CHEMBL3039540"))})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        die("排序表为空：扩展图中缺少 ARIA 概念节点，请检查 build 阶段")
    rdf = rdf.sort_values(["aria_concept", "risk_score"], ascending=[True, False])
    rdf["rank_in_concept"] = rdf.groupby("aria_concept")["risk_score"]\
        .rank(method="first", ascending=False).astype(int)
    # 去偏：模型对"报告量大的药"天然给高分，故补充概念内标准化分与超额分
    rdf["z_in_concept"] = rdf.groupby("aria_concept")["risk_score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    rdf["excess_score"] = rdf.groupby("aria_concept")["risk_score"].transform(
        lambda s: s - s.mean())
    rdf["rank_by_z"] = rdf.groupby("aria_concept")["z_in_concept"]\
        .rank(method="first", ascending=False).astype(int)
    _da_cnt = {}
    for _a, _b in G["da"]:
        _da_cnt[int(_a)] = _da_cnt.get(int(_a), 0) + 1
    rdf["n_known_ae"] = [int(_da_cnt.get(idx.get(str(r["drug_id"]), -1), 0))
                         for _, r in rdf.iterrows()]
    tdir = proj / "results" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(tdir / "gnn_candidate_aria_ranking.csv", index=False)
    log("[排序] 候选药 x ARIA -> %s（%s 行）", "gnn_candidate_aria_ranking.csv", human(len(rdf)))

    # 控制台摘要：每个概念 top5 非靶药候选
    for con in ARIA_CONCEPTS:
        sub = rdf[(rdf["aria_concept"] == con) & (rdf["is_target_drug"] == 0)].head(5)
        if len(sub):
            log("  %-24s top: %s", con,
                "; ".join("%s(%.3f)" % (x["drug"][:22], x["risk_score"]) for _, x in sub.iterrows()))

    # 热力图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        fp = None
        for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
                  "/System/Library/Fonts/PingFang.ttc"]:
            if os.path.exists(p):
                fp = font_manager.FontProperties(fname=p); break
        top_drugs = (rdf[rdf["is_target_drug"] == 0].groupby("drug")["risk_score"]
                     .max().sort_values(ascending=False).head(args.top_k).index.tolist())
        mat = np.zeros((len(top_drugs), len(ARIA_CONCEPTS)))
        for i, dn in enumerate(top_drugs):
            for j, con in enumerate(ARIA_CONCEPTS):
                v = rdf[(rdf["drug"] == dn) & (rdf["aria_concept"] == con)]["risk_score"]
                mat[i, j] = float(v.iloc[0]) if len(v) else 0.0
        fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.30 * len(top_drugs) + 1.8)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.6, mat.max()))
        ax.set_xticks(range(len(ARIA_CONCEPTS)))
        ax.set_xticklabels(ARIA_CONCEPTS, rotation=35, ha="right",
                           fontproperties=fp, fontsize=8)
        ax.set_yticks(range(len(top_drugs)))
        ax.set_yticklabels(top_drugs, fontproperties=fp, fontsize=8)
        ax.set_title("Candidate approved drugs x ARIA concept: predicted risk",
                     fontproperties=fp, fontsize=12)
        fig.colorbar(im, ax=ax, label="risk score")
        fig.tight_layout()
        fdir = proj / "results" / "figures"; fdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fdir / "gnn_candidate_heatmap.png", dpi=160)
        plt.close(fig)
        log("[图] %s", fdir / "gnn_candidate_heatmap.png")
    except Exception as e:
        log("[提示] 热力图生成失败（不影响结果）：%s", str(e)[:150])

    return rdf


# =============================================================================
# 6. explain 阶段：靶点级留一归因
# =============================================================================

def stage_explain(proj, args):
    import torch
    kgd = proj / "data" / "processed" / "kg_ext"
    ndf = pd.read_parquet(kgd / "nodes.parquet")
    edf = pd.read_parquet(kgd / "edges.parquet")
    G = _build_tensors(ndf, edf, seed=args.seed)
    model, ids = _load_model(proj, args)
    idx = {nid: i for i, nid in enumerate(ids)}
    name_of = dict(zip(ndf["node_id"], ndf["name"]))

    tdir = proj / "results" / "tables"
    rfile = tdir / "gnn_candidate_aria_ranking.csv"
    if not rfile.exists():
        die("未找到排序表，请先运行 rank 阶段")
    rdf = pd.read_csv(rfile)

    # 取每个概念风险最高的若干非靶药做解释
    kk = max(1, args.top_k // len(ARIA_CONCEPTS))
    sub = rdf[rdf["is_target_drug"] == 0].sort_values("risk_score", ascending=False)
    targets = sub.groupby("aria_concept", group_keys=False).head(kk)
    pairs = targets[["drug_id", "aria_concept"]].drop_duplicates()

    # 该药的全部 drug_target 边（全局索引）
    da_r = EDGE2IDX["drug_target"]
    dt = {}
    for s, d, t in zip(G["src"], G["dst"], G["et"]):
        if t == da_r:
            dt.setdefault(int(s), []).append(int(d))
    ae_r = EDGE2IDX["drug_ae"]
    da_edges = [(int(s), int(d)) for s, d, t in zip(G["src"], G["dst"], G["et"]) if t == ae_r]

    z0 = _recompute_z(model, G, args)
    rows = []
    for _, r in pairs.iterrows():
        did, con = r["drug_id"], r["aria_concept"]
        if did not in idx:
            continue
        aid = "AE:" + con
        if aid not in idx:
            continue
        di, ai = idx[did], idx[aid]
        base = float(model.score(z0, np.array([[di, ai]])).detach().cpu().numpy()[0])   # logit
        tgts = dt.get(di, [])
        for gi in tgts:
            # 留一：屏蔽该药的这一条 drug_target 边，用 logit 变化衡量靶点贡献
            drop = [(di, gi)]
            z1 = _recompute_z(model, G, args, drop_edges=drop)
            s1 = float(model.score(z1, np.array([[di, ai]])).detach().cpu().numpy()[0])
            rows.append({"drug": name_of.get(did, did), "drug_id": did,
                         "aria_concept": con,
                         "base_prob": round(float(1.0 / (1.0 + np.exp(-base))), 6),
                         "base_logit": round(base, 5),
                         "target_symbol": name_of.get(ids[gi], ids[gi]),
                         "target_node": ids[gi],
                         "mask_logit": round(s1, 5),
                         "attribution_logit_delta": round(base - s1, 5)})
    edf_out = pd.DataFrame(rows)
    if edf_out.empty:
        log("[解释] 未生成机制解释（候选药缺少靶点边）")
        return edf_out
    edf_out = edf_out.sort_values(["aria_concept", "drug", "attribution_logit_delta"],
                                  ascending=[True, True, False])
    edf_out.to_csv(tdir / "gnn_mechanism_explanations.csv", index=False)
    log("[解释] 靶点级机制解释 -> gnn_mechanism_explanations.csv（%s 行）", human(len(edf_out)))
    for _, r in edf_out.groupby(["drug", "aria_concept"], group_keys=False).head(1).head(5).iterrows():
        log("   %s x %s：主要贡献靶点 %s（logit Δ=%.4f）",
            str(r["drug"])[:22], r["aria_concept"], r["target_symbol"],
            r["attribution_logit_delta"])
    return edf_out


# =============================================================================
# 7. 训练曲线 / ROC-PR 图（复用 metrics.json）
# =============================================================================

def stage_figures(proj, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fp = None
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"]:
        if os.path.exists(p):
            fp = font_manager.FontProperties(fname=p); break

    mf = proj / "results" / "tables" / "gnn_metrics.json"
    if not mf.exists():
        log("[提示] 未找到 gnn_metrics.json，跳过训练曲线")
        return
    m = json.loads(mf.read_text(encoding="utf-8"))
    hist = m.get("history") or []
    fdir = proj / "results" / "figures"; fdir.mkdir(parents=True, exist_ok=True)

    if hist:
        ep = [h["epoch"] for h in hist]
        fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
        ax1.plot(ep, [h["loss"] for h in hist], color="#d62728", label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="#d62728")
        ax2 = ax1.twinx()
        ax2.plot(ep, [h["val_auc"] for h in hist], color="#1f77b4", label="val AUC")
        ax2.set_ylabel("val AUC-ROC", color="#1f77b4")
        ax1.set_title("R-GCN training curve", fontproperties=fp)
        fig.tight_layout(); fig.savefig(fdir / "gnn_training_curve.png", dpi=160); plt.close(fig)
        log("[图] %s", fdir / "gnn_training_curve.png")

    pf = proj / "results" / "tables" / "gnn_test_predictions.csv"
    if pf.exists():
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        df = pd.read_csv(pf)
        y = (df["type"] == "positive").astype(int).values
        sc = df["score"].values
        fpr, tpr, _ = roc_curve(y, sc)
        pr, rc, _ = precision_recall_curve(y, sc)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
        axes[0].plot(fpr, tpr, color="#1f77b4")
        axes[0].plot([0, 1], [0, 1], "--", color="#999999")
        axes[0].set_title("ROC (AUC=%.3f)" % auc(fpr, tpr), fontproperties=fp)
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[1].plot(rc, pr, color="#2ca02c")
        axes[1].set_title("Precision-Recall (AP=%.3f)" % auc(rc, pr), fontproperties=fp)
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        fig.tight_layout(); fig.savefig(fdir / "gnn_roc_pr.png", dpi=160); plt.close(fig)
        log("[图] %s", fdir / "gnn_roc_pr.png")


# =============================================================================
# 8. 主流程
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="机制感知的可解释 GNN 风险预测与老药候选排序")
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["fetch", "build", "train", "rank", "explain", "figures", "all"])
    ap.add_argument("--proj", default=None, help="项目根目录（默认自动查找）")
    ap.add_argument("--max-drugs", type=int, default=None, help="候选药上限（默认全部已上市）")
    ap.add_argument("--max-ind", type=int, default=12, help="每药保留适应症上限")
    ap.add_argument("--max-ae", type=int, default=30, help="每药保留不良事件上限")
    ap.add_argument("--dim", type=int, default=128, help="GNN 隐层维度")
    ap.add_argument("--layers", type=int, default=3, help="R-GCN 层数")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--split", default="random", choices=["random", "drug"],
                    help="drug_ae 划分：random=随机边；drug=整药留出（更严格）")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="测试集 AUC 自助法重采样次数")
    ap.add_argument("--patience", type=int, default=40,
                    help="验证集连续无提升的早停耐心（评估次数）")
    ap.add_argument("--force", action="store_true", help="忽略缓存强制重抓")
    return ap.parse_args()


def main():
    args = parse_args()
    proj = Path(args.proj).resolve() if args.proj else find_proj_root()
    log("=" * 74)
    log("项目根：%s", proj)
    log("阶段：%s | 设备：%s | 随机种子：%d", args.stage, args.device, args.seed)
    check_disk(proj, args.min_free_gb)

    if args.stage in ("fetch", "all"):
        stage_fetch(proj, args)
    if args.stage in ("build", "all"):
        stage_build(proj, args)
    if args.stage in ("train", "all"):
        stage_train(proj, args)
    if args.stage in ("rank", "all"):
        stage_rank(proj, args)
    if args.stage in ("explain", "all"):
        stage_explain(proj, args)
    if args.stage in ("figures", "all"):
        stage_figures(proj, args)

    log("=" * 74)
    log("完成")


if __name__ == "__main__":
    main()
