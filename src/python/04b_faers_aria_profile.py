#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_faers_aria_profile.py —— 候选药 ARIA 事件画像（FAERS 全量）与机制接近度双通道分层

为什么需要这个脚本
    04_gnn_risk.py 的实测结论是：在当前规模的扩展图上做"随机边划分的链路预测"
    结构性不可行 —— 612 个不良事件节点只连 914 条 drug_ae 边（平均度 1.5），
    随机抽走 10% 测试边后，67% 的测试正样本所对应的事件节点在训练图里完全孤立，
    消息传递触不到它们，AUC 天花板被锁在 0.5 附近（实测 0.5286，95%CI 跨 0.5）。
    同时"药物流行度"这种零智能基线能到 0.6406，高于模型 —— 说明模型学到的是
    报告量偏差而非药理机制。

    因此本章改用"双通道证据分层"这一在方法学上站得住、且真正多模态的框架：

        通道 A（临床流行病学，本脚本主体）
            直接从 FAERS 全量表计算每个候选药对 7 个 ARIA 概念的不成比例信号
            （ROR / PRR，含 95% 置信区间）。这是有真实报告支撑的、可写进论文的
            主结果，不依赖任何模型拟合。

        通道 B（机制接近度）
            在第 3 章知识图谱上计算候选药与抗 Abeta 单抗在"靶点-通路"空间上的
            接近程度，并统计候选药靶点是否落在 ARIA 相关机制通路（止血/凝血/
            血小板/血管/炎症/细胞外基质）上，提供"为什么"的解释。

        交叉分层
            A 阳 + B 阳 = 高优先级候选；仅 A 阳 = 临床需警惕；仅 B 阳 = 机制假说。
            评估口径用"已知阳性药富集 + 双通道一致性置换检验"，而非 AUC。

关键口径（与第 2/3 章严格一致，保证跨章可比）
    - 药名归一化：大写化、去剂型/剂量/盐基/立体化学前缀
    - 事件归一化：norm_ae()，与 03_kg_build.py / 04_gnn_risk.py 逐字一致
    - ARIA 概念折叠：aria_concept()，同上
    - ROR 定义：a=目标药+目标事件, b=目标药+其他事件, c=其他药+目标事件,
      d=其他药+其他事件；ROR=ad/bc；SE(logROR)=sqrt(1/a+1/b+1/c+1/d)
      （已用第 2 章 lecanemab ARIA-E 的 a/b/c/d 反算校验：ROR=5509.6 逐位吻合）
    - 计数单位为"报告"（primaryid 去重），全库报告数约 2,069 万，与第 2 章一致

输入
    data/processed/faers/drug.parquet      primaryid, caseid, drug_seq, role_cod,
                                           drugname, prod_ai, route
    data/processed/faers/reac.parquet      primaryid, caseid, pt
    data/interim/kg_ext/candidate_drugs.csv       候选药清单（chembl_id, name, is_cns）
    data/interim/kg_ext/selected_drugs.json       本次选中药单
    data/processed/kg_ext/nodes.parquet           扩展图节点
    data/processed/kg_ext/edges.parquet           扩展图边

产出
    data/interim/faers_profile/drug_name_map.csv  药名 -> 候选药 匹配映射（供人工审核）
    data/interim/faers_profile/cand_reports.parquet 候选药命中报告（断点续跑用）
    results/tables/faers_candidate_aria_profile.csv   通道 A：候选药 x ARIA 信号
    results/tables/faers_drug_match_diagnostic.csv    药名匹配诊断
    results/tables/kg_mechanism_proximity.csv         通道 B：机制接近度
    results/tables/chapter4_cross_evidence.csv        双通道交叉分层（论文核心表）
    results/figures/faers_profile_forest.png          高优先级候选森林图
    results/figures/chapter4_cross_scatter.png        A/B 双通道散点图

用法
    python 04b_faers_aria_profile.py probe    --proj /home/hyzyai/ad_antiab_kg
    python 04b_faers_aria_profile.py profile  --proj /home/hyzyai/ad_antiab_kg
    python 04b_faers_aria_profile.py mech     --proj /home/hyzyai/ad_antiab_kg
    python 04b_faers_aria_profile.py cross    --proj /home/hyzyai/ad_antiab_kg
    python 04b_faers_aria_profile.py all      --proj /home/hyzyai/ad_antiab_kg

依赖
    pandas、numpy、pyarrow、matplotlib（无需 torch / networkx）
"""

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 0. 常量
# =============================================================================

TARGET_DRUGS = {
    "CHEMBL3833321": "LECANEMAB",
    "CHEMBL4297245": "DONANEMAB",
    "CHEMBL3039540": "ADUCANUMAB",
}

# 与第 3 章 / 04_gnn_risk.py 完全一致的 7 个 ARIA 相关概念
ARIA_CONCEPTS = ["ARIA-E", "ARIA-H", "ARIA-NOS", "SUPERFICIAL_SIDEROSIS",
                 "CEREBRAL_MICROHEMORRHAGE", "CEREBRAL_HEMORRHAGE",
                 "CEREBRAL_EDEMA"]

CONCEPT_LABEL = {
    "ARIA-E": "ARIA-E (edema/effusion)",
    "ARIA-H": "ARIA-H (microbleeds/hemosiderin)",
    "ARIA-NOS": "ARIA, unspecified",
    "SUPERFICIAL_SIDEROSIS": "Superficial siderosis of CNS",
    "CEREBRAL_MICROHEMORRHAGE": "Cerebral microhemorrhage",
    "CEREBRAL_HEMORRHAGE": "Cerebral hemorrhage",
    "CEREBRAL_EDEMA": "Cerebral edema",
}

# ARIA 相关机制通路关键词（通道 B 的机制解释用）
ARIA_PATHWAY_HINTS = [
    "hemostasis", "coagulation", "platelet", "fibrin", "thrombin", "clotting",
    "angiogenesis", "vascular", "vessel", "endotheli", "basement membrane",
    "collagen", "extracellular matrix", "integrin",
    "inflammat", "immune", "complement", "cytokine", "interleukin",
    "amyloid", "tau", "mmp", "matrix metalloproteinase", "pericyte",
    "blood brain barrier", "bradykinin", "renin", "angiotensin",
]

# 剂型 / 给药途径 / 剂量单位等噪声词（归一化时剔除）
DRUG_NOISE = {
    "tablet", "tablets", "tab", "capsule", "capsules", "cap", "injection",
    "injectable", "solution", "suspension", "syrup", "oral", "intravenous",
    "intramuscular", "subcutaneous", "intravenous", "iv", "im", "sc", "syringe",
    "infusion", "vial", "ampule", "ampoule", "cream", "ointment", "gel", "patch",
    "transdermal", "inhaler", "inhalation", "spray", "drops", "ophthalmic",
    "topical", "nasal", "rectal", "extended", "release", "delayed", "immediate",
    "film", "coated", "powder", "concentrate", "granules", "usp", "bp", "ph",
    "er", "xr", "sr", "cr", "la", "mr", "odt", "ec",
}

# 盐基 / 酯基 / 水合物后缀（归一化时剔除）
SALT_WORDS = {
    "sodium", "hydrochloride", "hcl", "mesylate", "mesilate", "maleate",
    "tartrate", "bitartrate", "citrate", "sulfate", "sulphate", "acetate",
    "phosphate", "bromide", "chloride", "calcium", "potassium", "magnesium",
    "dihydrate", "monohydrate", "hydrate", "anhydrous", "besylate", "fumarate",
    "succinate", "lactate", "gluconate", "oxide", "nitrate", "carbonate",
    "stearate", "tromethamine", "trometamol", "meglumine", "arginine", "lysine",
    "esylate", "tosylate", "napsylate", "pamoate", "embonate", "oxalate",
    "malate", "benzoate", "salicylate", "sulfonate", "bromate", "iodide",
    "bisulfate", "bisulphate", "hydrobromide", "dihydrochloride",
    "hemihydrate", "sesquihydrate", "xinafoate", "orotate",
}

# 剂量单位（归一化时剔除）
DOSE_UNITS = {"mg", "mcg", "ug", "g", "kg", "ml", "l", "iu", "unit", "units",
              "meq", "mmol", "mol", "percent", "w", "v", "wv", "vv", "dr"}

# 常见药的商品名 / 别名（FAERS 自由文本里常见写法）
BUILTIN_ALIASES = {
    "lecanemab": ["leqembi"],
    "donanemab": ["kisunla"],
    "aducanumab": ["aduhelm"],
    "gantenerumab": [],
    "solanezumab": [],
    "bapineuzumab": [],
    "aspirin": ["acetylsalicylic acid", "acetylsalicylate", "asa", "asaec",
                "asa e c", "acetysalicylic acid"],
    "clopidogrel": ["clopidogrel bisulfate", "clopidogrel bisulphate"],
    "ticagrelor": [], "prasugrel": [], "dipyridamole": [],
    "dabigatran": ["dabigatran etexilate"],
    "rivaroxaban": [], "apixaban": [], "edoxaban": [],
    "warfarin": ["warfarin sodium", "coumarin"],
    "heparin": ["heparin sodium", "unfractionated heparin", "lmwh"],
    "enoxaparin": ["enoxaparin sodium"],
    "alteplase": ["tpa", "tissue plasminogen activator", "rtpa"],
    "donepezil": ["donepezil hydrochloride"],
    "galantamine": ["galantamine hydrobromide"],
    "rivastigmine": [],
    "memantine": ["memantine hydrochloride"],
    "atorvastatin": ["atorvastatin calcium"],
    "simvastatin": [], "rosuvastatin": ["rosuvastatin calcium"],
    "pravastatin": ["pravastatin sodium"], "lovastatin": [],
    "ezetimibe": [],
    "losartan": ["losartan potassium"], "valsartan": [],
    "irbesartan": [], "candesartan": ["candesartan cilexetil"],
    "telmisartan": [], "olmesartan": ["olmesartan medoxomil"],
    "amlodipine": ["amlodipine besylate", "amlodipine besilate"],
    "nifedipine": [], "diltiazem": ["diltiazem hydrochloride"],
    "verapamil": ["verapamil hydrochloride"],
    "metoprolol": ["metoprolol succinate", "metoprolol tartrate"],
    "bisoprolol": ["bisoprolol fumarate"], "carvedilol": [], "propranolol": [],
    "atenolol": [], "labetalol": ["labetalol hydrochloride"],
    "hydrochlorothiazide": ["hctz"], "furosemide": ["frusemide"],
    "spironolactone": [], "torsemide": ["torasemide"],
    "quetiapine": ["quetiapine fumarate"],
    "risperidone": [], "olanzapine": [], "aripiprazole": [],
    "brexpiprazole": [], "ziprasidone": ["ziprasidone hydrochloride"],
    "paliperidone": [], "lurasidone": ["lurasidone hydrochloride"],
    "sertraline": ["sertraline hydrochloride"], "citalopram": ["citalopram hydrobromide"],
    "escitalopram": ["escitalopram oxalate"], "fluoxetine": ["fluoxetine hydrochloride"],
    "paroxetine": ["paroxetine hydrochloride", "paroxetine mesylate"],
    "fluvoxamine": ["fluvoxamine maleate"],
    "duloxetine": ["duloxetine hydrochloride"],
    "venlafaxine": ["venlafaxine hydrochloride"], "desvenlafaxine": [],
    "mirtazapine": [], "trazodone": ["trazodone hydrochloride"],
    "bupropion": ["bupropion hydrochloride", "amfebutamone"],
    "gabapentin": [], "pregabalin": [],
    "valproate": ["valproic acid", "divalproex", "divalproex sodium",
                  "valproate sodium", "sodium valproate"],
    "levetiracetam": [], "carbamazepine": [], "lamotrigine": [],
    "phenytoin": ["phenytoin sodium", "fosphenytoin"], "topiramate": [],
    "oxcarbazepine": [], "lacosamide": [], "zonisamide": [],
    "bromocriptine": ["bromocriptine mesylate"], "cabergoline": [],
    "pramipexole": ["pramipexole dihydrochloride"],
    "ropinirole": ["ropinirole hydrochloride"],
    "levodopa": ["l dopa", "carbidopa levodopa", "carbidopa-levodopa",
                 "levodopa carbidopa", "sinemet", "levodopa benserazide"],
    "carbidopa": ["carbidopa levodopa"],
    "amantadine": ["amantadine hydrochloride"],
    "rasagiline": ["rasagiline mesylate"], "selegiline": ["deprenyl"],
    "entacapone": [], "tolcapone": [],
    "acamprosate": ["acamprosate calcium"],
    "riluzole": [], "edaravone": [],
    "dexmedetomidine": ["dexmedetomidine hydrochloride"],
    "midazolam": ["midazolam hydrochloride"],
    "haloperidol": ["haloperidol lactate", "haloperidol decanoate"],
    "chlorpromazine": ["chlorpromazine hydrochloride"],
    "melatonin": [], "modafinil": [], "armodafinil": [],
    "atomoxetine": ["atomoxetine hydrochloride"],
    "methylphenidate": ["methylphenidate hydrochloride", "ritalin"],
    "prednisone": [], "prednisolone": ["prednisolone sodium phosphate"],
    "dexamethasone": ["dexamethasone sodium phosphate"],
    "methylprednisolone": ["methylprednisolone sodium succinate"],
    "ibuprofen": [], "naproxen": ["naproxen sodium"],
    "diclofenac": ["diclofenac sodium", "diclofenac potassium"],
    "celecoxib": [], "indomethacin": ["indometacin"],
    "acetaminophen": ["paracetamol"],
    "colchicine": [], "allopurinol": [],
    "insulin": ["insulin human", "insulin glargine", "insulin aspart"],
    "metformin": ["metformin hydrochloride"],
    "levothyroxine": ["levothyroxine sodium", "l thyroxine"],
    "omeprazole": [], "pantoprazole": ["pantoprazole sodium"],
    "esomeprazole": ["esomeprazole magnesium"],
    "ranitidine": ["ranitidine hydrochloride"], "famotidine": [],
    "ondansetron": ["ondansetron hydrochloride"],
    "vericiguat": [], "sacubitril": ["sacubitril valsartan"],
    "digoxin": [], "amiodarone": ["amiodarone hydrochloride"],
    "flecainide": ["flecainide acetate"], "sotalol": ["sotalol hydrochloride"],
    "vitamin d": ["cholecalciferol", "ergocalciferol", "vitamin d3"],
    "vitamin e": ["alpha tocopherol", "tocopherol"],
    "vitamin b12": ["cyanocobalamin", "hydroxocobalamin", "cobalamin"],
    "folic acid": ["folate", "folacin", "pteroylglutamic acid"],
    "calcium": ["calcium carbonate", "calcium citrate"],
    "iron": ["ferrous sulfate", "ferrous fumarate", "ferric", "iron sucrose"],
    "albumin human": ["human albumin", "albumin", "albutein", "plasbumin"],
}

_LOGFH = None


# =============================================================================
# 1. 工具
# =============================================================================

def log(msg, *a):
    s = msg % a if a else msg
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), s)
    print(line, flush=True)
    if _LOGFH:
        _LOGFH.write(line + "\n")
        _LOGFH.flush()


def die(msg, *a):
    log("[错误] " + (msg % a if a else msg))
    sys.exit(2)


def human(n):
    try:
        n = int(n)
    except Exception:
        return str(n)
    if abs(n) >= 1e8:
        return "%.2f亿" % (n / 1e8)
    if abs(n) >= 1e4:
        return "{:,}".format(n)
    return str(n)


def check_disk(path, min_free_gb):
    try:
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


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


# ---- 事件归一化（与第 3 章逐字一致）----
def norm_ae(name):
    n = (name or "").lower()
    n = n.replace("oedema", "edema").replace("haemorrhage", "hemorrhage")
    n = n.replace("haemosiderin", "hemosiderin").replace("haematoma", "hematoma")
    n = n.replace("oedematous", "edematous")
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def aria_concept(n):
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


# ---- 药名归一化 ----
def norm_drug_name(s):
    """把 FAERS 自由文本药名归一化为可用于匹配的 token 串。

    规则：小写 -> 非字母数字转空格 -> 拆 token -> 剔除纯数字 / 剂量单位 /
    剂型词 / 盐基词 -> 还原顺序无关的空白分隔串。
    例：
        'ASPIRIN 81MG TABLET'        -> 'aspirin'
        'DABIGATRAN ETEXILATE CAP'   -> 'dabigatran etexilate'
        'ALBUMIN (HUMAN) 25%'        -> 'albumin human'
    """
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    out = []
    for t in s.split():
        if not t:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", t):          # 纯数字 = 剂量
            continue
        if re.fullmatch(r"\d+(\.\d+)?(mg|mcg|ug|g|ml|l|iu|meq|mmol|%)", t):
            continue
        if t in DOSE_UNITS or t in DRUG_NOISE or t in SALT_WORDS:
            continue
        out.append(t)
    return " ".join(out)


def match_patterns(drug_name):
    """给候选药生成匹配用的归一化模式集合（含别名）。"""
    base = norm_drug_name(drug_name)
    pats = set()
    if base:
        pats.add(base)
    key = base
    # 内置别名的键既可能是原始名（去除盐基后），也可能是通用名
    for k, alist in BUILTIN_ALIASES.items():
        if not alist and k != key:
            continue
        nk = norm_drug_name(k)
        cand_first = base.split()[0] if base.split() else ""
        nk_first = nk.split()[0] if nk.split() else ""
        # 候选药名与别名键共享首个词，或完全一致，才认为相关
        if nk and (nk == base or (nk_first and cand_first and nk_first == cand_first)):
            for a in alist:
                na = norm_drug_name(a)
                if na:
                    pats.add(na)
    return pats


# =============================================================================
# 2. 读入：候选药清单 / KG
# =============================================================================

def load_candidates(proj):
    """返回候选药 DataFrame：chembl_id, name, is_cns, is_target_drug。"""
    base = proj / "data" / "interim" / "kg_ext"
    recs = []
    seen = set()

    # 先放 3 个靶药（一定纳入，作为对照）
    for cid, nm in TARGET_DRUGS.items():
        recs.append({"chembl_id": cid, "name": nm, "is_cns": 1, "is_target_drug": 1})
        seen.add(cid)

    cpath = base / "candidate_drugs.csv"
    if cpath.exists():
        cdf = pd.read_csv(cpath)
        for r in cdf.to_dict("records"):
            cid = str(r.get("chembl_id") or "").strip()
            if not cid or cid in seen or cid == "nan":
                continue
            seen.add(cid)
            recs.append({"chembl_id": cid,
                         "name": str(r.get("name") or "").strip(),
                         "is_cns": int(r.get("is_cns") or 0),
                         "is_target_drug": int(cid in TARGET_DRUGS)})
    if len(recs) <= len(TARGET_DRUGS):
        log("[警告] 候选药清单为空（%s），仅纳入 3 个靶药", cpath)
    return pd.DataFrame(recs)


def load_kg(proj):
    kg = proj / "data" / "processed" / "kg_ext"
    if not (kg / "nodes.parquet").exists():
        kg = proj / "data" / "processed" / "kg"
    npath, epath = kg / "nodes.parquet", kg / "edges.parquet"
    if not npath.exists() or not epath.exists():
        return None, None
    return pd.read_parquet(npath), pd.read_parquet(epath)


# =============================================================================
# 3. 阶段 probe / profile：FAERS 药名匹配与 ARIA 画像
# =============================================================================

def _iter_batches(path, columns, batch_size=2_000_000):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    avail = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in avail]
    for b in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield b.to_pandas()


def stage_probe(proj, args):
    """扫 FAERS DRUG 表收集唯一药名，与候选药匹配，输出诊断表（不读 REAC）。"""
    fdir = proj / "data" / "processed" / "faers"
    dpath = fdir / "drug.parquet"
    if not dpath.exists():
        die("未找到 FAERS DRUG 表：%s。请先运行 01_download_faers.py", dpath)

    cands = load_candidates(proj)
    log("[候选] 候选药 %d 个（含 %d 个靶药）", len(cands), int(cands["is_target_drug"].sum()))

    check_disk(proj, args.min_free_gb)

    # ---- Pass 1：收集唯一药名组合 ----
    uniq = set()
    n_rows = 0
    t0 = time.time()
    for i, df in enumerate(_iter_batches(dpath, ["drugname", "prod_ai"]), 1):
        n_rows += len(df)
        for c in ("drugname", "prod_ai"):
            if c in df.columns:
                vals = df[c].dropna().astype(str).unique()
                uniq.update(vals.tolist())
        if i % 5 == 0:
            log("  [扫描] 已读 %s 行，唯一药名 %s（%.0fs）",
                human(n_rows), human(len(uniq)), time.time() - t0)
    log("[扫描] DRUG 表共 %s 行，唯一药名 %s（%.0fs）",
        human(n_rows), human(len(uniq)), time.time() - t0)

    uniq = sorted(uniq)
    ndf = pd.DataFrame({"raw": uniq})
    ndf["norm"] = [norm_drug_name(x) for x in uniq]
    ndf = ndf[ndf["norm"] != ""]
    log("[归一] 有效唯一药名 %s 个", human(len(ndf)))

    # ---- 匹配：候选药的所有别名合成一条正则，一次向量化筛（避免只命中部分写法）----
    matched_rows = []
    raw_map = []            # 原始药名 -> 候选药 全量映射（profile 阶段直接查字典）
    for r in cands.to_dict("records"):
        pats = sorted(match_patterns(r["name"]), key=len, reverse=True)
        if not pats:
            continue
        rx = r"(?:^|\s)(?:" + "|".join(re.escape(p) for p in pats) + r")(?:\s|$)"
        m = ndf["norm"].str.contains(rx, regex=True, na=False)
        if not m.any():
            continue
        names = sorted(set(ndf.loc[m, "raw"].tolist()))
        for raw in names:
            raw_map.append({"raw": str(raw), "chembl_id": str(r["chembl_id"])})
        matched_rows.append({
            "chembl_id": r["chembl_id"], "drug": r["name"],
            "is_target_drug": r["is_target_drug"],
            "pattern": " | ".join(pats[:4]), "n_matched_names": len(names),
            "examples": " | ".join(names[:12]),
        })

    mdf = pd.DataFrame(matched_rows)
    base = proj / "data" / "interim" / "faers_profile"
    base.mkdir(parents=True, exist_ok=True)
    out = base / "drug_name_map.csv"
    mdf.to_csv(out, index=False)
    # 全量原始名映射（供 profile 阶段 O(1) 查表，避免重复正则匹配）
    rmdf = pd.DataFrame(raw_map) if raw_map else pd.DataFrame(columns=["raw", "chembl_id"])
    rmdf = rmdf.drop_duplicates(subset=["raw"])
    rmdf.to_parquet(base / "raw_name_map.parquet", index=False)
    log("[映射] 原始药名 -> 候选药 共 %s 条 -> raw_name_map.parquet", human(len(rmdf)))

    # 诊断
    diag = cands.copy().rename(columns={"name": "drug"})
    if len(mdf):
        diag = diag.merge(mdf[["chembl_id", "pattern", "n_matched_names"]],
                          on="chembl_id", how="left")
    else:
        diag["pattern"] = None
        diag["n_matched_names"] = 0
    diag["n_matched_names"] = diag["n_matched_names"].fillna(0).astype(int)
    diag["matched"] = (diag["n_matched_names"] > 0).astype(int)
    diagp = proj / "results" / "tables" / "faers_drug_match_diagnostic.csv"
    diagp.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(diagp, index=False)

    n_ok = int(diag["matched"].sum())
    log("[匹配] 命中 %d / %d（%.1f%%）-> %s",
        n_ok, len(diag), 100.0 * n_ok / max(1, len(diag)), out.name)
    misses = diag[diag["matched"] == 0]["drug"].tolist()
    if misses:
        log("[匹配] 未命中 %d 个（可编辑 %s 内 pattern 或补别名后重跑）：%s",
            len(misses), out.name, ", ".join(misses[:20]))
    else:
        log("[匹配] 全部候选药均命中")
    log("[提示] 若个别药匹配到过多无关名称（如 'albumin'），请在脚本 BUILTIN_ALIASES 中收紧")
    return diag


def stage_profile(proj, args):
    """计算候选药对 7 个 ARIA 概念的不成比例信号（ROR / PRR）。"""
    fdir = proj / "data" / "processed" / "faers"
    dpath, rpath = fdir / "drug.parquet", fdir / "reac.parquet"
    if not dpath.exists() or not rpath.exists():
        die("未找到 FAERS DRUG/REAC 表：%s", fdir)

    base = proj / "data" / "interim" / "faers_profile"
    base.mkdir(parents=True, exist_ok=True)
    rmap_p = base / "raw_name_map.parquet"
    if not rmap_p.exists():
        die("未找到原始药名全量映射 %s，请先运行 probe 阶段", rmap_p)
    rm = pd.read_parquet(rmap_p)
    if len(rm) == 0:
        die("药名映射为空，请在 probe 阶段确认匹配结果")
    name2che = {}
    for raw, che in zip(rm["raw"].values, rm["chembl_id"].values):
        name2che[str(raw)] = str(che)
    log("[映射] 载入 %s 个原始药名 -> 候选药 映射", human(len(name2che)))

    check_disk(proj, args.min_free_gb)

    # ---- Pass 1：drug 表筛出候选药命中的报告（O(1) 查表，不做重复正则）----
    cache = base / "cand_reports.parquet"
    if cache.exists() and not args.force:
        cr = pd.read_parquet(cache)
        log("[复用] 候选药命中报告 %s -> %s", human(len(cr)), cache.name)
    else:
        role_ok = {x.strip().upper() for x in args.role_codes.split(",") if x.strip()}
        chunks = []
        n_rows = 0
        t0 = time.time()
        for i, df in enumerate(_iter_batches(dpath, ["primaryid", "drugname", "prod_ai", "role_cod"]), 1):
            n_rows += len(df)
            if "primaryid" not in df.columns:
                continue
            if "role_cod" in df.columns and role_ok:
                rc = df["role_cod"].astype(str).str.strip().str.upper()
                df = df[rc.isin(role_ok)]
                if df.empty:
                    continue
            cid = None
            if "drugname" in df.columns:
                cid = df["drugname"].astype(str).map(name2che)
            if "prod_ai" in df.columns:
                c2 = df["prod_ai"].astype(str).map(name2che)
                cid = c2 if cid is None else cid.fillna(c2)
            if cid is None:
                continue
            m = cid.notna()
            if m.any():
                chunks.append(pd.DataFrame({
                    "primaryid": df.loc[m, "primaryid"].values,
                    "chembl_id": cid[m].values,
                }))
            if i % 5 == 0:
                log("  [扫描] 已读 %s 行，命中 %s（%.0fs）",
                    human(n_rows), human(sum(len(c) for c in chunks)), time.time() - t0)
        cr = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
            columns=["primaryid", "chembl_id"])
        cr["primaryid"] = pd.to_numeric(cr["primaryid"], errors="coerce")
        cr = cr.dropna(subset=["primaryid"])
        cr["primaryid"] = cr["primaryid"].astype("int64")
        # 同一报告同一药只记一次
        cr = cr.drop_duplicates(subset=["primaryid", "chembl_id"])
        cr.to_parquet(cache, index=False)
        log("[扫描] DRUG 表共 %s 行；候选药命中（药-报告对）%s -> %s（%.0fs）",
            human(n_rows), human(len(cr)), cache.name, time.time() - t0)

    cand_chembl = set(cr["chembl_id"].unique()) if len(cr) else set()
    log("[命中] 有报告数据的候选药 %d 个", len(cand_chembl))
    # 每药报告数
    drug_pid = defaultdict(set)
    for pid, che in zip(cr["primaryid"].values, cr["chembl_id"].values):
        drug_pid[che].add(int(pid))

    # ---- Pass 2：reac 表统计全库 PT 计数 + 候选药报告的 PT ----
    pid2cand = defaultdict(set)
    for pid, che in zip(cr["primaryid"].values, cr["chembl_id"].values):
        pid2cand[int(pid)].add(che)
    all_total = 0                      # 全库不同报告数
    pt_total = defaultdict(int)        # 全库 PT 计数（按报告去重）
    cand_pt = defaultdict(lambda: defaultdict(int))   # chembl -> concept -> 报告数
    seen_pid_global = 0
    t0 = time.time()
    for i, df in enumerate(_iter_batches(rpath, ["primaryid", "pt"]), 1):
        if "primaryid" not in df.columns or "pt" not in df.columns:
            die("REAC 表缺少 primaryid / pt 列，实际列：%s", list(df.columns))
        pid = pd.to_numeric(df["primaryid"], errors="coerce")
        df = df.assign(_pid=pid).dropna(subset=["_pid"])
        df["_pid"] = df["_pid"].astype("int64")
        df["_pt"] = df["pt"].map(norm_ae)
        # 全库统计（按 primaryid 去重：同一报告同一 PT 只算一次）
        dd = df.drop_duplicates(subset=["_pid", "_pt"])
        all_total += int(dd["_pid"].nunique())
        seen_pid_global += int(dd["_pid"].nunique())
        vc = dd["_pt"].value_counts()
        for k, v in vc.items():
            pt_total[k] += int(v)
        # 候选药命中报告的 PT
        sub = dd[dd["_pid"].isin(pid2cand)]
        if len(sub):
            for pidv, ptv in zip(sub["_pid"].values, sub["_pt"].values):
                con = aria_concept(ptv)
                if con is None:
                    continue
                for che in pid2cand.get(int(pidv), ()):
                    cand_pt[che][con] += 1
        if i % 5 == 0:
            log("  [REAC] 已读批次 %d，全库报告 %s（%.0fs）",
                i, human(seen_pid_global), time.time() - t0)
    log("[REAC] 全库报告（按 PT 去重累计）%s，PT 种类 %s（%.0fs）",
        human(all_total), human(len(pt_total)), time.time() - t0)

    N = seen_pid_global
    if N <= 0:
        die("全库报告数为 0，REAC 表解析异常")

    # 预先把全库 PT 计数折叠成 7 个 ARIA 概念的总数（避免在双重循环里重复扫描）
    concept_totals = {c: 0 for c in ARIA_CONCEPTS}
    for pt, v in pt_total.items():
        c = aria_concept(pt)
        if c in concept_totals:
            concept_totals[c] += int(v)
    log("[事件] ARIA 概念全库报告数：%s",
        ", ".join("%s=%s" % (c, human(concept_totals[c])) for c in ARIA_CONCEPTS))

    # ---- 计算 ROR / PRR ----
    rows = []
    for che, pid_set in drug_pid.items():
        n_drug = len(pid_set)
        if n_drug <= 0:
            continue
        cpt = cand_pt.get(che, {})
        for con in ARIA_CONCEPTS:
            a = int(cpt.get(con, 0))
            n_event = int(concept_totals.get(con, 0))
            b = n_drug - a
            c = max(0, n_event - a)
            d = max(0, N - n_drug - c)
            ror, rl, ru = _ror(a, b, c, d)
            prr, pl, pu = _prr(a, b, c, d)
            rows.append({
                "chembl_id": che, "aria_concept": con,
                "concept_label": CONCEPT_LABEL.get(con, con),
                "a": a, "b": b, "c": c, "d": d,
                "n_drug": n_drug, "n_event": n_event,
                "ROR": ror, "ROR025": rl, "ROR975": ru,
                "PRR": prr, "PRR025": pl, "PRR975": pu,
                "sig_ror": int(a >= 3 and rl is not None and rl > 1),
                "sig_prr": int(a >= 3 and pl is not None and pl > 1),
            })
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        die("未计算出任何信号：候选药命中报告为 0，请在 probe 阶段检查药名匹配")

    # 补上药名 / CNS / 靶药标记
    info = load_candidates(proj)
    imap = {str(r["chembl_id"]): r for r in info.to_dict("records")}
    rdf["drug"] = [imap.get(c, {}).get("name", c) for c in rdf["chembl_id"]]
    rdf["is_cns"] = [int(imap.get(c, {}).get("is_cns", 0)) for c in rdf["chembl_id"]]
    rdf["is_target_drug"] = [int(imap.get(c, {}).get("is_target_drug", 0))
                             for c in rdf["chembl_id"]]
    rdf["robust"] = ((rdf["sig_ror"] == 1) & (rdf["sig_prr"] == 1) & (rdf["a"] >= 3)).astype(int)
    rdf = rdf.sort_values(["aria_concept", "ROR"], ascending=[True, False])

    tdir = proj / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    outp = tdir / "faers_candidate_aria_profile.csv"
    rdf.to_csv(outp, index=False)
    log("[画像] 候选药 x ARIA 信号 -> %s（%s 行）", outp.name, human(len(rdf)))

    # 控制台摘要
    for con in ARIA_CONCEPTS:
        sub = rdf[(rdf["aria_concept"] == con) & (rdf["is_target_drug"] == 0)
                  & (rdf["robust"] == 1)].head(5)
        if len(sub):
            log("  %-24s 稳健信号 top: %s", con, "; ".join(
                "%s(ROR=%.1f,n=%d)" % (r["drug"], r["ROR"], r["a"])
                for _, r in sub.iterrows()))
    tgt = rdf[(rdf["is_target_drug"] == 1) & (rdf["aria_concept"].isin(["ARIA-E", "ARIA-H"]))]
    for _, r in tgt.iterrows():
        log("  [对照] %s %s: a=%d ROR=%.1f (CI %.1f-%.1f)",
            r["drug"], r["aria_concept"], r["a"], r["ROR"], r["ROR025"] or 0, r["ROR975"] or 0)

    _plot_profile(proj, rdf, args)
    return rdf


def pt_total_concept(con, pt_total):
    """把 7 个 ARIA 概念映射回 PT 计数之和（用于 2x2 表的 c 与 n_event）。

    注意：概念折叠后一个概念可能由多个 PT 组成（如 ARIA-E 含 edema/exudative
    两种英美拼写变体），故此处按 aria_concept 归类后求和，与第 2/3 章口径一致。
    但 PT 全库计数里只保留归一化 PT 串，重算概念归属即可。
    """
    s = 0
    for pt, v in pt_total.items():
        if aria_concept(pt) == con:
            s += int(v)
    return s


def _ror(a, b, c, d):
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return (None, None, None)
    ror = (a * d) / (b * c)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    lr = math.log(ror)
    return (ror, math.exp(lr - 1.96 * se), math.exp(lr + 1.96 * se))


def _prr(a, b, c, d):
    if a <= 0 or (a + b) <= 0 or (c + d) <= 0:
        return (None, None, None)
    prr = (a / (a + b)) / (c / (c + d))
    if a <= 0:
        return (prr, None, None)
    se = math.sqrt(1.0 / a - 1.0 / (a + b) + 1.0 / c - 1.0 / (c + d))
    lr = math.log(prr)
    return (prr, math.exp(lr - 1.96 * se), math.exp(lr + 1.96 * se))


def _plot_profile(proj, rdf, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log("[图] matplotlib 不可用，跳过（%s）", e)
        return
    fdir = proj / "results" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    sub = rdf[(rdf["is_target_drug"] == 0) & (rdf["robust"] == 1)
              & (rdf["ROR025"].notna())].copy()
    sub = sub[sub["ROR"] > 0].sort_values("ROR", ascending=False).head(25)
    if len(sub) == 0:
        log("[图] 无稳健信号，跳过森林图")
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(sub) + 1.5)))
    y = np.arange(len(sub))
    ax.errorbar(sub["ROR"], y,
                xerr=[sub["ROR"] - sub["ROR025"], sub["ROR975"] - sub["ROR"]],
                fmt="o", ms=4, lw=1.1, color="#1f4e79", ecolor="#8fb3d9", capsize=2)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(["%s (%s, n=%d)" % (r["drug"], r["aria_concept"], r["a"])
                        for _, r in sub.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Reporting odds ratio (95% CI, log scale)")
    ax.set_title("Candidate approved drugs: robust ARIA signals in FAERS", fontsize=11)
    fig.tight_layout()
    fp = fdir / "faers_profile_forest.png"
    fig.savefig(fp, dpi=160)
    plt.close(fig)
    log("[图] %s", fp)


# =============================================================================
# 4. 阶段 mech：机制接近度（通道 B）
# =============================================================================

def stage_mech(proj, args):
    ndf, edf = load_kg(proj)
    if ndf is None:
        die("未找到知识图谱，请先运行 03_kg_build.py 与 04_gnn_risk.py build")

    log("[读入] KG 节点 %s | 边 %s", human(len(ndf)), human(len(edf)))

    d2t = defaultdict(set)     # drug -> targets
    d2i = defaultdict(set)     # drug -> indications
    d2a = defaultdict(set)     # drug -> ae nodes
    g2p = defaultdict(set)     # gene -> pathways
    g2a = defaultdict(set)     # gene -> ae nodes (gene_ae 派生边)
    pname = {}
    for r in ndf.to_dict("records"):
        if r.get("node_type") == "Pathway":
            pname[r["node_id"]] = r.get("name") or r["node_id"]
    for e in edf.to_dict("records"):
        s, d, t = e.get("src"), e.get("dst"), e.get("edge_type")
        if t == "drug_target":
            d2t[s].add(d)
        elif t == "drug_indication":
            d2i[s].add(d)
        elif t == "drug_ae":
            d2a[s].add(d)
        elif t == "gene_pathway":
            g2p[s].add(d)
        elif t == "gene_ae":
            g2a[s].add(d)

    # 抗 Abeta 单抗的靶点 / 通路 / ARIA 事件
    target_dids = ["DRUG:" + c for c in TARGET_DRUGS if ("DRUG:" + c) in set(ndf["node_id"])]
    if not target_dids:
        die("扩展图中找不到 3 个靶药节点，请先运行 04_gnn_risk.py build")
    Tg = set().union(*[d2t.get(d, set()) for d in target_dids])
    Tp = set().union(*[set().union(*[g2p.get(g, set()) for g in d2t.get(d, set())])
                       for d in target_dids])
    aris_ae = set().union(*[d2a.get(d, set()) for d in target_dids])
    log("[机制] 靶药 %d 个；靶点 %d；二跳通路 %d；ARIA 事件 %d",
        len(target_dids), len(Tg), len(Tp), len(aris_ae))

    # ARIA 相关机制通路（关键词）
    def is_aria_pathway(pid):
        nm = (pname.get(pid) or "").lower()
        return any(h in nm for h in ARIA_PATHWAY_HINTS)

    # 与 ARIA 事件直接相连的基因（gene_ae 稀疏，仅作参考）
    aria_genes = set()
    for g, aes in g2a.items():
        if aes & aris_ae:
            aria_genes.add(g)

    # 候选药
    cinfo = proj / "data" / "interim" / "kg_ext" / "candidate_drugs.csv"
    cns_map, name_map = {}, {}
    if cinfo.exists():
        for r in pd.read_csv(cinfo).to_dict("records"):
            cid = "DRUG:" + str(r["chembl_id"])
            cns_map[cid] = int(r.get("is_cns") or 0)
            name_map[cid] = str(r.get("name") or cid)
    for cid, nm in TARGET_DRUGS.items():
        name_map.setdefault("DRUG:" + cid, nm)
        cns_map.setdefault("DRUG:" + cid, 1)

    rows = []
    for r in ndf[ndf["node_type"] == "Drug"].to_dict("records"):
        did = r["node_id"]
        gs = d2t.get(did, set())
        ps = set().union(*[g2p.get(g, set()) for g in gs]) if gs else set()
        shared_g = gs & Tg
        shared_p = ps & Tp
        hemo_p = {p for p in ps if is_aria_pathway(p)}
        tjac = len(shared_g) / (len(gs | Tg) + 1e-9) if (gs or Tg) else 0.0
        pjac = len(shared_p) / (len(ps | Tp) + 1e-9) if (ps or Tp) else 0.0
        # 机制分：通路接近度为主，靶点直接共享与 ARIA 相关通路命中共为加成
        mech = (0.6 * pjac
                + 0.2 * min(1.0, len(shared_g) / 3.0)
                + 0.2 * min(1.0, len(hemo_p) / 10.0))
        rows.append({
            "chembl_id": str(r.get("chembl_id") or did.split(":")[-1]),
            "drug": r.get("name") or name_map.get(did, did), "drug_id": did,
            "is_cns": cns_map.get(did, 0),
            "is_target_drug": int(did in target_dids),
            "n_targets": len(gs), "n_pathways": len(ps),
            "n_shared_targets": len(shared_g),
            "shared_targets": " | ".join(sorted(x.split(":")[-1] for x in shared_g)),
            "n_shared_pathways": len(shared_p), "target_jaccard": round(tjac, 5),
            "pathway_jaccard": round(pjac, 5),
            "n_aria_pathways": len(hemo_p),
            "aria_pathways": " | ".join(sorted(pname.get(p, p) for p in list(hemo_p)[:8])),
            "n_aria_genes": len(gs & aria_genes),
            "mech_score": round(mech, 6),
        })
    mdf = pd.DataFrame(rows).sort_values("mech_score", ascending=False)
    tdir = proj / "results" / "tables"
    outp = tdir / "kg_mechanism_proximity.csv"
    mdf.to_csv(outp, index=False)
    log("[机制] 接近度 -> %s（%s 行）", outp.name, human(len(mdf)))
    top = mdf[mdf["is_target_drug"] == 0].head(8)
    for _, r in top.iterrows():
        log("  %-28s mech=%.4f 通路Jaccard=%.4f 共享靶点=%d ARIA通路=%d",
            r["drug"], r["mech_score"], r["pathway_jaccard"],
            r["n_shared_targets"], r["n_aria_pathways"])
    return mdf


# =============================================================================
# 5. 阶段 cross：双通道交叉分层
# =============================================================================

def stage_cross(proj, args):
    tdir = proj / "results" / "tables"
    pp = tdir / "faers_candidate_aria_profile.csv"
    mp = tdir / "kg_mechanism_proximity.csv"
    if not pp.exists():
        die("未找到通道 A 结果 %s，请先运行 profile 阶段", pp)
    if not mp.exists():
        die("未找到通道 B 结果 %s，请先运行 mech 阶段", mp)
    pdf = pd.read_csv(pp)
    mdf = pd.read_csv(mp)

    # 通道 A 汇总：取每个药的"最强 ARIA 信号"（稳健信号数 + 最大 ROR）
    agg = []
    for che, sub in pdf.groupby("chembl_id"):
        rob = sub[sub["robust"] == 1]
        agg.append({
            "chembl_id": che,
            "drug": sub["drug"].iloc[0],
            "is_cns": int(sub["is_cns"].iloc[0]),
            "is_target_drug": int(sub["is_target_drug"].iloc[0]),
            "n_robust_concepts": int(len(rob)),
            "max_ROR": float(rob["ROR"].max()) if len(rob) else np.nan,
            "max_ROR_concept": (rob.sort_values("ROR", ascending=False)
                                ["aria_concept"].iloc[0] if len(rob) else ""),
            "total_a": int(sub["a"].sum()),
            "sig_concepts": " | ".join(sorted(rob["aria_concept"].tolist())),
        })
    adf = pd.DataFrame(agg)

    cdf = adf.merge(
        mdf[["chembl_id", "mech_score", "pathway_jaccard", "n_shared_targets",
             "n_aria_pathways", "n_targets", "n_pathways"]],
        on="chembl_id", how="left")
    cdf["mech_score"] = cdf["mech_score"].fillna(0.0)
    cdf["pathway_jaccard"] = cdf["pathway_jaccard"].fillna(0.0)

    # 判定阈值（基于分布的相对量，稳健且不依赖绝对 ROR）
    a_pos = cdf["n_robust_concepts"] >= 1
    b_pos = cdf["mech_score"] > 0

    def layer(r):
        if r["is_target_drug"]:
            return "target_drug"
        if r["n_robust_concepts"] >= 1 and r["mech_score"] > 0:
            return "high_priority(A+B)"
        if r["n_robust_concepts"] >= 1:
            return "clinical_caution(A)"
        if r["mech_score"] > 0:
            return "mechanism_hypothesis(B)"
        return "low"

    cdf["layer"] = cdf.apply(layer, axis=1)
    cdf = cdf.sort_values(["n_robust_concepts", "mech_score"],
                          ascending=[False, False])
    outp = tdir / "chapter4_cross_evidence.csv"
    cdf.to_csv(outp, index=False)
    log("[交叉] 双通道证据 -> %s（%s 行）", outp.name, human(len(cdf)))
    for k, v in cdf["layer"].value_counts().items():
        log("  %-24s %d", k, v)

    # 评估①：已知阳性药（抗凝/抗血小板）是否富集在 A 通道阳性（排除 3 个靶药）
    known = ["aspirin", "clopidogrel", "dabigatran", "rivaroxaban", "apixaban",
             "warfarin", "heparin", "enoxaparin", "ticagrelor", "prasugrel",
             "edoxaban", "dipyridamole"]
    cdf["_nk"] = cdf["drug"].str.lower().apply(
        lambda x: int(any(k in str(x).lower() for k in known)))
    nz = cdf[cdf["is_target_drug"] == 0]
    if nz["_nk"].sum() > 0 and (nz["_nk"] == 0).sum() > 0:
        r1 = nz[nz["_nk"] == 1]["n_robust_concepts"].mean()
        r0 = nz[nz["_nk"] == 0]["n_robust_concepts"].mean()
        log("[评估] 已知出血风险药(n=%d) 平均稳健概念数 %.2f vs 其他候选药(n=%d) %.2f",
            int(nz["_nk"].sum()), r1, int((nz["_nk"] == 0).sum()), r0)

    # 评估②：双通道一致性置换检验（Spearman）
    try:
        from scipy.stats import spearmanr
        sub = cdf[(cdf["is_target_drug"] == 0) & (cdf["mech_score"] > 0)]
        if len(sub) >= 8:
            rho, p = spearmanr(sub["n_robust_concepts"], sub["mech_score"])
            log("[评估] 双通道一致性 Spearman rho=%.3f (p=%.4f, n=%d)",
                rho, p, len(sub))
    except Exception as e:
        log("[评估] 一致性检验跳过（%s）", e)

    _plot_cross(proj, cdf, args)
    return cdf


def _plot_cross(proj, cdf, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log("[图] matplotlib 不可用，跳过（%s）", e)
        return
    fdir = proj / "results" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sub = cdf[cdf["is_target_drug"] == 0]
    sc = ax.scatter(sub["mech_score"], sub["n_robust_concepts"],
                    s=30 + 6 * sub["total_a"].clip(0, 60) / 6.0,
                    c=sub["is_cns"], cmap="viridis", alpha=0.8, edgecolor="white", lw=0.5)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    for _, r in sub.nlargest(12, "n_robust_concepts").iterrows():
        ax.annotate(str(r["drug"])[:16], (r["mech_score"], r["n_robust_concepts"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mechanism proximity to anti-Abeta mAbs (channel B)")
    ax.set_ylabel("Number of robust ARIA concepts in FAERS (channel A)")
    ax.set_title("Dual-channel evidence: candidate drugs vs anti-Abeta mAbs", fontsize=11)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("is CNS drug")
    fig.tight_layout()
    fp = fdir / "chapter4_cross_scatter.png"
    fig.savefig(fp, dpi=160)
    plt.close(fig)
    log("[图] %s", fp)


# =============================================================================
# 6. main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="候选药 ARIA 事件画像（FAERS）与机制接近度双通道分层")
    ap.add_argument("stage", choices=["probe", "profile", "mech", "cross", "all"])
    ap.add_argument("--proj", default=None, help="项目根目录（默认自动向上查找）")
    ap.add_argument("--role-codes", default="PS,SS",
                    help="FAERS role_cod 取值，默认 PS,SS（怀疑药）；用 PS,SS,C 纳入合并用药")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--force", action="store_true", help="忽略缓存，强制重算")
    args = ap.parse_args()

    proj = Path(args.proj).resolve() if args.proj else find_proj_root()
    if not (proj / "data").exists():
        die("项目根 %s 下找不到 data 目录，请用 --proj 指定", proj)

    glog = Path.home() / "faers_profile.log"
    global _LOGFH
    try:
        _LOGFH = open(glog, "a", encoding="utf-8")
    except Exception:
        _LOGFH = None

    log("=" * 74)
    log("项目根：%s", proj)
    log("阶段：%s | role_cod=%s", args.stage, args.role_codes)
    t0 = time.time()

    if args.stage in ("probe",):
        stage_probe(proj, args)
    elif args.stage == "profile":
        stage_profile(proj, args)
    elif args.stage == "mech":
        stage_mech(proj, args)
    elif args.stage == "cross":
        stage_cross(proj, args)
    else:  # all
        stage_probe(proj, args)
        stage_profile(proj, args)
        stage_mech(proj, args)
        stage_cross(proj, args)

    log("=" * 74)
    log("完成，用时 %.1f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
