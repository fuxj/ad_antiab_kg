#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04c_ddi_coreport.py  --  第4章扩展：抗Aβ单抗合并用药的 FAERS 共报告（DDI）分析

为什么需要这一步
----------------
第4章通道A证明"抗凝/抗血小板药与出血性 ARIA 概念稳健关联"，但审稿人必问：
    "脑出血在 FAERS 里本就是常见事件，你怎么证明这些药是特异性加重 ARIA，
      而不是仅仅反映了泛化的出血风险、或报告量偏倚？"

本脚本在"含抗Aβ单抗(lecanemab / donanemab / aducanumab)的报告"这一队列内部，
做 case-case 比较，回答这个因果方向的问题。

核心设计
--------
    队列   = 所有含抗Aβ单抗的 FAERS 报告
    暴露   = 同一份报告中同时出现抗凝/抗血小板药（可切换暴露集）
    结局   = ARIA 相关事件（ARIA-E / ARIA-H / ARIA-NOS / 脑水肿 / 脑出血 /
             脑微出血 / 中枢神经系统表面铁沉积）
    统计量 = case-case OR（Fisher 精确检验，Mann-Whitney 秩检验）

内部对照（关键）
----------------
    同时检验 ARIA-E（水肿型，非出血性）与 ARIA-H（出血型）。
    若暴露显著抬高 ARIA-H 而不抬高 ARIA-E，则支持"直接出血机制"，
    而非"这批人本来就更爱报出血"的泛化偏倚。

分层
----
    按抗Aβ单抗品种 / 性别 / 年龄组 / 报告年代（FDA 收到年份）分层；
    并对每个合并用药单独给出 OR 表；利用 role_cod 区分主要怀疑药与伴随用药。

用法
----
    python 04c_ddi_coreport.py probe    --proj .
    python 04c_ddi_coreport.py coreport --proj . [--exposure anticoag_antiplatelet]
    python 04c_ddi_coreport.py all      --proj .

输出
----
    results/tables/ddi_probe.txt                  数据 schema 自检
    results/tables/ddi_report_flags.parquet       报告级标记表（可复核）
    results/tables/ddi_2x2_overall.csv            总体 case-case 2x2
    results/tables/ddi_by_concept.csv             逐 ARIA 概念
    results/tables/ddi_by_targetdrug.csv          逐抗Aβ单抗
    results/tables/ddi_by_exposure_drug.csv       逐合并用药
    results/tables/ddi_strata.csv                 性别/年龄/年代分层
    results/tables/ddi_role_sensitivity.csv       role_cod 敏感性分析
    results/figures/ddi_coreport_forest.png       逐合并用药 OR 森林图
    results/figures/ddi_concept_compare.png       逐概念暴露组 vs 单药组
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None

try:
    from scipy.stats import fisher_exact, mannwhitneyu
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


# ----------------------------------------------------------------------------
# 常量：目标药 与 暴露集
# ----------------------------------------------------------------------------

TARGET_DRUGS = {
    "lecanemab":  ["LECANEMAB", "LEQEMBI", "BAN2401"],
    "donanemab":  ["DONANEMAB", "KISUNLA", "LY3002813"],
    "aducanumab": ["ADUCANUMAB", "ADUHELM", "BIIB037"],
}

EXPOSURE_SETS = {
    "anticoag_antiplatelet": {
        "dabigatran":     ["DABIGATRAN", "PRADAXA"],
        "clopidogrel":    ["CLOPIDOGREL", "PLAVIX"],
        "aspirin":        ["ASPIRIN", "ACETYLSALICYLIC ACID", "ECOTRIN"],
        "warfarin":       ["WARFARIN", "COUMADIN", "JANTOVEN"],
        "rivaroxaban":    ["RIVAROXABAN", "XARELTO"],
        "apixaban":       ["APIXABAN", "ELIQUIS"],
        "edoxaban":       ["EDOXABAN", "SAVAYSA", "LIXIANA"],
        "ticagrelor":     ["TICAGRELOR", "BRILINTA"],
        "prasugrel":      ["PRASUGREL", "EFFIENT"],
        "dipyridamole":   ["DIPYRIDAMOLE", "PERSANTINE"],
        "heparin":        ["HEPARIN"],
        "enoxaparin":     ["ENOXAPARIN", "LOVENOX"],
        "ticlopidine":    ["TICLOPIDINE", "TICLID"],
        "cilostazol":     ["CILOSTAZOL", "PLETAL"],
    },
    "ssri": {
        "citalopram":     ["CITALOPRAM", "CELEXA"],
        "escitalopram":   ["ESCITALOPRAM", "LEXAPRO"],
        "sertraline":     ["SERTRALINE", "ZOLOFT"],
        "fluoxetine":     ["FLUOXETINE", "PROZAC"],
        "paroxetine":     ["PAROXETINE", "PAXIL"],
        "fluvoxamine":    ["FLUVOXAMINE", "LUVOX"],
        "venlafaxine":    ["VENLAFAXINE", "EFFEXOR"],
        "duloxetine":     ["DULOXETINE", "CYMBALTA"],
    },
    "nsaid": {
        "ibuprofen":      ["IBUPROFEN", "ADVIL", "MOTRIN"],
        "naproxen":       ["NAPROXEN", "NAPROSYN", "ALEVE"],
        "diclofenac":     ["DICLOFENAC", "VOLTAREN"],
        "celecoxib":      ["CELECOXIB", "CELEBREX"],
        "meloxicam":      ["MELOXICAM", "MOBIC"],
        "lornoxicam":     ["LORNOXICAM"],
        "indomethacin":   ["INDOMETHACIN", "INDOCIN"],
        "ketorolac":      ["KETOROLAC", "TORADOL"],
        "etodolac":       ["ETODOLAC", "LODINE"],
        "piroxicam":      ["PIROXICAM", "FELDENE"],
    },
}

# ARIA 相关事件概念（匹配归一化后的 MedDRA PT 文本，兼容英美拼写）
CONCEPT_RULES = [
    ("ARIA-NOS", lambda t: ("AMYLOID RELATED IMAGING ABNORMALIT" in t)
                           and ("EDEMA/EFFUSION" not in t) and ("OEDEMA/EFFUSION" not in t)
                           and ("MICROHAEMORRHAGES AND HAEMOSIDERIN" not in t)
                           and ("MICROHEMORRHAGES AND HEMOSIDERIN" not in t)),
    ("ARIA-E",   lambda t: ("AMYLOID RELATED IMAGING ABNORMALIT" in t)
                           and (("EDEMA/EFFUSION" in t) or ("OEDEMA/EFFUSION" in t))),
    ("ARIA-H",   lambda t: ("AMYLOID RELATED IMAGING ABNORMALIT" in t)
                           and (("MICROHAEMORRHAGES AND HAEMOSIDERIN" in t)
                                or ("MICROHEMORRHAGES AND HEMOSIDERIN" in t))),
    ("CEREBRAL_EDEMA", lambda t: ("BRAIN OEDEMA" in t) or ("BRAIN EDEMA" in t)
                                 or ("VASOGENIC CEREBRAL OEDEMA" in t)
                                 or ("VASOGENIC CEREBRAL EDEMA" in t)),
    ("CEREBRAL_HEMORRHAGE", lambda t: ("CEREBRAL HAEMORRHAGE" in t) or ("CEREBRAL HEMORRHAGE" in t)),
    ("CEREBRAL_MICROHEMORRHAGE", lambda t: ("CEREBRAL MICROHAEMORRHAGE" in t)
                                            or ("CEREBRAL MICROHEMORRHAGE" in t)),
    ("SUPERFICIAL_SIDEROSIS", lambda t: "SUPERFICIAL SIDEROSIS" in t),
]

# 出血性 vs 非出血性分组（用于内部对照：直接出血机制 vs 泛化偏倚）
BLEEDING_CONCEPTS = ["ARIA-H", "CEREBRAL_HEMORRHAGE", "CEREBRAL_MICROHEMORRHAGE", "SUPERFICIAL_SIDEROSIS"]
NONBLEEDING_CONCEPTS = ["ARIA-E", "CEREBRAL_EDEMA"]

# 已知需在敏感性分析中剔除的适应症混杂药物（阳性对照排除项）
KNOWN_CONFOUNDERS = ["FLORBETAPIR F 18", "FLORBETABEN F 18", "FLUTEMETAMOL F 18", "TACROLIMUS"]


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _norm_alias(a: str) -> str:
    a = str(a).upper()
    a = re.sub(r"[^A-Z0-9/ ]+", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a


def norm_series(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.upper()
    x = x.str.replace(r"[^A-Z0-9/ ]+", " ", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return x


def match_aliases(norm: pd.Series, aliases) -> pd.Series:
    """token 边界匹配，避免 'ASA' 命中 'ASACOL' 这类子串误配。"""
    mask = pd.Series(False, index=norm.index)
    for a in aliases:
        na = _norm_alias(a)
        if not na:
            continue
        pat = r"(?:^| )" + re.escape(na) + r"(?: |$)"
        mask = mask | norm.str.contains(pat, regex=True, na=False)
    return mask


def two_by_two(a: int, b: int, c: int, d: int, label: str = "") -> dict:
    """
    2x2:  a=暴露&事件  b=暴露&无事件  c=非暴露&事件  d=非暴露&无事件
    返回 OR / 95%CI / Fisher p / 卡方 p / 暴露组与对照组事件率。
    """
    out = {"label": label, "a": a, "b": b, "c": c, "d": d}
    n_exp = a + b
    n_ctl = c + d
    out["n_exposed"] = n_exp
    out["n_unexposed"] = n_ctl
    out["rate_exposed"] = (a / n_exp) if n_exp else float("nan")
    out["rate_unexposed"] = (c / n_ctl) if n_ctl else float("nan")

    zero_cell = (min(a, b, c, d) == 0)
    out["zero_cell"] = bool(zero_cell)
    if zero_cell:
        # 任一格为 0 时 OR 与 CI 不稳定，不报告数值，仅保留 Fisher 精确 p（仍有效）
        out["OR"] = float("nan")
        out["OR_lo"] = float("nan")
        out["OR_hi"] = float("nan")
    else:
        orv = (a * d) / (b * c)
        se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        out["OR"] = orv
        out["OR_lo"] = math.exp(math.log(orv) - 1.96 * se)
        out["OR_hi"] = math.exp(math.log(orv) + 1.96 * se)

    if HAS_SCIPY:
        out["fisher_p"] = float(fisher_exact([[a, b], [c, d]], alternative="greater")[1])
        out["fisher_p_two_sided"] = float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])
    else:
        out["fisher_p"] = float("nan")
        out["fisher_p_two_sided"] = float("nan")

    # 风险差
    out["risk_diff"] = out["rate_exposed"] - out["rate_unexposed"]
    return out


def iter_parquet(path: Path, columns=None, batch_size: int = 2_000_000):
    pf = pq.ParquetFile(str(path))
    names = pf.schema_arrow.names
    if columns is not None:
        columns = [c for c in columns if c in names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def pick_col(names, candidates, required=True, what=""):
    low = {str(n).lower(): n for n in names}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"找不到列 {what}（候选 {candidates}），实际列：{list(names)}")
    return None


def parquet_schema(path: Path):
    pf = pq.ParquetFile(str(path))
    return pf.schema_arrow.names


# ----------------------------------------------------------------------------
# 阶段 1：probe —— 数据自检
# ----------------------------------------------------------------------------

def stage_probe(proj: Path, args) -> None:
    fdir = proj / "data" / "processed" / "faers"
    tdir = proj / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def rec(s):
        lines.append(s)
        log(s)

    rec("=" * 70)
    rec("04c 共报告(DDI)分析 -- 数据自检")
    rec("=" * 70)
    rec(f"FAERS 目录：{fdir}")

    need = ["drug", "reac", "demo"]
    found = {}
    for name in need:
        p = fdir / f"{name}.parquet"
        if not p.exists():
            cand = list(fdir.glob(f"*{name}*.parquet")) or list(fdir.glob(f"*{name.upper()}*.parquet"))
            if cand:
                p = cand[0]
                rec(f"[提示] {name}.parquet 不存在，改用 {p.name}")
        if not p.exists():
            raise FileNotFoundError(f"缺 {name}.parquet，请确认 01 脚本产物在 {fdir}")
        found[name] = p
        cols = parquet_schema(p)
        pf = pq.ParquetFile(str(p))
        rec(f"[{name}] {p.name} | 行数 {pf.metadata.num_rows:,} | 列 {cols}")

    # 检查关键列
    drug_cols = parquet_schema(found["drug"])
    reac_cols = parquet_schema(found["reac"])
    demo_cols = parquet_schema(found["demo"])
    pid_d = pick_col(drug_cols, ["primaryid", "PRIMARYID", "primary_id"], what="drug.primaryid")
    dname_d = pick_col(drug_cols, ["drugname", "drug_name", "DRUGNAME"], what="drug.drugname")
    prodai_d = pick_col(drug_cols, ["prod_ai", "PROD_AI", "prodai"], required=False)
    role_d = pick_col(drug_cols, ["role_cod", "role_cod".upper(), "role"], required=False)
    pid_r = pick_col(reac_cols, ["primaryid", "PRIMARYID"], what="reac.primaryid")
    pt_r = pick_col(reac_cols, ["pt", "PT", "pt_name"], what="reac.pt")
    pid_m = pick_col(demo_cols, ["primaryid", "PRIMARYID"], what="demo.primaryid")

    rec(f"[列] drug.primaryid={pid_d} drug.drugname={dname_d} prod_ai={prodai_d} role_cod={role_d}")
    rec(f"[列] reac.primaryid={pid_r} reac.pt={pt_r} | demo.primaryid={pid_m}")
    rec(f"[列] demo 可选：age={pick_col(demo_cols, ['age'], required=False)} "
        f"sex={pick_col(demo_cols, ['sex'], required=False)} "
        f"fda_dt={pick_col(demo_cols, ['fda_dt'], required=False)} "
        f"event_dt={pick_col(demo_cols, ['event_dt'], required=False)}")

    # 抽样 300 万行检查目标药命中
    sample = []
    n = 0
    for df in iter_parquet(found["drug"], columns=[dname_d] + ([prodai_d] if prodai_d else []), batch_size=1_000_000):
        sample.append(df)
        n += len(df)
        if n >= 3_000_000:
            break
    s = pd.concat(sample, ignore_index=True)
    s["_nm"] = norm_series(s[dname_d])
    if prodai_d:
        s["_nm"] = s["_nm"].where(s["_nm"].str.len() > 0, norm_series(s[prodai_d]))
    rec(f"[抽样] 前 {len(s):,} 行用于匹配自检")
    for dname, aliases in TARGET_DRUGS.items():
        hit = int(match_aliases(s["_nm"], aliases).sum())
        rec(f"  目标药 {dname:12s} 命中 {hit:,} 行")

    exp_set = EXPOSURE_SETS[args.exposure]
    for ename, aliases in exp_set.items():
        hit = int(match_aliases(s["_nm"], aliases).sum())
        if hit:
            rec(f"  暴露药 {ename:12s} 命中 {hit:,} 行")
    rec("[提示] 命中为 0 的药可能在抽样范围内没有报告，属正常；以 coreport 全量结果为准")

    (tdir / "ddi_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[输出] {tdir / 'ddi_probe.txt'}")


# ----------------------------------------------------------------------------
# 阶段 2：coreport —— 主分析
# ----------------------------------------------------------------------------

def _scan_drug_table(proj: Path, exposure_set_name: str, keep_all: bool):
    """一遍扫 drug 表：收集每份报告中出现的 目标药 / 暴露药。"""
    fdir = proj / "data" / "processed" / "faers"
    ppath = fdir / "drug.parquet"
    if not ppath.exists():
        cand = list(fdir.glob("*drug*.parquet"))
        if not cand:
            raise FileNotFoundError(f"缺 drug.parquet（{fdir}）")
        ppath = cand[0]
    cols = parquet_schema(ppath)
    pid_c = pick_col(cols, ["primaryid", "PRIMARYID"], what="drug.primaryid")
    dn_c = pick_col(cols, ["drugname", "drug_name", "DRUGNAME"], what="drug.drugname")
    pa_c = pick_col(cols, ["prod_ai", "PROD_AI"], required=False)
    role_c = pick_col(cols, ["role_cod", "ROLE_COD"], required=False)
    use_cols = [pid_c, dn_c] + ([pa_c] if pa_c else []) + ([role_c] if role_c else [])

    exp_set = EXPOSURE_SETS[exposure_set_name]
    target_rows = []   # (primaryid, target_name)
    exp_rows = []      # (primaryid, exposure_name, role_cod)
    n = 0
    t0 = time.time()
    for df in iter_parquet(ppath, columns=use_cols, batch_size=2_000_000):
        n += len(df)
        nm = norm_series(df[dn_c])
        if pa_c:
            nm2 = norm_series(df[pa_c])
            nm = nm.where(nm.str.len() > 0, nm2)
        for tname, aliases in TARGET_DRUGS.items():
            m = match_aliases(nm, aliases)
            if m.any():
                sub = pd.DataFrame({pid_c: df.loc[m, pid_c].values, "_t": tname})
                if role_c:
                    sub["_role"] = df.loc[m, role_c].values
                target_rows.append(sub)
        for ename, aliases in exp_set.items():
            m = match_aliases(nm, aliases)
            if m.any():
                sub = pd.DataFrame({pid_c: df.loc[m, pid_c].values, "_e": ename})
                if role_c:
                    sub["_role"] = df.loc[m, role_c].values
                exp_rows.append(sub)
        if n % 20_000_000 < 2_000_000:
            log(f"  [扫 drug] {n:,} 行（{time.time()-t0:.0f}s）")
    log(f"  [扫 drug] 完成 {n:,} 行（{time.time()-t0:.0f}s）")

    tgt = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame(columns=[pid_c, "_t"])
    ex = pd.concat(exp_rows, ignore_index=True) if exp_rows else pd.DataFrame(columns=[pid_c, "_e"])
    return pid_c, tgt, ex


def stage_coreport(proj: Path, args) -> None:
    if pa is None:
        raise RuntimeError("需要 pyarrow")
    tdir = proj / "results" / "tables"
    fdir = proj / "results" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"04c 共报告(DDI)分析 | 暴露集 = {args.exposure}")
    log("=" * 70)

    pid_c, tgt, ex = _scan_drug_table(proj, args.exposure, args.keep_all)
    log(f"[命中] 目标药行 {len(tgt):,} | 暴露药行 {len(ex):,}")

    if len(tgt) == 0:
        raise RuntimeError("未匹配到任何抗Aβ单抗报告，请检查药名或数据")

    # 报告级汇总
    tgt["_pid"] = tgt[pid_c]
    target_pids = pd.Index(tgt["_pid"].unique())
    log(f"[队列] 含抗Aβ单抗的报告 {len(target_pids):,} 份")

    # 每份报告的目标药集合
    tset = tgt.groupby("_pid")["_t"].apply(lambda s: sorted(set(s))).rename("target_drugs")
    # 每份报告的暴露药集合
    if len(ex):
        ex["_pid"] = ex[pid_c]
        ex = ex[ex["_pid"].isin(target_pids)]
        eset = ex.groupby("_pid")["_e"].apply(lambda s: sorted(set(s))).rename("exposure_drugs")
    else:
        eset = pd.Series(dtype=object, name="exposure_drugs")

    rep = pd.DataFrame(index=target_pids)
    rep.index.name = "primaryid"
    rep = rep.join(tset).join(eset)
    rep["target_drugs"] = rep["target_drugs"].apply(lambda x: x if isinstance(x, list) else [])
    rep["exposure_drugs"] = rep["exposure_drugs"].apply(lambda x: x if isinstance(x, list) else [])
    rep["exposed"] = rep["exposure_drugs"].map(len) > 0
    rep["target_primary"] = rep["target_drugs"].map(lambda l: l[0] if l else "NA")

    # ---- 扫 reac：判定每份报告的事件概念 ----
    rpath = proj / "data" / "processed" / "faers" / "reac.parquet"
    if not rpath.exists():
        cand = list((proj / "data" / "processed" / "faers").glob("*reac*.parquet"))
        if not cand:
            raise FileNotFoundError("缺 reac.parquet")
        rpath = cand[0]
    rcols = parquet_schema(rpath)
    pid_r = pick_col(rcols, ["primaryid", "PRIMARYID"], what="reac.primaryid")
    pt_r = pick_col(rcols, ["pt", "PT", "pt_name"], what="reac.pt")

    concept_flags = {c: set() for c, _ in CONCEPT_RULES}
    n = 0
    t0 = time.time()
    tps = set(target_pids)
    for df in iter_parquet(rpath, columns=[pid_r, pt_r], batch_size=3_000_000):
        n += len(df)
        df = df[df[pid_r].isin(tps)]
        if len(df) == 0:
            continue
        nm = norm_series(df[pt_r])
        for cname, rule in CONCEPT_RULES:
            m = nm.map(lambda t: bool(rule(t)) if isinstance(t, str) else False)
            if m.any():
                concept_flags[cname].update(df.loc[m.values, pid_r].tolist())
        if n % 20_000_000 < 3_000_000:
            log(f"  [扫 reac] {n:,} 行（{time.time()-t0:.0f}s）")
    log(f"  [扫 reac] 完成 {n:,} 行（{time.time()-t0:.0f}s）")

    for cname in concept_flags:
        rep[f"ev_{cname}"] = rep.index.isin(concept_flags[cname])
    rep["ev_any_aria"] = rep[["ev_ARIA-E", "ev_ARIA-H"]].any(axis=1)
    rep["ev_any_bleeding"] = rep[[f"ev_{c}" for c in BLEEDING_CONCEPTS if f"ev_{c}" in rep.columns]].any(axis=1)
    rep["ev_any_nonbleeding"] = rep[[f"ev_{c}" for c in NONBLEEDING_CONCEPTS if f"ev_{c}" in rep.columns]].any(axis=1)

    # ---- 扫 demo：性别 / 年龄 / 年代 ----
    dpath = proj / "data" / "processed" / "faers" / "demo.parquet"
    if not dpath.exists():
        cand = list((proj / "data" / "processed" / "faers").glob("*demo*.parquet"))
        if cand:
            dpath = cand[0]
    have_demo = False
    if dpath.exists():
        dcols = parquet_schema(dpath)
        pid_m = pick_col(dcols, ["primaryid", "PRIMARYID"], required=False)
        age_m = pick_col(dcols, ["age"], required=False)
        sex_m = pick_col(dcols, ["sex"], required=False)
        fda_m = pick_col(dcols, ["fda_dt"], required=False)
        ev_m = pick_col(dcols, ["event_dt"], required=False)
        use = [c for c in [pid_m, age_m, sex_m, fda_m, ev_m] if c]
        if pid_m and len(use) > 1:
            parts = []
            for df in iter_parquet(dpath, columns=use, batch_size=3_000_000):
                df = df[df[pid_m].isin(tps)]
                if len(df):
                    parts.append(df)
            if parts:
                dm = pd.concat(parts, ignore_index=True).drop_duplicates(pid_m).set_index(pid_m)
                if age_m:
                    rep["age"] = dm[age_m].reindex(rep.index)
                if sex_m:
                    rep["sex"] = dm[sex_m].reindex(rep.index)
                yr_src = None
                for cc in [fda_m, ev_m]:
                    if cc:
                        v = pd.to_numeric(dm[cc].reindex(rep.index), errors="coerce")
                        yr = (v // 10000).where((v > 19000000) & (v < 21000000))
                        if yr.notna().sum() > 0:
                            yr_src = yr
                            break
                if yr_src is not None:
                    rep["year"] = yr_src
                have_demo = True
    if not have_demo:
        log("[提示] demo 表缺少可用字段，跳过性别/年龄/年代分层")

    if "age" in rep.columns:
        rep["age_grp"] = pd.cut(pd.to_numeric(rep["age"], errors="coerce"),
                                bins=[-1, 64, 74, 84, 200],
                                labels=["<65", "65-74", "75-84", ">=85"])
    if "year" in rep.columns:
        rep["era"] = pd.cut(rep["year"], bins=[-1, 2022, 2100], labels=["<=2022", ">=2023"])

    rep = rep.reset_index()
    rep.to_parquet(tdir / "ddi_report_flags.parquet", index=False)
    log(f"[输出] {tdir / 'ddi_report_flags.parquet'}（{len(rep):,} 行）")

    # ---- 统计 ----
    def stats_for(mask_rep: pd.DataFrame, label: str) -> dict:
        e = mask_rep["exposed"]
        y = mask_rep["ev_any_aria"]
        a = int((e & y).sum()); b = int((e & ~y).sum())
        c = int((~e & y).sum()); d = int((~e & ~y).sum())
        return two_by_two(a, b, c, d, label)

    rows = [stats_for(rep, "ALL_TARGET_REPORTS")]
    ov = pd.DataFrame(rows)
    ov.to_csv(tdir / "ddi_2x2_overall.csv", index=False)
    log(f"[统计] 总体 case-case OR = {ov.loc[0,'OR']:.3f} "
        f"({ov.loc[0,'OR_lo']:.3f}-{ov.loc[0,'OR_hi']:.3f}) Fisher p = {ov.loc[0,'fisher_p']:.4g}")
    log(f"  暴露组 ARIA 率 {ov.loc[0,'rate_exposed']:.4f} vs 单药组 {ov.loc[0,'rate_unexposed']:.4f}")

    # 逐概念
    crows = []
    for cname, _ in CONCEPT_RULES:
        col = f"ev_{cname}"
        if col not in rep.columns:
            continue
        e = rep["exposed"]; y = rep[col]
        a = int((e & y).sum()); b = int((e & ~y).sum())
        c = int((~e & y).sum()); d = int((~e & ~y).sum())
        r = two_by_two(a, b, c, d, cname)
        r["concept"] = cname
        r["is_bleeding"] = cname in BLEEDING_CONCEPTS
        crows.append(r)

    # 内部对照：出血性复合 vs 非出血性复合
    # 若暴露对出血性概念的效应显著大于非出血性，支持"直接出血机制"而非泛化报告偏倚
    for comp_name, comp_col, is_bleed in [
        ("ANY_BLEEDING_COMPOSITE", "ev_any_bleeding", True),
        ("ANY_NONBLEEDING_COMPOSITE", "ev_any_nonbleeding", False),
    ]:
        if comp_col not in rep.columns:
            continue
        e = rep["exposed"]; y = rep[comp_col]
        a = int((e & y).sum()); b = int((e & ~y).sum())
        c = int((~e & y).sum()); d = int((~e & ~y).sum())
        r = two_by_two(a, b, c, d, comp_name)
        r["concept"] = comp_name
        r["is_bleeding"] = is_bleed
        crows.append(r)

    cdf = pd.DataFrame(crows)
    cdf.to_csv(tdir / "ddi_by_concept.csv", index=False)
    log("[统计] 逐概念 OR：")
    for _, r in cdf.iterrows():
        log(f"   {r['concept']:26s} OR={r['OR']:8.3f} ({r['OR_lo']:.2f}-{r['OR_hi']:.2f}) "
            f"p={r['fisher_p']:.3g} 暴露率={r['rate_exposed']:.4f} 对照率={r['rate_unexposed']:.4f}")

    # 逐目标药
    trows = []
    for td, g in rep.groupby("target_primary"):
        if len(g) < 30:
            continue
        r = stats_for(g, td)
        r["target_drug"] = td
        trows.append(r)
    pd.DataFrame(trows).to_csv(tdir / "ddi_by_targetdrug.csv", index=False)

    # 逐暴露药
    erows = []
    for ename in EXPOSURE_SETS[args.exposure]:
        m = rep["exposure_drugs"].map(lambda l, e=ename: e in l)
        if m.sum() < 5:
            continue
        y = rep["ev_any_aria"]
        a = int((m & y).sum()); b = int((m & ~y).sum())
        c = int((~m & y).sum()); d = int((~m & ~y).sum())
        r = two_by_two(a, b, c, d, ename)
        r["exposure_drug"] = ename
        erows.append(r)
    edf = pd.DataFrame(erows).sort_values("OR", ascending=False) if erows else pd.DataFrame()
    if len(edf):
        edf.to_csv(tdir / "ddi_by_exposure_drug.csv", index=False)
        log("[统计] 逐合并用药 OR（Top5）：")
        for _, r in edf.head(5).iterrows():
            log(f"   {r['exposure_drug']:14s} OR={r['OR']:8.3f} ({r['OR_lo']:.2f}-{r['OR_hi']:.2f}) "
                f"p={r['fisher_p']:.3g} n={int(r['n_exposed'])}")

    # 分层
    srows = []
    for key in ["target_primary", "sex", "age_grp", "era"]:
        if key not in rep.columns:
            continue
        for lv, g in rep.groupby(key, observed=True):
            if len(g) < 30 or g["exposed"].sum() < 5:
                continue
            r = stats_for(g, f"{key}={lv}")
            r["stratum_var"] = key
            r["stratum_level"] = str(lv)
            srows.append(r)
    pd.DataFrame(srows).to_csv(tdir / "ddi_strata.csv", index=False)

    # role_cod 敏感性（目标药是否主要怀疑药）
    if "_role" in tgt.columns:
        ps_pids = set(tgt.loc[tgt["_role"].astype(str).str.upper().eq("PS"), "_pid"])
        if ps_pids:
            sub = rep[rep["primaryid"].isin(ps_pids)]
            if len(sub) >= 30:
                r = stats_for(sub, "target_drug_role=PS")
                pd.DataFrame([r]).to_csv(tdir / "ddi_role_sensitivity.csv", index=False)
                log(f"[统计] 仅目标药为主要怀疑药(PS)：OR={r['OR']:.3f} "
                    f"({r['OR_lo']:.2f}-{r['OR_hi']:.2f}) p={r['fisher_p']:.3g} n={len(sub):,}")

    # 摘要 JSON
    summary = {
        "exposure_set": args.exposure,
        "n_target_reports": int(len(rep)),
        "n_exposed_reports": int(rep["exposed"].sum()),
        "aria_rate_exposed": float(ov.loc[0, "rate_exposed"]),
        "aria_rate_unexposed": float(ov.loc[0, "rate_unexposed"]),
        "overall_OR": float(ov.loc[0, "OR"]),
        "overall_OR_ci": [float(ov.loc[0, "OR_lo"]), float(ov.loc[0, "OR_hi"])],
        "overall_fisher_p": float(ov.loc[0, "fisher_p"]),
        "has_scipy": HAS_SCIPY,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (tdir / "ddi_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[输出] {tdir / 'ddi_summary.json'}")
    return summary


# ----------------------------------------------------------------------------
# 阶段 3：figures
# ----------------------------------------------------------------------------

def stage_figures(proj: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tdir = proj / "results" / "tables"
    fdir = proj / "results" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    # 森林图：逐合并用药 OR
    p = tdir / "ddi_by_exposure_drug.csv"
    if p.exists():
        d = pd.read_csv(p).sort_values("OR")
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.34 * len(d) + 1.2)), dpi=160)
        y = np.arange(len(d))
        ax.errorbar(d["OR"], y,
                    xerr=[d["OR"] - d["OR_lo"], d["OR_hi"] - d["OR"]],
                    fmt="o", color="#1f5fa8", ecolor="#7aa6d6", capsize=3, ms=6)
        ax.axvline(1.0, color="grey", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.exposure_drug} (n={int(r.n_exposed)})" for r in d.itertuples()])
        ax.set_xscale("log")
        ax.set_xlabel("Case-case odds ratio for ARIA (log scale)")
        ax.set_title("ARIA risk in anti-Abeta mAb reports co-reported with each drug")
        for i, r in enumerate(d.itertuples()):
            ax.text(r.OR_hi * 1.15, i, f"{r.OR:.2f} ({r.OR_lo:.2f}-{r.OR_hi:.2f})", va="center", fontsize=7)
        ax.grid(axis="x", ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fdir / "ddi_coreport_forest.png")
        plt.close(fig)
        log(f"[图] {fdir / 'ddi_coreport_forest.png'}")

    # 概念对比：暴露组 vs 单药组 事件率
    p2 = tdir / "ddi_by_concept.csv"
    if p2.exists():
        d = pd.read_csv(p2)
        fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=160)
        x = np.arange(len(d)); w = 0.38
        ax.bar(x - w / 2, d["rate_exposed"] * 100, w, label="co-reported", color="#c0392b")
        ax.bar(x + w / 2, d["rate_unexposed"] * 100, w, label="mAb alone", color="#7f8c8d")
        ax.set_xticks(x)
        ax.set_xticklabels(d["concept"], rotation=25, ha="right")
        ax.set_ylabel("Event proportion (%)")
        ax.set_title("ARIA concepts: co-reported vs mAb-alone (case-case comparison)")
        ax.legend()
        ax.grid(axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fdir / "ddi_concept_compare.png")
        plt.close(fig)
        log(f"[图] {fdir / 'ddi_concept_compare.png'}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="抗Aβ单抗合并用药的 FAERS 共报告(DDI)分析")
    ap.add_argument("stage", choices=["probe", "coreport", "figures", "all"])
    ap.add_argument("--proj", default=".", help="项目根目录")
    ap.add_argument("--exposure", default="anticoag_antiplatelet",
                    choices=list(EXPOSURE_SETS.keys()), help="暴露药集")
    ap.add_argument("--keep-all", action="store_true", help="保留所有目标药角色")
    args = ap.parse_args(argv)

    proj = Path(args.proj).resolve()
    if not proj.exists():
        log(f"[错误] 项目根不存在：{proj}")
        return 2

    t0 = time.time()
    if args.stage in ("probe",):
        stage_probe(proj, args)
    elif args.stage == "coreport":
        stage_coreport(proj, args)
    elif args.stage == "figures":
        stage_figures(proj, args)
    elif args.stage == "all":
        stage_coreport(proj, args)
        stage_figures(proj, args)
    log("=" * 70)
    log(f"完成，用时 {time.time()-t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
