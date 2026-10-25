#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04e 共报告(DDI)分析 —— 缺失值敏感性分析层（年龄 + 性别）

背景
----
04d 的校正模型采用完整案例分析（complete-case）：要求 age、sex 等协变量
全部非缺失，因此样本量从 6,951 份报告降到约 4,450 份（64%）。FAERS 中
年龄/性别缺失率偏高属常态（严重报告未强制填写），但审稿人必然追问：

    "三分之一的协变量缺失会不会让你的校正结果产生偏倚？"

本脚本用两条互补的敏感性分析回答这个问题，并与主分析（完整案例）对照：

  模型 1  complete-case          仅用 age、female 均非缺失的报告
                                 （= 04d 主分析，重现其数值）
  模型 2  分类化（含"未知"档）    年龄切 <65/65-74/75-84/85+/未知 五档，
                                 性别切 F/M/未知 三档，用全部 6,951 份报告，
                                 不丢弃任何记录
  模型 3  中位数填补 + 缺失指示    连续 age、female（缺失填中位数）+ 两个缺失
                                 哑变量，用全部报告

三条模型若给出方向一致、量级接近的校正 OR，即说明结论不依赖于
"缺失完全随机"这一强假设。

另外输出一项关键诊断：协变量缺失是否与暴露、结局相关。若不相关，
完整案例分析基本无偏；若相关，则必须以模型 2/3 为准并在正文说明。

输入
  results/tables/ddi_report_flags.parquet    由 04c 产出（04d 已更新 age 列）

输出
  results/tables/ddi_missing_sensitivity.csv        逐结局 × 三模型对照
  results/tables/ddi_missing_check.csv              缺失机制诊断
  results/tables/ddi_missing_sensitivity_summary.json
  results/figures/ddi_missing_sensitivity.png       三模型对照森林图

用法
  python 04e_ddi_missing_sens.py selftest                  # 离线自检（不读数据）
  python 04e_ddi_missing_sens.py sensitivity --proj .      # 敏感性分析
  python 04e_ddi_missing_sens.py all --proj .              # sensitivity + figures
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ----------------------------------------------------------------------------
# 基础工具（与 04d 完全一致的实现，保证两章数值可比）
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_cdf(x: float) -> float:
    """标准正态分布函数。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


OUTCOMES = [
    ("any_aria", "Any ARIA"),
    ("any_bleeding", "Any bleeding-related ARIA"),
    ("any_nonbleeding", "Any non-bleeding ARIA"),
    ("ARIA-E", "ARIA-E"),
    ("ARIA-H", "ARIA-H"),
    ("ARIA-NOS", "ARIA-NOS"),
    ("CEREBRAL_EDEMA", "Cerebral edema"),
    ("CEREBRAL_HEMORRHAGE", "Cerebral hemorrhage"),
    ("CEREBRAL_MICROHEMORRHAGE", "Cerebral microhemorrhage"),
    ("SUPERFICIAL_SIDEROSIS", "Superficial siderosis"),
]

AGE_BINS = [0, 65, 75, 85, 200]
AGE_LABELS = ["<65", "65-74", "75-84", "85+"]
AGE_UNKNOWN = "unknown"
# 模型 2 的年龄哑变量：以 <65 为参照
AGE_CAT_LEVELS = ["65-74", "75-84", "85+", AGE_UNKNOWN]
# 模型 2 的性别哑变量：以 F 为参照
SEX_UNKNOWN = "unknown"


# ----------------------------------------------------------------------------
# IRLS logistic 回归（与 04d 相同，零额外依赖）
# ----------------------------------------------------------------------------

def irls_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 200,
               tol: float = 1e-9):
    """用迭代重加权最小二乘拟合 logistic 回归，返回 (beta, se)。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        XW = X * w[:, None]
        H = X.T @ XW
        try:
            Hinv = np.linalg.pinv(H, rcond=1e-12)
        except np.linalg.LinAlgError:
            break
        beta_new = Hinv @ (XW.T @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-10, None)
    H = X.T @ (X * w[:, None])
    cov = np.linalg.pinv(H, rcond=1e-12)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return beta, se


def crude_or(exposed: pd.Series, outcome: pd.Series) -> dict:
    """2x2 粗 OR 与 95%CI（Woolf 对数法）。"""
    e = exposed.astype(bool)
    y = outcome.astype(bool)
    a = int((e & y).sum())
    b = int((e & ~y).sum())
    c = int((~e & y).sum())
    d = int((~e & ~y).sum())
    out = {"a": a, "b": b, "c": c, "d": d}
    if min(a, b, c, d) == 0:
        out.update({"OR": float("nan"), "OR_lo": float("nan"),
                    "OR_hi": float("nan")})
        return out
    orv = (a * d) / (b * c)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    out.update({
        "OR": orv,
        "OR_lo": math.exp(math.log(orv) - 1.96 * se),
        "OR_hi": math.exp(math.log(orv) + 1.96 * se),
    })
    return out


def fisher_p(a: int, b: int, c: int, d: int) -> float:
    """2x2 表双侧 Fisher 精确检验（scipy 不可用时返回 nan）。"""
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------
# 数据准备
# ----------------------------------------------------------------------------

def prepare_base(rep: pd.DataFrame) -> pd.DataFrame:
    """
    构造三模型共用的基础列：year_c / 靶药哑变量 / 缺失标记。
    age、female 以原始形式保留，供三个模型各自处理。
    """
    d = pd.DataFrame(index=rep.index)
    d["exposed"] = rep["exposed"].astype(bool).astype(int)

    if "age" in rep.columns:
        a = pd.to_numeric(rep["age"], errors="coerce")
        d["age"] = a.where((a >= 18) & (a <= 110))
    else:
        d["age"] = np.nan

    if "sex" in rep.columns:
        s = rep["sex"].astype(str).str.upper().str.strip().str[:1]
        d["female"] = s.map({"F": 1.0, "M": 0.0})
    else:
        d["female"] = np.nan

    if "year" in rep.columns:
        y = pd.to_numeric(rep["year"], errors="coerce")
        d["year_c"] = (y - 2022.0).where((y >= 1990) & (y <= 2030))

    if "target_primary" in rep.columns:
        tp = rep["target_primary"].astype(str).str.upper()
        for key, short in [("LECANEMAB", "lecanemab"),
                           ("DONANEMAB", "donanemab"),
                           ("ADUCANUMAB", "aducanumab")]:
            if (tp == key).any():
                d[f"tgt_{short}"] = (tp == key).astype(float)

    d["age_missing"] = d["age"].isna().astype(int)
    d["female_missing"] = d["female"].isna().astype(int)
    return d


def common_covariates(design: pd.DataFrame) -> list:
    """三模型共用的校正变量（不含 age / female）。"""
    base = ["year_c"]
    tgt = [c for c in sorted(design.columns) if c.startswith("tgt_")]
    return [c for c in base if c in design.columns] + tgt


def build_missing_terms(design: pd.DataFrame) -> pd.DataFrame:
    """为模型 2/3 追加 age / female 的缺失处理列，返回新表。"""
    d = design.copy()

    # ---- 模型 2：age 5 分类（含"未知"）
    cat = pd.cut(d["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    cat = cat.cat.add_categories([AGE_UNKNOWN]).fillna(AGE_UNKNOWN)
    for lv in AGE_CAT_LEVELS:
        d[f"agecat_{lv}"] = (cat == lv).astype(float)

    # ---- 模型 2：female 3 分类（F 为参照）
    fs = d["female"].map({1.0: "F", 0.0: "M"}).fillna(SEX_UNKNOWN)
    d["sexcat_M"] = (fs == "M").astype(float)
    d[f"sexcat_{SEX_UNKNOWN}"] = (fs == SEX_UNKNOWN).astype(float)

    # ---- 模型 3：中位数填补 + 缺失指示
    amed = float(d["age"].median()) if d["age"].notna().any() else 0.0
    fmed = float(d["female"].median()) if d["female"].notna().any() else 0.0
    d["age_imp"] = d["age"].fillna(amed)
    d["female_imp"] = d["female"].fillna(fmed)
    d.attrs["age_median"] = amed
    d.attrs["female_median"] = fmed
    return d


def fit_model(design: pd.DataFrame, y: pd.Series, cols: list, min_n: int = 50):
    """
    拟合 logistic 回归，返回 exposed 的校正 OR。
    缺失值整行剔除；样本不足或结局无变异时返回 None。
    """
    use = ["exposed"] + [c for c in cols if c in design.columns]
    sub = design[use].dropna()
    if len(sub) < min_n:
        return None
    yv = y.loc[sub.index].astype(int).values
    if yv.min() == yv.max():
        return None

    X = np.column_stack([np.ones(len(sub)), sub.values.astype(float)])
    beta, se = irls_logit(X, yv)
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(se)):
        return None

    b, s = float(beta[1]), float(se[1])
    if s <= 0 or not math.isfinite(s):
        return None
    z = b / s
    return {
        "n_model": int(len(sub)),
        "n_event": int(yv.sum()),
        "OR": math.exp(b),
        "OR_lo": math.exp(b - 1.96 * s),
        "OR_hi": math.exp(b + 1.96 * s),
        "p": 2.0 * (1.0 - norm_cdf(abs(z))),
        "beta": b,
        "se": s,
    }


# ----------------------------------------------------------------------------
# 敏感性分析主逻辑
# ----------------------------------------------------------------------------

def analyze_sensitivity(design: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    common = common_covariates(design)
    ds = build_missing_terms(design)

    cols_cc = [c for c in ["age", "female"] if c in design.columns] + common
    cols_cat = ([f"agecat_{lv}" for lv in AGE_CAT_LEVELS if f"agecat_{lv}" in ds.columns]
                + [c for c in ["sexcat_M", f"sexcat_{SEX_UNKNOWN}"] if c in ds.columns]
                + common)
    cols_mi = ([c for c in ["age_imp", "age_missing", "female_imp", "female_missing"]
                if c in ds.columns] + common)

    log(f"[模型1 完整案例]   {cols_cc}")
    log(f"[模型2 分类化]     {cols_cat}")
    log(f"[模型3 填补+指示]  {cols_mi}")

    for c in ["age", "female"]:
        if c in ds.columns:
            nn = int(ds[c].notna().sum())
            log(f"[缺失] {c:7s} 非缺失 {nn:,} / {len(ds):,}（{ds[c].notna().mean():.1%}）")

    both = ds[["age", "female"]].notna().all(axis=1) if "female" in ds.columns \
        else ds[["age"]].notna().all(axis=1)
    log(f"[完整案例可用样本] {int(both.sum()):,} / {len(ds):,}（{both.mean():.1%}）")

    if "age" in ds.columns and ds["age"].notna().any():
        cat = pd.cut(ds["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        cat = cat.cat.add_categories([AGE_UNKNOWN]).fillna(AGE_UNKNOWN)
        cnt = cat.value_counts().reindex(AGE_LABELS + [AGE_UNKNOWN]).fillna(0).astype(int)
        log("[年龄分布] " + " | ".join(f"{k} {v}" for k, v in cnt.items()))

    rows = []
    for key, label in OUTCOMES:
        col = f"ev_{key}"
        if col not in rep.columns:
            log(f"[跳过] 缺列 {col}")
            continue
        y = rep[col].astype(bool)
        cr = crude_or(design["exposed"], y)
        row = {
            "outcome": key, "label": label, "n_total": int(len(rep)),
            "n_event": int(y.sum()),
            "crude_OR": cr["OR"], "crude_OR_lo": cr["OR_lo"],
            "crude_OR_hi": cr["OR_hi"],
        }
        for tag, cols in [("cc", cols_cc), ("cat", cols_cat), ("mi", cols_mi)]:
            r = fit_model(ds, y, cols)
            if r:
                for k in ("OR", "OR_lo", "OR_hi", "p", "n_model"):
                    row[f"{tag}_{k}"] = r[k]
        rows.append(row)

        def g(tag):
            v = row.get(f"{tag}_OR")
            return f"{v:.3f}" if isinstance(v, float) else "  -  "

        log(f"  {label:26s} crude {g('crude')} | 完整案例 {g('cc')} | "
            f"分类化 {g('cat')} | 填补+指示 {g('mi')}  "
            f"(n_cc={row.get('cc_n_model','-')}, n_cat={row.get('cat_n_model','-')})")
    return pd.DataFrame(rows)


def missingness_check(design: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    """
    诊断缺失机制：缺失是否与暴露、结局、另一协变量相关。
    若与暴露显著相关 → 完整案例分析可能有选择偏倚。
    """
    if "age_missing" not in design.columns:
        return pd.DataFrame()
    rows = []

    def add(label, m):
        e = design["exposed"].astype(bool).astype(int)
        a = int(((m == 1) & (e == 1)).sum())
        b = int(((m == 1) & (e == 0)).sum())
        c = int(((m == 0) & (e == 1)).sum())
        d = int(((m == 0) & (e == 0)).sum())
        rows.append({
            "comparison": label, "n_missing": a + b, "n_observed": c + d,
            "exposed_rate_missing": a / (a + b) if (a + b) else float("nan"),
            "exposed_rate_observed": c / (c + d) if (c + d) else float("nan"),
            "a": a, "b": b, "c": c, "d": d, "fisher_p": fisher_p(a, b, c, d),
        })

    add("age_missing vs exposed", design["age_missing"].astype(int))
    if "female_missing" in design.columns:
        add("female_missing vs exposed", design["female_missing"].astype(int))
        # age 缺失 vs female 缺失（缺失是否成簇）
        m = design["age_missing"].astype(int)
        f = design["female_missing"].astype(int)
        a = int(((m == 1) & (f == 1)).sum())
        b = int(((m == 1) & (f == 0)).sum())
        c = int(((m == 0) & (f == 1)).sum())
        d = int(((m == 0) & (f == 0)).sum())
        rows.append({
            "comparison": "age_missing vs female_missing",
            "n_missing": a + b, "n_observed": c + d,
            "exposed_rate_missing": a / (a + b) if (a + b) else float("nan"),
            "exposed_rate_observed": c / (c + d) if (c + d) else float("nan"),
            "a": a, "b": b, "c": c, "d": d,
            "fisher_p": fisher_p(a, b, c, d),
        })

    # 缺失 vs 结局
    m = design["age_missing"].astype(int)
    for key in ["any_aria", "any_bleeding", "any_nonbleeding", "ARIA-E", "ARIA-H"]:
        col = f"ev_{key}"
        if col not in rep.columns:
            continue
        y = rep[col].astype(bool).astype(int)
        a = int(((m == 1) & (y == 1)).sum())
        b = int(((m == 1) & (y == 0)).sum())
        c = int(((m == 0) & (y == 1)).sum())
        d = int(((m == 0) & (y == 0)).sum())
        rows.append({
            "comparison": f"age_missing vs {key}",
            "n_missing": a + b, "n_observed": c + d,
            "exposed_rate_missing": a / (a + b) if (a + b) else float("nan"),
            "exposed_rate_observed": c / (c + d) if (c + d) else float("nan"),
            "a": a, "b": b, "c": c, "d": d, "fisher_p": fisher_p(a, b, c, d),
        })

    df = pd.DataFrame(rows)
    log("[缺失机制诊断]（fisher_p 为双侧精确检验）")
    for _, r in df.iterrows():
        log(f"  {r['comparison']:34s} 缺失组 {r['exposed_rate_missing']:.3f} vs "
            f"观察组 {r['exposed_rate_observed']:.3f} | p={r['fisher_p']:.4f}")
    return df


def plot_forest(df: pd.DataFrame, outpath: Path) -> None:
    if not HAS_MPL:
        log("[跳过] 未安装 matplotlib")
        return
    d = df.dropna(subset=["crude_OR"]).copy()
    if d.empty:
        return
    d = d.iloc[::-1].reset_index(drop=True)
    ypos = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.8, 0.78 * len(d) + 2.3))
    series = [
        ("crude", "o", "#8a8a8a", "#bdbdbd", 0.30, "Crude OR"),
        ("cc", "s", "#c0392b", "#e6a6a0", 0.10,
         "Adjusted OR (complete case)"),
        ("cat", "D", "#1f6feb", "#a9c7f5", -0.10,
         "Adjusted OR (categorical, all reports)"),
        ("mi", "^", "#1a7f37", "#a8d5b5", -0.30,
         "Adjusted OR (imputation + missing flags)"),
    ]
    for tag, mk, cc_, ec, off, lab in series:
        orc, lo, hi = f"{tag}_OR", f"{tag}_OR_lo", f"{tag}_OR_hi"
        if orc not in d.columns or not d[orc].notna().any():
            continue
        m = d[orc].notna() & d[lo].notna() & d[hi].notna()
        ax.errorbar(d.loc[m, orc], ypos[m] + off,
                    xerr=[d.loc[m, orc] - d.loc[m, lo], d.loc[m, hi] - d.loc[m, orc]],
                    fmt=mk, color=cc_, ecolor=ec, elinewidth=1.8,
                    markersize=6, capsize=3, label=lab)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title("Sensitivity to missing age/sex: complete-case vs categorical "
                 "vs imputation\ncase/non-case analysis within anti-Abeta mAb reports")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    log(f"[图] {outpath}")


# ----------------------------------------------------------------------------
# 离线自检
# ----------------------------------------------------------------------------

def run_selftest() -> int:
    log("=" * 70)
    log("04e 自检：三种缺失处理能否在 MCAR / MNAR 下恢复已知效应")
    log("=" * 70)
    checks = []
    rng = np.random.RandomState(42)
    n = 60000
    TRUE_OR = 1.5

    x = (rng.rand(n) < 0.10).astype(float)
    age_full = np.clip(rng.normal(74, 9, n), 40, 98)
    female_full = (rng.rand(n) < 0.60).astype(float)
    year = rng.uniform(2018, 2026, n)
    lp = (math.log(0.30) + math.log(TRUE_OR) * x
          + 0.008 * (age_full - 74) - 0.10 * female_full + 0.05 * (year - 2022))
    y = (rng.rand(n) < 1.0 / (1.0 + np.exp(-lp))).astype(int)
    ys = pd.Series(y)

    def build(age_in, fem_in):
        d = pd.DataFrame({"exposed": x, "age": age_in, "female": fem_in,
                          "year_c": year - 2022.0})
        d["age_missing"] = d["age"].isna().astype(int)
        d["female_missing"] = d["female"].isna().astype(int)
        return d

    def three(d):
        ds = build_missing_terms(d)
        cc = fit_model(ds, ys, ["age", "female", "year_c"])
        cat = fit_model(ds, ys, [f"agecat_{lv}" for lv in AGE_CAT_LEVELS]
                        + ["sexcat_M", f"sexcat_{SEX_UNKNOWN}", "year_c"])
        mi = fit_model(ds, ys, ["age_imp", "age_missing", "female_imp",
                                "female_missing", "year_c"])
        return cc, cat, mi

    # --- MCAR：age 缺 35% + sex 缺 15%
    a1 = age_full.copy(); f1 = female_full.copy()
    a1[rng.rand(n) < 0.35] = np.nan
    f1[rng.rand(n) < 0.15] = np.nan
    cc, cat, mi = three(build(a1, f1))
    for tag, r, tol in [("完整案例", cc, 0.15), ("分类化", cat, 0.15), ("填补+指示", mi, 0.15)]:
        ok = r is not None and abs(r["OR"] - TRUE_OR) < tol
        checks.append((f"MCAR {tag}恢复真值 1.5", ok,
                       f"OR={r['OR']:.4f}" if r else "拟合失败"))

    # --- MCAR 下三模型应彼此接近
    if cc and cat and mi:
        spread = max(cc["OR"], cat["OR"], mi["OR"]) - min(cc["OR"], cat["OR"], mi["OR"])
        checks.append(("MCAR 三模型最大差 <0.10", spread < 0.10, f"最大差 {spread:.4f}"))

    # --- MNAR-lite：老年人更易缺年龄、女性更易缺性别
    pmiss_a = 1.0 / (1.0 + np.exp(-((age_full - 78) / 4.0)))
    a2 = age_full.copy(); a2[rng.rand(n) < pmiss_a * 0.6] = np.nan
    f2 = female_full.copy(); f2[(female_full == 1) & (rng.rand(n) < 0.30)] = np.nan
    cc2, cat2, mi2 = three(build(a2, f2))
    ok2 = all(r is not None and abs(r["OR"] - TRUE_OR) < 0.20 for r in (cc2, cat2, mi2))
    checks.append(("MNAR-lite 三模型均近似真值(±0.20)", ok2,
                   " | ".join(f"{r['OR']:.3f}" if r else "fail" for r in (cc2, cat2, mi2))))

    # --- 零效应：CI 覆盖 1
    lp0 = math.log(0.30) + 0.008 * (age_full - 74)
    y0 = pd.Series((rng.rand(n) < 1.0 / (1.0 + np.exp(-lp0))).astype(int))
    ds0 = build_missing_terms(build(a1, f1))
    r0 = fit_model(ds0, y0, ["age_imp", "age_missing", "female_imp",
                             "female_missing", "year_c"])
    ok0 = r0 and r0["OR_lo"] <= 1.0 <= r0["OR_hi"]
    checks.append(("零效应时 95%CI 覆盖 1", ok0,
                   f"OR={r0['OR']:.3f} ({r0['OR_lo']:.3f}-{r0['OR_hi']:.3f})"
                   if r0 else "拟合失败"))

    # --- 无缺失时，模型1与模型3一致
    dn = build(age_full, female_full)
    ccn = fit_model(build_missing_terms(dn), ys, ["age", "female", "year_c"])
    min_ = fit_model(build_missing_terms(dn), ys,
                     ["age_imp", "age_missing", "female_imp", "female_missing", "year_c"])
    ok3 = ccn and min_ and abs(ccn["OR"] - min_["OR"]) < 1e-6
    checks.append(("无缺失时模型1与模型3一致", ok3,
                   f"差 {abs(ccn['OR'] - min_['OR']):.2e}" if (ccn and min_) else "拟合失败"))

    # --- 分类化模型在 age 全缺时仍可运行（用 sex + year）
    a3 = np.full(n, np.nan)
    _, cat3, _ = three(build(a3, female_full))
    ok4 = cat3 is not None
    checks.append(("age 全缺时分类化模型仍可运行", ok4,
                   f"OR={cat3['OR']:.4f}" if cat3 else "失败"))

    # --- 完全分离不崩溃
    dsx = pd.DataFrame({"exposed": np.r_[np.ones(20), np.zeros(20)],
                        "age_imp": np.full(40, 70.0),
                        "age_missing": np.zeros(40)})
    yx = pd.Series(np.r_[np.ones(20), np.zeros(20)])
    try:
        fit_model(dsx, yx, ["age_imp", "age_missing"])
        checks.append(("完全分离不崩溃", True, "已优雅处理"))
    except Exception as e:
        checks.append(("完全分离不崩溃", False, f"{type(e).__name__}: {e}"))

    # --- 空数据不崩溃
    try:
        fit_model(pd.DataFrame({"exposed": []}), pd.Series([], dtype=float), [])
        checks.append(("空数据不崩溃", True, "已优雅处理"))
    except Exception as e:
        checks.append(("空数据不崩溃", False, f"{type(e).__name__}: {e}"))

    print()
    npass = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  —— {detail}")
        npass += int(ok)
    print()
    if npass == len(checks):
        print("=== selftest PASSED ===")
        return 0
    print(f"=== selftest FAILED（{npass}/{len(checks)}）===")
    return 1


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def load_flags(proj: Path) -> pd.DataFrame:
    p = proj / "results" / "tables" / "ddi_report_flags.parquet"
    if not p.exists():
        raise FileNotFoundError(f"缺 {p}，请先运行 04c 的 coreport 阶段")
    rep = pd.read_parquet(p)
    log(f"[读入] {p.name}：{len(rep):,} 行")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="04e 共报告分析 —— 缺失值敏感性分析")
    ap.add_argument("stage", choices=["sensitivity", "figures", "all", "selftest"])
    ap.add_argument("--proj", default=".")
    args = ap.parse_args()

    if args.stage == "selftest":
        return run_selftest()

    proj = Path(args.proj).resolve()
    tdir = proj / "results" / "tables"
    fdir = proj / "results" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"04e 缺失值敏感性分析（age + sex）| 项目根 {proj}")
    log("=" * 70)

    if args.stage in ("sensitivity", "all"):
        rep = load_flags(proj)
        design = prepare_base(rep)
        log(f"[样本] 报告 {len(rep):,} 份 | 暴露 {int(design['exposed'].sum()):,} 份")

        df = analyze_sensitivity(design, rep)
        out_csv = tdir / "ddi_missing_sensitivity.csv"
        df.to_csv(out_csv, index=False, float_format="%.6g")
        log(f"[输出] {out_csv}")

        chk = missingness_check(design, rep)
        if not chk.empty:
            out_chk = tdir / "ddi_missing_check.csv"
            chk.to_csv(out_chk, index=False, float_format="%.6g")
            log(f"[输出] {out_chk}")

        def nm(tag):
            v = df.get(f"{tag}_n_model")
            return int(v.dropna().max()) if v is not None and v.notna().any() else 0

        summary = {
            "n_reports": int(len(rep)),
            "n_exposed": int(design["exposed"].sum()),
            "n_age_observed": int(design["age"].notna().sum()),
            "age_observed_rate": float(design["age"].notna().mean()),
            "n_female_observed": int(design["female"].notna().sum()),
            "female_observed_rate": float(design["female"].notna().mean()),
            "n_model_complete_case": nm("cc"),
            "n_model_categorical": nm("cat"),
            "n_model_missing_indicator": nm("mi"),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_json = tdir / "ddi_missing_sensitivity_summary.json"
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        log(f"[输出] {out_json}")

    if args.stage in ("figures", "all"):
        p = tdir / "ddi_missing_sensitivity.csv"
        if p.exists():
            plot_forest(pd.read_csv(p), fdir / "ddi_missing_sensitivity.png")
        else:
            log(f"[跳过] 缺 {p}，请先运行 sensitivity 阶段")

    log(f"完成，用时 {time.time() - _T0:.1f} 秒")
    return 0


_T0 = time.time()

if __name__ == "__main__":
    raise SystemExit(main())
