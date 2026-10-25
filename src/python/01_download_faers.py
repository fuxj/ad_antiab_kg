#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_download_faers.py

FAERS 季度 ASCII 数据全流程：下载 -> 完整性校验 -> 抽取 -> 归一化 -> 去重 -> Parquet

设计要点（全部依据对 FDA 官方数据的实测，不是推测）
--------------------------------------------------
1) 下载地址两种前缀：
     2004Q1 ~ 2012Q3 : aers_ascii_YYYYqN.zip   (旧 AERS)
     2012Q4 ~ 至今    : faers_ascii_YYYYqN.zip  (新 FAERS)
   实测切换点：aers 用至 2012q3，faers 自 2012q4 起。

2) 包内文件名大小写不固定，且随年份变化：
     2004q1 -> ascii/DEMO04Q1.TXT
     2012q4 -> ascii/demo12q4.txt     (全小写)
     2013q1 -> ascii/DEMO13Q1.txt
   因此一律按“小写后比较”查找文件，绝不写死文件名。

3) 列结构跨年份在变（这是最容易静默出错的地方）：
     列名变体： ISR/CASE/GNDR_COD  ->  primaryid/caseid/sex
     列的增删： 2016Q1 起新增 auth_num、lit_ref、age_grp
     大小写与空格： 2012q4 的表头里出现过 ' wt_cod$ rept_dt'（字段名带前导空格）
   因此本脚本【读表头后按别名表映射】，不做任何列位置假设。

4) 去重规则（FDA 推荐）：
     同一 CASEID 保留 FDA_DT 最大者；若 FDA_DT 相同，保留 PRIMARYID 最大者。

5) 断点续传与磁盘保护：
     下载用 HTTP Range 续传；每次下载前检查剩余空间，低于阈值直接中止。

用法示例
--------
  # 先看计划（不下载、不写文件）
  python 01_download_faers.py plan --start 2004q1 --end 2026q2

  # 全流程（下载 + 校验 + 抽取 + 构建）
  python 01_download_faers.py all --start 2004q1 --end 2026q2 --workers 3

  # 只下载
  python 01_download_faers.py download --start 2026q1 --end 2026q2

  # 只构建（zip 已在本地时）
  python 01_download_faers.py build --start 2004q1 --end 2026q2

  # 只跑 3 个季度做冒烟测试
  python 01_download_faers.py all --quarters 2004q1,2012q4,2019q1

修订记录
--------
v1.2  修复包内文件名带 _new 后缀（如 2018q1 的 DEMO18Q1_new.txt）时整表被漏抽的问题。
v1.1  修复 build 阶段 PRIMARYID -> CASEID 映射表索引非唯一导致的
      InvalidIndexError（仅在几十个季度的大数据量下才会触发）。
      映射表改为按 PRIMARYID 去重后建索引；另对回填时未匹配到的行显式报警。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
BASE_URL = "https://fis.fda.gov/content/Exports"
UA = "Mozilla/5.0 (X11; Linux x86_64) faers-kg-research/1.0"
PREFIX_SWITCH = (2012, 4)          # 2012Q4 起改用 faers_ 前缀
TABLES = ["DEMO", "DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI"]

# 列别名表：canon -> [可能的原始名（小写、去空格后）]
ALIASES = {
    "primaryid": ["primaryid", "isr"],
    "caseid": ["caseid", "case"],
    "caseversion": ["caseversion"],
    "fda_dt": ["fda_dt"],
    "event_dt": ["event_dt"],
    "age": ["age"],
    "age_cod": ["age_cod"],
    "sex": ["sex", "gndr_cod"],
    "wt": ["wt"],
    "wt_cod": ["wt_cod"],
    "rept_cod": ["rept_cod"],
    "occp_cod": ["occp_cod"],
    "reporter_country": ["reporter_country"],
    "occr_country": ["occr_country"],
    "mfr_sndr": ["mfr_sndr"],
    "drug_seq": ["drug_seq", "dsg_drug_seq", "indi_drug_seq"],
    "role_cod": ["role_cod"],
    "drugname": ["drugname"],
    "prod_ai": ["prod_ai"],
    "route": ["route"],
    "pt": ["pt"],
    "outc_cod": ["outc_cod", "outc_code"],
    "rpsr_cod": ["rpsr_cod"],
    "start_dt": ["start_dt"],
    "end_dt": ["end_dt"],
    "dur": ["dur"],
    "dur_cod": ["dur_cod"],
    "indi_pt": ["indi_pt"],
}

# 每张表要保留的规范列
TABLE_COLS = {
    "DEMO": ["primaryid", "caseid", "caseversion", "fda_dt", "event_dt", "age",
             "age_cod", "sex", "wt", "wt_cod", "rept_cod", "occp_cod",
             "reporter_country", "occr_country", "mfr_sndr"],
    "DRUG": ["primaryid", "caseid", "drug_seq", "role_cod", "drugname",
             "prod_ai", "route"],
    "REAC": ["primaryid", "caseid", "pt"],
    "OUTC": ["primaryid", "caseid", "outc_cod"],
    "RPSR": ["primaryid", "caseid", "rpsr_cod"],
    "THER": ["primaryid", "caseid", "drug_seq", "start_dt", "end_dt", "dur",
             "dur_cod"],
    "INDI": ["primaryid", "caseid", "drug_seq", "indi_pt"],
}

_LOGFH = None


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOGFH is not None:
        _LOGFH.write(line + "\n")
        _LOGFH.flush()


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}PB"


def default_proj_root() -> Path:
    """脚本位于 <proj>/src/python/01_download_faers.py，上溯两级即项目根。"""
    here = Path(__file__).resolve()
    if here.parent.name == "python" and here.parent.parent.name == "src":
        return here.parent.parent.parent
    return Path.cwd()


def parse_quarter(s: str) -> tuple:
    """'2026q2' / '2026Q2' / '2026-2' -> (2026, 2)"""
    s = s.strip().lower().replace("-", "").replace(" ", "")
    if "q" in s:
        y, q = s.split("q")
        return int(y), int(q)
    if len(s) == 6:                     # 202602
        return int(s[:4]), int(s[4:])
    raise ValueError(f"无法解析季度: {s!r}，请用形如 2026q2")


def qstr(y: int, q: int) -> str:
    return f"{y}q{q}"


def quarter_range(start: tuple, end: tuple):
    y, q = start
    while (y, q) <= end:
        yield y, q
        q += 1
        if q > 4:
            y, q = y + 1, 1


def zip_name(y: int, q: int) -> str:
    prefix = "aers" if (y, q) < PREFIX_SWITCH else "faers"
    return f"{prefix}_ascii_{y}q{q}.zip"


def table_basename(table: str, y: int, q: int) -> str:
    """包内文件名，统一小写比较（真实数据存在全小写与全大写两种）。"""
    return f"{table.lower()}{y % 100:02d}q{q}.txt"


# ----------------------------------------------------------------------------
# 网络
# ----------------------------------------------------------------------------
def remote_size(url: str, timeout: int = 40) -> int:
    """用 Range 探测远端大小；失败返回 -1。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                tail = cr.rsplit("/", 1)[1].strip()
                if tail.isdigit():
                    return int(tail)
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else -1
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return -2               # 明确不存在
        return -1
    except Exception:
        return -1


def download(url: str, dest: Path, retries: int = 3, timeout: int = 90,
             quiet: bool = False, total: int = 0) -> bool:
    """带 HTTP Range 断点续传的下载。返回是否成功。

    total: 远端文件字节数（0 表示未知）。用于防御"续传起点越界"：
           本地文件若已达到或超过远端大小，必须截断重下 —— 否则
           Range 起点落在 EOF 之后，服务端返回 HTTP 416，永远下不完。
    """
    for attempt in range(1, retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        if total > 0 and existing >= total:
            existing = 0                      # 起点越界，从头下载
        headers = {"User-Agent": UA}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status = getattr(r, "status", 200)
                if existing > 0 and status != 206:
                    # 服务端不支持续传，从头来
                    mode, existing = "wb", 0
                last = time.time()
                with open(dest, mode) as fh:
                    while True:
                        chunk = r.read(1024 * 512)
                        if not chunk:
                            break
                        fh.write(chunk)
                        if not quiet and time.time() - last > 20:
                            log(f"      下载中 {dest.name} "
                                f"{human(dest.stat().st_size)}")
                            last = time.time()
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416:
                # 续传起点越界：通常是本地文件已完整但内容损坏。
                # 删掉重来，这一次调用内就能拿到干净文件。
                dest.unlink(missing_ok=True)
                if not quiet:
                    log(f"      [重来] {dest.name}: 续传起点越界(HTTP 416)，"
                        f"已删除本地文件，从头下载")
                continue
            if not quiet:
                log(f"      [重试 {attempt}/{retries}] {dest.name}: {e}")
            time.sleep(2 * attempt)
        except Exception as e:
            if not quiet:
                log(f"      [重试 {attempt}/{retries}] {dest.name}: {e}")
            time.sleep(2 * attempt)
    return False


def verify_zip(path: Path) -> bool:
    """校验 ZIP 完整性。真实数据里截断下载很常见，必须逐条校。"""
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            return bad is None
    except Exception:
        return False


# ----------------------------------------------------------------------------
# 抽取
# ----------------------------------------------------------------------------
_ZIP_TABLE_RE = re.compile(
    r"(demo|drug|reac|outc|rpsr|ther|indi)(\d{2})q(\d)(_[a-z0-9]+)?")


def extract_tables(zip_path: Path, y: int, q: int, dest_dir: Path,
                   tables) -> dict:
    """按“小写文件名比较”抽取需要的表，返回 {table: 抽取后的路径}。

    文件名匹配用正则而非精确相等：FDA 偶尔会重发季度包的修正版，把表文件命名为
    <表>NNqN_new.txt。已实测：2018q1 的 DEMO 就叫 DEMO18Q1_new.txt，
    精确相等匹配会整季漏掉该表（而且只在建表时静默少一份，不报错）。
    因此这里按 <表><两位年>q<季>[后缀].txt 匹配；带后缀的视为修正版，优先采用。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    want = {t.upper() for t in tables}
    yq = f"{y % 100:02d}q{q}"
    got, ranks = {}, {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name.lower()
            if not base.endswith(".txt"):
                continue
            m = _ZIP_TABLE_RE.fullmatch(base[:-4])
            if not m:
                continue
            table = m.group(1).upper()
            if table not in want or f"{m.group(2)}q{m.group(3)}" != yq:
                continue
            rank = 1 if m.group(4) else 0
            if ranks.get(table, -1) >= rank:
                continue
            out = dest_dir / base
            if not out.exists() or out.stat().st_size != info.file_size:
                with z.open(info) as src, open(out, "wb") as fh:
                    shutil.copyfileobj(src, fh, 1024 * 256)
            got[table] = out
            ranks[table] = rank
    return got


def collect_deleted(zip_path: Path, dest_dir: Path) -> list:
    """抽取 deleted/ 下的累计删除病例表（2019Q1 起才有）。返回文件路径列表。"""
    out_paths = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir():
                continue
            if name.lower().endswith(".txt") and "deleted" in info.filename.lower():
                out = dest_dir / name
                with z.open(info) as src, open(out, "wb") as fh:
                    shutil.copyfileobj(src, fh, 1024 * 256)
                out_paths.append(out)
    return out_paths


# ----------------------------------------------------------------------------
# 读表 + 归一化
# ----------------------------------------------------------------------------
def _clean_col(c: str) -> str:
    """清洗列名。

    真实数据里踩过的坑：
      · 2012q4 的 drug12q4.txt 带 UTF-8 BOM；
        若用 latin-1 读，BOM 会变成 3 个可见字符 'i>?'，导致首列名不再是 primaryid；
      · 2012q4 的表头里出现过 ' wt_cod$ rept_dt'，字段名带前导空格。
    因此这里统一：去 BOM、去首尾空白、转小写。
    """
    s = str(c)
    for bom in ("\ufeff", "\xef\xbb\xbf", "i>?"):
        if s.startswith(bom):
            s = s[len(bom):]
    return s.strip().lower()


def read_ascii(path: Path) -> pd.DataFrame:
    """按 $ 分隔读取。

    编码用 utf-8-sig：既自动吞掉 UTF-8 BOM，又不会像 latin-1 那样把 BOM 读成乱码；
    配 encoding_errors='replace' 保证遇到个别非 UTF-8 字节也不会中断整表读取。
    引号一律不解析（药名里可能含引号）。

    index_col=False 是必须的：旧 AERS（2004Q1~2012Q3）的数据行末尾多一个 '$'，
    数据行字段数比表头多 1。不加这个参数，pandas 会把首列当成行索引，
    导致【所有列整体右移一格】——表照样读得出来，但每一列都是错的，极难察觉。
    """
    df = pd.read_csv(
        path,
        sep="$",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8-sig",
        encoding_errors="replace",
        index_col=False,
        engine="c",
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="warn",
        low_memory=False,
    )
    df.columns = [_clean_col(c) for c in df.columns]

    # 列对齐体检：首列应为纯数字 ID（ISR / PRIMARYID）。
    # 一旦出现大量非数字，说明列发生了错位（例如忘了 index_col=False），
    # 这种情况表仍能读出，但每一列的值都是错的，必须显式报警。
    if len(df) and df.shape[1] > 1:
        col0 = df.columns[0]
        sample = df[col0].head(500)
        n_bad = int((~sample.str.fullmatch(r"\d+")).sum())
        if n_bad > 50:
            log(f"      [警告] {Path(path).name}: 首列 {col0!r} 有 "
                f"{n_bad}/{len(sample)} 行不是纯数字 ID，疑似列对齐异常，请核对！")
    return df


def normalize(df: pd.DataFrame, table: str) -> tuple:
    """按别名表把原始列名映射为规范列名，只保留需要的列。"""
    present = set(df.columns)
    rename = {}
    for canon, aliases in ALIASES.items():
        if canon in present:
            continue
        for a in aliases:
            if a in present and a not in rename:
                rename[a] = canon
                break
    if rename:
        df = df.rename(columns=rename)
    keep = [c for c in TABLE_COLS[table] if c in df.columns]
    missing = [c for c in TABLE_COLS[table] if c not in df.columns]
    return df[keep], missing


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def stage_plan(quarters, out: Path) -> None:
    log("计划（不下载、不写文件）:")
    for y, q in quarters:
        url = f"{BASE_URL}/{zip_name(y, q)}"
        log(f"  {qstr(y, q):>8}  {zip_name(y, q)}")
        log(f"            {url}")
        log(f"            抽取目标: "
            + ", ".join(table_basename(t, y, q) for t in TABLES))
    log(f"共 {len(quarters)} 个季度；输出根目录: {out}")


def _fetch_one(y: int, q: int, raw_dir: Path, min_free_gb: float,
               lock: threading.Lock, quiet: bool):
    """下载单个季度并校验。返回 (y, q, 路径或 None, 结果码)。"""
    name = zip_name(y, q)
    dest = raw_dir / name
    url = f"{BASE_URL}/{name}"

    with lock:
        if dest.exists():
            if verify_zip(dest):
                log(f"[跳过] {name} 已存在且完整 ({human(dest.stat().st_size)})")
                return (y, q, dest, "skip")
            # 本地文件损坏或截断：必须先删除再重下。
            # 否则下面的 Range 续传会从文件末尾开始请求，服务端返回 416，
            # 重试多少次都没用 —— 该季度会被永久卡住并在建表时静默丢失。
            log(f"[清理] {name} 本地文件损坏或截断 "
                f"({human(dest.stat().st_size)})，删除后重新下载")
            dest.unlink(missing_ok=True)

    # 磁盘保护：每个季度下载前检查
    free_gb = shutil.disk_usage(str(raw_dir)).free / 1024 ** 3
    if free_gb < min_free_gb:
        with lock:
            log(f"[中止] 剩余空间 {free_gb:.1f}G < 阈值 {min_free_gb}G，跳过 {name}")
        return (y, q, None, "nospace")

    rs = remote_size(url)
    if rs == -2:
        with lock:
            log(f"[缺失] {name} 远端不存在（该季度未发布），跳过")
        return (y, q, None, "404")

    with lock:
        log(f"[下载] {name}" + (f"  远端 {human(rs)}" if rs > 0 else ""))

    if not download(url, dest, quiet=quiet, total=rs if rs > 0 else 0):
        with lock:
            log(f"[失败] {name} 下载未完成")
        return (y, q, None, "download")

    if rs > 0 and dest.stat().st_size != rs:
        with lock:
            log(f"[失败] {name} 大小不符 本地 {dest.stat().st_size} != 远端 {rs}，删除待重下")
        dest.unlink(missing_ok=True)
        return (y, q, None, "size")
    if not verify_zip(dest):
        with lock:
            log(f"[失败] {name} ZIP 完整性校验不通过，删除待重下")
        dest.unlink(missing_ok=True)
        return (y, q, None, "zip")

    with lock:
        log(f"[完成] {name}  {human(dest.stat().st_size)}")
    return (y, q, dest, "ok")


def stage_download(quarters, raw_dir: Path, min_free_gb: float,
                   workers: int) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    lock = threading.Lock()
    quiet = workers > 1

    if workers <= 1:
        for y, q in quarters:
            _, _, path, code = _fetch_one(y, q, raw_dir, min_free_gb, lock, quiet)
            (ok if path is not None else fail).append(
                (y, q, path if path is not None else code))
    else:
        log(f"并行下载: {workers} 个并发连接")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_one, y, q, raw_dir, min_free_gb, lock, quiet): (y, q)
                    for y, q in quarters}
            for fu in as_completed(futs):
                y, q = futs[fu]
                try:
                    _, _, path, code = fu.result()
                except Exception as e:                 # 单季度异常不影响整轮
                    with lock:
                        log(f"[失败] {qstr(y, q)} 异常: {e}")
                    fail.append((y, q, "exception"))
                    continue
                (ok if path is not None else fail).append(
                    (y, q, path if path is not None else code))

    log(f"下载阶段结束: 成功 {len(ok)} 个，失败/缺失 {len(fail)} 个")
    if fail:
        log("  未完成: " + ", ".join(f"{qstr(y, q)}({r})" for y, q, r in fail))
    return ok


def stage_build(quarters, raw_dir: Path, interim_dir: Path,
                processed_dir: Path, tables, force: bool = False) -> dict:
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    norm_dir = interim_dir / "norm"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 逐季度归一化 ----
    built = {}
    for y, q in quarters:
        zpath = raw_dir / zip_name(y, q)
        if not zpath.exists() or not verify_zip(zpath):
            log(f"[跳过] {qstr(y, q)} 无可用 zip")
            continue
        ex_dir = interim_dir / "extract" / qstr(y, q)
        files = extract_tables(zpath, y, q, ex_dir, tables)
        missing_tables = [t for t in tables if t not in files]
        if missing_tables:
            log(f"[注意] {qstr(y, q)} 缺表: {missing_tables}")

        per_q = {}
        for table, path in files.items():
            dest = norm_dir / f"{table}.{qstr(y, q)}.parquet"
            if dest.exists() and not force:
                per_q[table] = dest
                continue
            try:
                raw = read_ascii(path)
                df, miss = normalize(raw, table)
                if miss:
                    log(f"        {qstr(y, q)} {table}: 缺列 {miss}")
                if df.empty:
                    continue
                df.to_parquet(dest, index=False)
                per_q[table] = dest
            except Exception as e:
                log(f"[错误] {qstr(y, q)} {table} 读取失败: {e}")
        built[qstr(y, q)] = per_q
        log(f"[归一化] {qstr(y, q)} 完成: {sorted(per_q)}")

        # 顺手抽出 deleted 累计表（2019Q1 起存在）
        collect_deleted(zpath, interim_dir / "deleted" / qstr(y, q))

    # ---- 2) DEMO 去重 ----
    demo_parts = [built[k]["DEMO"] for k in built if "DEMO" in built[k]]
    if not demo_parts:
        log("[中止] 没有任何可用的 DEMO 表")
        return {}

    log(f"[组装] DEMO 共 {len(demo_parts)} 个季度文件，开始合并 ...")
    frames = []
    for p in demo_parts:
        d = pd.read_parquet(p)
        frames.append(d)
    demo = pd.concat(frames, ignore_index=True)
    del frames
    n_before = len(demo)
    log(f"        DEMO 原始行数: {n_before:,}")

    demo["_fda"] = pd.to_numeric(demo.get("fda_dt", pd.Series(index=demo.index)),
                                 errors="coerce")
    demo["_pid"] = pd.to_numeric(demo["primaryid"], errors="coerce")

    # CASEID 保留 FDA_DT 最大；FDA_DT 相同保留 PRIMARYID 最大
    demo = demo.sort_values(["caseid", "_fda", "_pid"],
                            na_position="first", kind="mergesort")
    demo = demo.drop_duplicates(subset=["caseid"], keep="last")
    n_after = len(demo)
    log(f"        去重后行数:   {n_after:,}   (删除 {n_before - n_after:,} 条重复报告)")

    demo = demo.drop(columns=["_fda", "_pid"])
    demo.to_parquet(processed_dir / "demo.parquet", index=False)
    kept = pd.Index(demo["primaryid"].unique())
    log(f"        保留 PRIMARYID: {len(kept):,} 个 -> {processed_dir / 'demo.parquet'}")

    summary = {"demo_raw": int(n_before), "demo_dedup": int(n_after)}

    # ---- 3) 其余表按保留的 PRIMARYID 过滤 ----
    # 说明：2004Q1~2012Q3 的旧 AERS 子表只有 ISR、没有 CASE，因此这些季度的
    #       DRUG/REAC/OUTC/RPSR/THER/INDI 归一化后会缺 caseid。
    #       这里统一用 DEMO 建立的 primaryid -> caseid 映射补回来，
    #       保证后续所有表都能按 caseid 关联。
    # 映射表必须先压成「唯一索引」再用于 Series.map：
    # 真实数据里存在「同一个 PRIMARYID 同时挂在多个 CASEID 下」的脏记录
    # （全量 89 个季度约 1.8 千条），此时 primaryid 索引非唯一，
    # Series.map 会直接抛：
    #   InvalidIndexError: Reindexing only valid with uniquely valued Index objects
    # 因此这里按 PRIMARYID 去重后再建索引；冲突记录取排序后第一个 CASEID，
    # 结果可复现。注意：此去重只作用于映射表，不影响已落盘的 demo.parquet。
    pid2cid = (demo[["primaryid", "caseid"]]
               .dropna(subset=["primaryid"])
               .drop_duplicates(subset=["primaryid"], keep="first")
               .set_index("primaryid")["caseid"])
    n_dup_pid = n_after - len(pid2cid)
    if n_dup_pid:
        log(f"[注意] {n_dup_pid:,} 个 PRIMARYID 对应多个 CASEID，"
            f"映射表已按字典序最小的 CASEID 归一（占 {n_dup_pid / n_after:.4%}）")

    for table in [t for t in tables if t != "DEMO"]:
        parts = [built[k][table] for k in built if table in built[k]]
        if not parts:
            log(f"[跳过] {table} 无数据")
            continue
        out_frames = []
        total = 0
        n_backfill = 0
        for p in parts:
            d = pd.read_parquet(p)
            total += len(d)
            if "primaryid" not in d.columns:
                log(f"[警告] {p.name} 无 primaryid 列，跳过")
                continue
            d = d[d["primaryid"].isin(kept)]
            if "caseid" not in d.columns:
                d = d.copy()
                cid = d["primaryid"].map(pid2cid)
                n_miss = int(cid.isna().sum())
                if n_miss:
                    # 未匹配通常意味着 PRIMARYID 的类型/格式与 DEMO 侧不一致
                    # （例如一侧是 '1001'、另一侧是 '1001.0'）。这类问题会让
                    # 回填出的 caseid 大面积为空，下游按 caseid 关联时静默丢数据，
                    # 因此必须显式报警，而不是默默通过。
                    lvl = "[错误]" if n_miss > 0.5 * len(d) else "[警告]"
                    log(f"{lvl} {p.name}: {n_miss:,}/{len(d):,} 行 PRIMARYID "
                        f"未匹配到 CASEID，请核对 PRIMARYID 的字符串格式")
                d["caseid"] = cid
                n_backfill += len(d)
            out_frames.append(d)
        if not out_frames:
            log(f"[跳过] {table} 过滤后无数据")
            continue
        merged = pd.concat(out_frames, ignore_index=True)
        del out_frames
        merged.to_parquet(processed_dir / f"{table.lower()}.parquet", index=False)
        extra = f"（其中 {n_backfill:,} 行由 primaryid 回填 caseid）" if n_backfill else ""
        log(f"[完成] {table}: {total:,} -> {len(merged):,} 行{extra} "
            f"-> {processed_dir / (table.lower() + '.parquet')}")
        summary[table.lower()] = int(len(merged))

    with open(processed_dir / "build_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    global _LOGFH

    ap = argparse.ArgumentParser(
        description="FAERS 季度 ASCII 数据下载/校验/抽取/归一化/去重",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "download", "build", "all"],
                    help="要执行的阶段")
    ap.add_argument("--start", default="2004q1", help="起始季度，默认 2004q1")
    ap.add_argument("--end", default="2026q2", help="结束季度，默认 2026q2")
    ap.add_argument("--quarters", default="",
                    help="显式季度列表，逗号分隔，给出后忽略 --start/--end")
    ap.add_argument("--project", default=str(default_proj_root()),
                    help="项目根目录")
    ap.add_argument("--out", default="", help="输出根目录，默认 <project>/data/raw/faers")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="低于该剩余空间即中止下载（GB）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发下载数，1=顺序；建议 3-4，过大可能被限速")
    ap.add_argument("--tables", default=",".join(TABLES),
                    help="要抽取的表，逗号分隔")
    ap.add_argument("--force", action="store_true",
                    help="build 阶段强制重建（忽略已存在的中间 parquet）")
    args = ap.parse_args(argv)

    proj = Path(args.project).resolve()
    out_root = Path(args.out).resolve() if args.out else proj / "data" / "raw" / "faers"
    interim_dir = proj / "data" / "interim" / "faers"
    processed_dir = proj / "data" / "processed" / "faers"
    log_dir = proj / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    _LOGFH = open(log_dir / f"faers_{args.stage}_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                  "w", encoding="utf-8")

    tables = [t.strip().upper() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in tables if t not in TABLES]
    if bad:
        log(f"[错误] 不支持的表: {bad}，可选 {TABLES}")
        return 2

    if args.quarters.strip():
        quarters = [parse_quarter(x) for x in args.quarters.split(",") if x.strip()]
    else:
        quarters = list(quarter_range(parse_quarter(args.start),
                                      parse_quarter(args.end)))

    log("=" * 74)
    log(f"FAERS 处理  stage={args.stage}  季度数={len(quarters)}  "
        f"范围={qstr(*quarters[0])}~{qstr(*quarters[-1])}")
    log(f"项目根: {proj}")
    log(f"原始目录: {out_root}")
    log(f"表: {tables}")
    log("=" * 74)

    if args.stage == "plan":
        stage_plan(quarters, out_root)
        return 0

    if args.stage in ("download", "all"):
        stage_download(quarters, out_root, args.min_free_gb, args.workers)

    if args.stage in ("build", "all"):
        summary = stage_build(quarters, out_root, interim_dir,
                              processed_dir, tables, force=args.force)
        if summary:
            log("汇总: " + json.dumps(summary, ensure_ascii=False))

    log("全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
