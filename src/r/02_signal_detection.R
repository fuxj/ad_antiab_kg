#!/usr/bin/env Rscript
# =============================================================================
#  02_signal_detection.R
#  FAERS 四法不成比例分析（disproportionality analysis）：ROR / PRR / BCPNN / MGPS
#
#  方法学依据
#  ----------
#  · ROR / PRR：病例-非病例设计的标准公式（FDA Sentinel 与药物警戒文献通用）
#  · BCPNN：Bate et al. 1998 原始精确式（digamma / trigamma 解析解），
#           与 R 包 PhViD、pvm 的实现同源；输出 IC 与 IC025（2.5% 分位）
#  · MGPS：默认给出文献通用的闭式 EBGM / EB05（相对报告比口径）；
#          若已安装 openEBGM，可用 --mgps true 切换为 DuMouchel 真正的
#          两步先验经验贝叶斯收缩估计
#
#  四格表约定（a/b/c/d）
#  --------------------
#                   目标事件   其他事件
#      目标药物        a          b
#      其他药物        c          d
#  a = 同时含目标药物与目标事件的报告数；b = 含目标药物但不含目标事件的报告数
#  c = 含目标事件但不含目标药物的报告数；d = 两者都不含的报告数；N = a+b+c+d
#
#  阳性信号判定（四项全部为阳性才算稳健信号）
#  ------------------------------------------
#      ROR : a >= min.a 且 ROR 的 95%CI 下限 > 1
#      PRR : a >= min.a 且 PRR >= 2 且 chi2 >= 4
#      BCPNN: IC025 > 0
#      MGPS : EB05 > 2
#
#  输入（由 01_download_faers.py 生成）
#  -----------------------------------
#      <proj>/data/processed/faers/demo.parquet
#      <proj>/data/processed/faers/drug.parquet
#      <proj>/data/processed/faers/reac.parquet
#      <proj>/data/processed/faers/ther.parquet   (仅 TTO 分析需要)
#
#  输出
#  ----
#      <proj>/results/tables/faers_pt_level_signals.csv
#      <proj>/results/tables/faers_signal_summary.csv
#      <proj>/results/tables/faers_year_stratified.csv
#      <proj>/results/tables/faers_tto_summary.csv
#      <proj>/results/figures/faers_forest_aria.png
#      <proj>/results/figures/faers_signal_counts.png
#
#  运行
#  ----
#      Rscript src/r/02_signal_detection.R
#      Rscript src/r/02_signal_detection.R --role PS --min-a 3
#      Rscript src/r/02_signal_detection.R --mgps true      # 需先安装 openEBGM
#      Rscript src/r/02_signal_detection.R --mgps true --mgps-timeout 300
#      Rscript src/r/02_signal_detection.R --proj /path/to/ad_antiab_kg
#
#  依赖：data.table, arrow, ggplot2（stats 为 R 自带）
#      conda install -c conda-forge r-data.table r-arrow r-ggplot2
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

# =============================================================================
# 0. 可调配置
# =============================================================================

## 目标药物：键为规范名，值为 FAERS 中可能出现的药名/商品名/研发代号
TARGET_DRUGS <- list(
  lecanemab  = c("LECANEMAB", "LEQEMBI", "BAN2401"),
  donanemab  = c("DONANEMAB", "KISUNLA", "LY3002813"),
  aducanumab = c("ADUCANUMAB", "ADUHELM", "BIIB037")
)

## 目标事件（可在此扩充；不在此列表中的 PT 仍会被全量筛查并标注为 Other）
ARIA_PTS <- c(
  "AMYLOID RELATED IMAGING ABNORMALITIES",
  "AMYLOID RELATED IMAGING ABNORMALITY",
  "AMYLOID-RELATED IMAGING ABNORMALITY EDEMA/EXUDATIVE TYPE",
  "AMYLOID-RELATED IMAGING ABNORMALITY OEDEMA/EXUDATIVE TYPE",
  "AMYLOID-RELATED IMAGING ABNORMALITY MICROBLEEDS AND IRON DEPOSITION TYPE",
  "CEREBRAL MICROHAEMORRHAGE",
  "CEREBRAL HAEMORRHAGE",
  "CEREBRAL HAEMATOMA",
  "BRAIN OEDEMA",
  "CEREBRAL OEDEMA",
  "VASOGENIC CEREBRAL OEDEMA",
  "SUPERFICIAL SIDEROSIS OF THE CENTRAL NERVOUS SYSTEM"
)

INFUSION_PTS <- c(
  "INFUSION RELATED REACTION",
  "INFUSION SITE REACTION",
  "HEADACHE",
  "CHILLS",
  "PYREXIA",
  "FATIGUE",
  "FLUSHING",
  "BACK PAIN",
  "NAUSEA",
  "HYPOTENSION"
)

NEURO_PTS <- c(
  "SEIZURE",
  "CONFUSIONAL STATE",
  "ISCHAEMIC STROKE",
  "CEREBRAL INFARCTION",
  "HAEMORRHAGIC STROKE",
  "SYNCOPE",
  "FALL",
  "DIZZINESS",
  "ATAXIA",
  "APHASIA",
  "ENCEPHALOPATHY",
  "DEMENTIA",
  "COGNITIVE DISORDER",
  "HIPPOCAMPAL ATROPHY"
)

## 默认角色：PS = 首要怀疑药；如需 PS+SS 传 --role PS,SS
DEFAULT_ROLES <- c("PS")

# =============================================================================
# 1. 命令行参数
# =============================================================================
parse_args <- function() {
  a <- commandArgs(trailingOnly = TRUE)
  opt <- list(proj = NULL, role = NULL, min_a = 3L, mgps = FALSE,
              mgps_timeout = 600L, tto = TRUE, top_n = 30L)
  i <- 1L
  while (i <= length(a)) {
    k <- a[i]
    v <- if (i < length(a)) a[i + 1L] else ""
    if (k == "--proj") {
      opt$proj <- v; i <- i + 2L
    } else if (k == "--role") {
      opt$role <- toupper(trimws(strsplit(v, ",")[[1]])); i <- i + 2L
    } else if (k == "--min-a") {
      opt$min_a <- as.integer(v); i <- i + 2L
    } else if (k == "--mgps") {
      opt$mgps <- tolower(v) %in% c("true", "t", "1", "yes"); i <- i + 2L
    } else if (k == "--mgps-timeout") {
      opt$mgps_timeout <- max(30L, as.integer(v)); i <- i + 2L
    } else if (k == "--tto") {
      opt$tto <- tolower(v) %in% c("true", "t", "1", "yes"); i <- i + 2L
    } else if (k == "--top-n") {
      opt$top_n <- as.integer(v); i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  if (is.null(opt$role)) opt$role <- DEFAULT_ROLES
  opt
}

find_proj_root <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) {
    p <- normalizePath(dirname(f[1]), mustWork = FALSE)
    if (basename(p) %in% c("r", "R") && basename(dirname(p)) == "src") {
      return(normalizePath(file.path(p, "..", ".."), mustWork = FALSE))
    }
    return(p)
  }
  getwd()
}

msg <- function(...) cat(sprintf(...), "\n", sep = "")

## 带硬超时的求值器：openEBGM 的先验估计会反复重试多种优化方法与初值，
## 在个别数据上可能长时间不出结果。这里给它套一层墙钟护栏，超时即抛错，
## 由上层 tryCatch 捕获后降档，保证整条流程不会无限等待。
time_limited <- function(expr, secs) {
  setTimeLimit(elapsed = secs, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = TRUE), add = TRUE)
  expr
}

# =============================================================================
# 2. 读数与通用小工具
# =============================================================================
read_parquet_dt <- function(path) {
  if (!file.exists(path)) stop(sprintf("找不到文件：%s", path))
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("读取 parquet 需要 arrow 包，请执行：conda install -c conda-forge r-arrow")
  }
  as.data.table(arrow::read_parquet(path))
}

as_chr <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x
}

## 药名/成分归一化：转大写、非字母数字统一换成空格、两侧补空格，便于按词边界匹配
norm_name <- function(x) {
  s <- toupper(as_chr(x))
  s <- gsub("[^A-Z0-9]+", " ", s)
  s <- trimws(s)
  paste0(" ", s, " ")
}

## 用词边界正则匹配某个药的全部别名，返回该药在指定角色下命中的报告 primaryid
drug_reports <- function(drug_dt, aliases, roles) {
  ok_role <- drug_dt$role_cod %in% roles
  nm <- ifelse(ok_role, drug_dt$name_norm, " ")
  pat <- paste0(" (", paste(aliases, collapse = "|"), ") ")
  hit <- ok_role & grepl(pat, nm, perl = TRUE)
  unique(as_chr(drug_dt$primaryid[hit]))
}

## 事件分组标签
event_group <- function(pt) {
  g <- rep("Other", length(pt))
  g[pt %in% INFUSION_PTS] <- "Infusion"
  g[pt %in% NEURO_PTS] <- "Neuro"
  g[pt %in% ARIA_PTS] <- "ARIA"
  g[grepl("AMYLOID", pt)] <- "ARIA"          # 兜底：任何含 AMYLOID 的 PT
  g[grep("^CEREBRAL (MICRO)?HAEMORRHAGE|^BRAIN OEDEMA|OEDEMA$|SIDEROSIS", pt)] <- "ARIA"
  g
}

# =============================================================================
# 3. 四格表 -> 四法指标
# =============================================================================
## Haldane-Anscombe 校正：任一格为 0 时四格同加 0.5，仅用于计算，不改动原始计数
haldane <- function(a, b, cc, dd) {
  z <- (a == 0 | b == 0 | cc == 0 | dd == 0)
  list(a = a + 0.5 * z, b = b + 0.5 * z, cc = cc + 0.5 * z, dd = dd + 0.5 * z)
}

## BCPNN 精确式（Bate 1998）：返回 IC 与 IC025
bcpnn_ic <- function(a, b, cc, dd) {
  n1. <- a + cc; n.1 <- a + b; n <- a + b + cc + dd
  p1 <- 1 + n1.; p2 <- 1 + n - n1.
  q1 <- 1 + n.1; q2 <- 1 + n - n.1
  r1 <- a + 1
  r2b <- n - a - 1 + (2 + n)^2 / (q1 * p1)
  IC <- (digamma(r1) - digamma(r1 + r2b) -
           (digamma(p1) - digamma(p1 + p2) + digamma(q1) - digamma(q1 + q2))) / log(2)
  VIC <- (trigamma(r1) - trigamma(r1 + r2b) +
            (trigamma(p1) - trigamma(p1 + p2) + trigamma(q1) - trigamma(q1 + q2))) / log(2)^2
  list(IC = IC, IC025 = IC - 1.96 * sqrt(pmax(VIC, 0)))
}

## 对整张四格表批量计算四法指标
compute_metrics <- function(a, b, cc, dd) {
  N <- a + b + cc + dd
  h <- haldane(a, b, cc, dd)

  ## ---- ROR ----
  ROR <- (h$a * h$dd) / (h$b * h$cc)
  se_ror <- sqrt(1 / h$a + 1 / h$b + 1 / h$cc + 1 / h$dd)
  ROR025 <- exp(log(ROR) - 1.96 * se_ror)
  ROR975 <- exp(log(ROR) + 1.96 * se_ror)

  ## ---- PRR ----
  PRR <- (h$a / (h$a + h$b)) / (h$cc / (h$cc + h$dd))
  se_prr <- sqrt(1 / h$a - 1 / (h$a + h$b) + 1 / h$cc - 1 / (h$cc + h$dd))
  PRR025 <- exp(log(PRR) - 1.96 * se_prr)
  PRR975 <- exp(log(PRR) + 1.96 * se_prr)
  chi2 <- (N * (h$a * h$dd - h$b * h$cc)^2) /
    ((h$a + h$b) * (h$cc + h$dd) * (h$a + h$cc) * (h$b + h$dd))

  ## ---- BCPNN ----
  ic <- bcpnn_ic(h$a, h$b, h$cc, h$dd)

  ## ---- MGPS / EBGM（文献通用闭式）----
  EBGM <- (h$a * N) / ((h$a + h$cc) * (h$a + h$b))
  EB05 <- exp(log(EBGM) - 1.96 * se_ror)

  data.frame(ROR = ROR, ROR025 = ROR025, ROR975 = ROR975,
             PRR = PRR, PRR025 = PRR025, PRR975 = PRR975, chi2 = chi2,
             IC = ic$IC, IC025 = ic$IC025,
             EBGM = EBGM, EB05 = EB05)
}

# =============================================================================
# 4. 主流程
# =============================================================================
main <- function() {
  opt <- parse_args()
  proj <- normalizePath(opt$proj %||% find_proj_root(), mustWork = FALSE)
  fdir <- file.path(proj, "data", "processed", "faers")
  tdir <- file.path(proj, "results", "tables")
  gdir <- file.path(proj, "results", "figures")
  dir.create(tdir, recursive = TRUE, showWarnings = FALSE)
  dir.create(gdir, recursive = TRUE, showWarnings = FALSE)

  msg("======================================================================")
  msg("FAERS 不成比例分析（ROR / PRR / BCPNN / MGPS）")
  msg("项目根: %s", proj)
  msg("角色过滤: %s   最小报告数 a >= %d   MGPS真值: %s",
      paste(opt$role, collapse = "+"), opt$min_a, opt$mgps)
  msg("时间: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  msg("======================================================================")

  # ---- 4.1 读入 ----
  demo <- read_parquet_dt(file.path(fdir, "demo.parquet"))
  drug <- read_parquet_dt(file.path(fdir, "drug.parquet"))
  reac <- read_parquet_dt(file.path(fdir, "reac.parquet"))
  msg("[读入] DEMO %s 行 | DRUG %s 行 | REAC %s 行",
      format(nrow(demo), big.mark = ","), format(nrow(drug), big.mark = ","),
      format(nrow(reac), big.mark = ","))

  for (tb in c("demo", "drug", "reac")) {
    x <- get(tb)
    need <- switch(tb, demo = c("primaryid"), drug = c("primaryid", "drugname"),
                   reac = c("primaryid", "pt"))
    miss <- setdiff(need, names(x))
    if (length(miss)) stop(sprintf("%s.parquet 缺少必需列：%s", tb, paste(miss, collapse = ", ")))
  }
  ## 可选列兜底，保证不同年份/不同来源的表结构都能跑通
  if (!"prod_ai" %in% names(drug)) drug[, prod_ai := ""]
  if (!"role_cod" %in% names(drug)) drug[, role_cod := "PS"]
  if (!"fda_dt" %in% names(demo)) demo[, fda_dt := ""]
  if (!"event_dt" %in% names(demo)) demo[, event_dt := ""]

  # ---- 4.2 归一化与报告集合 ----
  demo[, primaryid := as_chr(primaryid)]
  drug[, primaryid := as_chr(primaryid)]
  reac[, primaryid := as_chr(primaryid)]
  drug[, name_norm := norm_name(paste(drugname %||% "", prod_ai %||% ""))]
  reac[, pt := toupper(trimws(as_chr(pt)))]
  reac <- reac[pt != ""]
  drug <- drug[!is.na(role_cod)]

  ## 分析集 = DEMO 中的去重报告
  N_total <- uniqueN(demo$primaryid)
  msg("[分析集] 去重报告数 N = %s（角色 %s）",
      format(N_total, big.mark = ","), paste(opt$role, collapse = "+"))

  ## 全库每个事件的目标报告数 n_e（含该事件的报告数）
  reac_u <- unique(reac[, .(primaryid, pt)])
  n_e <- reac_u[, .(n_e = uniqueN(primaryid)), by = pt]
  setorder(n_e, -n_e)
  msg("[背景] 事件 PT 种类 %s，最高频 PT：%s (%s)",
      format(nrow(n_e), big.mark = ","), n_e$pt[1],
      format(n_e$n_e[1], big.mark = ","))

  ## 供后续核对 PT 拼写
  pt_amyloid <- n_e[grepl("AMYLOID", pt)]
  if (nrow(pt_amyloid)) {
    msg("[核对] 含 AMYLOID 的 PT 共 %d 个，前 5：", nrow(pt_amyloid))
    print(head(pt_amyloid, 5))
  }

  # ---- 4.3 逐药计算 ----
  all_pt <- list()
  summ <- list()
  for (dname in names(TARGET_DRUGS)) {
    aliases <- TARGET_DRUGS[[dname]]
    ids <- drug_reports(drug, aliases, opt$role)
    ids <- intersect(ids, demo$primaryid)
    n_d <- length(ids)
    msg("[药物] %-11s 命中报告 %s（别名 %s）",
        dname, format(n_d, big.mark = ","), paste(aliases, collapse = "/"))
    if (n_d == 0) {
      msg("        无报告，跳过"); next
    }

    a_tab <- reac_u[primaryid %in% ids, .(a = uniqueN(primaryid)), by = pt]
    m <- merge(a_tab, n_e, by = "pt", all.x = TRUE)
    m[, b := n_d - a]
    m[, cc := n_e - a]
    m[, dd := N_total - a - b - cc]
    m <- m[a >= 1]                       # 保留所有至少出现一次的事件（便于附材料追溯）

    met <- compute_metrics(m$a, m$b, m$cc, m$dd)

    out <- data.table(
      drug = dname, pt = m$pt, event_group = event_group(m$pt),
      a = m$a, b = m$b, c = m$cc, d = m$dd, n_drug = n_d, n_event = m$n_e,
      as.data.table(met)
    )
    out[, `:=`(
      sig_ROR = a >= opt$min_a & ROR025 > 1,
      sig_PRR = a >= opt$min_a & PRR >= 2 & chi2 >= 4,
      sig_BCPNN = a >= opt$min_a & IC025 > 0,
      sig_MGPS = a >= opt$min_a & EB05 > 2
    )]
    out[, n_sig := as.integer(sig_ROR) + as.integer(sig_PRR) +
          as.integer(sig_BCPNN) + as.integer(sig_MGPS)]
    out[, robust := sig_ROR & sig_PRR & sig_BCPNN & sig_MGPS]

    all_pt[[dname]] <- out
    summ[[dname]] <- data.table(
      drug = dname, n_reports = n_d,
      n_PT_observed = nrow(out),
      n_PT_screened = sum(out$a >= opt$min_a),
      n_sig_ROR = sum(out$sig_ROR), n_sig_PRR = sum(out$sig_PRR),
      n_sig_BCPNN = sum(out$sig_BCPNN), n_sig_MGPS = sum(out$sig_MGPS),
      n_sig_any = sum(out$n_sig > 0), n_sig_robust = sum(out$robust),
      n_sig_ARIA = sum(out$robust & out$event_group == "ARIA")
    )
  }

  if (!length(all_pt)) stop("所有目标药物均未命中报告，请核对 TARGET_DRUGS 别名。")
  signals <- rbindlist(all_pt)
  summary_tb <- rbindlist(summ)

  # ---- 4.4 MGPS 真值（可选，DuMouchel 两步先验）----
  if (opt$mgps) {
    if (!requireNamespace("openEBGM", quietly = TRUE)) {
      msg("[MGPS] 未安装 openEBGM，沿用闭式 EBGM。安装：install.packages('openEBGM')")
    } else {
      msg("[MGPS] 使用 openEBGM 重新估计（全库药物-事件对）...")
      signals <- mgps_true(signals, drug, reac_u, N_total, opt$role, opt$min_a,
                           opt$mgps_timeout)
    }
  }

  # ---- 4.5 年度分层敏感性分析 ----
  yr <- year_stratified(demo, reac_u, drug, n_e, N_total, opt)

  # ---- 4.6 至事件发生时间（TTO）----
  tto_tb <- NULL
  if (opt$tto && file.exists(file.path(fdir, "ther.parquet"))) {
    tto_tb <- tto_analysis(demo, drug, reac, file.path(fdir, "ther.parquet"), opt)
  }

  # ---- 4.7 输出 ----
  f1 <- file.path(tdir, "faers_pt_level_signals.csv")
  fwrite(signals, f1)
  f2 <- file.path(tdir, "faers_signal_summary.csv")
  fwrite(summary_tb, f2)
  msg("[输出] %s （%s 行）", f1, format(nrow(signals), big.mark = ","))
  msg("[输出] %s", f2)

  if (!is.null(yr)) {
    f3 <- file.path(tdir, "faers_year_stratified.csv")
    fwrite(yr, f3); msg("[输出] %s", f3)
  }
  if (!is.null(tto_tb)) {
    f4 <- file.path(tdir, "faers_tto_summary.csv")
    fwrite(tto_tb, f4); msg("[输出] %s", f4)
  }

  make_figures(signals, summary_tb, gdir, opt$top_n)

  msg("======================================================================")
  msg("完成。稳健信号（四法全阳性）汇总：")
  print(summary_tb)
  msg("======================================================================")
  invisible(0)
}

# =============================================================================
# 5. 真值 MGPS（openEBGM，DuMouchel 两步先验）
# =============================================================================
mgps_true <- function(signals, drug, reac_u, N_total, roles, min_a = 3L,
                      timeout = 600L) {
  ## 构建全库「报告-药物-事件」长表（openEBGM 输入格式：id / var1 / var2）
  ## 注意：经验贝叶斯先验描述的是"全库所有药物-事件对"的效应分布，
  ##      因此这里【不做角色过滤】；角色过滤只用于目标药的暴露定义。
  d2 <- unique(drug[name_norm != " ", .(primaryid, name_norm)])
  de <- merge(d2, reac_u, by = "primaryid", allow.cartesian = TRUE)
  raw <- data.frame(id = de$primaryid, var1 = de$name_norm, var2 = de$pt,
                    stringsAsFactors = FALSE)
  msg("       药物-事件长表 %s 行，开始估计先验 ...",
      format(nrow(raw), big.mark = ","))

  proc <- openEBGM::processRaw(raw)

  theta_init <- data.frame(
    alpha1 = c(1, 2, 3),
    beta1  = c(1, 2, 3),
    alpha2 = c(2, 4, 5),
    beta2  = c(2, 4, 5),
    p      = c(0.1, 0.2, 0.3)
  )

  ## ---- 先验估计：按 openEBGM 官方 vignette03 配方，失败则逐级降档 ----
  ## 官方推荐：对低计数对做两轮压缩（先 count=1 档，再 count=2 档）后估计先验。
  ## 单一 bin_size 并不能适配所有数据，故按由细到粗给出配方阶梯，
  ## 取第一个收敛的结果；全部失败才回退闭式 EBGM。
  recipes <- list(
    list(bs1 = 25, kp1 = 10, bs2 = 10, kp2 = 10),   # 官方默认
    list(bs1 = 50, kp1 = 50, bs2 = 50, kp2 = 50),   # 中等分箱
    list(bs1 =  5, kp1 =  5, bs2 =  5, kp2 =  5)    # 细分箱（低计数对极多时）
  )
  ## 每次 autoHyper 调用的墙钟上限：把总预算按最多 6 次调用均摊，
  ## 避免"3 套配方 × 2 组初值 × 3 组兜底"叠加后拖到几小时。
  per_call <- max(30L, as.integer(timeout / 6L))
  msg("       先验估计单次超时上限 %ds（总预算 %ds）", per_call, timeout)
  th <- NULL
  for (rc in recipes) {
    sq <- tryCatch({
      s1 <- openEBGM::squashData(proc, count = 1, bin_size = rc$bs1, keep_pts = rc$kp1)
      openEBGM::squashData(s1, count = 2, bin_size = rc$bs2, keep_pts = rc$kp2)
    }, error = function(e) NULL)
    for (ti in list(theta_init, theta_init[1:2, , drop = FALSE])) {
      if (is.null(sq)) break
      th <- tryCatch(time_limited(openEBGM::autoHyper(sq, theta_init = ti,
                                                      squashed = TRUE, max_pts = 1e5),
                                  per_call),
                     error = function(e) NULL)
      if (!is.null(th)) {
        msg("       先验收敛：squash bin_size %d/%d -> %d 个压缩点",
            rc$bs1, rc$bs2, nrow(sq))
        break
      }
    }
    if (!is.null(th)) break
  }
  if (is.null(th)) {
    ## 兜底一：官方 autoSquash（自动分箱）
    th <- tryCatch(time_limited(openEBGM::autoHyper(openEBGM::autoSquash(proc),
                                                    theta_init = theta_init),
                                per_call),
                   error = function(e) NULL)
    if (!is.null(th)) msg("       先验收敛：autoSquash 自动分箱")
  }
  if (is.null(th)) {
    ## 兜底二：随机抽样到 2 万点以内，不压缩直接估计
    set.seed(20260101L)
    n_p <- nrow(proc)
    idx <- if (n_p > 20000L) sample.int(n_p, 20000L) else seq_len(n_p)
    th <- tryCatch(time_limited(openEBGM::autoHyper(proc[idx, , drop = FALSE],
                                                    theta_init = theta_init,
                                                    squashed = FALSE),
                                per_call),
                   error = function(e) NULL)
    if (!is.null(th)) msg("       先验收敛：随机抽样 %d 点全似然", length(idx))
  }
  if (is.null(th)) {
    msg("       [警告] 先验估计未收敛（数据库过小或计数分布异常），保留闭式 EBGM")
    return(signals[])
  }
  msg("       先验估计完成：%s", paste(round(th$estimates, 4), collapse = ", "))

  sc <- openEBGM::ebScores(proc, hyper_estimate = th, quantiles = c(5, 95))
  d <- as.data.table(sc$data)
  d <- d[!is.na(EBGM)]
  ## processRaw 会去掉药名首尾空格，这里重新做一次词边界归一化以便匹配
  d[, name_norm := norm_name(as.character(var1))]
  d[, pt := as.character(var2)]

  out <- copy(signals)
  out[, `:=`(EBGM_mgps = NA_real_, EB05_mgps = NA_real_)]
  for (dn in unique(signals$drug)) {
    pat <- paste0(" (", paste(TARGET_DRUGS[[dn]], collapse = "|"), ") ")
    ks <- d[grepl(pat, name_norm, perl = TRUE), .(pt, EBGM_mgps = EBGM,
                                                  EB05_mgps = QUANT_05)]
    if (!nrow(ks)) next
    ks <- ks[, .(EBGM_mgps = max(EBGM_mgps), EB05_mgps = max(EB05_mgps)), by = pt]
    out[drug == dn, `:=`(EBGM_mgps = ks$EBGM_mgps[match(pt, ks$pt)],
                         EB05_mgps = ks$EB05_mgps[match(pt, ks$pt)])]
  }

  hit <- !is.na(out$EB05_mgps)
  msg("       匹配到真值 EBGM 的药物-事件对：%s / %s",
      format(sum(hit), big.mark = ","), format(nrow(out), big.mark = ","))
  out[hit, `:=`(EBGM = EBGM_mgps, EB05 = EB05_mgps,
                sig_MGPS = a >= min_a & EB05_mgps > 2)]
  out[, n_sig := as.integer(sig_ROR) + as.integer(sig_PRR) +
        as.integer(sig_BCPNN) + as.integer(sig_MGPS)]
  out[, robust := sig_ROR & sig_PRR & sig_BCPNN & sig_MGPS]
  out[, c("EBGM_mgps", "EB05_mgps") := NULL]
  out[]
}

# =============================================================================
# 6. 年度分层敏感性分析
# =============================================================================
year_stratified <- function(demo, reac_u, drug, n_e, N_total, opt) {
  if (!"fda_dt" %in% names(demo)) return(NULL)
  dd <- copy(demo)[, yr := suppressWarnings(as.integer(substr(as_chr(fda_dt), 1, 4)))]
  dd <- dd[!is.na(yr) & yr >= 2004 & yr <= 2100]
  if (!nrow(dd)) return(NULL)

  keep_pt <- unique(c(ARIA_PTS, INFUSION_PTS[1:3]))
  res <- list()
  for (dname in names(TARGET_DRUGS)) {
    ids <- drug_reports(drug, TARGET_DRUGS[[dname]], opt$role)
    ids <- intersect(ids, dd$primaryid)
    if (!length(ids)) next
    for (y in sort(unique(dd$yr))) {
      pid_y <- dd[yr == y, primaryid]
      Ny <- length(pid_y)
      ids_y <- intersect(ids, pid_y)
      ndy <- length(ids_y)
      if (ndy == 0) next
      reac_y <- reac_u[primaryid %in% pid_y]
      a_tab <- reac_y[primaryid %in% ids_y, .(a = uniqueN(primaryid)), by = pt]
      ne_y <- reac_y[, .(n_e = uniqueN(primaryid)), by = pt]
      m <- merge(a_tab, ne_y, by = "pt")
      m <- m[pt %in% keep_pt]
      if (!nrow(m)) next
      m[, `:=`(b = ndy - a, cc = n_e - a)]
      m[, dd_ := Ny - a - b - cc]
      met <- compute_metrics(m$a, m$b, m$cc, m$dd_)
      res[[length(res) + 1L]] <- data.table(
        drug = dname, year = y, pt = m$pt, a = m$a, n_drug_year = ndy,
        as.data.table(met)
      )
    }
  }
  if (!length(res)) return(NULL)
  rbindlist(res)
}

# =============================================================================
# 7. 至事件发生时间（TTO）
# =============================================================================
parse_ymd <- function(x) {
  x <- as_chr(x)
  x[nchar(x) != 8 | x == "0"] <- NA_character_
  as.Date(x, format = "%Y%m%d")
}

tto_analysis <- function(demo, drug, reac, ther_path, opt) {
  ther <- read_parquet_dt(ther_path)
  if (!all(c("primaryid", "drug_seq", "start_dt") %in% names(ther)) ||
      !"drug_seq" %in% names(drug)) {
    msg("[TTO] 缺少 drug_seq 或 start_dt 列，跳过 TTO 分析")
    return(NULL)
  }
  ther[, primaryid := as_chr(primaryid)]
  demo[, primaryid := as_chr(primaryid)]
  drug[, primaryid := as_chr(primaryid)]

  ev <- copy(demo)[, event_dt2 := parse_ymd(event_dt)]
  ev <- ev[!is.na(event_dt2), .(primaryid, event_dt2)]

  res <- list()
  for (dname in names(TARGET_DRUGS)) {
    ids <- drug_reports(drug, TARGET_DRUGS[[dname]], opt$role)
    ids <- intersect(ids, ev$primaryid)
    if (!length(ids)) next
    ds <- unique(drug[primaryid %in% ids & name_norm != " ",
                      .(primaryid, drug_seq)])
    st <- merge(ds, ther[primaryid %in% ids, .(primaryid, drug_seq, start_dt)],
                by = c("primaryid", "drug_seq"))
    st[, start2 := parse_ymd(start_dt)]
    st <- st[!is.na(start2)]
    if (!nrow(st)) next
    st1 <- st[, .(start_dt2 = min(start2)), by = primaryid]
    ev1 <- ev[primaryid %in% ids]
    j <- merge(st1, ev1, by = "primaryid")
    j[, tto_days := as.numeric(event_dt2 - start_dt2)]
    j <- j[tto_days >= 0 & tto_days <= 3650]
    if (!nrow(j)) next
    agg <- reac[primaryid %in% ids & pt %in% ARIA_PTS]
    any_aria <- unique(agg$primaryid)
    j[, aria := primaryid %in% any_aria]
    res[[length(res) + 1L]] <- data.table(
      drug = dname, stratum = c("All", "ARIA", "Non-ARIA"),
      n = c(nrow(j), sum(j$aria), sum(!j$aria)),
      median_days = c(median(j$tto_days), median(j$tto_days[j$aria]),
                      median(j$tto_days[!j$aria])),
      q1_days = c(quantile(j$tto_days, .25), quantile(j$tto_days[j$aria], .25),
                  quantile(j$tto_days[!j$aria], .25)),
      q3_days = c(quantile(j$tto_days, .75), quantile(j$tto_days[j$aria], .75),
                  quantile(j$tto_days[!j$aria], .75))
    )
  }
  if (!length(res)) return(NULL)
  rbindlist(res)
}

# =============================================================================
# 8. 作图
# =============================================================================
make_figures <- function(signals, summary_tb, gdir, top_n) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    msg("[作图] 未安装 ggplot2，跳过作图"); return(invisible(NULL))
  }
  library(ggplot2)

  ## 8.1 森林图：ARIA 相关 PT 的 ROR（含 95%CI）
  aria <- signals[event_group == "ARIA"]
  if (nrow(aria)) {
    aria <- aria[order(drug, -ROR)]
    aria[, lab := paste0(drug, " | ", pt)]
    aria[, lab := factor(lab, levels = rev(unique(lab)))]
    p <- ggplot(aria, aes(x = ROR, y = lab, colour = drug)) +
      geom_vline(xintercept = 1, linetype = 2, colour = "grey50") +
      geom_errorbar(aes(xmin = ROR025, xmax = ROR975), orientation = "y",
                    width = 0.25) +
      geom_point(aes(size = a)) +
      scale_x_log10() +
      labs(x = "Reporting odds ratio (95% CI, log scale)", y = NULL,
           size = "Reports (a)", colour = "Drug",
           title = "ARIA-related disproportionality signals, FAERS",
           subtitle = "Anti-Abeta monoclonal antibodies") +
      theme_bw(base_size = 11) +
      theme(legend.position = "right")
    ggsave(file.path(gdir, "faers_forest_aria.png"), p,
           width = 10, height = max(4, 0.33 * nrow(aria) + 2), dpi = 300)
    msg("[作图] faers_forest_aria.png")
  }

  ## 8.2 信号计数条形图
  long <- melt(summary_tb,
               id.vars = "drug",
               measure.vars = c("n_sig_ROR", "n_sig_PRR", "n_sig_BCPNN",
                                "n_sig_MGPS", "n_sig_robust"),
               variable.name = "method", value.name = "n")
  p2 <- ggplot(long, aes(x = drug, y = n, fill = method)) +
    geom_col(position = position_dodge(width = .8)) +
    labs(x = NULL, y = "Number of signals", fill = "Method",
         title = "Signal counts by detection method") +
    theme_bw(base_size = 11)
  ggsave(file.path(gdir, "faers_signal_counts.png"), p2,
         width = 8, height = 5, dpi = 300)
  msg("[作图] faers_signal_counts.png")
  invisible(NULL)
}

# =============================================================================
if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
