#!/usr/bin/env Rscript
# 06_coloc.R  ——  cis-eQTL × GWAS 共定位分析（coloc.abf）
# -----------------------------------------------------------------------------
# 为第 5 章 MR 的关键基因 × 结局对做 colocalization，区分
#   PP.H3 : 两组信号由不同因果变异驱动（MR 结果更可能是 LD 假信号）
#   PP.H4 : 两组信号由同一因果变异驱动（支持因果，MR 更可信）
#
# 数据 : OpenGWAS REST API（/associations，按基因窗口取 summary stats）
# 暴露 : cis-eQTL 数据集（eqtl-a-<ENSG>，eQTLGen 全血）
# 结局 : 第 5 章配置的 9 个 GWAS（AD/ICH/SAH/SVD/IS/LDL）
#
# 依赖 : coloc ≥ 5.0、data.table、httr、jsonlite、arrow
#   缺包时脚本会停下来给出安装命令。
#
# 用法 :
#   Rscript src/r/06_coloc.R [stage] [options]
#     stage: probe | fetch | analyse | all  (default all)
#   常用参数 :
#     --proj DIR          项目根（默认 .）
#     --pairs GENE=OUTCOME,GENE=OUTCOME,...   指定要跑的对（默认预置关键对）
#     --window KB         提取基因两侧窗口大小（默认 250，即 ±250kb，总 500kb）
#     --no-cache          强制重新拉取 summary stats
#     --no-figs           不画 LocusZoom 风格 Manhattan
#     --quiet             少日志
#
# 输出 :
#   data/interim/coloc_cache/*.rds            窗口级 summary stats 缓存
#   results/tables/coloc_summary.csv          PP.H0–H4 总表
#   results/tables/coloc_top_snps.csv         每个对的 top SNP 对比
#   results/tables/coloc_summary.json         JSON 摘要
#   results/figures/coloc_*.png               LocusZoom 风格对比图（若启用）
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  need <- c("data.table", "httr", "jsonlite")
  for (p in need) if (!requireNamespace(p, quietly=TRUE))
    stop(sprintf("missing R package: %s; install with\n  conda install -c conda-forge r-%s", p, tolower(p)))
  library(data.table)
  library(httr)
  library(jsonlite)
})
if (!requireNamespace("coloc", quietly=TRUE))
  stop("missing R package: coloc; install with\n  conda install -c conda-forge r-coloc\nor\n  install.packages('coloc')")
library(coloc)

# ---- 常量 ------------------------------------------------------------------
API_DEFAULT <- "https://api.opengwas.io/api"
KB_PER_MB <- 1000

# 第 5 章的基因面板（与 05_mr_validation.R 一致，仅保留关键/候选集）
GENE_ENSEMBL <- c(
  # ---- 凝血/抗血小板（第 4 章核心药物靶点）----
  VKORC1   = "ENSG00000167397",
  F10      = "ENSG00000126218",
  PROC     = "ENSG00000115718",
  ITGB3    = "ENSG00000259207",
  P2RY12   = "ENSG00000169313",
  # ---- NSAID ----
  PTGS1    = "ENSG00000095303",
  PTGS2    = "ENSG00000073756",
  PDGFRB   = "ENSG00000113721",
  # ---- 他汀阳性对照 ----
  HMGCR    = "ENSG00000113161",
  # ---- BBB/炎症（MR 阴性，但共定位能排除 LD 干扰）----
  CLDN5    = "ENSG00000184113",
  MMP9     = "ENSG00000100985",
  IL1B     = "ENSG00000125538",
  # ---- AD 位点（流程阳性对照）----
  BIN1     = "ENSG00000136717",
  ACE      = "ENSG00000159640",
  APP      = "ENSG00000142192",
  PLCG2    = "ENSG00000197943"
)

# 第 5 章的结局 GWAS（与 05_mr_validation.R 一致）
OUTCOMES <- list(
  list(short = "AD",       id = "ebi-a-GCST90027158", role = "primary"),
  list(short = "AD_repl1", id = "ieu-b-5067",         role = "replication"),
  list(short = "AD_repl2", id = "ieu-a-297",          role = "replication"),
  list(short = "ICH",      id = "ebi-a-GCST90018870", role = "safety"),
  list(short = "ICH_fin",  id = "finn-b-I9_ICH",      role = "safety"),
  list(short = "SAH",      id = "ebi-a-GCST90018923", role = "safety"),
  list(short = "SVD",      id = "ieu-a-1111",         role = "safety"),
  list(short = "IS",       id = "ebi-a-GCST90018864", role = "contrast"),
  list(short = "LDL",      id = "ieu-b-110",          role = "positive_control")
)

# 默认关键对：第 4 章/第 5 章核心候选
DEFAULT_PAIRS <- c(
  "VKORC1=ICH", "VKORC1=ICH_fin", "VKORC1=LDL",
  "PTGS1=ICH", "PTGS1=ICH_fin", "PTGS1=SAH",
  "F10=ICH", "F10=ICH_fin", "F10=LDL",
  "P2RY12=ICH", "P2RY12=ICH_fin", "P2RY12=SVD",
  "ITGB3=ICH", "ITGB3=ICH_fin",
  "PTGS2=ICH", "PDGFRB=ICH", "HMGCR=ICH_fin", "HMGCR=LDL",
  "MMP9=ICH", "CLDN5=ICH", "IL1B=ICH",
  "BIN1=AD", "BIN1=AD_repl2", "ACE=AD", "ACE=AD_repl2",
  "PLCG2=AD", "APP=SAH"
)

# ---- 工具 ------------------------------------------------------------------
`%||%` <- function(a, b) if (!is.null(a) && length(a) && !is.na(a) && nzchar(a)) a else b

log_msg <- function(...) {
  ts <- format(Sys.time(), "[%H:%M:%S]")
  cat(ts, ..., "\n", sep = "", file = "", append = FALSE)
  flush.console()
}

ensure_dir <- function(p) { if (!dir.exists(p)) dir.create(p, recursive = TRUE) }

pick_col <- function(df, candidates) {
  for (c in candidates) if (c %in% names(df)) return(c)
  NA_character_
}

# ---- API -------------------------------------------------------------------
# 发 POST 请求，返回 parsed JSON 或 NULL
api_post <- function(endpoint, body, cfg, tag = NULL) {
  url <- paste0(cfg$api, "/", sub("^/", "", endpoint))
  raw_path <- if (!is.null(tag)) file.path(cfg$raw_dir, paste0(tag, ".json")) else NULL
  resp <- tryCatch(
    POST(url,
         add_headers(Authorization = paste("Bearer", cfg$token),
                     "Content-Type" = "application/json"),
         body = toJSON(body, auto_unbox = TRUE),
         encode = "raw", timeout(60)),
    error = function(e) {
      log_msg(" [error] POST ", endpoint, ": ", conditionMessage(e)); NULL
    }
  )
  if (is.null(resp)) return(NULL)
  txt <- content(resp, as = "text", encoding = "UTF-8")
  if (!is.null(raw_path)) writeLines(txt, raw_path)
  if (http_error(resp)) {
    log_msg(" [error] ", endpoint, " HTTP ", status_code(resp), " ", substr(txt, 1, 200))
    return(NULL)
  }
  parsed <- tryCatch(fromJSON(txt, flatten = FALSE, simplifyVector = TRUE),
                     error = function(e) { log_msg(" [warn] parse JSON: ", conditionMessage(e)); NULL })
  if (is.null(parsed)) return(NULL)
  parsed
}

# ---- 拉取一个 (dataset, chr, start, end) 窗口的 associations ---------------
fetch_window <- function(dataset, chrom, start, end, cfg) {
  # OpenGWAS 不支持 ranges 参数；改用 chr:pos1-pos2 变体查询格式（associations endpoint 接受 variant 为区间）
  # 实测 OpenGWAS 的 associations/variant 可接受区间字符串，但对大窗口可能超限，这里改成按 rsid 取。
  # 为避免大量 rsid 请求，直接用 tophits 在 pval=1 下取整个 chr:interval 的所有 SNP（用 get-vs-associations 的方式）。
  #
  # 做法：用 associations 端点把 variant 设为区间字符串
  var <- sprintf("%s:%d-%d", chrom, start, end)
  body <- list(variant = I(var), id = I(dataset), proxies = 0, flatten = TRUE)
  j <- api_post("associations", body, cfg,
                tag = sprintf("win_%s_%s_%d_%d", gsub("[^A-Za-z0-9._-]", "_", dataset), chrom, start, end))
  if (is.null(j) || length(j) == 0) return(data.table())
  if (is.data.frame(j)) rec <- j else {
    # 如果是 list of records，转成 df
    rec <- tryCatch(as.data.frame(rbindlist(lapply(j, as.data.table), fill = TRUE),
                                 stringsAsFactors = FALSE),
                    error = function(e) NULL)
  }
  if (is.null(rec) || nrow(rec) == 0) return(data.table())
  setDT(rec)
  # 标准化列名（列名随端点版本略有差异）
  c_snp <- pick_col(rec, c("rsid", "SNP", "variant"))
  c_chr <- pick_col(rec, c("chr", "chrom", "chromosome"))
  c_pos <- pick_col(rec, c("position", "pos", "bp"))
  c_b   <- pick_col(rec, c("beta", "b"))
  c_se  <- pick_col(rec, c("se", "sebeta"))
  c_ea  <- pick_col(rec, c("ea", "effect_allele", "alt", "a1"))
  c_oa  <- pick_col(rec, c("nea", "other_allele", "ref", "a0"))
  c_eaf <- pick_col(rec, c("eaf", "freq", "af"))
  c_p   <- pick_col(rec, c("p", "pval"))
  c_n   <- pick_col(rec, c("n", "samplesize", "n_total"))
  if (any(is.na(c(c_snp, c_b, c_se, c_ea, c_oa)))) {
    # 缺关键字段 → 尝试用另一版列名
    need <- c("rsid","beta","se","ea","nea","position","chr")
    if (all(need %in% names(rec))) {
      c_snp="rsid"; c_b="beta"; c_se="se"; c_ea="ea"; c_oa="nea"; c_pos="position"; c_chr="chr"
    } else {
      log_msg(" [warn] unexpected schema for ", dataset, ": ", paste(names(rec), collapse=","))
      return(data.table())
    }
  }
  out <- data.table(
    SNP    = as.character(rec[[c_snp]]),
    chr    = as.character(if (!is.na(c_chr)) rec[[c_chr]] else chrom),
    pos    = as.numeric(if (!is.na(c_pos)) rec[[c_pos]] else NA_real_),
    beta   = as.numeric(rec[[c_b]]),
    se     = as.numeric(rec[[c_se]]),
    EA     = as.character(rec[[c_ea]]),
    OA     = as.character(rec[[c_oa]]),
    eaf    = if (!is.na(c_eaf)) suppressWarnings(as.numeric(rec[[c_eaf]])) else NA_real_,
    p      = if (!is.na(c_p)) suppressWarnings(as.numeric(rec[[c_p]])) else NA_real_,
    n      = if (!is.na(c_n)) suppressWarnings(as.numeric(rec[[c_n]])) else NA_real_
  )
  out <- out[!is.na(beta) & !is.na(se) & se > 0 & !is.na(SNP) & !is.na(p) & is.finite(p) & p > 0 & p <= 1]
  out
}

# ---- 获取基因坐标（从 OpenGWAS eqtl-a 数据集的 tophits 推断；或用固定窗口于 TSS 附近）
# 这里我们直接用基因在 GRCh37 上的近似 TSS 坐标表（从 OpenTargets 已抓取过的数据，下面硬编码）
# 若缺失则走 tophits 最大 p 阈值返回第一个 SNP 来定 pos（备用）
GENE_B37 <- list(
  VKORC1 = list(chr = "16", tss = 31100000),
  F10    = list(chr = "13", tss = 113700000),
  PROC   = list(chr = "2",  tss = 128100000),
  ITGB3  = list(chr = "17", tss = 45300000),
  P2RY12 = list(chr = "3",  tss = 151000000),
  PTGS1  = list(chr = "9",  tss = 125100000),
  PTGS2  = list(chr = "1",  tss = 186600000),
  PDGFRB = list(chr = "5",  tss = 149500000),
  HMGCR  = list(chr = "5",  tss = 74600000),
  CLDN5  = list(chr = "22", tss = 19500000),
  MMP9   = list(chr = "20", tss = 44600000),
  IL1B   = list(chr = "2",  tss = 113500000),
  BIN1   = list(chr = "2",  tss = 127800000),
  ACE    = list(chr = "17", tss = 61500000),
  APP    = list(chr = "21", tss = 27200000),
  PLCG2  = list(chr = "16", tss = 81700000)
)

get_gene_window <- function(gene, cfg) {
  info <- GENE_B37[[gene]]
  if (is.null(info)) {
    # 备用：通过 eqtl-a 数据集的最显著 SNP 定位
    e <- GENE_ENSEMBL[[gene]]
    j <- api_post("tophits",
                  list(id = I(paste0("eqtl-a-", e)), pval = 1, clump = 0,
                       preclumped = 0, pval_filter = 1),
                  cfg, tag = sprintf("loc_%s", gene))
    if (is.null(j) || length(j) == 0) stop("cannot locate gene ", gene)
    rec <- as.data.table(j)
    c_chr <- pick_col(rec, c("chr","chrom")); c_pos <- pick_col(rec, c("position","pos"))
    if (is.na(c_chr) || is.na(c_pos)) stop("cannot parse position for ", gene)
    info <- list(chr = as.character(rec[[c_chr]][1]), tss = as.integer(rec[[c_pos]][1]))
  }
  start <- max(1L, as.integer(info$tss - cfg$window_kb * 1000L))
  end   <- as.integer(info$tss + cfg$window_kb * 1000L)
  list(chr = info$chr, start = start, end = end)
}

# ---- 抓/读 一对 (exposure, outcome) 的两个数据集窗口 -----------------------
fetch_pair <- function(gene, outcome_short, cfg) {
  ens <- GENE_ENSEMBL[[gene]]
  o   <- OUTCOMES[[which(vapply(OUTCOMES, function(x) x$short == outcome_short, logical(1)))]]
  if (is.null(o)) stop("unknown outcome: ", outcome_short)
  win <- get_gene_window(gene, cfg)
  cache_exp <- file.path(cfg$cache_dir, sprintf("exp_%s_%s_%d_%d.rds", ens, win$chr, win$start, win$end))
  cache_out <- file.path(cfg$cache_dir, sprintf("out_%s_%s_%d_%d.rds", o$id, win$chr, win$start, win$end))

  fetch_one <- function(cache, dataset) {
    if (file.exists(cache) && cfg$reuse) {
      obj <- readRDS(cache); if (cfg$verbose) log_msg(" [cache] ", dataset, " (", nrow(obj), " rows)")
      return(obj)
    }
    log_msg(" [fetch] ", dataset, " ", win$chr, ":", win$start, "-", win$end)
    df <- fetch_window(dataset, win$chr, win$start, win$end, cfg)
    Sys.sleep(0.3)
    if (is.null(df) || nrow(df) == 0) {
      log_msg(" [warn] no SNP for ", dataset, " in window")
      df <- data.table()
    }
    saveRDS(df, cache)
    df
  }
  exp <- fetch_one(cache_exp, paste0("eqtl-a-", ens))
  out <- fetch_one(cache_out, o$id)
  list(gene = gene, ensembl = ens, outcome_short = outcome_short, outcome_id = o$id,
       outcome_role = o$role, chr = win$chr, start = win$start, end = win$end,
       exp = exp, out = out)
}

# ---- 等位基因对齐 + coloc.abf 输入准备 -------------------------------------
dedup_snp <- function(dt, pcol = "p") {
  if (is.null(dt) || nrow(dt) == 0) return(dt)
  dt <- if (is.data.table(dt)) copy(dt) else as.data.table(dt)
  if (!anyDuplicated(dt$SNP)) return(dt)
  n0 <- nrow(dt)
  ord_cols <- if (pcol %in% names(dt) && "se" %in% names(dt)) c(pcol, "se") else pcol
  setorderv(dt, ord_cols, na.last = TRUE)
  dt <- dt[!duplicated(SNP)]
  log_msg(" [dedup] duplicated rsID collapsed: ", n0, " -> ", nrow(dt), " rows")
  dt
}

align_and_prepare <- function(pair) {
  if (is.null(pair$exp) || nrow(pair$exp) == 0 ||
      is.null(pair$out) || nrow(pair$out) == 0) return(NULL)
  exp <- dedup_snp(pair$exp, "p"); out <- dedup_snp(pair$out, "p")
  m <- merge(exp[, .(SNP, beta.exp=beta, se.exp=se, EA.exp=EA, OA.exp=OA, eaf.exp=eaf, p.exp=p, n.exp=n)],
             out[, .(SNP, beta.out=beta, se.out=se, EA.out=EA, OA.out=OA, eaf.out=eaf, p.out=p, n.out=n)],
             by = "SNP", all = FALSE)
  if (nrow(m) < 50) {
    log_msg(" [warn] ", pair$gene, " x ", pair$outcome_short, " only ", nrow(m),
            " overlapping SNPs in window (need >=50 for stable coloc)")
    if (nrow(m) < 10) return(NULL)
  }
  same <- m$EA.exp == m$EA.out & m$OA.exp == m$OA.out
  swap <- m$EA.exp == m$OA.out & m$OA.exp == m$EA.out
  same[is.na(same)] <- FALSE; swap[is.na(swap)] <- FALSE
  amb  <- !same & !swap
  if (any(amb)) {
    palin <- (m$EA.exp %in% c("A","T") & m$OA.exp %in% c("A","T")) |
             (m$EA.exp %in% c("C","G") & m$OA.exp %in% c("C","G"))
    drop <- amb & palin
    if (any(drop)) {
      log_msg(" [align] drop ", sum(drop), " palindromic ambiguous SNP(s) out of ", nrow(m), " overlapping")
      keep <- !drop
      m <- m[keep]; same <- same[keep]; swap <- swap[keep]; amb <- amb[keep]
    }
    if (any(amb)) {
      e1 <- m$eaf.exp; e2 <- m$eaf.out
      to_same <- amb & is.finite(e1) & is.finite(e2) & abs(e1 - e2) < 0.2
      to_swap <- amb & is.finite(e1) & is.finite(e2) & abs(e1 - (1 - e2)) < 0.2
      same[to_same] <- TRUE; swap[to_swap] <- TRUE
      amb <- !same & !swap
      if (any(amb)) {
        log_msg(" [align] drop ", sum(amb), " ambiguous SNP(s) with unresolvable allele coding")
        keep <- !amb
        m <- m[keep]; same <- same[keep]; swap <- swap[keep]
      }
    }
  }
  if (nrow(m) < 10) return(NULL)
  idx_swap <- which(swap)
  flips <- length(idx_swap)
  if (flips > 0) m[idx_swap, c("beta.out","eaf.out") := .(-beta.out, 1 - eaf.out)]
  n_exp <- median(m$n.exp, na.rm=TRUE); if (!is.finite(n_exp) || n_exp <= 0) n_exp <- 31470
  n_out <- median(m$n.out, na.rm=TRUE)
  if (!is.finite(n_out) || n_out <= 0) {
    n_out <- if (grepl("^ieu-b-110", pair$outcome_id)) 173082 else 473513
  }
  maf <- ifelse(!is.na(m$eaf.out), pmin(m$eaf.out, 1 - m$eaf.out),
         ifelse(!is.na(m$eaf.exp), pmin(m$eaf.exp, 1 - m$eaf.exp), NA_real_))
  ok  <- is.finite(m$beta.exp) & is.finite(m$se.exp) & is.finite(m$beta.out) & is.finite(m$se.out) &
         m$se.exp > 0 & m$se.out > 0 & is.finite(maf) & maf > 0 & maf < 0.5
  if (!all(ok)) { m <- m[ok]; maf <- maf[ok] }
  if (nrow(m) < 10) return(NULL)
  dup <- duplicated(m$SNP)
  if (any(dup)) {
    log_msg(" [align] drop ", sum(dup), " residual duplicated rsID(s) before coloc.abf")
    m <- m[!dup]; maf <- maf[!dup]
  }
  stopifnot(length(maf) == nrow(m), length(m$SNP) == nrow(m))
  list(dataset1 = list(beta = m$beta.exp, varbeta = m$se.exp^2, N = n_exp, MAF = maf, type = "quant", snp = m$SNP),
       dataset2 = list(beta = m$beta.out, varbeta = m$se.out^2, N = n_out, MAF = maf,
                       type = if (pair$outcome_short %in% c("AD","ICH","ICH_fin","SAH","SVD","IS")) "cc" else "quant",
                       s = 0.15, snp = m$SNP),
       overlap = nrow(m), flips = flips, merged = m)
}

# ---- 主跑 coloc.abf ---------------------------------------------------------
run_coloc <- function(prepared) {
  if (is.null(prepared)) return(NULL)
  res <- tryCatch(
    coloc.abf(dataset1 = prepared$dataset1, dataset2 = prepared$dataset2),
    error = function(e) { log_msg(" [error] coloc.abf: ", conditionMessage(e)); NULL }
  )
  if (is.null(res)) return(NULL)
  summary <- res$summary
  # SNP.PP.H4 是每个 SNP 的 H4 后验
  pp <- res$results[, c("snp", "SNP.PP.H4")]
  setDT(pp); setorder(pp, -SNP.PP.H4)
  list(summary = summary, top = head(pp, 10), overlap = prepared$overlap, flips = prepared$flips)
}

# ---- 图：LocusZoom 风格对比（若 ggplot2 可用）--------------------------------
plot_pair <- function(pair, prepared, res, out_png) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(invisible(FALSE))
  library(ggplot2)
  m <- prepared$merged
  m[, p.exp  := pmin(p.exp,  1)]
  m[, p.out  := pmin(p.out,  1)]
  m[, p.exp  := pmax(p.exp,  1e-300)]
  m[, p.out  := pmax(p.out,  1e-300)]
  pp <- res$top[, .(SNP=snp, SNP.PP.H4)]
  m <- merge(m, pp, by.x="SNP", by.y="SNP", all.x=TRUE)
  m[is.na(SNP.PP.H4), SNP.PP.H4 := 0]
  m <- m[order(SNP)]
  # 为了画 Manhattan，需要 pos；但我们没取 pos，这里按 SNP 顺序排
  m[, idx := .I]
  top <- m[which.max(-log10(p.exp))]
  plt <- ggplot(m, aes(x = idx)) +
    geom_point(aes(y = -log10(p.exp)), color = "#1f77b4", alpha = 0.6, size = 1.2) +
    geom_point(aes(y = -log10(p.out)), color = "#d62728", alpha = 0.6, size = 1.2) +
    geom_point(data = m[SNP.PP.H4 > 0.5], aes(y = -log10(p.exp)),
               color = "#2ca02c", size = 2.2, shape = 17) +
    labs(title = sprintf("%s x %s  PP.H3=%.3f  PP.H4=%.3f",
                         pair$gene, pair$outcome_short,
                         res$summary["PP.H3.abf"], res$summary["PP.H4.abf"]),
         subtitle = sprintf("overlapping SNPs = %d, flipped = %d, window = %s:%d-%d",
                            res$overlap, res$flips, pair$chr, pair$start, pair$end),
         x = "SNP (ordered)", y = "-log10(p)", color = "") +
    theme_bw(base_size = 11)
  ggsave(out_png, plt, width = 9, height = 4.2, dpi = 160)
  invisible(TRUE)
}

# ---- stages ----------------------------------------------------------------
stage_probe <- function(cfg) {
  log_msg("=== STAGE probe ===")
  log_msg("api = ", cfg$api)
  log_msg("token: ", if (nchar(cfg$token)) paste("present (", nchar(cfg$token), " chars )", sep="") else "MISSING")
  log_msg("pairs = ", length(cfg$pairs), " | window_kb = ±", cfg$window_kb, "kb")
  for (p in cfg$pairs) cat("  ", p$gene, "->", p$outcome, "\n")
  if (!nchar(cfg$token)) stop("OPENGWAS_JWT missing; export it or pass --token")
  # 测试 token
  r <- GET(paste0(cfg$api, "/user"), add_headers(Authorization = paste("Bearer", cfg$token)))
  if (http_error(r)) stop("token invalid: HTTP ", status_code(r))
  log_msg("token OK")
  invisible(TRUE)
}

stage_fetch <- function(cfg) {
  log_msg("=== STAGE fetch ===")
  for (i in seq_along(cfg$pairs)) {
    p <- cfg$pairs[[i]]
    log_msg(sprintf("[%d/%d] %s -> %s", i, length(cfg$pairs), p$gene, p$outcome))
    dat <- tryCatch(fetch_pair(p$gene, p$outcome, cfg), error = function(e) {
      log_msg(" [error] fetch: ", conditionMessage(e)); NULL })
    if (!is.null(dat)) log_msg(" exp=", nrow(dat$exp), " out=", nrow(dat$out),
                               " chr:", dat$chr, " ", dat$start, "-", dat$end)
  }
  invisible(TRUE)
}

stage_analyse <- function(cfg) {
  log_msg("=== STAGE analyse ===")
  rows <- list(); top_rows <- list(); n_pass <- 0L
  for (i in seq_along(cfg$pairs)) {
    p <- cfg$pairs[[i]]
    log_msg(sprintf("[%d/%d] %s -> %s", i, length(cfg$pairs), p$gene, p$outcome))
    ens <- GENE_ENSEMBL[[p$gene]]
    o   <- OUTCOMES[[which(vapply(OUTCOMES, function(x) x$short == p$outcome, logical(1)))]]
    win <- get_gene_window(p$gene, cfg)
    cache_exp <- file.path(cfg$cache_dir, sprintf("exp_%s_%s_%d_%d.rds", ens, win$chr, win$start, win$end))
    cache_out <- file.path(cfg$cache_dir, sprintf("out_%s_%s_%d_%d.rds", o$id, win$chr, win$start, win$end))
    if (!file.exists(cache_exp) || !file.exists(cache_out)) {
      log_msg(" [skip] cache missing (rerun --stage fetch)"); next
    }
    exp <- readRDS(cache_exp); out <- readRDS(cache_out)
    pair <- list(gene = p$gene, ensembl = ens, outcome_short = p$outcome, outcome_id = o$id,
                 outcome_role = o$role, chr = win$chr, start = win$start, end = win$end,
                 exp = exp, out = out)
    prep <- tryCatch(align_and_prepare(pair),
                    error = function(e) { log_msg(" [error] align failed: ", conditionMessage(e)); NULL })
    if (is.null(prep)) { log_msg(" [skip] no overlapping SNPs or too few"); next }
    res <- run_coloc(prep)
    if (is.null(res)) { log_msg(" [skip] coloc.abf failed"); next }
    n_pass <- n_pass + 1L
    s <- res$summary
    verdict <- if (s[["PP.H4.abf"]] >= 0.8) "H4_shared"
               else if (s[["PP.H3.abf"]] >= 0.8) "H3_distinct"
               else "inconclusive"
    rows[[length(rows) + 1]] <- data.table(
      gene = p$gene, ensembl = ens,
      outcome_short = p$outcome, outcome_id = o$id, outcome_role = o$role,
      chr = win$chr, start = win$start, end = win$end,
      n_overlap = res$overlap, n_flip = res$flips,
      PP_H0 = s[["PP.H0.abf"]], PP_H1 = s[["PP.H1.abf"]], PP_H2 = s[["PP.H2.abf"]],
      PP_H3 = s[["PP.H3.abf"]], PP_H4 = s[["PP.H4.abf"]],
      verdict = verdict
    )
    top1 <- res$top[1]
    top_rows[[length(top_rows) + 1]] <- data.table(
      gene = p$gene, outcome_short = p$outcome,
      top_SNP = top1$snp, top_PP_H4 = top1$SNP.PP.H4
    )
    if (cfg$figs) {
      png_path <- file.path(cfg$fig_dir, sprintf("coloc_%s_%s.png", p$gene, p$outcome))
      try(plot_pair(pair, prep, res, png_path), silent = TRUE)
    }
  }
  summary_df <- if (length(rows)) rbindlist(rows) else data.table()
  top_df     <- if (length(top_rows)) rbindlist(top_rows) else data.table()
  # 输出
  fwrite(summary_df, file.path(cfg$tab_dir, "coloc_summary.csv"))
  fwrite(top_df,     file.path(cfg$tab_dir, "coloc_top_snps.csv"))
  # JSON 摘要
  json_txt <- toJSON(list(
    generated = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    n_pairs = length(cfg$pairs), n_run = n_pass,
    h4_ge_80 = sum(summary_df$PP_H4 >= 0.8, na.rm = TRUE),
    h3_ge_80 = sum(summary_df$PP_H3 >= 0.8, na.rm = TRUE),
    pairs = summary_df
  ), auto_unbox = TRUE, pretty = TRUE, digits = 6)
  writeLines(json_txt, file.path(cfg$tab_dir, "coloc_summary.json"))
  # 打印摘要
  cat("\n--- coloc summary ---\n")
  if (nrow(summary_df) == 0) { cat("(no successful coloc)\n") } else {
    print(summary_df[order(-PP_H4), .(gene, outcome_short, n_overlap, PP_H3, PP_H4, verdict)],
          row.names = FALSE, digits = 3)
    cat(sprintf("\nH4≥0.8: %d  H3≥0.8: %d  total run: %d / %d\n",
                sum(summary_df$PP_H4>=0.8), sum(summary_df$PP_H3>=0.8),
                n_pass, length(cfg$pairs)))
  }
  invisible(list(summary = summary_df, top = top_df))
}

# ---- CLI -------------------------------------------------------------------
parse_args <- function(a) {
  cfg <- list(stage = "all", proj = ".", pairs_str = NULL, window_kb = 250L,
              api = API_DEFAULT, token = Sys.getenv("OPENGWAS_JWT"),
              reuse = TRUE, verbose = TRUE, figs = TRUE)
  i <- 1L; stage_seen <- FALSE
  while (i <= length(a)) {
    k <- a[i]
    nxt <- function() { if (i+1L > length(a)) stop(paste("missing value for", k)); a[i+1L] }
    if (substr(k,1,2) != "--" && !stage_seen && k %in% c("probe","fetch","analyse","analyze","all")) {
      cfg$stage <- if (k=="analyze") "analyse" else k; stage_seen <- TRUE; i <- i+1L
    } else if (k == "--proj")    { cfg$proj <- nxt(); i <- i+2L
    } else if (k == "--pairs")   { cfg$pairs_str <- nxt(); i <- i+2L
    } else if (k == "--window")  { cfg$window_kb <- as.integer(nxt()); i <- i+2L
    } else if (k == "--api")     { cfg$api <- sub("/+$","",nxt()); i <- i+2L
    } else if (k == "--token")   { cfg$token <- nxt(); i <- i+2L
    } else if (k == "--no-cache"){ cfg$reuse <- FALSE; i <- i+1L
    } else if (k == "--no-figs") { cfg$figs <- FALSE; i <- i+1L
    } else if (k == "--quiet")   { cfg$verbose <- FALSE; i <- i+1L
    } else if (k == "--help" || k == "-h") {
      cat("Usage: Rscript 06_coloc.R [stage] [options]\n",
          " stage probe | fetch | analyse | all  (default all)\n",
          " --proj DIR project root (default .)\n",
          " --pairs G=O,G=O,... pairs to run (default built-in 27 key pairs)\n",
          " --window KB flank in kb each side (default 250, total 500kb)\n",
          " --api URL OpenGWAS API base\n",
          " --token JWT or set env OPENGWAS_JWT\n",
          " --no-cache / --no-figs / --quiet\n")
      quit(save="no", status=0)
    } else stop(paste("unknown arg:", k))
  }
  # 展开 pairs
  if (is.null(cfg$pairs_str)) {
    pairs_strs <- DEFAULT_PAIRS
  } else {
    pairs_strs <- strsplit(cfg$pairs_str, ",")[[1]]
  }
  cfg$pairs <- lapply(pairs_strs, function(s) {
    kv <- strsplit(s, "=")[[1]]
    if (length(kv) != 2) stop("bad pair spec: ", s)
    g <- toupper(trimws(kv[1])); o <- trimws(kv[2])
    if (!(g %in% names(GENE_ENSEMBL))) stop("unknown gene: ", g)
    if (!(o %in% vapply(OUTCOMES, function(x) x$short, character(1))))
      stop("unknown outcome short name: ", o)
    list(gene = g, outcome = o)
  })
  # 路径
  cfg$cache_dir <- file.path(cfg$proj, "data", "interim", "coloc_cache")
  cfg$raw_dir   <- file.path(cfg$proj, "data", "interim", "coloc_raw")
  cfg$tab_dir   <- file.path(cfg$proj, "results", "tables")
  cfg$fig_dir   <- file.path(cfg$proj, "results", "figures")
  for (d in c(cfg$cache_dir, cfg$raw_dir, cfg$tab_dir, cfg$fig_dir)) ensure_dir(d)
  cfg
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  cfg <- parse_args(argv)
  if (cfg$stage == "probe")   stage_probe(cfg)
  if (cfg$stage %in% c("fetch", "all")) { stage_probe(cfg); stage_fetch(cfg) }
  if (cfg$stage %in% c("analyse", "analyze", "all")) stage_analyse(cfg)
  log_msg("done.")
}

main()
