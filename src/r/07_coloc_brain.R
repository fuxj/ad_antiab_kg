#!/usr/bin/env Rscript
# 07_coloc_brain.R  ——  脑组织 cis-eQTL × GWAS 共定位（敏感性分析）
# -----------------------------------------------------------------------------
# 目的
#   第 5 章基于全血 eQTLGen 的共定位显示 BIN1 / PLCG2 / ACE / HMGCR 等强位点为
#   PP.H3（两组信号由不同因果变异驱动）。本脚本把“暴露”换成**脑组织 cis-eQTL**
#   重跑 coloc.abf，用以判断该 H3 是否由“组织错配（全血 vs 脑）”造成：
#     - 若改用脑组织后 PP.H4 明显上升 → 支持“共享因果变异”，H3 系组织错配伪影
#     - 若仍为 PP.H3 → 该位点确为不同因果变异，MR 关联更可能受 LD 驱动
#
# 数据
#   暴露(dataset1) : eQTL Catalogue release 7 脑组织 cis-eQTL
#                    （tabix 远程按窗口取；ROSMAP / CommonMind / BrainSeq / GTEx 各脑区）
#   结局(dataset2) : 复用 06_coloc.R 已抓好的 GWAS 窗口缓存
#                    data/interim/coloc_cache/out_<id>_<chr>_<start>_<end>.rds （GRCh37）
#
# 关键设计
#   * eQTL Catalogue 用 GRCh38，GWAS 缓存用 GRCh37 —— 一律按 **rsID 合并**，规避坐标体系差异
#   * 需系统里有 tabix（htslib），并能访问 https://ftp.ebi.ac.uk
#   * 逐 (基因 × 数据集) 缓存 RDS，可断点续跑
#   * 单对失败不影响整批（tryCatch 隔离）
#
# 依赖
#   R 包：data.table、coloc(≥5.0)
#   系统：tabix（`conda install -c bioconda htslib` 会一并安装 tabix / bgzip）
#
# 用法
#   Rscript src/r/07_coloc_brain.R [probe|fetch|analyse|all] [options]
#     --proj DIR        项目根（默认 .）
#     --pairs GENE=OUTCOME,...   默认见 DEFAULT_PAIRS
#     --datasets NAME,NAME,...   脑 eQTL 数据集，默认 4 个 DLPFC/cortex 大样本集
#     --window KB       两侧窗口（默认 250，即 ±250kb）
#     --tabix PATH      tabix 可执行文件路径（默认从 PATH 找）
#     --gwas-cache DIR  GWAS 缓存目录（默认 data/interim/coloc_cache，即 06 的输出）
#     --no-cache        强制重抓脑 eQTL
#     --quiet           少日志
#
# 输出
#   data/interim/coloc_brain_cache/*.rds        脑 eQTL 窗口缓存
#   results/tables/coloc_brain_summary.csv      PP.H0–H4 总表（含数据集列）
#   results/tables/coloc_brain_top_snps.csv     每对 top SNP
#   results/tables/coloc_brain_summary.json     JSON 摘要
#   results/tables/coloc_brain_vs_blood.csv     与全血版（若存在）并排对照
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  need <- c("data.table")
  for (p in need) if (!requireNamespace(p, quietly = TRUE))
    stop(sprintf("missing R package: %s; install with\n  conda install -c conda-forge r-%s", p, tolower(p)))
  library(data.table)
})
if (!requireNamespace("coloc", quietly = TRUE))
  stop("missing R package: coloc; install with\n  conda install -c conda-forge r-coloc")
suppressPackageStartupMessages(library(coloc))

# ---- 常量：eQTL Catalogue 脑组织数据集 --------------------------------------
EQTL_BASE <- "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats"

BRAIN_DATASETS <- list(
  ROSMAP_DLPFC      = list(qts = "QTS000025", qtd = "QTD000434", n = 560, tissue = "brain (DLPFC)",        study = "ROSMAP"),
  CommonMind_DLPFC  = list(qts = "QTS000008", qtd = "QTD000075", n = 586, tissue = "brain (DLPFC)",        study = "CommonMind"),
  BrainSeq_DLPFC    = list(qts = "QTS000005", qtd = "QTD000051", n = 479, tissue = "brain (DLPFC)",        study = "BrainSeq"),
  GTEx_cortex       = list(qts = "QTS000015", qtd = "QTD000171", n = 205, tissue = "brain (cortex)",       study = "GTEx"),
  GTEx_hippocampus  = list(qts = "QTS000015", qtd = "QTD000181", n = 165, tissue = "brain (hippocampus)",  study = "GTEx"),
  GTEx_DLPFC        = list(qts = "QTS000015", qtd = "QTD000176", n = 175, tissue = "brain (DLPFC)",        study = "GTEx"),
  Braineac2_putamen = list(qts = "QTS000004", qtd = "QTD000041", n = 102, tissue = "brain (putamen)",      study = "Braineac2")
)

# 默认使用的脑数据集（优先样本量大的 DLPFC/cortex）
DEFAULT_DATASETS <- c("ROSMAP_DLPFC", "CommonMind_DLPFC", "BrainSeq_DLPFC")

# eQTL Catalogue 名义汇总统计 *.all.tsv.gz 的真实列顺序（2026-10 真机实测，19 列）
# 实测原始表头：molecular_trait_id, chromosome, position, ref, alt, variant, ma_samples,
#              maf, pvalue, beta, se, type, ac, an, r2,
#              molecular_trait_object_id, gene_id, median_tpm, rsid
EQTL_COLS <- c("molecular_trait_id", "chromosome", "position", "ref", "alt", "variant",
               "ma_samples", "maf", "pvalue", "beta", "se", "type", "ac", "an", "r2",
               "molecular_trait_object_id", "gene_id", "median_tpm", "rsid")

# ---- 常量：基因与结局（与 06_coloc.R 保持一致）-----------------------------
GENE_ENSEMBL <- c(
  VKORC1 = "ENSG00000167397", F10 = "ENSG00000126218", PROC = "ENSG00000115718",
  ITGB3  = "ENSG00000259207", P2RY12 = "ENSG00000169313",
  PTGS1  = "ENSG00000095303", PTGS2 = "ENSG00000073756", PDGFRB = "ENSG00000113721",
  HMGCR  = "ENSG00000113161",
  CLDN5  = "ENSG00000184113", MMP9 = "ENSG00000100985", IL1B = "ENSG00000125538",
  BIN1   = "ENSG00000136717", ACE = "ENSG00000159640", APP = "ENSG00000142192",
  PLCG2  = "ENSG00000197943"
)

# 基因在 **GRCh37** 上的近似 TSS（与 06_coloc.R 完全一致，用于定位 GWAS 缓存文件名）
GENE_B37 <- list(
  VKORC1 = list(chr = "16", tss = 31100000), F10 = list(chr = "13", tss = 113700000),
  PROC   = list(chr = "2",  tss = 128100000), ITGB3 = list(chr = "17", tss = 45300000),
  P2RY12 = list(chr = "3",  tss = 151000000), PTGS1 = list(chr = "9",  tss = 125100000),
  PTGS2  = list(chr = "1",  tss = 186600000), PDGFRB = list(chr = "5", tss = 149500000),
  HMGCR  = list(chr = "5",  tss = 74600000),  CLDN5 = list(chr = "22", tss = 19500000),
  MMP9   = list(chr = "20", tss = 44600000),  IL1B = list(chr = "2",  tss = 113500000),
  BIN1   = list(chr = "2",  tss = 127800000), ACE = list(chr = "17", tss = 61500000),
  APP    = list(chr = "21", tss = 27200000),  PLCG2 = list(chr = "16", tss = 81700000)
)

# 基因在 **GRCh38** 上的坐标（来自 GTEx v26 注释；用于向 eQTL Catalogue 取窗口）
# 值 = list(chr, start, end)，窗口取 [start-window, end+window]
GENE_HG38 <- list(
  VKORC1 = list(chr = "16", start = 31090842, end = 31095980),
  F10    = list(chr = "13", start = 113122814, end = 113149529),
  ITGB3  = list(chr = "17", start = 47253846, end = 47311816),
  P2RY12 = list(chr = "3",  start = 151337380, end = 151384812),
  PTGS1  = list(chr = "9",  start = 122370530, end = 122395703),
  PDGFRB = list(chr = "5",  start = 150113837, end = 150155872),
  HMGCR  = list(chr = "5",  start = 75336329, end = 75362104),
  CLDN5  = list(chr = "22", start = 19523024, end = 19527545),
  MMP9   = list(chr = "20", start = 46008908, end = 46016561),
  IL1B   = list(chr = "2",  start = 112829751, end = 112836903),
  BIN1   = list(chr = "2",  start = 127048027, end = 127107355),
  ACE    = list(chr = "17", start = 63477061, end = 63498380),
  APP    = list(chr = "21", start = 25880550, end = 26171128),
  PLCG2  = list(chr = "16", start = 81739097, end = 81962693)
)

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

# 默认要跑的“关键对”：优先选择第 5 章里 PP.H4/PP.H3 结果最需要脑组织复核的位点
DEFAULT_PAIRS <- c(
  "BIN1=AD", "BIN1=AD_repl2", "PLCG2=AD", "ACE=AD", "ACE=AD_repl2", "APP=SAH",
  "VKORC1=ICH", "PTGS1=ICH", "PTGS1=SAH", "F10=ICH", "P2RY12=ICH",
  "ITGB3=ICH", "PDGFRB=ICH", "HMGCR=LDL", "MMP9=ICH", "CLDN5=ICH", "IL1B=ICH"
)

# ---- 工具 ------------------------------------------------------------------
log_msg <- function(...) {
  cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n", sep = "", file = "", append = FALSE)
  flush.console()
}
ensure_dir <- function(p) { if (!dir.exists(p)) dir.create(p, recursive = TRUE) }

# ---- tabix 远程取窗口 -------------------------------------------------------
# 返回该区域的所有原始文本行（未解析）。contig 名自动兼容 "2" 与 "chr2"。
tabix_fetch_lines <- function(url, chrom, start, end, cfg) {
  regs <- c(sprintf("%s:%d-%d", chrom, start, end),
            sprintf("chr%s:%d-%d", chrom, start, end))
  for (reg in regs) {
    cmd <- sprintf("%s %s %s 2>/dev/null", shQuote(cfg$tabix), shQuote(url), shQuote(reg))
    lines <- tryCatch(system(cmd, intern = TRUE), error = function(e) character(0))
    if (length(lines) > 0) return(list(lines = lines, region = reg))
  }
  list(lines = character(0), region = regs[1])
}

# 把 tabix 文本行解析为“该基因”的 eQTL 记录
parse_eqtl_lines <- function(lines, gene_ens) {
  if (!length(lines)) return(data.table())
  # tabix 可能把表头行一起吐出来（不同版本 htslib 行为不同）；含列名则先丢弃，避免整体错位
  if (grepl("molecular_trait_id", lines[1], fixed = TRUE)) lines <- lines[-1]
  if (!length(lines)) return(data.table())
  dt <- tryCatch(
    fread(text = lines, header = FALSE, sep = "\t", quote = "", fill = TRUE,
          na.strings = c("NA", "na", ".")),
    error = function(e) { log_msg(" [warn] fread failed: ", conditionMessage(e)); NULL })
  if (is.null(dt) || nrow(dt) == 0) return(data.table())
  nc <- min(ncol(dt), length(EQTL_COLS))
  setnames(dt, seq_len(nc), EQTL_COLS[seq_len(nc)])
  # 只保留目标基因（gene_id 与 molecular_trait_id 均为无版本 ENSG）
  keep <- (dt$gene_id %in% gene_ens) | (dt$molecular_trait_id %in% gene_ens)
  dt <- dt[keep]
  if (nrow(dt) == 0) return(data.table())
  out <- data.table(
    SNP  = as.character(dt$rsid),
    beta = suppressWarnings(as.numeric(dt$beta)),
    se   = suppressWarnings(as.numeric(dt$se)),
    EA   = toupper(as.character(dt$alt)),
    OA   = toupper(as.character(dt$ref)),
    eaf  = suppressWarnings(as.numeric(dt$maf)),
    p    = suppressWarnings(as.numeric(dt$pvalue)),
    n    = suppressWarnings(as.numeric(dt$an)) / 2
  )
  out <- out[!is.na(SNP) & nzchar(SNP) & SNP != "." &
             !is.na(beta) & !is.na(se) & se > 0 &
             !is.na(p) & is.finite(p) & p > 0 & p <= 1]
  out <- out[!is.na(EA) & !is.na(OA) & nzchar(EA) & nzchar(OA)]
  out
}

# ---- 等位基因对齐 + coloc.abf 输入准备（与 06_coloc.R 完全一致，已修 bug）----
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
  amb <- !same & !swap
  if (any(amb)) {
    palin <- (m$EA.exp %in% c("A","T") & m$OA.exp %in% c("A","T")) |
             (m$EA.exp %in% c("C","G") & m$OA.exp %in% c("C","G"))
    drop <- amb & palin
    if (any(drop)) {
      log_msg(" [align] drop ", sum(drop), " palindromic ambiguous SNP(s) out of ", nrow(m))
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
  idx_swap <- which(swap); flips <- length(idx_swap)
  if (flips > 0) m[idx_swap, c("beta.out", "eaf.out") := .(-beta.out, 1 - eaf.out)]
  # 样本量：eQTL 侧优先用数据集中值，缺失则回退到数据集标称 n
  n_exp <- median(m$n.exp, na.rm = TRUE)
  if (!is.finite(n_exp) || n_exp <= 0) n_exp <- if (!is.null(pair$exp_n) && is.finite(pair$exp_n)) pair$exp_n else 500
  n_out <- median(m$n.out, na.rm = TRUE)
  if (!is.finite(n_out) || n_out <= 0) n_out <- if (grepl("^ieu-b-110", pair$outcome_id)) 173082 else 473513
  maf <- ifelse(!is.na(m$eaf.exp), pmin(m$eaf.exp, 1 - m$eaf.exp),
         ifelse(!is.na(m$eaf.out), pmin(m$eaf.out, 1 - m$eaf.out), NA_real_))
  ok <- is.finite(m$beta.exp) & is.finite(m$se.exp) & is.finite(m$beta.out) & is.finite(m$se.out) &
        m$se.exp > 0 & m$se.out > 0 & is.finite(maf) & maf > 0 & maf < 0.5
  if (!all(ok)) { m <- m[ok]; maf <- maf[ok] }
  if (nrow(m) < 10) return(NULL)
  dup <- duplicated(m$SNP)
  if (any(dup)) {
    log_msg(" [align] drop ", sum(dup), " residual duplicated rsID(s) before coloc.abf")
    m <- m[!dup]; maf <- maf[!dup]
  }
  stopifnot(length(maf) == nrow(m), length(m$SNP) == nrow(m))
  list(dataset1 = list(beta = m$beta.exp, varbeta = m$se.exp^2,
                       N = n_exp, MAF = maf, type = "quant", snp = m$SNP),
       dataset2 = list(beta = m$beta.out, varbeta = m$se.out^2,
                       N = n_out, MAF = maf,
                       type = if (pair$outcome_short %in% c("AD","ICH","ICH_fin","SAH","SVD","IS")) "cc" else "quant",
                       s = 0.15, snp = m$SNP),
       overlap = nrow(m), flips = flips, merged = m)
}

run_coloc <- function(prepared) {
  if (is.null(prepared)) return(NULL)
  res <- tryCatch(
    coloc.abf(dataset1 = prepared$dataset1, dataset2 = prepared$dataset2),
    error = function(e) { log_msg(" [error] coloc.abf: ", conditionMessage(e)); NULL })
  if (is.null(res)) return(NULL)
  pp <- res$results[, c("snp", "SNP.PP.H4")]
  setDT(pp); setorder(pp, -SNP.PP.H4)
  list(summary = res$summary, top = head(pp, 10),
       overlap = prepared$overlap, flips = prepared$flips)
}

# ---- 窗口计算 ---------------------------------------------------------------
win_gwas_b37 <- function(gene, cfg) {          # 用于定位 06 的 GWAS 缓存
  info <- GENE_B37[[gene]]
  list(chr = info$chr,
       start = max(1L, as.integer(info$tss - cfg$window_kb * 1000L)),
       end   = as.integer(info$tss + cfg$window_kb * 1000L))
}
win_eqtl_hg38 <- function(gene, cfg) {         # 用于向 eQTL Catalogue 取窗口
  info <- GENE_HG38[[gene]]
  list(chr = info$chr,
       start = max(1L, as.integer(info$start - cfg$window_kb * 1000L)),
       end   = as.integer(info$end + cfg$window_kb * 1000L))
}

# ---- 抓取脑 eQTL 窗口（带缓存）---------------------------------------------
fetch_brain_one <- function(ds_name, gene, cfg) {
  ds  <- BRAIN_DATASETS[[ds_name]]
  ens <- GENE_ENSEMBL[[gene]]
  win <- win_eqtl_hg38(gene, cfg)
  cache <- file.path(cfg$brain_cache_dir, sprintf("brain_%s_%s_%s_%d_%d.rds",
                                                  ds_name, gene, win$chr, win$start, win$end))
  if (file.exists(cache) && cfg$reuse) {
    obj <- readRDS(cache)
    if (cfg$verbose) log_msg("  [cache] ", ds_name, " ", gene, " (", nrow(obj), " rows)")
    return(obj)
  }
  url <- sprintf("%s/%s/%s/%s.all.tsv.gz", EQTL_BASE, ds$qts, ds$qtd, ds$qtd)
  log_msg("  [fetch] ", ds_name, " ", gene, "  ", win$chr, ":", win$start, "-", win$end)
  got <- tabix_fetch_lines(url, win$chr, win$start, win$end, cfg)
  df  <- parse_eqtl_lines(got$lines, ens)
  if (nrow(df) == 0)
    log_msg("  [warn] no cis-eQTL rows for ", gene, " in ", ds_name, " (region ", got$region, ")")
  saveRDS(df, cache)
  Sys.sleep(0.2)
  df
}

fetch_pair_brain <- function(gene, outcome_short, ds_name, cfg) {
  ens <- GENE_ENSEMBL[[gene]]
  o   <- OUTCOMES[[which(vapply(OUTCOMES, function(x) x$short == outcome_short, logical(1)))]]
  if (is.null(o)) stop("unknown outcome: ", outcome_short)
  win_b37 <- win_gwas_b37(gene, cfg)
  cache_out <- file.path(cfg$gwas_cache_dir,
                         sprintf("out_%s_%s_%d_%d.rds", o$id, win_b37$chr, win_b37$start, win_b37$end))
  if (!file.exists(cache_out)) {
    log_msg("  [skip] GWAS cache missing: ", basename(cache_out), " (先跑 06_coloc.R fetch)")
    return(NULL)
  }
  out <- readRDS(cache_out)
  exp <- tryCatch(fetch_brain_one(ds_name, gene, cfg),
                  error = function(e) { log_msg("  [error] fetch brain: ", conditionMessage(e)); data.table() })
  list(gene = gene, ensembl = ens, outcome_short = outcome_short, outcome_id = o$id,
       outcome_role = o$role, dataset = ds_name,
       chr = win_b37$chr, start = win_b37$start, end = win_b37$end,
       exp_n = BRAIN_DATASETS[[ds_name]]$n,
       exp = exp, out = out)
}

# ---- stages ----------------------------------------------------------------
stage_probe <- function(cfg) {
  log_msg("=== STAGE probe (brain) ===")
  log_msg("tabix = ", cfg$tabix)
  if (!nzchar(Sys.which(cfg$tabix)) && !file.exists(cfg$tabix))
    stop("tabix not found. 安装：conda install -c bioconda htslib  （会带 tabix）")
  log_msg("datasets = ", length(cfg$datasets), " | ", paste(cfg$datasets, collapse = ", "))
  log_msg("pairs = ", length(cfg$pairs), " | window = ±", cfg$window_kb, "kb")
  # 用第一个基因试抓一个小窗口，验证 tabix 远程可用
  g <- cfg$pairs[[1]]$gene
  ds <- cfg$datasets[1]
  win <- win_eqtl_hg38(g, cfg)
  url <- sprintf("%s/%s/%s/%s.all.tsv.gz", EQTL_BASE, BRAIN_DATASETS[[ds]]$qts,
                 BRAIN_DATASETS[[ds]]$qtd, BRAIN_DATASETS[[ds]]$qtd)
  log_msg("probe fetch: ", ds, " ", g, " ", win$chr, ":", win$start, "-", win$end)
  got <- tabix_fetch_lines(url, win$chr, win$start, win$end, cfg)
  log_msg("probe rows returned: ", length(got$lines), " (region tried: ", got$region, ")")
  if (length(got$lines) == 0)
    log_msg(" [warn] 没取到数据：可能是网络不可达 ftp.ebi.ac.uk，或 contig 命名不符。\n",
            "        请先单独测试：tabix ", url, " 2:127048027-127107355 | head")
  else
    log_msg("probe OK —— tabix 远程可用")
  invisible(TRUE)
}

stage_fetch <- function(cfg) {
  log_msg("=== STAGE fetch (brain) ===")
  n <- 0L
  for (i in seq_along(cfg$pairs)) {
    p <- cfg$pairs[[i]]
    for (ds in cfg$datasets) {
      log_msg(sprintf("[%d/%d] %s -> %s  @ %s", i, length(cfg$pairs), p$gene, p$outcome, ds))
      dat <- tryCatch(fetch_pair_brain(p$gene, p$outcome, ds, cfg),
                      error = function(e) { log_msg(" [error] ", conditionMessage(e)); NULL })
      if (!is.null(dat)) {
        n <- n + 1L
        log_msg("  exp=", nrow(dat$exp), " out=", nrow(dat$out))
      }
    }
  }
  log_msg("fetched windows: ", n)
  invisible(TRUE)
}

stage_analyse <- function(cfg) {
  log_msg("=== STAGE analyse (brain) ===")
  rows <- list(); top_rows <- list(); n_pass <- 0L
  for (i in seq_along(cfg$pairs)) {
    p <- cfg$pairs[[i]]
    win_b37 <- win_gwas_b37(p$gene, cfg)
    o <- OUTCOMES[[which(vapply(OUTCOMES, function(x) x$short == p$outcome, logical(1)))]]
    for (ds in cfg$datasets) {
      log_msg(sprintf("[%d/%d] %s -> %s  @ %s", i, length(cfg$pairs), p$gene, p$outcome, ds))
      win <- win_eqtl_hg38(p$gene, cfg)
      cache_exp <- file.path(cfg$brain_cache_dir,
                             sprintf("brain_%s_%s_%s_%d_%d.rds", ds, p$gene, win$chr, win$start, win$end))
      cache_out <- file.path(cfg$gwas_cache_dir,
                             sprintf("out_%s_%s_%d_%d.rds", o$id, win_b37$chr, win_b37$start, win_b37$end))
      if (!file.exists(cache_exp) || !file.exists(cache_out)) {
        log_msg("  [skip] cache missing"); next
      }
      pair <- list(gene = p$gene, ensembl = GENE_ENSEMBL[[p$gene]], outcome_short = p$outcome,
                   outcome_id = o$id, outcome_role = o$role, dataset = ds,
                   chr = win_b37$chr, start = win_b37$start, end = win_b37$end,
                   exp_n = BRAIN_DATASETS[[ds]]$n,
                   exp = readRDS(cache_exp), out = readRDS(cache_out))
      prep <- tryCatch(align_and_prepare(pair),
                       error = function(e) { log_msg("  [error] align failed: ", conditionMessage(e)); NULL })
      if (is.null(prep)) { log_msg("  [skip] no overlapping SNPs or too few"); next }
      res <- run_coloc(prep)
      if (is.null(res)) { log_msg("  [skip] coloc.abf failed"); next }
      n_pass <- n_pass + 1L
      s <- res$summary
      verdict <- if (s[["PP.H4.abf"]] >= 0.8) "H4_shared"
                 else if (s[["PP.H3.abf"]] >= 0.8) "H3_distinct" else "inconclusive"
      rows[[length(rows) + 1]] <- data.table(
        dataset = ds, study = BRAIN_DATASETS[[ds]]$study, tissue = BRAIN_DATASETS[[ds]]$tissue,
        dataset_n = BRAIN_DATASETS[[ds]]$n,
        gene = p$gene, ensembl = GENE_ENSEMBL[[p$gene]],
        outcome_short = p$outcome, outcome_id = o$id, outcome_role = o$role,
        n_overlap = res$overlap, n_flip = res$flips,
        PP_H0 = s[["PP.H0.abf"]], PP_H1 = s[["PP.H1.abf"]], PP_H2 = s[["PP.H2.abf"]],
        PP_H3 = s[["PP.H3.abf"]], PP_H4 = s[["PP.H4.abf"]], verdict = verdict
      )
      top1 <- res$top[1]
      top_rows[[length(top_rows) + 1]] <- data.table(
        dataset = ds, gene = p$gene, outcome_short = p$outcome,
        top_SNP = top1$snp, top_PP_H4 = top1$SNP.PP.H4)
    }
  }
  summary_df <- if (length(rows)) rbindlist(rows) else data.table()
  top_df     <- if (length(top_rows)) rbindlist(top_rows) else data.table()
  fwrite(summary_df, file.path(cfg$tab_dir, "coloc_brain_summary.csv"))
  fwrite(top_df,     file.path(cfg$tab_dir, "coloc_brain_top_snps.csv"))

  # 与全血版并排对照（若 06 的结果存在）
  blood_path <- file.path(cfg$tab_dir, "coloc_summary.csv")
  if (file.exists(blood_path) && nrow(summary_df) > 0) {
    bl <- tryCatch(fread(blood_path), error = function(e) NULL)
    if (!is.null(bl) && all(c("gene", "outcome_short", "PP_H3", "PP_H4") %in% names(bl))) {
      bl2 <- bl[, .(gene, outcome_short, blood_PP_H3 = PP_H3, blood_PP_H4 = PP_H4,
                    blood_verdict = verdict)]
      cmp <- merge(summary_df[, .(dataset, gene, outcome_short, brain_PP_H3 = PP_H3,
                                  brain_PP_H4 = PP_H4, brain_verdict = verdict)],
                   bl2, by = c("gene", "outcome_short"), all.x = TRUE)
      # 每个基因-结局取脑组织里 PP.H4 最高的数据集
      cmp[, best_brain_PP_H4 := max(brain_PP_H4, na.rm = TRUE), by = .(gene, outcome_short)]
      cmp[, delta_PP_H4 := brain_PP_H4 - blood_PP_H4]
      fwrite(cmp, file.path(cfg$tab_dir, "coloc_brain_vs_blood.csv"))
      log_msg("wrote coloc_brain_vs_blood.csv")
    }
  }

  json_txt <- jsonlite::toJSON(list(
    generated = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    n_pairs = length(cfg$pairs), n_datasets = length(cfg$datasets), n_run = n_pass,
    h4_ge_80 = sum(summary_df$PP_H4 >= 0.8, na.rm = TRUE),
    h3_ge_80 = sum(summary_df$PP_H3 >= 0.8, na.rm = TRUE),
    results = summary_df
  ), auto_unbox = TRUE, pretty = TRUE, digits = 6)
  writeLines(json_txt, file.path(cfg$tab_dir, "coloc_brain_summary.json"))

  cat("\n--- coloc (brain) summary ---\n")
  if (nrow(summary_df) == 0) cat("(no successful coloc)\n") else {
    print(summary_df[order(-PP_H4),
          .(dataset, gene, outcome_short, n_overlap, PP_H3, PP_H4, verdict)],
          row.names = FALSE, digits = 3)
    cat(sprintf("\nH4>=0.8: %d  H3>=0.8: %d  total run: %d / (%d pairs x %d datasets)\n",
                sum(summary_df$PP_H4 >= 0.8), sum(summary_df$PP_H3 >= 0.8),
                n_pass, length(cfg$pairs), length(cfg$datasets)))
  }
  invisible(list(summary = summary_df, top = top_df))
}

# ---- CLI -------------------------------------------------------------------
parse_args <- function(a) {
  cfg <- list(stage = "all", proj = ".", pairs_str = NULL, datasets_str = NULL,
              window_kb = 250L, tabix = "tabix", reuse = TRUE, verbose = TRUE,
              gwas_cache_dir = NULL)
  i <- 1L; stage_seen <- FALSE
  while (i <= length(a)) {
    k <- a[i]
    nxt <- function() { if (i + 1L > length(a)) stop(paste("missing value for", k)); a[i + 1L] }
    if (substr(k, 1, 2) != "--" && !stage_seen && k %in% c("probe", "fetch", "analyse", "analyze", "all")) {
      cfg$stage <- if (k == "analyze") "analyse" else k; stage_seen <- TRUE; i <- i + 1L
    } else if (k == "--proj")    { cfg$proj <- nxt(); i <- i + 2L
    } else if (k == "--pairs")   { cfg$pairs_str <- nxt(); i <- i + 2L
    } else if (k == "--datasets"){ cfg$datasets_str <- nxt(); i <- i + 2L
    } else if (k == "--window")  { cfg$window_kb <- as.integer(nxt()); i <- i + 2L
    } else if (k == "--tabix")   { cfg$tabix <- nxt(); i <- i + 2L
    } else if (k == "--gwas-cache") { cfg$gwas_cache_dir <- nxt(); i <- i + 2L
    } else if (k == "--no-cache"){ cfg$reuse <- FALSE; i <- i + 1L
    } else if (k == "--quiet")   { cfg$verbose <- FALSE; i <- i + 1L
    } else if (k == "--help" || k == "-h") {
      cat("Usage: Rscript 07_coloc_brain.R [stage] [options]\n",
          " stage probe | fetch | analyse | all  (default all)\n",
          " --proj DIR             project root (default .)\n",
          " --pairs G=O,G=O,...    pairs (default built-in key pairs)\n",
          " --datasets N,N,...     brain eQTL datasets (default ",
          paste(DEFAULT_DATASETS, collapse = ","), ")\n",
          " --window KB            flank each side (default 250)\n",
          " --tabix PATH           tabix executable (default: from PATH)\n",
          " --gwas-cache DIR       GWAS cache dir (default data/interim/coloc_cache)\n",
          " --no-cache / --quiet\n",
          " available brain datasets:\n  ",
          paste(sprintf("%s(%s,n=%d)", names(BRAIN_DATASETS),
                        vapply(BRAIN_DATASETS, function(x) x$tissue, character(1)),
                        vapply(BRAIN_DATASETS, function(x) x$n, numeric(1))), collapse = "\n  "), "\n",
          sep = "")
      quit(save = "no", status = 0)
    } else stop(paste("unknown arg:", k))
  }
  # pairs
  pairs_strs <- if (is.null(cfg$pairs_str)) DEFAULT_PAIRS else strsplit(cfg$pairs_str, ",")[[1]]
  cfg$pairs <- lapply(pairs_strs, function(s) {
    kv <- strsplit(s, "=")[[1]]
    if (length(kv) != 2) stop("bad pair spec: ", s)
    g <- toupper(trimws(kv[1])); o <- trimws(kv[2])
    if (!(g %in% names(GENE_ENSEMBL))) stop("unknown gene: ", g)
    if (!(g %in% names(GENE_HG38)))    stop("gene lacks hg38 coords: ", g)
    if (!(o %in% vapply(OUTCOMES, function(x) x$short, character(1)))) stop("unknown outcome: ", o)
    list(gene = g, outcome = o)
  })
  # datasets
  cfg$datasets <- if (is.null(cfg$datasets_str)) DEFAULT_DATASETS else trimws(strsplit(cfg$datasets_str, ",")[[1]])
  bad <- setdiff(cfg$datasets, names(BRAIN_DATASETS))
  if (length(bad)) stop("unknown dataset(s): ", paste(bad, collapse = ", "))
  # paths
  cfg$gwas_cache_dir  <- cfg$gwas_cache_dir %||% file.path(cfg$proj, "data", "interim", "coloc_cache")
  cfg$brain_cache_dir <- file.path(cfg$proj, "data", "interim", "coloc_brain_cache")
  cfg$tab_dir         <- file.path(cfg$proj, "results", "tables")
  for (d in c(cfg$brain_cache_dir, cfg$tab_dir)) ensure_dir(d)
  cfg
}

`%||%` <- function(a, b) if (!is.null(a) && length(a) && !is.na(a) && nzchar(a)) a else b

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  cfg <- parse_args(argv)
  if (cfg$stage == "probe") probe_only <- TRUE else probe_only <- FALSE
  if (cfg$stage %in% c("fetch", "all")) { stage_probe(cfg); stage_fetch(cfg) }
  if (cfg$stage == "probe") stage_probe(cfg)
  if (cfg$stage %in% c("analyse", "analyze", "all")) stage_analyse(cfg)
  log_msg("done.")
}

main()
