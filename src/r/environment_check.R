# 环境自检: 打印 R 版本与关键包版本
# 用法:
#   cd $PROJ
#   Rscript src/r/environment_check.R            # 自动推断项目根
#   Rscript src/r/environment_check.R /data/ad_kg # 显式指定

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  proj <- args[1]
} else if (nzchar(Sys.getenv("PROJ"))) {
  proj <- Sys.getenv("PROJ")
} else {
  # 从脚本路径推断: src/r/xxx.R -> 上两级
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  proj <- if (length(f) == 1) normalizePath(file.path(dirname(f), "..", "..")) else getwd()
}

pkgs <- c("tidyverse", "data.table", "DESeq2", "edgeR", "limma", "clusterProfiler",
          "fgsea", "Seurat", "coloc", "pROC", "rms", "dcurves", "ggplot2",
          "TwoSampleMR", "ieugwasr", "shiny", "forestploter")

res <- data.frame(package = pkgs, installed = NA, version = NA, stringsAsFactors = FALSE)
for (i in seq_along(pkgs)) {
  if (requireNamespace(pkgs[i], quietly = TRUE)) {
    res$installed[i] <- TRUE
    res$version[i]   <- as.character(packageVersion(pkgs[i]))
  } else {
    res$installed[i] <- FALSE
    res$version[i]   <- "-"
  }
}
print(res)

cat("\nOpenGWAS JWT:", if (nzchar(Sys.getenv("OPENGWAS_JWT"))) "已配置" else "未配置", "\n")
cat("R version:", R.version.string, "\n")

dir.create(file.path(proj, "logs"), showWarnings = FALSE, recursive = TRUE)
out <- file.path(proj, "logs", "env_check_R.csv")
write.csv(res, out, row.names = FALSE)
cat("已写入", out, "\n")
