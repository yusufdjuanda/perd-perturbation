# PERD's differential-enhancer test (perd.R): limma lmFit ~0 + group, contrast treated - control, eBayes.
# Usage: Rscript limma_diff.R <matrix.tsv> <samples.tsv> <out_dir>
#   matrix.tsv: first column enhancer_id, then one column per RNA replicate (predicted openness)
#   samples.tsv: columns accession, time_h (0 = control)
# Writes <out_dir>/limma_t<time>.tsv for every time point other than 0 (all rows, input order).
suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly = TRUE)
values <- read.delim(args[1], check.names = FALSE, row.names = 1)
samples <- read.delim(args[2], colClasses = c("character", "numeric"))
out_dir <- args[3]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
control <- samples$accession[samples$time_h == 0]
for (t in sort(unique(samples$time_h[samples$time_h != 0]))) {
  treated <- samples$accession[samples$time_h == t]
  m <- as.matrix(values[, c(control, treated)])
  group <- factor(c(rep("control", length(control)), rep("treated", length(treated))), levels = c("control", "treated"))
  design <- model.matrix(~0 + group)
  colnames(design) <- levels(group)
  fit <- lmFit(m, design)
  fit2 <- eBayes(contrasts.fit(fit, makeContrasts(treated - control, levels = design)))
  table <- topTable(fit2, coef = 1, number = Inf, sort.by = "none")
  write.table(data.frame(enhancer_id = rownames(table), table, check.names = FALSE), file.path(out_dir, sprintf("limma_t%s.tsv", format(t))), sep = "\t", quote = FALSE, row.names = FALSE)
}
cat("limma", as.character(packageVersion("limma")), "rows", nrow(values), "\n")
