#!/usr/bin/env Rscript
# Drive the R package scCDC on a fixed synthetic dataset so pysccdc can
# be compared against it.
#
# Usage:
#   Rscript r_reference_driver.R <counts.csv> <clusters.csv> <out_dir> [seed]
#
# Inputs:
#   counts.csv    genes x cells integer count matrix (row 1 = cell ids,
#                 col 1 = gene ids)
#   clusters.csv  two columns: cell, cluster
#
# Outputs (in out_dir):
#   detection.csv     ContaminationDetection() degree-of-contamination
#                     table for the detected GCGs (row names = GCGs)
#   gcgs.txt          the detected GCG names, one per line
#   entropy.bin       per-gene, per-cluster Shannon entropy of counts
#                     (genes x clusters, column-major doubles)
#   distance.bin      per-gene, per-cluster entropy divergence + mean
#                     (genes x (clusters + 1), column-major doubles)
#   thresholds.csv    per-GCG Youden subtraction threshold
#   corrected.csv     genes x cells corrected count matrix
#   genes.txt / clusters.txt   row and column names of the binary tables
#   ratio.txt         dataset contamination ratio

suppressPackageStartupMessages({
  library(scCDC)
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
counts_csv   <- args[[1]]
clusters_csv <- args[[2]]
out_dir      <- args[[3]]
seed         <- if (length(args) > 3) as.integer(args[[4]]) else 0L
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


# --- read the synthetic dataset -------------------------------------
counts <- as.matrix(read.csv(counts_csv, row.names = 1, check.names = FALSE))
mode(counts) <- "integer"
clusters <- read.csv(clusters_csv, stringsAsFactors = FALSE)

# --- build a clustered Seurat object --------------------------------
obj <- CreateSeuratObject(counts = as(counts, "dgCMatrix"))
cl <- clusters$cluster
names(cl) <- clusters$cell
obj$cluster <- cl[colnames(obj)]
Idents(obj) <- factor(obj$cluster)

# --- per-gene, per-cluster entropy ----------------------------------
levels_ <- levels(Idents(obj))
ent_list <- lapply(levels_, function(x) {
  tmp <- subset(obj, idents = x)
  m <- as.matrix(GetAssayData(tmp, layer = "counts"))
  scCDC:::MatrixToEntropy(m)
})
ent <- do.call(cbind, ent_list)
rownames(ent) <- rownames(obj)
colnames(ent) <- levels_
writeBin(as.vector(ent), file.path(out_dir, "entropy.bin"))
writeLines(rownames(ent), file.path(out_dir, "genes.txt"))
writeLines(colnames(ent), file.path(out_dir, "clusters.txt"))

# --- ContaminationDetection -----------------------------------------
set.seed(seed)
detection <- ContaminationDetection(
  obj, restriction_factor = 0.5, min.cell = 50, percent.cutoff = 0.2
)
write.csv(detection, file.path(out_dir, "detection.csv"))
gcgs <- rownames(detection)
writeLines(gcgs, file.path(out_dir, "gcgs.txt"))

# --- per-gene entropy divergence (distance) -------------------------
# re-run the internal curve fit, from the same seed, to export the full distance table
set.seed(seed)
ave <- log(scCDC:::CalAverageExpression.Seurat(obj, rownames(obj)) + 1)
entropy_result <- scCDC:::CalEnt.Seurat(obj, rownames(obj))
total_cluster <- colnames(ave)
all <- lapply(seq_along(total_cluster), function(i) {
  cluster <- total_cluster[i]
  gea <- tibble::tibble(
    Gene = rownames(entropy_result),
    mean.expr = ave[, cluster],
    entropy = entropy_result[, cluster]
  )
  scCDC:::generate_curve(gea)
})
genes <- rownames(obj)
dist_list <- lapply(all, function(tmp) tmp[genes, "distance"])
distance <- as.data.frame(do.call(cbind, dist_list))
colnames(distance) <- total_cluster
rownames(distance) <- genes
distance$mean_distance <- rowMeans(distance)
writeBin(as.vector(as.matrix(distance)), file.path(out_dir, "distance.bin"))

# --- ContaminationCorrection ----------------------------------------
corrected_obj <- ContaminationCorrection(obj, gcgs, auc_thres = 0.9,
                                         min.cell = 50)
corr <- as.matrix(GetAssayData(corrected_obj, assay = "Corrected",
                               layer = "counts"))
write.csv(corr, file.path(out_dir, "corrected.csv"))

# --- per-GCG Youden thresholds (recovered from the count delta) -----
orig <- counts[gcgs, , drop = FALSE]
corr_g <- corr[gcgs, , drop = FALSE]
thr <- sapply(seq_along(gcgs), function(i) {
  d <- orig[i, ] - corr_g[i, ]
  max(d)  # the subtracted (rounded) threshold
})
thr_df <- data.frame(gene = gcgs, threshold = thr)
write.csv(thr_df, file.path(out_dir, "thresholds.csv"), row.names = FALSE)

# --- ContaminationQuantification ------------------------------------
ratio <- tryCatch(
  ContaminationQuantification(obj, gcgs, auc_thres = 0.9, min.cell = 50),
  error = function(e) NA
)
writeLines(sprintf("%.17g", ratio), file.path(out_dir, "ratio.txt"))

cat("R scCDC reference done\n")
