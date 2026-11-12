# scCDC run on one task: ContaminationDetection + ContaminationCorrection with defaults.
# Usage: Rscript 02_run_sccdc.R <task_dir> <out_dir> [seed] [diagnostics 0/1]
suppressPackageStartupMessages({library(Matrix); library(Seurat); library(scCDC); library(jsonlite)})
a <- commandArgs(TRUE)
td <- a[1]; od <- a[2]; seed <- if (length(a) > 2) as.integer(a[3]) else 1L
diag <- length(a) > 3 && a[4] == "1"
dir.create(od, recursive = TRUE, showWarnings = FALSE)
tod <- as(readMM(gzfile(file.path(td, "tod.mtx.gz"))), "CsparseMatrix")
genes <- read.delim(file.path(td, "genes.tsv"))
droplets <- readLines(file.path(td, "droplets.tsv"))
cl <- read.delim(file.path(td, "clusters.tsv"), colClasses = "character")
toc <- tod[, match(cl$cell, droplets)]
rownames(toc) <- paste0("g", seq_len(nrow(toc)))      # Seurat-safe names; row i of the task matrix
colnames(toc) <- cl$cell
obj <- suppressWarnings(CreateSeuratObject(toc, min.cells = 3))   # genes seen in >= 3 cells
Idents(obj) <- factor(cl$cluster, levels = as.character(sort(unique(as.integer(cl$cluster)))))
set.seed(seed)
t0 <- proc.time()[["elapsed"]]
gcg <- suppressMessages(ContaminationDetection(obj))
t1 <- proc.time()[["elapsed"]]
cor <- suppressMessages(suppressWarnings(ContaminationCorrection(obj, rownames(gcg))))
t2 <- proc.time()[["elapsed"]]
idx <- as.integer(sub("^g", "", rownames(gcg)))
write.table(data.frame(row = idx, gcg, check.names = FALSE), file.path(od, "gcg.tsv"), sep = "\t", quote = FALSE,
            row.names = FALSE)
m <- GetAssayData(cor, assay = "Corrected", layer = "counts")
orig <- GetAssayData(obj, layer = "counts")
sub_ <- as(m[rownames(gcg), , drop = FALSE], "CsparseMatrix")
writeBin(as.integer(sub_@p), file.path(od, "corr_p.bin"), size = 4)
writeBin(as.integer(sub_@i), file.path(od, "corr_i.bin"), size = 4)
writeBin(as.double(sub_@x), file.path(od, "corr_x.bin"))
others <- setdiff(rownames(m), rownames(gcg))
same_others <- identical(m[others, ], orig[others, ])
write_json(list(engine = "scCDC", version = as.character(packageVersion("scCDC")), R = R.version.string,
                seed = seed, n_gcg = nrow(gcg), dim = dim(toc), others_unchanged = same_others,
                seconds_detection = t1 - t0, seconds_correction = t2 - t1, seconds = t2 - t0),
           file.path(od, "result.json"), auto_unbox = TRUE, digits = NA)
if (diag) {                                       # per-cluster curves and per-GCG thresholds, same RNG stream
  set.seed(seed)
  keep <- names(table(obj@active.ident))[as.numeric(table(obj@active.ident)) >= 100]
  so <- subset(obj, idents = keep)
  ent <- scCDC:::CalEnt.Seurat(so, rownames(so))
  ave <- log(suppressMessages(scCDC:::CalAverageExpression.Seurat(so, rownames(so))) + 1)
  writeLines(colnames(ave), file.path(od, "diag_clusters.txt"))
  for (k in seq_along(colnames(ave))) {
    cc <- colnames(ave)[k]
    g <- tibble::tibble(Gene = rownames(ent), mean.expr = ave[, cc], entropy = ent[, cc])
    r <- scCDC:::generate_curve(g)
    r$row <- as.integer(sub("^g", "", r$Gene))
    writeLines(as.character(r$row), file.path(od, sprintf("diag_curve_%s_rows.txt", cc)))
    writeBin(c(r$mean.expr, r$entropy, r$fit, r$distance, r$p.value, r$p.adj),
             file.path(od, sprintf("diag_curve_%s.bin", cc)))
  }
  on <- NormalizeData(obj, verbose = FALSE)
  nc <- table(Idents(on)); q <- names(nc[nc >= 50])
  th <- sapply(rownames(gcg), function(x) {
    au <- suppressMessages(scCDC:::Cal_AUCs(on, x, q))
    suppressWarnings(suppressMessages(scCDC:::Cal_thres(on, x, au[[1]], 0.9)))
  })
  write.table(data.frame(row = idx, threshold = th), file.path(od, "diag_thresholds.tsv"), sep = "\t",
              quote = FALSE, row.names = FALSE)
}
cat(basename(td), "GCGs", nrow(gcg), sprintf("detection %.1fs correction %.1fs\n", t1 - t0, t2 - t1))
