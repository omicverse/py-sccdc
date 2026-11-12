# Benchmark: pysccdc vs R scCDC 1.4

Five 10x Genomics human–mouse mixtures (hgmm 1k/5k/10k v3, 6k/12k v2). The tasks are those of the
py-soupx benchmark (`py-soupx/report/scripts/01_prepare.py`, copied here): Cell Ranger's cells, one
scanpy/Leiden clustering given to both packages, and species labels. Counts of the other species in a
cell can only be ambient RNA.

```
python  scripts/01_prepare.py <dir with extracted 10x archives> <tasks_dir>
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 10 1,2    # idle cores, 1 thread each
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures
```

Both build the gene set as `CreateSeuratObject(counts, min.cells = 3)` does (genes with counts in at
least 3 cells) and run `ContaminationDetection` and `ContaminationCorrection` with their defaults.
scCDC always fits its curves from `set.seed(1)` (Seurat's `subset` calls `WhichCells(seed = 1)`), and
pysccdc's default `random_state=1` matches it. In hgmm 10k one cluster (104 cells) has 81% of the genes
without counts; `smooth.spline` then stops in R and in pysccdc, and the mixture is left out.
R: scCDC 1.4 (GitHub) with Seurat 5.5.1 on R 4.5.3. Times are the faster of two runs, each on an idle
core with one BLAS thread. `figures/comparison_per_dataset.tsv` holds the per-dataset numbers.

`02_run_sccdc.R <task> <out> <seed> 1` also writes R's per-cluster curves and per-GCG thresholds; with
`OPENBLAS_CORETYPE=Nehalem` (no FMA kernels) they equal pysccdc's to the last bit.
