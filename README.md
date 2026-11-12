# pysccdc

A **pure-Python re-implementation of [scCDC](https://github.com/ZJU-UoE-CCW-LAB/scCDC)** (Wang et al., *Genome Biology* 2024) for entropy-based, **gene-specific** ambient-RNA contamination detection and correction in single-cell / single-nucleus RNA-seq data.

- AnnData-native — drop-in for the scanpy / omicverse ecosystem
- **No `rpy2`**, no R install — the Shannon-entropy core, R's `smooth.spline`, R's random number generator and `sample()`, `pnorm`, the FDR, pROC's AUROC and the Youden-index thresholding are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`ContaminationDetection` → `ContaminationQuantification` → `ContaminationCorrection`)
- Same results as R scCDC 1.4, including the bootstrapped curve fit: same GCGs, same entropy divergences, same corrected counts (see below), 11–22× faster

Unlike DecontX, SoupX, CellBender or scAR — which correct *every* gene — scCDC detects the small set of **Global Contamination-causing Genes (GCGs)** and corrects only those, avoiding the over-correction of lowly / non-contaminating genes (many of which are real cell-type markers). It needs **no empty-droplet data**.

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here for users who want scCDC without the full omicverse stack.

## Install

```bash
pip install pysccdc
```

or, from a checkout:

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`. No R, no `rpy2`.

## How it works

1. **Observed entropy** — for every gene in every cell cluster, compute the Shannon entropy (base 2) of its count distribution across droplets. A gene smeared across many droplets at a near-constant low ambient level has a *concentrated* count distribution → low entropy.
2. **Expected entropy curve** — fit the expected entropy as a smooth function of `log1p(mean expression)` with a bootstrapped, outlier-trimmed smoothing spline learnt from presumed-clean genes.
3. **Entropy divergence** = expected − observed entropy. A gene with significant positive divergence (normal-tail p, FDR ≤ 0.05) in more than `restriction_factor` of clusters — and expressed in enough cells in every cluster — is flagged a **GCG**.
4. **Gene-specific correction** — for each GCG, rank clusters by log-normalized expression, compute per-cluster **AUROC** vs the lowest-expressing cluster, split into eGCG-positive / -negative, then take the **Youden-index** count threshold on the pooled count distributions and subtract `round(threshold)` (floored at zero). **Non-GCG genes are left untouched** — scCDC's anti-over-correction design.

## Quick-start

```python
import pysccdc as cd

# bundled synthetic dataset: 4 clusters x 200 cells, 120 genes,
# 4 deliberately-spiked contaminating genes
adata = cd.datasets.simulate_contaminated(random_state=0)

# 1) detect GCGs
detection = cd.ContaminationDetection(adata, cluster_key="cluster")
detection                       # degree-of-contamination table (GCGs)
detection.attrs["GCGs"]         # the GCG list

# 2) quantify dataset-level contamination
ratio = cd.ContaminationQuantification(adata, detection,
                                       cluster_key="cluster")

# 3) correct only the GCGs
corrected = cd.ContaminationCorrection(adata, detection,
                                       cluster_key="cluster")
corrected.layers["Corrected"]          # decontaminated count matrix
corrected.uns["sccdc"]["thresholds"]   # per-GCG subtraction thresholds
```

scCDC works on a **filtered, clustered** count matrix; any AnnData with raw integer counts in `.X` (or a named `layer`) and a categorical cluster label in `.obs` works. See `examples/tutorial_standalone.py` for an end-to-end run on the bundled clustered PBMC 3k dataset (`data/pbmc3k_clustered.h5ad`).

## Low-level functional API (mirrors R one-to-one)

```python
from pysccdc import (
    ContaminationDetection, ContaminationQuantification, ContaminationCorrection,
    generate_curve, vector_entropy, matrix_entropy,
    SmoothSpline, smooth_spline, simple_roc, youden_threshold,
)

# Shannon entropy of a single gene's count distribution
matrix_entropy(counts_genes_by_cells)        # one entropy per gene

# Fit one cluster's entropy-vs-expression curve directly
generate_curve(df_with_Gene_meanexpr_entropy, spar=1.0)

# AUROC and the Youden-index cut point
simple_roc(expr, cls)
youden_threshold(neg_counts, pos_counts)
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `ContaminationDetection` | `ContaminationDetection` | detect GCGs; per-cluster entropy divergence table |
| `ContaminationQuantification` | `ContaminationQuantification` | dataset-level contamination ratio from the GCGs |
| `ContaminationCorrection` | `ContaminationCorrection` | Youden-threshold correction of the GCGs only |
| `generate_curve` | `generate_curve` | fit one cluster's entropy-vs-expression curve |
| `vector_entropy` / `matrix_entropy` | `VectorToEntropy` / `MatrixToEntropy` | Shannon entropy of count distributions |
| `SmoothSpline` / `smooth_spline` | `smooth.spline` | R's smoothing spline and `predict` |
| `simple_roc` / `youden_threshold` | `simple_roc` / `Cal_thres` | AUROC and Youden-index cut point |
| `datasets.simulate_contaminated` | — | synthetic clustered counts with spiked GCGs |

## Reproducing R results exactly

scCDC fits each cluster's entropy-vs-expression curve ten times on 80% of the genes, drawn with R's
`sample()`, using `smooth.spline(spar = 1)`. pysccdc ports every step of that path:

* R's Mersenne-Twister seeding and `sample()` (rejection sampling), so the same genes are drawn;
* `smooth.spline` operation for operation — tie pooling, knot choice (`.nknots.smspl`, `seq.int`),
  the `X'WX` and penalty bands of `sbart.c`, the `spar → lambda` ratio, the LINPACK banded Cholesky
  solve and `bvalue`, with R's linear extrapolation;
* `mean`, `sd` and `rowMeans` in long double, `round(x, 3)`, `pnorm` (Cody's algorithm), `p.adjust`,
  and C-library `log`/`log2`, as R computes them;
* pROC's ROC table, trapezoid AUROC and the first Youden maximum, and Seurat 5's `AverageExpression`
  (mean of `expm1` of the log-normalized data) for ordering the clusters.

**Seed.** R scCDC does not depend on the user's `set.seed`: before the curves are fitted it subsets
the Seurat object, and Seurat's `subset` calls `WhichCells(seed = 1)`, which runs `set.seed(1)`.
`ContaminationDetection(random_state=1)` (the default) therefore gives R's result; other values give
other bootstrap draws.

`tests/test_r_reference.py` compares with stored scCDC 1.4 output on the bundled synthetic dataset
(entropy, divergences, GCGs, corrected counts and contamination ratio, all identical);
`tests/test_r_parity.py` runs R live (`PYSCCDC_RSCRIPT=/path/to/Rscript`). The spline solve uses
BLAS `ddot`; with an OpenBLAS whose kernels use FMA instructions R's divergences differ from
pysccdc's in the last ~12 digits, which leaves GCGs and corrected counts unchanged.

On four 10x human–mouse mixtures (1,046–12,820 cells), pysccdc found the same GCGs as scCDC 1.4, in
the same order, and returned identical corrected counts; it ran 11–22× faster on one thread.
Scripts and figure: [`report/`](report/). scCDC needs fewer than 75% of the genes with zero counts
in every analysed cluster (otherwise `smooth.spline` stops, in R and in pysccdc); filter genes
detected in few cells first.

## Examples

`examples/` mirrors the reference layout:

* `r_driver_sccdc.R` — drives R scCDC end-to-end, dumps entropy / GCG / distance / corrected-matrix outputs
* `compare_R_vs_Python.ipynb` (+ `.executed.ipynb`) — runs R scCDC via `Rscript` and `pysccdc` on the bundled clustered PBMC 3k dataset and visualizes the agreement (entropy bit-exact, divergence correlation, GCG-set Venn, bit-exact corrected matrix) via `omicverse.pl.*`
* `tutorial_standalone.py` — minimal end-to-end pysccdc pipeline
* `benchmark.py` — head-to-head speed comparison

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: omicverse single-cell decontamination
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original scCDC paper:

> Wang, W., Cen, Y., Lu, Z. *et al.* **scCDC: a computational method for gene-specific contamination detection and correction in single-cell and single-nucleus RNA-seq data.** *Genome Biology* **25**, 136 (2024). https://doi.org/10.1186/s13059-024-03284-w

and acknowledge omicverse / this repo for the Python port.

## License

Apache-2.0. The upstream R package scCDC is GPL (≥ 2); `pysccdc` is an independent re-implementation from the published algorithm and the scCDC source.
