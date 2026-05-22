# py-sccdc

Pure-Python port of the R package
**[scCDC](https://github.com/ZJU-UoE-CCW-LAB/scCDC)** — Wang *et al.*,
"scCDC: a computational method for gene-specific contamination detection
and correction in single-cell and single-nucleus RNA-seq data",
*Genome Biology* **25**, 122 (2024).

`pysccdc` is an entropy-based, **gene-specific** ambient-RNA
decontamination method for scRNA-seq / snRNA-seq. Unlike DecontX, SoupX,
CellBender or scAR — which correct *every* gene — scCDC detects the small
set of **Global Contamination-causing Genes (GCGs)** and corrects only
those, avoiding the over-correction of lowly / non-contaminating genes
(many of which are real cell-type markers). It needs **no empty-droplet
data**.

| | |
|---|---|
| PyPI / import name | `pysccdc` |
| Repository | `omicverse/py-sccdc` |
| License | Apache-2.0 |
| Upstream | scCDC 1.4 (GPL ≥ 2, R) |
| Numerical parity | entropy & corrected matrix **bit-exact** vs scCDC |

## Install

```bash
pip install pysccdc              # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`, `scikit-learn`.
No R, no `rpy2`.

## How it works

1. **Observed entropy** — for every gene in every cell cluster, compute
   the Shannon entropy (base 2) of its count distribution across
   droplets. A gene smeared across many droplets at a near-constant low
   ambient level has a *concentrated* count distribution → low entropy.
2. **Expected entropy curve** — fit the expected entropy as a smooth
   function of `log1p(mean expression)` with a bootstrapped,
   outlier-trimmed smoothing spline learnt from presumed-clean genes.
3. **Entropy divergence** = expected − observed entropy. A gene with
   significant positive divergence (normal-tail p, FDR ≤ 0.05) in more
   than `restriction_factor` of clusters — and expressed in enough cells
   in every cluster — is flagged a **GCG**.
4. **Gene-specific correction** — for each GCG, rank clusters by
   log-normalized expression, compute per-cluster **AUROC** vs the
   lowest-expressing cluster, split into eGCG-positive / -negative, then
   take the **Youden-index** count threshold on the pooled count
   distributions and subtract `round(threshold)` (floored at zero).
   **Non-GCG genes are left untouched** — scCDC's anti-over-correction
   design.

## Quick start

```python
import pysccdc as cd

# bundled synthetic dataset: 4 clusters x 200 cells, 120 genes,
# 4 deliberately-spiked contaminating genes
adata = cd.datasets.simulate_contaminated(random_state=0)

# 1. detect GCGs
detection = cd.ContaminationDetection(adata, cluster_key="cluster")
detection                       # degree-of-contamination table (GCGs)
detection.attrs["GCGs"]         # the GCG list

# 2. quantify dataset-level contamination
ratio = cd.ContaminationQuantification(adata, detection,
                                       cluster_key="cluster")

# 3. correct only the GCGs
corrected = cd.ContaminationCorrection(adata, detection,
                                       cluster_key="cluster")
corrected.layers["Corrected"]   # decontaminated count matrix
corrected.uns["sccdc"]["thresholds"]   # per-GCG subtraction thresholds
```

## API

| Function | scCDC equivalent | Purpose |
|---|---|---|
| `ContaminationDetection` | `ContaminationDetection` | detect GCGs; per-cluster entropy divergence table |
| `ContaminationQuantification` | `ContaminationQuantification` | dataset-level contamination ratio from GCGs |
| `ContaminationCorrection` | `ContaminationCorrection` | Youden-threshold correction of the GCGs only |
| `generate_curve` | `generate_curve` | fit one cluster's entropy-vs-expression curve |
| `vector_entropy` / `matrix_entropy` | `VectorToEntropy` / `MatrixToEntropy` | Shannon entropy of count distributions |
| `SmoothSpline` / `smooth_spline` | `smooth.spline` | penalized cubic B-spline |
| `simple_roc` / `youden_threshold` | `simple_roc` / `Cal_thres` | AUROC and Youden-index cut point |
| `datasets.simulate_contaminated` | — | synthetic clustered counts with spiked GCGs |

All functions take an `anndata.AnnData` with **raw integer counts** in
`X` (or in a named `layer`) and a categorical cluster label in `obs`.

## R parity

`tests/` runs the **same** synthetic dataset through the R package
scCDC 1.4 (`tests/r_reference_driver.R`) and `pysccdc`, and asserts
agreement:

* **per-gene / per-cluster Shannon entropy — bit-exact** (the Rcpp
  `MatrixToEntropy` reduces to a deterministic `numpy.bincount`);
* **detected GCG list — identical**;
* **corrected count matrix — bit-exact** (the Youden-threshold path is
  fully deterministic);
* **contamination ratio — bit-exact**;
* **per-gene entropy divergence — Pearson r > 0.99**.

**Unavoidable difference.** The entropy-vs-expression curve is fit by a
*bootstrapped* smoothing spline (10 rounds, 80% gene resampling). Two
things differ from R: (i) R's `sample()` (Mersenne-Twister) and NumPy's
PCG64 draw different bootstrap subsets, and (ii) R's `smooth.spline` uses
an internal knot-thinning heuristic and GCV machinery that the scipy
penalized cubic B-spline reproduces only up to ~1e-3 in entropy units.
These propagate into the entropy *divergence* (hence r > 0.99 rather than
bit-exact), but **not** into the GCG list or the corrected matrix, which
match exactly. Fix `random_state` for reproducible Python runs.

## References

* Wang, W. *et al.* scCDC: a computational method for gene-specific
  contamination detection and correction in single-cell and
  single-nucleus RNA-seq data. *Genome Biology* **25**, 122 (2024).
* scCDC: <https://github.com/ZJU-UoE-CCW-LAB/scCDC>

## License

Apache-2.0. The upstream R package scCDC is GPL (≥ 2); `pysccdc` is an
independent re-implementation from the published algorithm and the scCDC
source.
