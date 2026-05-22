"""pysccdc -- pure-Python port of the R package **scCDC**.

scCDC (Wang *et al.*, *Genome Biology* 2024) is an entropy-based,
*gene-specific* ambient-RNA decontamination method for scRNA-seq and
snRNA-seq.  Instead of correcting every gene, scCDC detects the small
set of **Global Contamination-causing Genes (GCGs)** and corrects only
those -- avoiding the over-correction of lowly/non-contaminating genes
seen in DecontX / SoupX / CellBender.

The pipeline:

1. :func:`ContaminationDetection` -- per cluster, compute the Shannon
   entropy of every gene's count distribution, fit the expected
   entropy-vs-expression curve with an outlier-trimmed smoothing spline,
   score the *entropy divergence* (expected - observed) and flag genes
   significant in > ``restriction_factor`` of clusters as GCGs.
2. :func:`ContaminationQuantification` -- a dataset-level contamination
   ratio from the GCGs.
3. :func:`ContaminationCorrection` -- for each GCG, a Youden-index
   threshold on the per-cell count distributions decides how much to
   subtract (floored at zero); non-GCGs are untouched.

All functions are AnnData-friendly and depend only on
numpy / scipy / pandas / anndata / scikit-learn.
"""
from __future__ import annotations

from . import datasets
from .correction import (
    ContaminationCorrection,
    ContaminationQuantification,
    simple_roc,
    youden_threshold,
)
from .detection import ContaminationDetection, generate_curve
from .entropy import matrix_entropy, vector_entropy
from .spline import SmoothSpline, smooth_spline

__version__ = "0.1.0"

__all__ = [
    "ContaminationDetection",
    "ContaminationCorrection",
    "ContaminationQuantification",
    "generate_curve",
    "vector_entropy",
    "matrix_entropy",
    "SmoothSpline",
    "smooth_spline",
    "simple_roc",
    "youden_threshold",
    "datasets",
    "__version__",
]
