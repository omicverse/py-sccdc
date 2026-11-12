"""AnnData glue used across :mod:`pysccdc`.

scCDC reasons in the Seurat layout (genes x cells); AnnData stores cells x genes. These helpers
return the count layer as a sparse genes x cells matrix and the cluster labels with their level
order (R iterates over ``levels(Idents(object))``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["counts_matrix", "get_clusters", "to_dense"]


def to_dense(x) -> np.ndarray:
    """Return a dense 2-D ``float64`` array from dense or sparse input."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def counts_matrix(adata, layer: Optional[str] = None) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """The (genes x cells) count matrix of an AnnData, as a CSR matrix, with gene and cell names."""
    mat = adata.layers[layer] if layer is not None else adata.X
    counts = sp.csr_matrix(sp.csc_matrix(mat, dtype=np.float64).T)
    return counts, np.asarray(adata.var_names), np.asarray(adata.obs_names)


def _levels(ser: pd.Series) -> list:
    if isinstance(ser.dtype, pd.CategoricalDtype):
        return [str(c) for c in ser.cat.categories]
    u = pd.unique(ser.astype(str))
    try:
        return [str(v) for v in sorted(u, key=int)]
    except ValueError:
        return sorted(u)


def get_clusters(adata, cluster_key: Optional[str] = None, return_levels: bool = False):
    """Per-cell cluster labels as strings (and, with ``return_levels``, the level order).

    When ``cluster_key`` is ``None`` the first categorical/object column of ``adata.obs`` is used.
    Categorical columns keep their category order; other labels are ordered numerically when they
    all look like integers, else alphabetically.
    """
    if cluster_key is None:
        for col in adata.obs.columns:
            ser = adata.obs[col]
            if isinstance(ser.dtype, pd.CategoricalDtype) or ser.dtype == object:
                cluster_key = col
                break
        if cluster_key is None:
            raise ValueError("No categorical column found in adata.obs; pass cluster_key explicitly.")
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not in adata.obs.")
    ser = adata.obs[cluster_key]
    labels = ser.astype(str).to_numpy()
    if return_levels:
        return labels, _levels(ser)
    return labels
