"""AnnData <-> dense-matrix glue used across :mod:`pysccdc`.

scCDC reasons in the Seurat layout (genes x cells); AnnData stores
cells x genes.  These helpers convert and pick out the count layer /
cluster labels so the rest of the package can work with plain NumPy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["counts_matrix", "get_clusters", "to_dense"]


def to_dense(x) -> np.ndarray:
    """Return a dense 2-D ``float64`` array from dense or sparse input."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def counts_matrix(
    adata, layer: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the (genes x cells) count matrix from an AnnData.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with raw counts.
    layer
        ``adata.layers`` key; ``None`` uses ``adata.X``.

    Returns
    -------
    counts : ndarray
        Dense ``genes x cells`` matrix (transposed from AnnData).
    genes : ndarray
        ``var_names``.
    cells : ndarray
        ``obs_names``.
    """
    mat = adata.layers[layer] if layer is not None else adata.X
    counts = to_dense(mat).T  # genes x cells
    genes = np.asarray(adata.var_names)
    cells = np.asarray(adata.obs_names)
    return counts, genes, cells


def get_clusters(adata, cluster_key: Optional[str] = None) -> np.ndarray:
    """Return the per-cell cluster labels as a string array.

    When ``cluster_key`` is ``None`` the first categorical/object column
    of ``adata.obs`` is used.
    """
    if cluster_key is None:
        for col in adata.obs.columns:
            ser = adata.obs[col]
            if isinstance(ser.dtype, pd.CategoricalDtype) or \
                    ser.dtype == object:
                cluster_key = col
                break
        if cluster_key is None:
            raise ValueError(
                "No categorical column found in adata.obs; "
                "pass cluster_key explicitly."
            )
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not in adata.obs.")
    return adata.obs[cluster_key].astype(str).to_numpy()
