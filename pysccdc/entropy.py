"""Shannon-entropy computation on droplet count distributions.

scCDC's contamination signal is the *entropy of the count distribution*
of a gene within a cell cluster.  The R/Rcpp implementation
(``MatrixToEntropy`` / ``VectorToEntropy``) computes, for each gene row::

    counts  = integer counts across the droplets of one cluster
    values  = unique counts
    out[k]  = number of droplets whose count equals values[k]
    p[k]    = out[k] / n_droplets
    H       = -sum_k  p[k] * log2(p[k])

i.e. the Shannon entropy (base 2) of the *empirical distribution of the
count values themselves* -- not of a probability vector over genes.  A
gene that is uniformly low in every droplet has near-zero entropy; an
ambient/contaminating gene smeared across many droplets at varying low
counts has elevated entropy for its mean expression.

This module reproduces the Rcpp result exactly using NumPy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["vector_entropy", "matrix_entropy"]


def vector_entropy(x) -> float:
    """Shannon entropy (base 2) of the count distribution of one gene.

    Parameters
    ----------
    x
        1-D array-like of (integer) counts across droplets.

    Returns
    -------
    float
        ``-sum p log2 p`` over the empirical distribution of the count
        values.  Matches scCDC's ``VectorToEntropy``.
    """
    x = np.asarray(x).ravel()
    n = x.size
    if n == 0:
        return 0.0
    # bincount of value frequencies; round to int as R coerces to integer
    counts = np.round(x).astype(np.int64)
    counts = np.clip(counts, 0, None)
    freq = np.bincount(counts)
    freq = freq[freq > 0].astype(np.float64)
    p = freq / freq.sum()
    return float(-np.sum(p * np.log2(p)))


def matrix_entropy(mat) -> np.ndarray:
    """Per-row entropy of a (genes x droplets) count matrix.

    Parameters
    ----------
    mat
        2-D array-like, rows = genes, columns = droplets.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``n_genes`` with the entropy of each gene.
        Matches scCDC's ``MatrixToEntropy``.
    """
    mat = np.asarray(mat)
    if mat.ndim == 1:
        return np.array([vector_entropy(mat)])
    return np.array([vector_entropy(mat[i]) for i in range(mat.shape[0])])
