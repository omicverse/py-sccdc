"""Shannon entropy of each gene's count distribution (scCDC ``MatrixToEntropy``).

For one gene in one cluster the Rcpp code tabulates how many cells hold each distinct count,
turns the tallies into proportions ``p`` and returns ``-sum p log2 p``, adding the terms in
increasing order of the count. A gene that is uniformly low in every cell has near-zero entropy;
an ambient gene smeared over many cells at varying low counts has a higher entropy than its mean
expression predicts.

The port works on the sparse matrix: the zeros of a gene form one tally, the stored values are
tallied with one sort, and the terms are added per gene in the same order as the C++ loop.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from ._rbase import c_apply

__all__ = ["vector_entropy", "matrix_entropy"]


def matrix_entropy(mat) -> np.ndarray:
    """Per-row entropy of a genes x cells count matrix (dense or sparse).

    Matches scCDC's ``MatrixToEntropy`` (counts are taken as integers, as Rcpp's
    ``IntegerMatrix`` does)."""
    m = sp.csr_matrix(mat)
    m.sum_duplicates()
    n_genes, n_cells = m.shape
    if n_cells == 0:
        return np.zeros(n_genes)
    vals = np.trunc(m.data).astype(np.int64)
    rows = np.repeat(np.arange(n_genes), np.diff(m.indptr))
    nz = vals != 0
    rows, vals = rows[nz], vals[nz]
    # tallies of each (gene, count) pair, ordered by gene then count
    key = rows * (vals.max() + 1 if vals.size else 1) + vals
    ukey, tally = np.unique(key, return_counts=True)
    g_of = ukey // (vals.max() + 1 if vals.size else 1)
    n_zero = n_cells - np.bincount(rows, minlength=n_genes)
    # zero is the smallest count, so its term comes first; genes without zeros skip it
    g_all = np.concatenate([np.flatnonzero(n_zero > 0), g_of])
    c_all = np.concatenate([n_zero[n_zero > 0], tally]).astype(np.float64)
    pos = np.concatenate([np.zeros(int((n_zero > 0).sum()), dtype=np.int64),
                          np.ones(g_of.size, dtype=np.int64)])
    order = np.lexsort((pos, g_all))                 # stable: zeros first, then counts ascending
    g_all, c_all = g_all[order], c_all[order]
    p = c_all / n_cells
    term = p * c_apply(math.log2, p)
    # add the terms of each gene in order: one pass per term rank (the j-th term of every gene)
    start = np.searchsorted(g_all, np.arange(n_genes))
    rank = np.arange(g_all.size) - start[g_all]
    by_rank = np.argsort(rank, kind="stable")
    bounds = np.searchsorted(rank[by_rank], np.arange(int(rank.max()) + 2 if rank.size else 1))
    ent = np.zeros(n_genes)
    for j in range(bounds.size - 1):
        sel = by_rank[bounds[j]:bounds[j + 1]]
        ent[g_all[sel]] = ent[g_all[sel]] - term[sel]
    return ent


def vector_entropy(x) -> float:
    """Entropy of one gene's counts (scCDC ``VectorToEntropy``)."""
    return float(matrix_entropy(np.asarray(x).reshape(1, -1))[0])
