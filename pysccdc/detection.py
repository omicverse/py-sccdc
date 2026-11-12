"""GCG detection -- the :func:`ContaminationDetection` API.

Port of scCDC's ``Contamination_Detection.R``.

Pipeline (per cluster, clusters in factor-level order):

1. ``CalEnt``        -- entropy of every gene's count distribution.
2. ``CalAverageExpression`` -- ``log(mean_counts + 1)`` per gene.
3. ``generate_curve`` -- fit the expected entropy-vs-expression curve: ten times, take 80% of the
   genes with R's ``sample()``, fit ``smooth.spline(spar = 1)``, drop genes far below the curve
   twice and refit; the ten curves, evaluated on a 0.001 grid, are averaged. The *entropy
   divergence* ``distance = expected - observed`` gives a normal-tail p-value and an FDR per gene.
4. A gene is a candidate in a cluster when ``p.adj <= 0.05``.
5. A **Global Contamination-causing Gene (GCG)** is a gene flagged in
   ``>= round(restriction_factor * n_clusters)`` clusters and expressed in
   ``>= percent.cutoff`` of cells in *every* cluster.

R's bootstrap does not depend on the user's ``set.seed``: before fitting the curves scCDC subsets
the Seurat object (once to drop small clusters, then per cluster), and Seurat's ``subset`` calls
``WhichCells(seed = 1)``, which runs ``set.seed(1)``. The curves therefore always come from the
``set.seed(1)`` stream. ``random_state=1`` (the default) draws the same genes as R, and the spline,
means, standard deviations, ``pnorm`` and FDR follow R's arithmetic, so the curves, divergences and
GCGs agree with R.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import math

from ._anndata import counts_matrix, get_clusters
from ._rbase import c_apply, p_adjust_fdr, pnorm, r_mean, r_sd, round3_index, row_means
from .entropy import matrix_entropy
from .rrng import RRandom
from .spline import SmoothSpline

__all__ = ["generate_curve", "ContaminationDetection"]


def _upper_tail(d: np.ndarray) -> np.ndarray:
    """``1 - pnorm(d, mean(d), sd(d))``."""
    return 1.0 - pnorm(d, r_mean(d), r_sd(d))


def _rng(random_state) -> RRandom:
    if isinstance(random_state, RRandom):
        return random_state
    if random_state is None:
        random_state = int(np.random.SeedSequence().entropy % (2 ** 31))
    return RRandom(int(random_state))


def generate_curve(
    df: pd.DataFrame,
    filter: float = 0.01,
    select_factor: float = 0.8,
    n_boot: int = 10,
    spar: float = 1.0,
    random_state=None,
) -> pd.DataFrame:
    """Fit the entropy-vs-expression curve and score entropy divergence (scCDC ``generate_curve``).

    Parameters
    ----------
    df
        Columns ``Gene``, ``mean.expr``, ``entropy`` (one cluster).
    filter
        Outlier p-value cutoff used while trimming the bootstrap fits.
    select_factor
        Fraction of genes sampled per bootstrap round.
    n_boot
        Number of bootstrap rounds (scCDC uses 10).
    spar
        Smoothing-spline ``spar`` (scCDC uses 1).
    random_state
        An int (as given to R's ``set.seed``) or a :class:`~pysccdc.rrng.RRandom` stream, which
        is advanced -- pass the same stream for consecutive clusters, as R does.

    Returns
    -------
    pandas.DataFrame
        Indexed by gene, columns ``mean.expr`` (rounded to 3 decimals), ``entropy``,
        ``distance`` (expected - observed entropy), ``fit``, ``p.value``, ``p.adj`` and ``Gene``;
        sorted by decreasing ``distance``.
    """
    rng = _rng(random_state)
    x = df["mean.expr"].to_numpy(dtype=np.float64)
    y = df["entropy"].to_numpy(dtype=np.float64)
    genes = df["Gene"].to_numpy()
    n = x.size
    number = int(np.rint(n * select_factor))
    # the averaged curve is looked up at round(mean.expr, 3): grid point k / 1000
    k_gene = round3_index(x)
    k_need, k_inv = np.unique(k_gene, return_inverse=True)
    grid = k_need * 0.001
    curves = np.empty((k_need.size, n_boot))
    for b in range(n_boot):
        idx = rng.sample(n, number)
        bx, by = x[idx], y[idx]
        for _ in range(2):
            d = SmoothSpline(bx, by, spar=spar).predict(bx) - by
            ok = np.isfinite(d)
            bx, by, d = bx[ok], by[ok], d[ok]
            keep = _upper_tail(d) > filter
            bx, by = bx[keep], by[keep]
        curves[:, b] = SmoothSpline(bx, by, spar=spar).predict(grid)
    fit_at_gene = row_means(curves)[k_inv]
    distance = fit_at_gene - y
    pval = _upper_tail(distance)
    out = pd.DataFrame(
        {
            "mean.expr": k_gene / 1000.0,
            "entropy": y,
            "distance": distance,
            "fit": fit_at_gene,
            "p.value": pval,
            "p.adj": p_adjust_fdr(pval),
            "Gene": genes,
        },
        index=pd.Index(genes),
    )
    return out.iloc[np.argsort(-distance, kind="stable")]


def _row_mean_r(m: sp.csr_matrix) -> np.ndarray:
    """``apply(m, 1, mean)`` on integer counts: the long-double sum is exact, so R's mean is the
    sum divided by n in long double, rounded once (its correction pass adds ~1e-19)."""
    s = np.asarray(m.sum(axis=1), dtype=np.float64).ravel().astype(np.longdouble)
    return (s / m.shape[1]).astype(np.float64)


def ContaminationDetection(
    adata,
    cluster_key: Optional[str] = None,
    restriction_factor: float = 0.5,
    min_cell: int = 100,
    percent_cutoff: float = 0.2,
    layer: Optional[str] = None,
    spar: float = 1.0,
    random_state=1,
    return_full: bool = False,
):
    """Detect Global Contamination-causing Genes (GCGs).

    Faithful port of scCDC ``ContaminationDetection``.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with **raw integer counts** in ``X`` (or
        in ``layer``) and a cluster label in ``obs``.
    cluster_key
        Column of ``adata.obs`` holding the cluster labels.  If ``None``,
        the first categorical/object column is used.  Clusters are taken in
        the order of the categories (as R takes the factor levels); other
        labels are ordered numerically when they all look like integers,
        else alphabetically.
    restriction_factor
        Fraction of clusters in which a gene must be flagged to become a
        GCG (scCDC default 0.5).
    min_cell
        Clusters with fewer cells are dropped before analysis.
    percent_cutoff
        A GCG must be expressed in at least this fraction of cells in
        *every* qualifying cluster.
    layer
        ``adata.layers`` key holding counts; ``None`` -> ``adata.X``.
    spar
        Smoothing-spline ``spar`` passed to :func:`generate_curve`.
    random_state
        Seed of R's generator for the bootstrap. The default, 1, reproduces R
        scCDC, which always fits from ``set.seed(1)`` (Seurat's ``subset``
        resets the generator); other values give other bootstrap draws.
    return_full
        If ``True`` also return per-cluster diagnostics.

    Returns
    -------
    pandas.DataFrame
        The contamination-degree table for the detected GCGs: per-cluster
        entropy divergence plus a ``mean_distance`` column, sorted by
        descending ``mean_distance``.  ``.attrs['GCGs']`` holds the GCG
        list.  When ``return_full`` is set a ``dict`` is returned with the
        keys ``result``, ``GCGs``, ``per_cluster``, ``all_distance`` and
        ``qualified_clusters``.
    """
    counts, genes, _ = counts_matrix(adata, layer=layer)
    labels, levels = get_clusters(adata, cluster_key, return_levels=True)
    sizes = pd.Series(labels).value_counts().reindex(levels, fill_value=0)
    qualified = [c for c in levels if sizes[c] >= min_cell]
    if len(qualified) == 0:
        raise ValueError(
            f"No cluster reaches min_cell={min_cell} "
            f"(largest cluster has {int(sizes.max())} cells)."
        )
    rng = _rng(random_state)
    counts = sp.csc_matrix(counts)
    per_cluster = {}
    pct = {}
    for cl in qualified:
        sub = sp.csr_matrix(counts[:, np.flatnonzero(labels == cl)])
        sub.sum_duplicates()
        ent = matrix_entropy(sub)
        mean_expr = c_apply(math.log, _row_mean_r(sub) + 1.0)
        cdf = pd.DataFrame({"Gene": genes, "mean.expr": mean_expr, "entropy": ent})
        try:
            per_cluster[cl] = generate_curve(cdf, spar=spar, random_state=rng)
        except ValueError as e:
            if "IQR" not in str(e):
                raise
            zero = float(np.mean(_row_mean_r(sub) == 0))
            raise ValueError(
                f"cluster {cl!r}: {zero:.0%} of the genes have no counts in its {sub.shape[1]} cells, so the "
                "interquartile range of mean expression is 0 and smooth.spline cannot be fitted (R scCDC "
                "stops with the same error). Remove genes detected in few cells or raise min_cell."
            ) from e
        nz = np.diff((sub != 0).tocsr().indptr)
        pct[cl] = 1.0 - ((sub.shape[1] - nz) / sub.shape[1])
    # candidate genes: p.adj <= 0.05 in a cluster, flagged in enough clusters
    flagged = np.concatenate([per_cluster[cl].index[per_cluster[cl]["p.adj"] <= 0.05].to_numpy()
                              for cl in qualified])
    counter = pd.Series(flagged).value_counts()
    threshold = int(np.rint(len(qualified) * restriction_factor))
    candidates = set(counter.index[counter >= threshold])
    if len(candidates) == 0:
        raise ValueError("No contaminated genes found.")
    # expressed in >= percent_cutoff of the cells of every cluster
    in_all = np.all(np.column_stack([pct[cl] >= percent_cutoff for cl in qualified]), axis=1)
    gcgs = {g for g, ok in zip(genes, in_all) if ok and g in candidates}
    if len(gcgs) == 0:
        raise ValueError("No contaminated genes found after percent filter.")
    # contamination-degree table: per-cluster distance for every gene
    dist = np.column_stack([per_cluster[cl]["distance"].reindex(genes).to_numpy() for cl in qualified])
    dist_df = pd.DataFrame(dist, index=pd.Index(genes), columns=qualified)
    dist_df["mean_distance"] = row_means(dist)
    dist_df = dist_df.iloc[np.argsort(-dist_df["mean_distance"].to_numpy(), kind="stable")]
    result = dist_df.loc[[g in gcgs for g in dist_df.index]].copy()
    result.attrs["GCGs"] = list(result.index)
    if return_full:
        return {
            "result": result,
            "GCGs": list(result.index),
            "per_cluster": per_cluster,
            "all_distance": dist_df,
            "qualified_clusters": list(qualified),
        }
    return result
