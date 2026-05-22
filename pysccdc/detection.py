"""GCG detection -- the :func:`ContaminationDetection` API.

Port of scCDC's ``Contamination_Detection.R``.

Pipeline (per cluster):

1. ``CalEnt``        -- entropy of every gene's count distribution.
2. ``CalAverageExpression`` -- ``log(mean_counts + 1)`` per gene.
3. ``generate_curve`` -- fit the expected entropy-vs-expression curve via a
   bootstrapped, outlier-trimmed smoothing spline; the *entropy divergence*
   ``distance = expected - observed`` and a normal-tail p-value / FDR are
   returned per gene.
4. A gene is a candidate in a cluster when ``p.adj <= 0.05``.
5. A **Global Contamination-causing Gene (GCG)** is a gene flagged in
   ``>= restriction_factor`` of the qualifying clusters and expressed in
   ``>= percent.cutoff`` of cells in *every* qualifying cluster.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._anndata import counts_matrix, get_clusters
from .entropy import matrix_entropy
from .spline import SmoothSpline

__all__ = ["generate_curve", "ContaminationDetection"]


# ----------------------------------------------------------------------
def _fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR (R ``p.adjust(method='fdr')``)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def generate_curve(
    df: pd.DataFrame,
    filter: float = 0.01,
    select_factor: float = 0.8,
    n_boot: int = 10,
    spar: float = 1.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Fit the entropy-vs-expression curve and score entropy divergence.

    Parameters
    ----------
    df
        Columns ``Gene``, ``mean.expr``, ``entropy`` (one cluster).
    filter
        Outlier p-value cutoff used while trimming bootstrap fits.
    select_factor
        Fraction of genes sampled per bootstrap round.
    n_boot
        Number of bootstrap rounds (R uses 10).
    spar
        Smoothing-spline ``spar`` (scCDC uses 1).
    random_state
        Seed for the bootstrap sampling.

    Returns
    -------
    pandas.DataFrame
        Indexed by gene, columns ``mean.expr``, ``entropy``, ``distance``
        (expected - observed entropy = entropy divergence), ``fit``,
        ``p.value``, ``p.adj``; sorted by descending ``distance``.
    """
    rng = np.random.default_rng(random_state)
    x = df["mean.expr"].to_numpy(dtype=float)
    y = df["entropy"].to_numpy(dtype=float)
    genes = df["Gene"].to_numpy()
    n = len(df)
    number = int(round(n * select_factor))
    number = max(number, 4)

    max_number = float(np.max(x))
    discrete_x = np.arange(0.0, max_number + 0.1, 0.001)

    boot_curves = np.full((len(discrete_x), n_boot), np.nan)
    for b in range(n_boot):
        idx = rng.choice(n, size=number, replace=False)
        bx, by = x[idx], y[idx]
        # two outlier-trimming passes, then the final fit
        for _ in range(2):
            if len(bx) < 4:
                break
            fit = SmoothSpline(bx, by, spar=spar)
            pred = fit.predict(bx)
            dist = pred - by
            ok = np.isfinite(dist)
            bx, by, dist = bx[ok], by[ok], dist[ok]
            if dist.size < 2:
                break
            sd = dist.std(ddof=1)
            mu = dist.mean()
            if not np.isfinite(sd) or sd == 0:
                adj_p = np.ones_like(dist)
            else:
                adj_p = 1.0 - norm.cdf(dist, loc=mu, scale=sd)
            keep = adj_p > filter
            bx, by = bx[keep], by[keep]
        if len(bx) < 4:
            continue
        fit = SmoothSpline(bx, by, spar=spar)
        boot_curves[:, b] = fit.predict(discrete_x)

    curve = np.nanmean(boot_curves, axis=1)

    # look the fitted curve up at each gene's (rounded) mean expression
    mean_expr_r = np.round(x, 3)
    grid_idx = np.clip(
        np.round(mean_expr_r / 0.001).astype(int), 0, len(discrete_x) - 1
    )
    fit_at_gene = curve[grid_idx]

    distance = fit_at_gene - y
    finite = np.isfinite(distance)
    mu = distance[finite].mean()
    sd = distance[finite].std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        pval = np.ones_like(distance)
    else:
        pval = 1.0 - norm.cdf(distance, loc=mu, scale=sd)
    padj = _fdr(np.where(np.isfinite(pval), pval, 1.0))

    out = pd.DataFrame(
        {
            "mean.expr": mean_expr_r,
            "entropy": y,
            "distance": distance,
            "fit": fit_at_gene,
            "p.value": pval,
            "p.adj": padj,
            "Gene": genes,
        },
        index=genes,
    )
    out = out.sort_values("distance", ascending=False)
    return out


# ----------------------------------------------------------------------
def ContaminationDetection(
    adata,
    cluster_key: Optional[str] = None,
    restriction_factor: float = 0.5,
    min_cell: int = 100,
    percent_cutoff: float = 0.2,
    layer: Optional[str] = None,
    spar: float = 1.0,
    random_state: Optional[int] = 0,
    return_full: bool = False,
):
    """Detect Global Contamination-causing Genes (GCGs).

    Faithful port of scCDC ``ContaminationDetection``.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with **raw integer counts** in ``X`` (or
        in ``layer``) and a categorical cluster label in ``obs``.
    cluster_key
        Column of ``adata.obs`` holding the cluster labels.  If ``None``,
        the first categorical/object column is used.
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
        Seed for the bootstrap curve fitting (deterministic when set).
    return_full
        If ``True`` also return per-cluster diagnostics.

    Returns
    -------
    pandas.DataFrame
        The contamination-degree table for the detected GCGs: per-cluster
        entropy divergence plus a ``mean_distance`` column, sorted by
        descending ``mean_distance``.  ``.attrs['GCGs']`` holds the GCG
        list.  When ``return_full`` is set a ``dict`` is returned with the
        keys ``result``, ``GCGs``, ``per_cluster`` and ``all_distance``.
    """
    counts, genes, _ = counts_matrix(adata, layer=layer)
    labels = get_clusters(adata, cluster_key)
    labels = np.asarray(labels)

    # drop small clusters
    uniq, cnts = np.unique(labels, return_counts=True)
    qualified = [c for c, n in zip(uniq, cnts) if n >= min_cell]
    if len(qualified) == 0:
        raise ValueError(
            f"No cluster reaches min_cell={min_cell} "
            f"(largest cluster has {cnts.max()} cells)."
        )

    per_cluster = {}
    for cl in qualified:
        sub = counts[:, labels == cl]  # genes x cells
        ent = matrix_entropy(sub)
        mean_expr = np.log(sub.mean(axis=1) + 1.0)
        cdf = pd.DataFrame(
            {"Gene": genes, "mean.expr": mean_expr, "entropy": ent}
        )
        per_cluster[cl] = generate_curve(
            cdf, spar=spar, random_state=random_state
        )

    # candidate genes: p.adj <= 0.05 in each cluster
    flagged = []
    for cl in qualified:
        tab = per_cluster[cl]
        flagged.extend(tab.index[tab["p.adj"] <= 0.05].tolist())
    counter = pd.Series(flagged).value_counts()
    threshold = int(round(len(qualified) * restriction_factor))
    candidates = counter.index[counter >= threshold].tolist()
    if len(candidates) == 0:
        raise ValueError("No contaminated genes found.")

    # expression-percent filter (must pass in EVERY cluster)
    gene_pos = {g: i for i, g in enumerate(genes)}
    keep = []
    for g in candidates:
        gi = gene_pos[g]
        ok = True
        for cl in qualified:
            sub = counts[gi, labels == cl]
            pct = 1.0 - np.mean(sub == 0)
            if pct < percent_cutoff:
                ok = False
                break
        if ok:
            keep.append(g)
    gcgs = keep
    if len(gcgs) == 0:
        raise ValueError("No contaminated genes found after percent filter.")

    # contamination-degree table: per-cluster distance for every gene
    dist_cols = {}
    for cl in qualified:
        tab = per_cluster[cl]
        dist_cols[cl] = tab["distance"].reindex(genes).to_numpy()
    dist_df = pd.DataFrame(dist_cols, index=genes)
    dist_df["mean_distance"] = dist_df.mean(axis=1)
    dist_df = dist_df.sort_values("mean_distance", ascending=False)

    result = dist_df.loc[[g for g in dist_df.index if g in set(gcgs)]].copy()
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
