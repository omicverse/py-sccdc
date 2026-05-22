"""Synthetic example data for :mod:`pysccdc`.

:func:`simulate_contaminated` builds a small clustered count matrix in
which a handful of genes are deliberately spiked with *ambient*
contamination -- each contaminating gene is a marker of one cluster but
also leaks low counts into every other cluster.  This is the input used
by the R-parity test suite.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["simulate_contaminated"]


def simulate_contaminated(
    n_clusters: int = 4,
    n_cells_per_cluster: int = 200,
    n_genes: int = 120,
    n_contaminating: int = 4,
    contamination_strength: float = 6.0,
    random_state: Optional[int] = 0,
):
    """Simulate a clustered scRNA-seq count matrix with spiked GCGs.

    Parameters
    ----------
    n_clusters
        Number of cell clusters.
    n_cells_per_cluster
        Cells per cluster.
    n_genes
        Total genes (markers + housekeeping + contaminating).
    n_contaminating
        Number of deliberately-spiked contaminating genes (GCGs).
    contamination_strength
        Mean ambient leakage count of each contaminating gene into
        non-source clusters.
    random_state
        Seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        ``X`` holds integer counts (cells x genes); ``obs['cluster']``
        the cluster labels; ``uns['true_GCGs']`` the spiked gene names.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(random_state)
    n_cells = n_clusters * n_cells_per_cluster

    labels = np.repeat(
        [f"c{i}" for i in range(n_clusters)], n_cells_per_cluster
    )

    gene_names = [f"gene{i}" for i in range(n_genes)]
    counts = np.zeros((n_cells, n_genes), dtype=np.int64)

    # ---- housekeeping / background genes -----------------------------
    # genuinely variable expression -> a broad spread of count values,
    # i.e. HIGH entropy for their mean expression (non-contaminating).
    n_bg = n_genes - n_contaminating
    for j in range(n_bg):
        # negative-binomial-like over-dispersion gives a wide count
        # distribution (high entropy) typical of real genes
        rate = rng.gamma(shape=1.2, scale=rng.uniform(1.0, 8.0),
                         size=n_cells)
        counts[:, j] = rng.poisson(rate)

    # add cluster-marker structure to some background genes so clusters
    # are biologically distinct (these are NOT contaminating) -- still a
    # broad (high-entropy) distribution within the marker cluster.
    markers_per_cluster = max(1, n_bg // (n_clusters * 4))
    for ci in range(n_clusters):
        cmask = labels == f"c{ci}"
        for k in range(markers_per_cluster):
            j = ci * markers_per_cluster + k
            if j >= n_bg:
                break
            rate = rng.gamma(shape=2.0, scale=12.0, size=cmask.sum())
            counts[cmask, j] += rng.poisson(rate)

    # ---- contaminating genes (the true GCGs) -------------------------
    # A GCG is a strong marker of ONE cluster but leaks *ambient* RNA
    # into every other cluster.  The ambient leakage is near-constant
    # low counts (it reflects the soup composition, not real biology),
    # so the gene's overall count distribution is unusually CONCENTRATED
    # for its mean expression -> entropy BELOW the fitted curve, the
    # positive entropy divergence scCDC keys on.
    true_gcgs = []
    for k in range(n_contaminating):
        j = n_bg + k
        gene_names[j] = f"CONT{k}"
        true_gcgs.append(f"CONT{k}")
        source = k % n_clusters
        src_mask = labels == f"c{source}"
        # high, fairly uniform expression in the source cluster
        counts[src_mask, j] = rng.poisson(45.0, size=src_mask.sum())
        # ambient leakage everywhere else: a near-constant low count
        # (Bernoulli presence x a tight integer level) -> low entropy
        other = ~src_mask
        lvl = int(round(contamination_strength))
        present = rng.random(other.sum()) < 0.85
        counts[other, j] += np.where(present, lvl, lvl - 1)

    obs = pd.DataFrame(
        {"cluster": pd.Categorical(labels)},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(
        X=counts.astype(np.float64), obs=obs, var=var
    )
    adata.uns["true_GCGs"] = true_gcgs
    return adata
