"""GCG-specific decontamination -- :func:`ContaminationCorrection`.

Port of scCDC's ``Contamination_Correction.R`` (and the contamination
ratio of ``Contamination_Quantification.R``).

For each GCG, scCDC

1. ranks clusters by their mean **log-normalized** expression of the
   gene and takes the lowest-expressing qualified cluster as the first
   *eGCG-negative* reference;
2. computes, for every cluster, an **AUROC** of the per-cell counts of
   that cluster vs. the reference cluster (``Cal_AUCs``);
3. splits clusters into *eGCG-positive* (AUROC >= ``auc_thres``) and
   *eGCG-negative* (AUROC < ``auc_thres``);
4. pools the counts of the eGCG-negative cells with those of the
   **least** eGCG-positive cluster, builds a ROC of that pooled vector
   labelled neg(0)/pos(1), and takes the count value maximising the
   **Youden index** (sensitivity + specificity - 1) as the subtraction
   threshold (``Cal_thres``);
5. subtracts ``round(threshold)`` from every cell's count of that gene,
   floored at zero.

Only the GCGs are touched; all other genes pass through unchanged --
scCDC's anti-over-correction design.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._anndata import counts_matrix, get_clusters

__all__ = [
    "simple_roc",
    "youden_threshold",
    "ContaminationCorrection",
    "ContaminationQuantification",
]


# ----------------------------------------------------------------------
def _log_normalize(counts: np.ndarray, scale_factor: float = 1e4) -> np.ndarray:
    """Seurat ``LogNormalize``: ``log1p(count / colsum * scale_factor)``.

    ``counts`` is genes x cells; column sums are per-cell library sizes.
    """
    lib = counts.sum(axis=0)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib[None, :] * scale_factor)


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    """ROC thresholds / sensitivities / specificities (pROC, direction '<').

    ``direction='<'`` in pROC means the positive class is predicted to
    have *higher* scores.  pROC inserts ``+/-Inf`` boundary thresholds
    and uses midpoints between consecutive unique scores; we replicate
    that so the Youden-optimal threshold matches exactly.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    uniq = np.unique(scores)
    if uniq.size == 0:
        return np.array([np.inf]), np.array([0.0]), np.array([1.0])
    # pROC threshold grid: -Inf, midpoints of consecutive uniques, +Inf
    if uniq.size == 1:
        thr = np.array([-np.inf, np.inf])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        thr = np.concatenate(([-np.inf], mids, [np.inf]))

    sens = np.empty(thr.size)
    spec = np.empty(thr.size)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    for i, t in enumerate(thr):
        # predict positive when score >= t
        tp = np.sum(pos_scores >= t)
        tn = np.sum(neg_scores < t)
        sens[i] = tp / n_pos if n_pos > 0 else 0.0
        spec[i] = tn / n_neg if n_neg > 0 else 0.0
    return thr, sens, spec


def simple_roc(exp: np.ndarray, cls: np.ndarray) -> float:
    """AUROC of ``exp`` separating the two classes in ``cls`` (pROC '<').

    The positive class is the label ``1``.  Ties contribute 0.5 each
    (Mann-Whitney U / pROC convention).
    """
    exp = np.asarray(exp, dtype=float)
    cls = np.asarray(cls)
    pos = exp[cls == 1]
    neg = exp[cls == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-sum AUC with tie handling
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = ranks[:n_pos].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def youden_threshold(neg: np.ndarray, pos: np.ndarray) -> float:
    """Youden-index optimal cut count separating ``neg`` from ``pos``.

    Mirrors scCDC's ``Cal_thres`` -- builds the ROC of the pooled counts
    labelled 0(neg)/1(pos) and returns the pROC threshold that maximises
    ``sensitivity + specificity - 1``.
    """
    scores = np.concatenate([neg, pos]).astype(float)
    labels = np.concatenate([
        np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)
    ])
    thr, sens, spec = _roc_curve(scores, labels)
    yd = sens + spec - 1.0
    best = thr[int(np.argmax(yd))]
    return float(best)


# ----------------------------------------------------------------------
def _cal_aucs(counts, lognorm, labels, gene_idx, qualified):
    """``Cal_AUCs`` -- AUROC of each cluster vs the lowest-expressing one.

    Returns ``(aucs_by_cluster, eGCG_neg_reference_cluster)``.
    """
    # mean log-normalized expression per cluster (Seurat AverageExpression
    # on the LogNormalize 'data' slot used by Cal_AUCs to *order* clusters)
    clusters = np.unique(labels)
    mean_exp = {}
    for cl in clusters:
        mask = labels == cl
        # AverageExpression = expm1(mean(log-data)) then implicitly compared;
        # ordering by the log-data mean is monotone-equivalent.
        mean_exp[cl] = lognorm[gene_idx, mask].mean()
    order = sorted(clusters, key=lambda c: mean_exp[c])
    qset = set(qualified)
    ref = next((c for c in order if c in qset), order[0])

    g_counts = counts[gene_idx]
    ref_vals = g_counts[labels == ref]
    aucs = {}
    for cl in order:
        cl_vals = g_counts[labels == cl]
        scores = np.concatenate([cl_vals, ref_vals])
        cls = np.concatenate([
            np.ones(len(cl_vals), dtype=int),
            np.zeros(len(ref_vals), dtype=int),
        ])
        aucs[cl] = simple_roc(scores, cls)
    # preserve the expression-sorted order
    return order, aucs, ref


def _cal_thres(counts, labels, gene_idx, order, aucs, auc_thres):
    """``Cal_thres`` -- the per-gene Youden subtraction threshold."""
    auc_arr = np.array([aucs[c] for c in order])
    if np.sum(auc_arr >= auc_thres) == 0:
        auc_thres = np.nanmax(auc_arr)

    neg_cls = [c for c in order if aucs[c] < auc_thres]
    pos_cls = [c for c in order if aucs[c] >= auc_thres]
    if len(pos_cls) == 0:  # degenerate -> nothing to subtract
        return 0.0
    low_pos = pos_cls[0]

    g_counts = counts[gene_idx]
    neg_mask = np.isin(labels, neg_cls) if neg_cls else np.zeros(
        labels.shape, dtype=bool
    )
    pos_mask = labels == low_pos
    neg_vals = g_counts[neg_mask]
    pos_vals = g_counts[pos_mask]
    if neg_vals.size == 0 or pos_vals.size == 0:
        return 0.0
    return youden_threshold(neg_vals, pos_vals)


def _cont_level(counts, lognorm, labels, gene_idx, order, aucs, auc_thres):
    """``Cal_Cont_level`` -- per-gene contamination ratio.

    Total expression of the gene in eGCG-negative cells divided by the
    total expression of *all* genes in those same cells.
    """
    auc_arr = np.array([aucs[c] for c in order])
    if np.sum(auc_arr >= auc_thres) == 0:
        auc_thres = np.nanmax(auc_arr)
    neg_cls = [c for c in order if aucs[c] < auc_thres]
    if not neg_cls:
        return 0.0
    neg_mask = np.isin(labels, neg_cls)
    gene_total = counts[gene_idx, neg_mask].sum()
    all_total = counts[:, neg_mask].sum()
    if all_total == 0:
        return 0.0
    return float(gene_total / all_total)


# ----------------------------------------------------------------------
def ContaminationCorrection(
    adata,
    cont_genes: Union[Sequence[str], pd.DataFrame],
    cluster_key: Optional[str] = None,
    auc_thres: float = 0.9,
    min_cell: int = 50,
    layer: Optional[str] = None,
    copy: bool = True,
    corrected_layer: str = "Corrected",
):
    """Decontaminate the GCG counts of an AnnData (scCDC ``ContaminationCorrection``).

    Parameters
    ----------
    adata
        AnnData with raw integer counts (in ``X`` or ``layer``).
    cont_genes
        GCG names; either a sequence, or the DataFrame returned by
        :func:`~pysccdc.ContaminationDetection` (its index is used).
    cluster_key
        ``adata.obs`` column with cluster labels (auto-detected if None).
    auc_thres
        AUROC threshold splitting eGCG-positive / -negative clusters.
    min_cell
        Clusters smaller than this are not used as references.
    layer
        Count layer; ``None`` -> ``adata.X``.
    copy
        Return a corrected copy (``True``) or modify ``adata`` in place.
    corrected_layer
        Name of the layer that receives the corrected matrix.

    Returns
    -------
    anndata.AnnData
        AnnData with a ``layers[corrected_layer]`` holding the corrected
        counts.  ``.uns['sccdc']`` records the per-gene thresholds.
    """
    if isinstance(cont_genes, pd.DataFrame):
        cont_genes = list(cont_genes.index)
    cont_genes = list(cont_genes)

    out = adata.copy() if copy else adata
    counts, genes, _ = counts_matrix(out, layer=layer)
    labels = get_clusters(out, cluster_key)
    labels = np.asarray(labels)

    gene_pos = {g: i for i, g in enumerate(genes)}
    missing = [g for g in cont_genes if g not in gene_pos]
    if missing:
        raise KeyError(f"cont_genes not in adata.var_names: {missing}")

    uniq, cnts = np.unique(labels, return_counts=True)
    qualified = [c for c, n in zip(uniq, cnts) if n >= min_cell]
    if not qualified:
        raise ValueError(f"No cluster reaches min_cell={min_cell}.")

    lognorm = _log_normalize(counts)

    corrected = counts.copy()
    thresholds = {}
    for g in cont_genes:
        gi = gene_pos[g]
        order, aucs, _ref = _cal_aucs(
            counts, lognorm, labels, gi, qualified
        )
        thr = _cal_thres(counts, labels, gi, order, aucs, auc_thres)
        thresholds[g] = thr
        corrected[gi] = np.maximum(counts[gi] - round(thr), 0.0)

    # write back in AnnData orientation (cells x genes)
    out.layers[corrected_layer] = corrected.T
    info = out.uns.get("sccdc", {})
    info = dict(info)
    info["thresholds"] = thresholds
    info["GCGs"] = cont_genes
    info["auc_thres"] = auc_thres
    out.uns["sccdc"] = info
    return out


# ----------------------------------------------------------------------
def ContaminationQuantification(
    adata,
    cont_genes: Union[Sequence[str], pd.DataFrame],
    cluster_key: Optional[str] = None,
    auc_thres: float = 0.9,
    min_cell: int = 50,
    top10_gcg: bool = False,
    layer: Optional[str] = None,
    return_per_gene: bool = False,
):
    """Dataset-level contamination ratio from the GCGs (scCDC ``ContaminationQuantification``).

    For each GCG the contamination level is its total expression in the
    eGCG-negative cells divided by the total expression of all genes in
    those cells; the dataset ratio is the **maximum** across GCGs.  A
    ratio above ~3e-4 indicates high contamination (scCDC heuristic).

    Parameters
    ----------
    adata, cont_genes, cluster_key, auc_thres, min_cell, layer
        As in :func:`ContaminationCorrection`.
    top10_gcg
        Use only the first 10 GCGs (scCDC option).
    return_per_gene
        Also return the per-gene ratio :class:`pandas.Series`.

    Returns
    -------
    float or tuple
        The maximum contamination ratio, or ``(max_ratio, per_gene)``
        when ``return_per_gene`` is set.
    """
    if isinstance(cont_genes, pd.DataFrame):
        cont_genes = list(cont_genes.index)
    cont_genes = list(cont_genes)
    if top10_gcg and len(cont_genes) >= 10:
        cont_genes = cont_genes[:10]

    counts, genes, _ = counts_matrix(adata, layer=layer)
    labels = np.asarray(get_clusters(adata, cluster_key))
    gene_pos = {g: i for i, g in enumerate(genes)}

    uniq, cnts = np.unique(labels, return_counts=True)
    qualified = [c for c, n in zip(uniq, cnts) if n >= min_cell]
    if not qualified:
        raise ValueError(f"No cluster reaches min_cell={min_cell}.")

    lognorm = _log_normalize(counts)
    ratios = {}
    for g in cont_genes:
        gi = gene_pos[g]
        order, aucs, _ref = _cal_aucs(counts, lognorm, labels, gi, qualified)
        ratios[g] = _cont_level(
            counts, lognorm, labels, gi, order, aucs, auc_thres
        )
    per_gene = pd.Series(ratios, name="contamination_ratio")
    max_ratio = float(per_gene.max()) if len(per_gene) else 0.0
    if return_per_gene:
        return max_ratio, per_gene
    return max_ratio
