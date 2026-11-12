"""GCG-specific decontamination -- :func:`ContaminationCorrection`.

Port of scCDC's ``Contamination_Correction.R`` (and the contamination
ratio of ``Contamination_Quantification.R``).

For each GCG, scCDC

1. ranks all clusters by their ``AverageExpression`` of the gene (Seurat 5:
   the mean of ``expm1`` of the log-normalized data) and takes the
   lowest-expressing cluster with at least ``min_cell`` cells as the
   *eGCG-negative* reference;
2. computes, for every cluster, the AUROC (pROC, ``direction = "<"``) of the
   per-cell counts of that cluster against the reference cluster
   (``Cal_AUCs``);
3. splits clusters into *eGCG-positive* (AUROC >= ``auc_thres``) and
   *eGCG-negative* (AUROC < ``auc_thres``);
4. pools the counts of the eGCG-negative cells with those of the
   lowest-expressing eGCG-positive cluster, builds their ROC and takes the
   threshold with the largest Youden index (the first one, as
   ``which.max``) (``Cal_thres``);
5. subtracts ``round(threshold)`` from every cell's count of that gene,
   floored at zero.

Only the GCGs are touched; all other genes pass through unchanged --
scCDC's anti-over-correction design.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._anndata import counts_matrix, get_clusters
from ._rbase import c_apply

__all__ = [
    "simple_roc",
    "youden_threshold",
    "ContaminationCorrection",
    "ContaminationQuantification",
]


# ----------------------------------------------------------------------
def _roc_curve(cases: np.ndarray, controls: np.ndarray):
    """pROC ``roc(..., direction = "<")``: thresholds (increasing), sensitivities, specificities.

    Thresholds are ``-Inf``, the midpoints of consecutive unique values and ``+Inf``; a case is
    called positive when its value exceeds the threshold."""
    cases = np.asarray(cases, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    u = np.unique(np.concatenate([cases, controls]))
    thr = (np.concatenate(([-np.inf], u)) + np.concatenate((u, [np.inf]))) / 2
    # value > midpoint(u[k-1], u[k])  <=>  value >= u[k]
    tp = cases.size - np.searchsorted(np.sort(cases), u, side="left")
    fp = controls.size - np.searchsorted(np.sort(controls), u, side="left")
    se = np.concatenate((tp, [0])) / cases.size
    sp_ = (controls.size - np.concatenate((fp, [0]))) / controls.size
    return thr, se, sp_


def _auc(se: np.ndarray, sp_: np.ndarray) -> float:
    """pROC ``auc``: trapezoids over the curve (specificity increasing), summed in long double."""
    if np.any(np.diff(sp_) < 0):
        se, sp_ = se[::-1], sp_[::-1]
    terms = ((se[1:] + se[:-1]) / 2) * (sp_[1:] - sp_[:-1])
    return float(np.cumsum(terms.astype(np.longdouble))[-1]) if terms.size else float("nan")


def simple_roc(exp: np.ndarray, cls: np.ndarray) -> float:
    """scCDC ``simple_roc``: AUROC of ``exp`` for class ``1`` against class ``0`` (pROC)."""
    exp = np.asarray(exp, dtype=np.float64)
    cls = np.asarray(cls)
    if not ((cls == 1).any() and (cls == 0).any()):
        return float("nan")
    _, se, sp_ = _roc_curve(exp[cls == 1], exp[cls == 0])
    return _auc(se, sp_)


def youden_threshold(neg: np.ndarray, pos: np.ndarray) -> float:
    """Youden-index optimal cut separating ``neg`` (controls) from ``pos`` (cases).

    Mirrors scCDC's ``Cal_thres``: the first pROC threshold maximising
    ``sensitivity + specificity - 1``."""
    thr, se, sp_ = _roc_curve(pos, neg)
    return float(thr[int(np.argmax(se + sp_ - 1))])


# ----------------------------------------------------------------------
class _Clusters:
    """Cells of each cluster (level order) and the per-cell library sizes."""

    def __init__(self, counts: sp.csr_matrix, labels: np.ndarray, levels: list, min_cell: int):
        self.levels = [c for c in levels if (labels == c).any()]
        self.cells = {c: np.flatnonzero(labels == c) for c in self.levels}
        self.qualified = [c for c in self.levels if self.cells[c].size >= min_cell]
        self.lib = np.asarray(counts.sum(axis=0)).ravel()


def _gene_row(counts: sp.csr_matrix, gi: int) -> np.ndarray:
    row = np.zeros(counts.shape[1])
    s, e = counts.indptr[gi], counts.indptr[gi + 1]
    row[counts.indices[s:e]] = counts.data[s:e]
    return row


def _cal_aucs(row: np.ndarray, cl: _Clusters):
    """``Cal_AUCs``: clusters sorted by ``AverageExpression`` of the gene, their AUROC against
    the reference cluster, and the reference."""
    data = c_apply(math.log1p, row / cl.lib * 1e4)        # Seurat LogNormalize
    lin = c_apply(math.expm1, data)
    avg = np.array([lin[cl.cells[c]].sum() / cl.cells[c].size for c in cl.levels])
    order = [cl.levels[i] for i in np.argsort(avg, kind="stable")]
    ref = next(c for c in order if c in set(cl.qualified))
    ref_vals = row[cl.cells[ref]]
    aucs = {}
    for c in order:
        _, se, sp_ = _roc_curve(row[cl.cells[c]], ref_vals)
        aucs[c] = _auc(se, sp_)
    return order, aucs, ref


def _split(order, aucs, auc_thres):
    vals = np.array([aucs[c] for c in order])
    if np.sum(vals >= auc_thres) == 0:
        auc_thres = float(np.max(vals))
    neg = [c for c in order if aucs[c] < auc_thres]
    pos = [c for c in order if aucs[c] >= auc_thres]
    return neg, pos


def _cal_thres(row: np.ndarray, cl: _Clusters, order, aucs, auc_thres) -> float:
    """``Cal_thres``: the Youden threshold between the eGCG-negative cells and the
    lowest-expressing eGCG-positive cluster."""
    neg, pos = _split(order, aucs, auc_thres)
    neg_cells = np.concatenate([cl.cells[c] for c in neg]) if neg else np.array([], dtype=np.int64)
    return youden_threshold(row[neg_cells], row[cl.cells[pos[0]]])


# ----------------------------------------------------------------------
def _prepare(adata, cont_genes, cluster_key, layer, min_cell):
    if isinstance(cont_genes, pd.DataFrame):
        cont_genes = list(cont_genes.index)
    cont_genes = list(cont_genes)
    counts, genes, _ = counts_matrix(adata, layer=layer)
    labels, levels = get_clusters(adata, cluster_key, return_levels=True)
    gene_pos = {g: i for i, g in enumerate(genes)}
    missing = [g for g in cont_genes if g not in gene_pos]
    if missing:
        raise KeyError(f"cont_genes not in adata.var_names: {missing}")
    cl = _Clusters(counts, labels, levels, min_cell)
    if not cl.qualified:
        raise ValueError(f"No cluster reaches min_cell={min_cell}.")
    return cont_genes, counts, gene_pos, cl


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
        Name of the layer that receives the corrected matrix (sparse when
        the counts are sparse, else dense).

    Returns
    -------
    anndata.AnnData
        AnnData with a ``layers[corrected_layer]`` holding the corrected
        counts.  ``.uns['sccdc']`` records the per-gene thresholds.
    """
    out = adata.copy() if copy else adata
    cont_genes, counts, gene_pos, cl = _prepare(out, cont_genes, cluster_key, layer, min_cell)
    thresholds, new_rows = {}, {}
    for g in cont_genes:
        gi = gene_pos[g]
        row = _gene_row(counts, gi)
        order, aucs, _ = _cal_aucs(row, cl)
        thr = _cal_thres(row, cl, order, aucs, auc_thres)
        thresholds[g] = thr
        if thr == -np.inf:          # R: x - round(-Inf) = Inf for every cell
            warnings.warn(f"{g}: the Youden threshold is -Inf; R scCDC sets every count of this gene to Inf")
        new_rows[gi] = np.maximum(row - np.rint(thr), 0.0)
    # GCG rows replaced, every other row kept as it is
    rows = np.fromiter(new_rows, dtype=np.int64, count=len(new_rows))
    keep = np.ones(counts.shape[0])
    keep[rows] = 0.0
    repl = sp.coo_matrix(np.vstack([new_rows[r] for r in rows])) if rows.size else sp.coo_matrix((0, counts.shape[1]))
    corrected = sp.diags(keep) @ counts + sp.coo_matrix(
        (repl.data, (rows[repl.row], repl.col)), shape=counts.shape)
    corrected = sp.csr_matrix(corrected)
    corrected.eliminate_zeros()
    src = out.layers[layer] if layer is not None else out.X
    out.layers[corrected_layer] = sp.csr_matrix(corrected.T) if sp.issparse(src) else corrected.T.toarray()
    info = dict(out.uns.get("sccdc", {}))
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

    For each GCG the contamination level is its total count in the
    eGCG-negative cells divided by the total count of all genes in
    those cells; the dataset ratio is the **maximum** across GCGs.  scCDC
    calls a ratio above 3e-4 high contamination.

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
    cont_genes, counts, gene_pos, cl = _prepare(adata, cont_genes, cluster_key, layer, min_cell)
    ratios = {}
    for g in cont_genes:
        row = _gene_row(counts, gene_pos[g])
        order, aucs, _ = _cal_aucs(row, cl)
        neg, _ = _split(order, aucs, auc_thres)
        cells = np.concatenate([cl.cells[c] for c in neg]) if neg else np.array([], dtype=np.int64)
        total = cl.lib[cells].sum()
        ratios[g] = float(row[cells].sum() / total) if total else float("nan")
    per_gene = pd.Series(ratios, name="contamination_ratio")
    max_ratio = float(per_gene.max()) if len(per_gene) else 0.0
    if return_per_gene:
        return max_ratio, per_gene
    return max_ratio
