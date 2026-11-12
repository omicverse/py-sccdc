"""Build benchmark tasks from the 10x Genomics human-mouse mixtures (hgmm).

Usage: python 01_prepare.py <dir with extracted 10x archives> <tasks_dir>
Each task: tod.mtx.gz (genes x all droplets), genes.tsv (id, symbol, genome), droplets.tsv, cells.tsv
(Cell Ranger's cell calls), clusters.tsv (one clustering of the cells, given to both packages),
species.tsv (human / mouse / mixed from the share of human counts, for the ground truth).
"""
from __future__ import annotations

import gzip
import pathlib
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

SRC, OUT = map(pathlib.Path, sys.argv[1:3])


def read_dir(d):
    mtx = next(d.glob("matrix.mtx*"))
    m = sp.csc_matrix(scipy.io.mmread(gzip.open(mtx) if mtx.suffix == ".gz" else mtx))
    feats = next(d.glob("features.tsv*"), None) or next(d.glob("genes.tsv*"))
    genes = pd.read_csv(feats, sep="\t", header=None).iloc[:, :2]
    genes.columns = ["id", "symbol"]
    bcs = pd.read_csv(next(d.glob("barcodes.tsv*")), header=None)[0].to_numpy()
    return m, genes, bcs


def load(name, kind):
    base = SRC / f"{name}_{kind}"
    v3 = list(base.glob("*/matrix.mtx.gz"))
    if v3:                                                   # v3: one matrix, genome prefix in the id
        m, genes, bcs = read_dir(v3[0].parent)
        genes["genome"] = np.where(genes.id.str.startswith("mm10"), "mouse", "human")
        return m, genes, bcs
    parts = [read_dir(p.parent) for p in sorted(base.glob("*/*/matrix.mtx"))]   # v2: hg19 and mm10
    if "filtered" in kind:                                   # v2 calls cells per genome: take the union
        return None, None, np.unique(np.concatenate([p[2] for p in parts]))
    assert all((p[2] == parts[0][2]).all() for p in parts)
    for (_, g, _), gen in zip(parts, ["human", "mouse"]):
        g["genome"] = gen
    return sp.vstack([p[0] for p in parts]).tocsc(), pd.concat([p[1] for p in parts], ignore_index=True), parts[0][2]


for name in ["hgmm_1k_v3", "hgmm_5k_v3", "hgmm_6k", "hgmm_10k_v3", "hgmm_12k"]:
    kind = "raw_feature_bc_matrix" if name.endswith("v3") else "raw_gene_bc_matrices"
    tod, genes, droplets = load(name, kind)
    keep = np.asarray(tod.sum(axis=0)).ravel() > 0                     # droplets with any count
    tod, droplets = tod[:, keep], droplets[keep]
    _, _, cells = load(name, kind.replace("raw", "filtered"))
    idx = pd.Index(droplets).get_indexer(cells)
    assert (idx >= 0).all()
    toc = tod[:, idx]
    hum = np.asarray(toc[(genes.genome == "human").to_numpy()].sum(axis=0)).ravel()
    tot = np.asarray(toc.sum(axis=0)).ravel()
    frac = hum / tot
    species = np.select([frac >= 0.9, frac <= 0.1], ["human", "mouse"], "mixed")
    ad = sc.AnnData(sp.csr_matrix(toc.T))
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
    sc.pp.pca(ad, n_comps=30, random_state=0)
    sc.pp.neighbors(ad, n_neighbors=15, random_state=0)
    sc.tl.leiden(ad, resolution=1.0, random_state=0, flavor="igraph", n_iterations=2, directed=False)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(d / "tod.mtx", tod)
    with open(d / "tod.mtx", "rb") as fi, gzip.open(d / "tod.mtx.gz", "wb", compresslevel=3) as fo:
        fo.write(fi.read())
    (d / "tod.mtx").unlink()
    genes.to_csv(d / "genes.tsv", sep="\t", index=False)
    pd.Series(droplets).to_csv(d / "droplets.tsv", index=False, header=False)
    pd.Series(cells).to_csv(d / "cells.tsv", index=False, header=False)
    pd.DataFrame({"cell": cells, "cluster": ad.obs.leiden.to_numpy()}).to_csv(d / "clusters.tsv", sep="\t", index=False)
    pd.DataFrame({"cell": cells, "species": species, "human_frac": frac}).to_csv(d / "species.tsv", sep="\t", index=False)
    print(name, f"{tod.shape[1]:,} droplets, {len(cells):,} cells, {ad.obs.leiden.nunique()} clusters,",
          dict(pd.Series(species).value_counts()), flush=True)
