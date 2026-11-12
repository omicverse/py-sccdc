"""pysccdc run on one task, mirroring 02_run_sccdc.R.

Usage: python 03_run_pysccdc.py <task_dir> <out_dir> [run]

The bootstrap uses the default random_state=1: R scCDC always fits from set.seed(1).
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pysccdc


def main():
    td, od = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    od.mkdir(parents=True, exist_ok=True)
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")), dtype=np.float64)
    droplets = (td / "droplets.tsv").read_text().split()
    cl = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str)
    toc = tod[:, pd.Index(droplets).get_indexer(cl.cell)].tocsr()
    keep = np.flatnonzero(np.diff((toc != 0).tocsr().indptr) >= 3)     # CreateSeuratObject(min.cells = 3)
    adata = ad.AnnData(sp.csr_matrix(toc[keep].T))
    adata.var_names = [f"g{i + 1}" for i in keep]
    adata.obs_names = cl.cell.to_numpy()
    adata.obs["cluster"] = pd.Categorical(cl.cluster, categories=sorted(cl.cluster.unique(), key=int))
    t0 = time.perf_counter()
    gcg = pysccdc.ContaminationDetection(adata, cluster_key="cluster")      # random_state=1, as R
    t1 = time.perf_counter()
    out = pysccdc.ContaminationCorrection(adata, gcg, cluster_key="cluster")
    t2 = time.perf_counter()
    rows = np.array([int(g[1:]) for g in gcg.index])
    gcg.insert(0, "row", rows)
    gcg.to_csv(od / "gcg.tsv", sep="\t", index=False)
    m = sp.csc_matrix(out.layers["Corrected"].T)                       # genes x cells
    pos = pd.Index(adata.var_names).get_indexer(list(gcg.index))
    sub = sp.csc_matrix(m[pos])
    sub.sort_indices()
    sub.indptr.astype(np.int32).tofile(od / "corr_p.bin")
    sub.indices.astype(np.int32).tofile(od / "corr_i.bin")
    sub.data.astype(np.float64).tofile(od / "corr_x.bin")
    others = np.setdiff1d(np.arange(m.shape[0]), pos)
    orig = sp.csc_matrix(adata.X.T)
    same_others = (m[others] != orig[others]).nnz == 0
    th = out.uns["sccdc"]["thresholds"]
    pd.DataFrame({"row": rows, "threshold": [th[g] for g in gcg.index]}).to_csv(
        od / "thresholds.tsv", sep="\t", index=False)
    res = {"engine": "pysccdc", "version": pysccdc.__version__, "python": sys.version.split()[0], "seed": seed,
           "n_gcg": int(len(gcg)), "dim": [int(tod.shape[0]), int(toc.shape[1])], "others_unchanged": bool(same_others),
           "seconds_detection": t1 - t0, "seconds_correction": t2 - t1, "seconds": t2 - t0}
    (od / "result.json").write_text(json.dumps(res))
    print(td.name, "GCGs", len(gcg), f"detection {t1 - t0:.1f}s correction {t2 - t1:.1f}s", flush=True)


if __name__ == "__main__":
    main()
