"""Head-to-head speed benchmark: R scCDC vs pysccdc.

Runs both pipelines on the same clustered count matrix and reports wall
time for:

  * ContaminationDetection  — entropy + bootstrapped spline curve fit
  * ContaminationCorrection — Youden-threshold decontamination

The R side is driven through ``r_driver_sccdc.R``; the Python side calls
``pysccdc`` directly. Uses the bundled clustered PBMC 3k dataset (or the
synthetic dataset as a fallback when the h5ad is absent).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import pysccdc as cd

HERE = Path(__file__).parent
WORK = HERE / "compare_out"
DRIVER = HERE / "r_driver_sccdc.R"
RSCRIPT = "/scratch/users/steorra/env/CMAP/bin/Rscript"
R_LIBS = "/scratch/users/steorra/env/CMAP/R_extra_libs"
R_LD = "/scratch/users/steorra/env/CMAP/lib"


def load_adata():
    """Bundled clustered PBMC 3k, or the synthetic dataset as a fallback."""
    h5ad = HERE.parent / "data" / "pbmc3k_clustered.h5ad"
    if h5ad.exists():
        return ad.read_h5ad(h5ad), "leiden"
    return cd.datasets.simulate_contaminated(random_state=0), "cluster"


def time_r(counts_tsv: Path, clusters_csv: Path) -> float:
    """Wall time of the full R scCDC driver."""
    out = WORK / "r_bench"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    env = os.environ.copy()
    env["R_LIBS_USER"] = R_LIBS
    env["LD_LIBRARY_PATH"] = R_LD + ":" + env.get("LD_LIBRARY_PATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [RSCRIPT, str(DRIVER), str(counts_tsv), str(clusters_csv), str(out)],
        env=env, capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"R driver failed:\n{proc.stderr[-1500:]}")
    return dt


def time_python(adata, cluster_key: str) -> tuple[float, float]:
    """Return (detection_s, correction_s) for pysccdc."""
    t0 = time.perf_counter()
    det = cd.ContaminationDetection(
        adata, cluster_key=cluster_key, min_cell=50, random_state=0,
    )
    t_det = time.perf_counter() - t0

    t1 = time.perf_counter()
    cd.ContaminationCorrection(adata, det, cluster_key=cluster_key)
    t_cor = time.perf_counter() - t1
    return t_det, t_cor


def main() -> None:
    adata, cluster_key = load_adata()
    print(f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"cluster_key='{cluster_key}'")

    WORK.mkdir(exist_ok=True)
    counts_tsv = WORK / "counts.tsv"
    clusters_csv = WORK / "clusters.csv"
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    pd.DataFrame(X.T.astype(int), index=adata.var_names,
                 columns=adata.obs_names).to_csv(counts_tsv, sep="\t")
    pd.DataFrame({"cell": adata.obs_names,
                  "cluster": adata.obs[cluster_key].astype(str)}
                 ).to_csv(clusters_csv, index=False)

    print("\n[R]  timing scCDC end-to-end...")
    r_total = time_r(counts_tsv, clusters_csv)
    print(f"  detect + correct + quantify: {r_total:8.2f} s")

    print("\n[py] timing pysccdc...")
    py_det, py_cor = time_python(adata, cluster_key)
    print(f"  ContaminationDetection:      {py_det:8.2f} s")
    print(f"  ContaminationCorrection:     {py_cor:8.2f} s")
    print(f"  detect + correct:            {py_det + py_cor:8.2f} s")

    print(f"\nSpeed-up (R end-to-end / Python detect+correct): "
          f"{r_total / (py_det + py_cor):5.2f}x")
    print("Note: the R total also includes Seurat object construction; "
          "the Python figure is the algorithm only.")


if __name__ == "__main__":
    main()
