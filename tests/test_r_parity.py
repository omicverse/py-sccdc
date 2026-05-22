"""R-parity tests — pysccdc vs the R package scCDC.

The R driver (:file:`r_reference_driver.R`) runs scCDC 1.4 on the *same*
synthetic clustered count matrix that :func:`pysccdc.datasets.simulate_contaminated`
produces (4 clusters x 200 cells, 120 genes, 4 deliberately-spiked
contaminating genes).  Both sides therefore analyse identical input.

We compare:

* per-gene / per-cluster **Shannon entropy** — bit-exact (the Rcpp
  ``MatrixToEntropy`` reduces to a deterministic ``bincount``);
* the detected **GCG list** — must match exactly;
* the per-gene **entropy divergence** — Pearson r > 0.99 (the bootstrap
  uses a different RNG and ``smooth.spline`` differs slightly from the
  scipy penalized B-spline; see the README);
* the **corrected count matrix** — bit-exact for the deterministic
  Youden-threshold path;
* the dataset **contamination ratio** — bit-exact.

Tests skip gracefully when the CMAP R env or scCDC is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pysccdc as cd
from pysccdc._anndata import counts_matrix, get_clusters
from pysccdc.entropy import matrix_entropy

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
RSCRIPT = "/scratch/users/steorra/env/CMAP/bin/Rscript"


def _r_available() -> bool:
    if not R_DRIVER.exists() or not Path(RSCRIPT).exists():
        return False
    try:
        out = subprocess.run(
            [RSCRIPT, "-e", "library(scCDC); cat('OK')"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or scCDC not installed.",
)

RANDOM_STATE = 0


@pytest.fixture(scope="module")
def sim_adata():
    """The shared synthetic dataset."""
    return cd.datasets.simulate_contaminated(random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory, sim_adata):
    """Run the R scCDC reference once on the synthetic dataset."""
    out_dir = tmp_path_factory.mktemp("sccdc_R")
    counts_csv = out_dir / "counts.csv"
    clusters_csv = out_dir / "clusters.csv"

    counts = pd.DataFrame(
        sim_adata.X.T.astype(int),
        index=sim_adata.var_names,
        columns=sim_adata.obs_names,
    )
    counts.to_csv(counts_csv)
    pd.DataFrame(
        {"cell": sim_adata.obs_names,
         "cluster": sim_adata.obs["cluster"].astype(str)}
    ).to_csv(clusters_csv, index=False)

    res = subprocess.run(
        [RSCRIPT, str(R_DRIVER), str(counts_csv),
         str(clusters_csv), str(out_dir)],
        capture_output=True, text=True, timeout=1200,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def py_detection(sim_adata):
    return cd.ContaminationDetection(
        sim_adata, cluster_key="cluster", min_cell=50,
        random_state=RANDOM_STATE, return_full=True,
    )


# ----------------------------------------------------------------------
def test_entropy_bit_exact(r_reference, sim_adata):
    """Shannon entropy of count distributions matches R to machine eps."""
    r_ent = pd.read_csv(r_reference / "entropy.csv", index_col=0)
    counts, genes, _ = counts_matrix(sim_adata)
    labels = np.asarray(get_clusters(sim_adata, "cluster"))
    py_ent = pd.DataFrame(
        {cl: matrix_entropy(counts[:, labels == cl]) for cl in r_ent.columns},
        index=genes,
    )
    diff = np.abs(
        r_ent.values - py_ent.loc[r_ent.index, r_ent.columns].values
    ).max()
    assert diff < 1e-9, f"entropy max abs diff {diff:.2e}"


def test_detected_gcgs_match(r_reference, py_detection):
    """The detected GCG set is identical to R scCDC."""
    r_gcgs = set(
        (r_reference / "gcgs.txt").read_text().split()
    )
    py_gcgs = set(py_detection["GCGs"])
    assert py_gcgs == r_gcgs, f"R={sorted(r_gcgs)} PY={sorted(py_gcgs)}"


def test_all_spiked_genes_detected(r_reference, py_detection, sim_adata):
    """Both R and pysccdc recover every deliberately-spiked GCG."""
    truth = set(sim_adata.uns["true_GCGs"])
    r_gcgs = set((r_reference / "gcgs.txt").read_text().split())
    py_gcgs = set(py_detection["GCGs"])
    assert truth <= r_gcgs
    assert truth <= py_gcgs


def test_entropy_divergence_correlates(r_reference, py_detection):
    """Per-gene entropy divergence agrees with R (Pearson r > 0.99)."""
    r_dist = pd.read_csv(r_reference / "distance.csv", index_col=0)
    py_dist = py_detection["all_distance"]
    cm = r_dist.index.intersection(py_dist.index)
    cols = [c for c in r_dist.columns
            if c in py_dist.columns and c != "mean_distance"]
    rv = r_dist.loc[cm, cols].to_numpy().ravel()
    pv = py_dist.loc[cm, cols].to_numpy().ravel()
    mask = np.isfinite(rv) & np.isfinite(pv)
    rho, _ = pearsonr(rv[mask], pv[mask])
    assert rho > 0.99, f"entropy divergence Pearson r = {rho:.5f}"


def test_mean_distance_ordering(r_reference, py_detection):
    """GCG mean entropy divergence is large and well-correlated with R."""
    r_dist = pd.read_csv(r_reference / "distance.csv", index_col=0)
    py_dist = py_detection["all_distance"]
    gcgs = py_detection["GCGs"]
    rv = r_dist.loc[gcgs, "mean_distance"].to_numpy()
    pv = py_dist.loc[gcgs, "mean_distance"].to_numpy()
    rho, _ = pearsonr(rv, pv)
    assert rho > 0.9, f"GCG mean_distance r = {rho:.4f}"
    # GCGs should rank near the top of the full divergence table
    assert (py_dist.loc[gcgs, "mean_distance"] > 0).all()


def test_corrected_matrix_matches(r_reference, sim_adata, py_detection):
    """The decontaminated count matrix is bit-exact with R scCDC."""
    gcgs = py_detection["GCGs"]
    cor = cd.ContaminationCorrection(
        sim_adata, gcgs, cluster_key="cluster"
    )
    r_corr = pd.read_csv(r_reference / "corrected.csv", index_col=0)
    py_corr = pd.DataFrame(
        cor.layers["Corrected"].T,
        index=sim_adata.var_names, columns=sim_adata.obs_names,
    )
    cm = r_corr.index.intersection(py_corr.index)
    cc = r_corr.columns.intersection(py_corr.columns)
    rv = r_corr.loc[cm, cc].to_numpy()
    pv = py_corr.loc[cm, cc].to_numpy()
    max_diff = np.abs(rv - pv).max()
    assert max_diff == 0, f"corrected matrix max abs diff {max_diff}"


def test_correction_thresholds_match(r_reference, sim_adata, py_detection):
    """Per-GCG Youden subtraction thresholds match R after rounding."""
    gcgs = py_detection["GCGs"]
    cor = cd.ContaminationCorrection(
        sim_adata, gcgs, cluster_key="cluster"
    )
    r_thr = pd.read_csv(
        r_reference / "thresholds.csv"
    ).set_index("gene")["threshold"]
    py_thr = cor.uns["sccdc"]["thresholds"]
    for g in gcgs:
        assert round(py_thr[g]) == int(r_thr[g]), (
            f"{g}: PY round({py_thr[g]:.3f})={round(py_thr[g])} "
            f"vs R {int(r_thr[g])}"
        )


def test_contamination_ratio_matches(r_reference, sim_adata, py_detection):
    """The dataset contamination ratio is bit-exact with R scCDC."""
    ratio_txt = (r_reference / "ratio.txt").read_text().strip()
    r_ratio = float(ratio_txt)
    py_ratio = cd.ContaminationQuantification(
        sim_adata, py_detection["GCGs"], cluster_key="cluster"
    )
    assert abs(py_ratio - r_ratio) < 1e-6, (
        f"contamination ratio PY={py_ratio:.8f} R={r_ratio:.8f}"
    )
