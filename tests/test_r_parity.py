"""R-parity tests — pysccdc vs the R package scCDC (live R).

The R driver (:file:`r_reference_driver.R`) runs scCDC 1.4 on the *same*
synthetic clustered count matrix that :func:`pysccdc.datasets.simulate_contaminated`
produces (4 clusters x 200 cells, 120 genes, 4 deliberately-spiked
contaminating genes), after ``set.seed(R_SEED)``. R's bootstrap ignores that
seed -- Seurat's ``subset`` calls ``WhichCells(seed = 1)``, so scCDC always fits
from ``set.seed(1)`` -- and pysccdc's default ``random_state=1`` draws the same
genes, so the comparisons are exact:

* per-gene / per-cluster **Shannon entropy** — identical;
* the detected **GCG list** and its order — identical;
* the per-gene **entropy divergence** — equal to 1e-9 (bit-identical when
  R's BLAS does not use FMA instructions; OpenBLAS' FMA kernels change the
  last bits of the spline solve);
* the **corrected count matrix** and thresholds — identical;
* the dataset **contamination ratio** — identical.

Set ``PYSCCDC_RSCRIPT`` to an ``Rscript`` with scCDC installed (default: the
``Rscript`` on ``PATH``); the tests skip when it is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pysccdc as cd
from pysccdc._anndata import counts_matrix, get_clusters
from pysccdc.entropy import matrix_entropy

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
RSCRIPT = os.environ.get("PYSCCDC_RSCRIPT") or shutil.which("Rscript")


def _r_available() -> bool:
    if not R_DRIVER.exists() or not RSCRIPT or not Path(RSCRIPT).exists():
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
    reason="Rscript with scCDC not available (set PYSCCDC_RSCRIPT).",
)

RANDOM_STATE = 0      # the simulated data
R_SEED = 0            # set.seed() before the R calls; scCDC resets it to 1


def run_r_reference(sim_adata, out_dir: Path, seed: int = R_SEED) -> Path:
    """Run the R driver on ``sim_adata``; outputs land in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = out_dir / "counts.csv"
    clusters_csv = out_dir / "clusters.csv"
    pd.DataFrame(sim_adata.X.T.astype(int), index=sim_adata.var_names,
                 columns=sim_adata.obs_names).to_csv(counts_csv)
    pd.DataFrame({"cell": sim_adata.obs_names,
                  "cluster": sim_adata.obs["cluster"].astype(str)}).to_csv(clusters_csv, index=False)
    res = subprocess.run([RSCRIPT, str(R_DRIVER), str(counts_csv), str(clusters_csv), str(out_dir), str(seed)],
                         capture_output=True, text=True, timeout=1200)
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    counts_csv.unlink()
    clusters_csv.unlink()
    return out_dir


def read_table(out_dir: Path, name: str, extra=()):
    genes = (out_dir / "genes.txt").read_text().split()
    cols = (out_dir / "clusters.txt").read_text().split() + list(extra)
    v = np.fromfile(out_dir / name).reshape(len(cols), len(genes)).T
    return pd.DataFrame(v, index=genes, columns=cols)


@pytest.fixture(scope="module")
def sim_adata():
    """The shared synthetic dataset."""
    return cd.datasets.simulate_contaminated(random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory, sim_adata):
    """Run the R scCDC reference once on the synthetic dataset."""
    return run_r_reference(sim_adata, tmp_path_factory.mktemp("sccdc_R"))


@pytest.fixture(scope="module")
def py_detection(sim_adata):
    return cd.ContaminationDetection(
        sim_adata, cluster_key="cluster", min_cell=50, return_full=True,
    )


# ----------------------------------------------------------------------
def test_entropy_bit_exact(r_reference, sim_adata):
    """Shannon entropy of count distributions is identical to R."""
    r_ent = read_table(r_reference, "entropy.bin")
    counts, genes, _ = counts_matrix(sim_adata)
    labels = np.asarray(get_clusters(sim_adata, "cluster"))
    py_ent = pd.DataFrame({cl: matrix_entropy(counts[:, labels == cl]) for cl in r_ent.columns}, index=genes)
    assert np.array_equal(r_ent.values, py_ent.loc[r_ent.index, r_ent.columns].values)


def test_detected_gcgs_match(r_reference, py_detection):
    """The detected GCG set is identical to R scCDC."""
    r_gcgs = set(
        (r_reference / "gcgs.txt").read_text().split()
    )
    py_gcgs = set(py_detection["GCGs"])
    assert py_gcgs == r_gcgs, f"R={sorted(r_gcgs)} PY={sorted(py_gcgs)}"
    assert (r_reference / "gcgs.txt").read_text().split() == py_detection["GCGs"]


def test_all_spiked_genes_detected(r_reference, py_detection, sim_adata):
    """Both R and pysccdc recover every deliberately-spiked GCG."""
    truth = set(sim_adata.uns["true_GCGs"])
    r_gcgs = set((r_reference / "gcgs.txt").read_text().split())
    py_gcgs = set(py_detection["GCGs"])
    assert truth <= r_gcgs
    assert truth <= py_gcgs


def test_entropy_divergence_matches(r_reference, py_detection):
    """Per-gene, per-cluster entropy divergence and its mean equal R's."""
    r_dist = read_table(r_reference, "distance.bin", extra=["mean_distance"])
    py_dist = py_detection["all_distance"].loc[r_dist.index, r_dist.columns]
    np.testing.assert_allclose(py_dist.to_numpy(), r_dist.to_numpy(), rtol=0, atol=1e-9)


def test_corrected_matrix_matches(r_reference, sim_adata, py_detection):
    """The decontaminated count matrix is bit-exact with R scCDC."""
    gcgs = py_detection["GCGs"]
    cor = cd.ContaminationCorrection(
        sim_adata, gcgs, cluster_key="cluster"
    )
    r_corr = pd.read_csv(r_reference / "corrected.csv", index_col=0)
    py_corr = pd.DataFrame(
        np.asarray(cor.layers["Corrected"]).T,
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
        assert np.rint(py_thr[g]) == int(r_thr[g]), (
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
    assert py_ratio == r_ratio, (
        f"contamination ratio PY={py_ratio:.8f} R={r_ratio:.8f}"
    )
