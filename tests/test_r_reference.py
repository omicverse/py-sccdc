"""pysccdc against stored scCDC 1.4 output (no R needed).

``tests/data/r_sim`` holds what ``r_reference_driver.R`` wrote for the simulated
dataset (``simulate_contaminated(random_state=0)``, ``set.seed(0)``), run with
R 4.5.3 and an OpenBLAS without FMA kernels (``OPENBLAS_CORETYPE=Nehalem``), so
every number is compared exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pysccdc as cd
from pysccdc._anndata import counts_matrix, get_clusters
from pysccdc.entropy import matrix_entropy

REF = Path(__file__).parent / "data" / "r_sim"


def _table(name, extra=()):
    genes = (REF / "genes.txt").read_text().split()
    cols = (REF / "clusters.txt").read_text().split() + list(extra)
    v = np.fromfile(REF / name).reshape(len(cols), len(genes)).T
    return pd.DataFrame(v, index=genes, columns=cols)


@pytest.fixture(scope="module")
def sim():
    return cd.datasets.simulate_contaminated(random_state=0)


@pytest.fixture(scope="module")
def det(sim):
    return cd.ContaminationDetection(sim, cluster_key="cluster", min_cell=50, return_full=True)


def test_entropy(sim):
    ref = _table("entropy.bin")
    counts, genes, _ = counts_matrix(sim)
    labels = get_clusters(sim, "cluster")
    got = np.column_stack([matrix_entropy(counts[:, labels == c]) for c in ref.columns])
    assert np.array_equal(got, ref.loc[list(genes)].to_numpy())


def test_divergence_and_gcgs(det):
    ref = _table("distance.bin", extra=["mean_distance"])
    got = det["all_distance"].loc[ref.index, ref.columns]
    assert np.array_equal(got.to_numpy(), ref.to_numpy())
    assert det["GCGs"] == (REF / "gcgs.txt").read_text().split()


def test_correction_and_ratio(sim, det):
    gcgs = det["GCGs"]
    out = cd.ContaminationCorrection(sim, gcgs, cluster_key="cluster")
    pos = [list(sim.var_names).index(g) for g in gcgs]
    got = np.asarray(out.layers["Corrected"])[:, pos].T
    ref = np.fromfile(REF / "corrected_gcg.bin", dtype=np.int32).reshape(len(gcgs), -1)
    assert np.array_equal(got, ref)
    thr = pd.read_csv(REF / "thresholds.csv").set_index("gene")["threshold"]
    assert all(np.rint(out.uns["sccdc"]["thresholds"][g]) == thr[g] for g in gcgs)
    ratio = cd.ContaminationQuantification(sim, gcgs, cluster_key="cluster")
    assert ratio == float((REF / "ratio.txt").read_text())
