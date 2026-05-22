"""Smoke / unit tests for :mod:`pysccdc` (no R required)."""
from __future__ import annotations

import numpy as np
import pytest

import pysccdc as cd


# ----------------------------------------------------------------------
def test_vector_entropy_basics():
    # a constant vector has zero entropy
    assert cd.vector_entropy(np.zeros(50)) == 0.0
    assert cd.vector_entropy(np.full(20, 7)) == 0.0
    # two equally-frequent values -> 1 bit
    v = np.array([0, 0, 1, 1])
    assert abs(cd.vector_entropy(v) - 1.0) < 1e-12
    # four equally-frequent values -> 2 bits
    v = np.array([0, 1, 2, 3])
    assert abs(cd.vector_entropy(v) - 2.0) < 1e-12


def test_matrix_entropy_matches_rows():
    rng = np.random.default_rng(0)
    m = rng.poisson(3, size=(10, 200))
    ent = cd.matrix_entropy(m)
    assert ent.shape == (10,)
    for i in range(10):
        assert abs(ent[i] - cd.vector_entropy(m[i])) < 1e-12


def test_smooth_spline_fits_a_line():
    x = np.linspace(0, 5, 60)
    y = 2.0 * x + 1.0
    sp = cd.smooth_spline(x, y, spar=1.0)
    pred = sp.predict(x)
    # a strong-penalty smoothing spline reproduces a linear trend well
    assert np.corrcoef(pred, y)[0, 1] > 0.999


def test_simple_roc_perfect_separation():
    pos = np.array([5, 6, 7, 8])
    neg = np.array([0, 1, 2, 3])
    exp = np.concatenate([pos, neg])
    cls = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert cd.simple_roc(exp, cls) == 1.0


def test_youden_threshold_separates():
    neg = np.array([0, 0, 1, 1, 0])
    pos = np.array([10, 11, 12, 9, 13])
    thr = cd.youden_threshold(neg, pos)
    assert 1 < thr < 10


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim():
    return cd.datasets.simulate_contaminated(random_state=0)


def test_simulate_shape(sim):
    assert sim.shape == (800, 120)
    assert "cluster" in sim.obs
    assert len(sim.uns["true_GCGs"]) == 4


def test_detection_recovers_spiked_gcgs(sim):
    res = cd.ContaminationDetection(
        sim, cluster_key="cluster", min_cell=50, random_state=0
    )
    detected = set(res.index)
    truth = set(sim.uns["true_GCGs"])
    assert truth <= detected, f"missed {truth - detected}"
    assert "mean_distance" in res.columns
    # GCGs should have clearly positive entropy divergence
    assert (res["mean_distance"] > 0).all()


def test_correction_only_touches_gcgs(sim):
    res = cd.ContaminationDetection(
        sim, cluster_key="cluster", min_cell=50, random_state=0
    )
    cor = cd.ContaminationCorrection(sim, res, cluster_key="cluster")
    assert "Corrected" in cor.layers
    orig = sim.X
    corr = cor.layers["Corrected"]
    gcg_set = set(res.index)
    gpos = {g: i for i, g in enumerate(sim.var_names)}
    for j, g in enumerate(sim.var_names):
        if g in gcg_set:
            continue
        # non-GCG columns must be byte-identical (anti-over-correction)
        assert np.array_equal(orig[:, j], corr[:, j]), f"{g} changed"
    # GCG columns must not increase
    for g in gcg_set:
        j = gpos[g]
        assert (corr[:, j] <= orig[:, j]).all()
        assert (corr[:, j] >= 0).all()


def test_correction_reduces_total_counts(sim):
    res = cd.ContaminationDetection(
        sim, cluster_key="cluster", min_cell=50, random_state=0
    )
    cor = cd.ContaminationCorrection(sim, res, cluster_key="cluster")
    assert cor.layers["Corrected"].sum() < sim.X.sum()


def test_quantification_returns_ratio(sim):
    res = cd.ContaminationDetection(
        sim, cluster_key="cluster", min_cell=50, random_state=0
    )
    ratio, per_gene = cd.ContaminationQuantification(
        sim, res, cluster_key="cluster", return_per_gene=True
    )
    assert ratio > 0
    assert len(per_gene) == len(res)
    assert abs(ratio - per_gene.max()) < 1e-12


def test_detection_determinism(sim):
    a = cd.ContaminationDetection(
        sim, cluster_key="cluster", min_cell=50, random_state=0
    )
    b = cd.ContaminationDetection(
        sim, cluster_key="cluster", min_cell=50, random_state=0
    )
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(
        a["mean_distance"].to_numpy(), b["mean_distance"].to_numpy()
    )
