"""Smoothing-spline fit of the entropy-vs-expression curve.

scCDC fits the expected entropy as a smooth function of the mean
(log1p) expression with R's ``smooth.spline(x, y, spar = 1)``.

R's ``smooth.spline`` is a *penalized cubic B-spline* (a natural smoothing
spline).  The smoothing parameter ``lambda`` is derived from ``spar`` by

    lambda = ratio * 256 ** (3 * spar - 1)

where ``ratio = tr(X'WX) / tr(Omega)`` (``X`` the B-spline design at the
unique x, ``Omega`` the integrated-squared-second-derivative penalty).
With ``spar = 1`` this is ``lambda = ratio * 256**2`` -- a very strong
penalty, so the fit is close to (but not exactly) the least-squares
line through the entropy/expression cloud.

We reproduce this with a penalized cubic B-spline solved in closed form.
A knot at every distinct x value (R's default for moderate n) plus the
R ``spar -> lambda`` mapping gives a curve numerically very close to
``smooth.spline``; the residual difference (typically < 1e-3 in entropy
units) comes from R's exact knot-thinning heuristic for large n and is
documented in the test suite.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

__all__ = ["SmoothSpline", "smooth_spline"]

_R_NKNOTS_SMALL = 49  # below this many unique x, R uses all of them


def _r_nknots(n_unique: int) -> int:
    """Number of interior knots R's smooth.spline picks (`.nknots.smspl`)."""
    if n_unique < _R_NKNOTS_SMALL + 1:
        return n_unique
    a1, a2, a3, a4 = np.log2(50), np.log2(100), np.log2(140), np.log2(200)
    if n_unique < 200:
        return int(2 ** (a1 + (a2 - a1) * (n_unique - 50) / 150))
    if n_unique < 800:
        return int(2 ** (a2 + (a3 - a2) * (n_unique - 200) / 600))
    if n_unique < 3200:
        return int(2 ** (a3 + (a4 - a3) * (n_unique - 800) / 2400))
    return int(200 + (n_unique - 3200) ** 0.2)


class SmoothSpline:
    """A fitted penalized cubic B-spline (mirrors R ``smooth.spline``).

    Parameters
    ----------
    x, y
        Training points.  Repeated ``x`` values are aggregated (weighted
        mean of ``y``) exactly as R does.
    spar
        Smoothing parameter on R's scale (``smooth.spline`` ``spar``).
        ``spar = 1`` is scCDC's setting.
    """

    degree = 3

    def __init__(self, x, y, spar: float = 1.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        order = np.argsort(x, kind="mergesort")
        x, y = x[order], y[order]

        # aggregate ties (R collapses duplicate x, weight = multiplicity)
        ux, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        uy = np.zeros_like(ux)
        np.add.at(uy, inv, y)
        uy /= cnt
        w = cnt.astype(float)

        self.spar = float(spar)
        self.x_min, self.x_max = ux[0], ux[-1]
        self._fit(ux, uy, w)

    # ------------------------------------------------------------------
    def _fit(self, x, y, w):
        n = x.size
        if n < 4:  # too few points -> linear/least-squares fallback
            self._linear = np.polyfit(x, y, min(1, n - 1)) if n > 1 \
                else np.array([0.0, y[0] if n else 0.0])
            self._spline = None
            return
        self._linear = None

        rng = self.x_max - self.x_min
        if rng <= 0:
            self._linear = np.array([0.0, y.mean()])
            self._spline = None
            return

        # ---- knot placement (R's quantile-based interior knots) -------
        nk = _r_nknots(n)
        if nk >= n:
            interior = x[1:-1]
        else:
            qs = np.linspace(0, 1, nk)
            interior = np.quantile(x, qs)[1:-1]
        k = self.degree
        t = np.concatenate((
            np.repeat(self.x_min, k + 1),
            interior,
            np.repeat(self.x_max, k + 1),
        ))
        self.knots = t
        m = len(t) - k - 1  # number of B-spline coefficients

        # ---- B-spline design matrix B (n x m) -------------------------
        B = np.empty((n, m))
        eye = np.eye(m)
        for j in range(m):
            B[:, j] = BSpline(t, eye[j], k, extrapolate=True)(x)

        # ---- penalty Omega = integral of (B'')^2 ----------------------
        Omega = self._penalty_matrix(t, k, m)

        # ---- spar -> lambda mapping (R smooth.spline) -----------------
        W = np.diag(w)
        BtWB = B.T @ W @ B
        r = np.trace(BtWB)
        tr_omega = np.trace(Omega)
        ratio = r / tr_omega if tr_omega > 0 else 1.0
        lam = ratio * 256.0 ** (3.0 * self.spar - 1.0)
        self.lam = lam

        # ---- solve penalized least squares ----------------------------
        A = BtWB + lam * Omega
        rhs = B.T @ W @ y
        coef = np.linalg.solve(A, rhs)
        self.coef = coef
        self._spline = BSpline(t, coef, k, extrapolate=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _penalty_matrix(t, k, m):
        """Integrated squared 2nd-derivative penalty for the B-spline basis."""
        Omega = np.zeros((m, m))
        eye = np.eye(m)
        d2 = [BSpline(t, eye[j], k, extrapolate=True).derivative(2)
              for j in range(m)]
        # 3-point Gauss-Legendre quadrature on each knot span
        gx = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        gw = np.array([5 / 9, 8 / 9, 5 / 9])
        uknots = np.unique(t)
        for a, b in zip(uknots[:-1], uknots[1:]):
            if b <= a:
                continue
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            pts = mid + half * gx
            vals = np.array([d2[j](pts) for j in range(m)])  # m x 3
            for q in range(3):
                v = vals[:, q]
                Omega += gw[q] * half * np.outer(v, v)
        return Omega

    # ------------------------------------------------------------------
    def predict(self, x):
        """Evaluate the fitted curve at ``x`` (clamped to the fit range)."""
        x = np.asarray(x, dtype=float)
        xc = np.clip(x, self.x_min, self.x_max)
        if self._spline is None:
            return np.polyval(self._linear, xc)
        return self._spline(xc)

    __call__ = predict


def smooth_spline(x, y, spar: float = 1.0) -> SmoothSpline:
    """Fit a :class:`SmoothSpline` (functional alias of the constructor)."""
    return SmoothSpline(x, y, spar=spar)
