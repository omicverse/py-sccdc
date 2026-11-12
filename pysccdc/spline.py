"""R ``smooth.spline(x, y, spar = ...)`` and its ``predict`` method.

scCDC fits the expected entropy-vs-expression curve with ``smooth.spline(spar = 1)``. This module
follows R 4.5's ``stats`` code step by step so the fitted values match R to rounding:

* the R front end (``smooth.spline``): ties in ``x`` are pooled after rounding ``(x - mean(x)) / tol``
  with ``tol = 1e-6 * IQR(x)``; ``x`` is mapped to [0, 1]; the knots are the pooled points at
  ``seq.int(1, nx, length.out = .nknots.smspl(nx))``;
* ``sbart.c``: the ``X'WX`` and penalty bands (``stxwx.f``, ``sgram.f``, including its
  ``.3330`` constant in place of 1/3), ``lambda = ratio * 16^(6 spar - 2)`` with ``ratio`` the ratio of
  the band traces, and the banded Cholesky solve (LINPACK ``dpbfa``/``dpbsl``) in ``sslvrg.f``;
* ``bvalue.f`` for evaluation, with R's linear extrapolation outside the data range.

B-spline values (de Boor's ``bsplvb``/``bsplvd``) and ``bvalue`` are evaluated for all points at
once; only the Cholesky factorisation of the ``nk x nk`` band (``nk`` ~ 200) runs as a scalar loop,
as in LINPACK.
"""
from __future__ import annotations

import math

import numpy as np

from ._rbase import iqr, r_mean

__all__ = ["SmoothSpline", "smooth_spline"]

_K = 4
_THIRD = 0.333                     # sgram.f: ``.3330d0``


def _nknots(n: int) -> int:
    """``stats:::.nknots.smspl``."""
    if n < 50:
        return n
    a1, a2, a3, a4 = math.log2(50), math.log2(100), math.log2(140), math.log2(200)
    if n < 200:
        v = 2 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        v = 2 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        v = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        v = 200 + (n - 3200) ** 0.2
    return int(math.trunc(v))


def _seq_length_out(start: float, stop: float, n: int) -> np.ndarray:
    """``seq.int(start, stop, length.out = n)`` (R 4.5 ``do_seq``: ``start + i * by``)."""
    out = np.empty(n)
    if n > 0:
        out[0] = start
    if n > 1:
        out[-1] = stop
    if n > 2:
        by = (stop - start) / (n - 1)
        out[1:-1] = start + np.arange(1, n - 1) * by
    return out


def _bsplvb_init(x):
    return {"j": 1, "b": {1: np.ones_like(x)}, "dl": {}, "dr": {}}


def _bsplvb(t, x, left, jhigh, st):
    """de Boor's ``bsplvb``: raise the order of the values in ``st`` to ``jhigh``.

    ``t`` is 0-based; ``left`` holds the 1-based Fortran interval index for every point."""
    b, dl, dr = st["b"], st["dl"], st["dr"]
    j = st["j"]
    while j < jhigh:
        jp1 = j + 1
        dr[j] = t[left + j - 1] - x
        dl[j] = x - t[left - j]
        saved = 0.0
        for i in range(1, j + 1):
            term = b[i] / (dr[i] + dl[jp1 - i])
            b[i] = saved + dr[i] * term
            saved = dl[jp1 - i] * term
        b[jp1] = saved
        j = jp1
    st["j"] = j
    return st


def _bsplvd(t, x, left, nderiv):
    """de Boor's ``bsplvd`` for cubic splines: values and derivatives up to ``nderiv - 1``.

    Returns ``d`` with ``d[(i, m)]`` the (m-1)-th derivative of the i-th non-zero B-spline."""
    k, kp1 = _K, _K + 1
    mhigh = max(min(nderiv, k), 1)
    st = _bsplvb(t, x, left, kp1 - mhigh, _bsplvb_init(x))
    col1 = st["b"]
    d = {}
    if mhigh == 1:
        return {(i, 1): col1[i] for i in range(1, k + 1)}
    ideriv = mhigh
    for _m in range(2, mhigh + 1):
        jp1mid = 1
        for j in range(ideriv, k + 1):
            d[(j, ideriv)] = col1[jp1mid]
            jp1mid += 1
        ideriv -= 1
        st = _bsplvb(t, x, left, kp1 - ideriv, st)
    for i in range(1, k + 1):
        d[(i, 1)] = col1[i]
    a = {(i, j): (1.0 if i == j else 0.0) for i in range(1, k + 1) for j in range(1, k + 1)}
    for m in range(2, mhigh + 1):
        kp1mm = kp1 - m
        fkp1mm = float(kp1mm)
        il = left
        i = k
        for _ in range(kp1mm):
            factor = fkp1mm / (t[il + kp1mm - 1] - t[il - 1])
            for j in range(1, i + 1):
                a[(i, j)] = (a[(i, j)] - a[(i - 1, j)]) * factor
            il = il - 1
            i -= 1
        for i in range(1, k + 1):
            s = 0.0
            for j in range(max(i, m), k + 1):
                s = a[(j, i)] * d[(j, m)] + s
            d[(i, m)] = s
    return d


def _sgram(t, nk):
    """``sgram.f``: the four bands of the integrated squared second-derivative Gram matrix."""
    i = np.arange(1, nk + 1)
    left = np.maximum(i, 4)
    x0, x1 = t[i - 1], t[i]
    v0 = _bsplvd(t, x0, left, 3)
    v1 = _bsplvd(t, x1, left, 3)
    y1 = {ii: v0[(ii, 3)] for ii in range(1, 5)}
    y2 = {ii: v1[(ii, 3)] - y1[ii] for ii in range(1, 5)}
    wpt = x1 - x0
    bands = [np.zeros(nk) for _ in range(4)]
    for band in range(4):
        idx, val = [], []
        for ii in range(1, 5 - band):
            jj = ii + band
            term = wpt * ((y1[ii] * y1[jj] + (y2[ii] * y1[jj] + y2[jj] * y1[ii]) * 0.5)
                          + y2[ii] * y2[jj] * _THIRD)
            idx.append(left - 4 + ii - 1)
            val.append(term)
        # contributions reach each entry in Fortran loop order: interval first, then ii
        order = np.argsort(np.repeat(np.arange(nk), len(idx)).reshape(nk, -1).T.ravel(), kind="stable")
        np.add.at(bands[band], np.concatenate(idx)[order], np.concatenate(val)[order])
    return bands


def _stxwx(x, z, w, t, nk):
    """``stxwx.f``: ``X'W z`` and the four bands of ``X'WX`` (weights enter squared)."""
    left = np.minimum(np.searchsorted(t[:nk + 1], x, side="right"), nk)
    v = _bsplvd(t, x, left, 1)
    v = [v[(r, 1)] for r in range(1, 5)]
    w2 = w * w
    y = np.zeros(nk)
    hs = [np.zeros(nk) for _ in range(4)]
    # an entry gets its contributions in data order; points sorted by x have non-decreasing
    # ``left``, so the later rows (r = 3, 2, ...) of the 4 x 4 block reach it first
    for r in range(3, -1, -1):
        j = left - 4 + r                  # 0-based coefficient index
        np.add.at(y, j, w2 * z * v[r])
        for band in range(4 - r):
            if band == 0:
                np.add.at(hs[0], j, w2 * (v[r] * v[r]))
            else:
                np.add.at(hs[band], j, w2 * v[r] * v[r + band])
    return y, hs


def _dpbfa_dpbsl(abd, b):
    """LINPACK ``dpbfa`` + ``dpbsl`` for a symmetric band matrix with 3 super-diagonals.

    ``abd`` is the Fortran (4, n) band array, row 3 (0-based) the diagonal. Scalar loops as in
    LINPACK; ``ddot``/``daxpy`` of length <= 3 accumulate in order."""
    m = 3
    n = abd.shape[1]
    a = abd.tolist()
    for j in range(n):
        s = 0.0
        ik = m
        jk = max(j - m, 0)
        mu = max(m + 1 - j - 1 + 1, 1) - 1          # 0-based row of the first used entry
        for kk in range(mu, m):
            dot = 0.0
            for l in range(kk - mu):
                dot = dot + a[ik + l][jk] * a[mu + l][j]
            tt = a[kk][j] - dot
            tt = tt / a[m][jk]
            a[kk][j] = tt
            s = s + tt * tt
            ik -= 1
            jk += 1
        s = a[m][j] - s
        if s <= 0.0:
            raise np.linalg.LinAlgError(f"smooth.spline: band matrix not positive definite at {j + 1}")
        a[m][j] = math.sqrt(s)
    bb = list(map(float, b))
    for k in range(n):
        lm = min(k, m)
        la = m - lm
        lb = k - lm
        dot = 0.0
        for l in range(lm):
            dot = dot + a[la + l][k] * bb[lb + l]
        bb[k] = (bb[k] - dot) / a[m][k]
    for k in range(n - 1, -1, -1):
        lm = min(k, m)
        la = m - lm
        lb = k - lm
        bb[k] = bb[k] / a[m][k]
        tt = -bb[k]
        for l in range(lm):
            bb[lb + l] = bb[lb + l] + tt * a[la + l][k]
    return np.asarray(bb)


def _bvalue(t, coef, nk, x, jderiv=0):
    """``bvalue.f`` (cubic) at points ``x`` inside [t(4), t(nk+1)]."""
    nkk = nk + _K
    i = np.searchsorted(t[:nkk], x, side="right")       # largest 1-based i with t(i) <= x
    at_end = (x == t[nk]) & (t[nk] == t[nkk - 1])
    i = np.where(at_end, nk, i)
    i = np.minimum(i, nk)
    dl = {j: x - t[i + 1 - j - 1] for j in range(1, _K)}
    dr = {j: t[i + j - 1] - x for j in range(1, _K)}
    aj = {jc: coef[i - _K + jc - 1] for jc in range(1, _K + 1)}
    for j in range(1, jderiv + 1):
        kmj = _K - j
        fkmj = float(kmj)
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = ((aj[jj + 1] - aj[jj]) / (dl[ilo] + dr[jj])) * fkmj
            ilo -= 1
    for j in range(jderiv + 1, _K):
        kmj = _K - j
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = (aj[jj + 1] * dl[ilo] + aj[jj] * dr[jj]) / (dl[ilo] + dr[jj])
            ilo -= 1
    return aj[1]


class SmoothSpline:
    """A fitted ``smooth.spline(x, y, spar = spar)`` (no weights), with R's ``predict``."""

    def __init__(self, x, y, spar: float = 1.0):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("missing or infinite values in inputs are not allowed")
        n = x.size
        tol = 1e-6 * iqr(x)
        if not (np.isfinite(tol) and tol > 0):
            raise ValueError("'tol' must be strictly positive and finite (IQR(x) is 0)")
        xx = np.rint((x - r_mean(x)) / tol)
        iox = np.argsort(x, kind="stable")
        xxs = xx[iox]
        nd = np.concatenate(([True], xxs[:-1] < xxs[1:]))
        ux = x[iox][nd]
        nx = ux.size
        if nx <= 3:
            raise ValueError("need at least four unique 'x' values")
        if nx == n:
            wbar = np.ones(nx)
            ybar = y[iox]
        else:
            grp = np.cumsum(nd) - 1                       # group of each sorted point
            g = np.empty(n, dtype=np.int64)
            g[iox] = grp
            wbar = np.bincount(g, minlength=nx).astype(np.float64)
            sy = np.zeros(nx, dtype=np.longdouble)
            np.add.at(sy, g, y.astype(np.longdouble))     # R sum(): long double, original order
            ybar = sy.astype(np.float64) / np.where(wbar > 0, wbar, 1.0)
        r_ux = ux[-1] - ux[0]
        xbar = (ux - ux[0]) / r_ux
        nknots = _nknots(nx)
        sel = _seq_length_out(1.0, float(nx), nknots).astype(np.int64) - 1
        knot = np.concatenate((np.repeat(xbar[0], 3), xbar[sel], np.repeat(xbar[-1], 3)))
        nk = nknots + 2
        ws = np.sqrt(wbar)
        sg = _sgram(knot, nk)
        xwy, hs = _stxwx(xbar, ybar, ws, knot, nk)
        t1 = np.cumsum(hs[0][2:nk - 3])[-1] if nk > 5 else 0.0
        t2 = np.cumsum(sg[0][2:nk - 3])[-1] if nk > 5 else 0.0
        ratio = t1 / t2
        lam = ratio * (16.0 ** (spar * 6.0 - 2.0))
        abd = np.zeros((4, nk))
        abd[3] = hs[0] + lam * sg[0]
        abd[2, 1:] = hs[1][:-1] + lam * sg[1][:-1]
        abd[1, 2:] = hs[2][:-2] + lam * sg[2][:-2]
        abd[0, 3:] = hs[3][:-3] + lam * sg[3][:-3]
        self.coef = _dpbfa_dpbsl(abd, xwy)
        self.knot, self.nk = knot, nk
        self.min, self.range = ux[0], r_ux
        self.spar, self.ratio, self.lambda_ = float(spar), float(ratio), float(lam)
        self.x, self.w, self.yin = ux, wbar, ybar

    def _eval(self, xs, deriv):
        return _bvalue(self.knot, self.coef, self.nk, xs, deriv)

    def predict(self, x, deriv: int = 0):
        """``predict(fit, x)$y`` (R ``predict.smooth.spline.fit``), linear outside the data."""
        x = np.asarray(x, dtype=np.float64)
        xs = (x - self.min) / self.range
        left, right = xs < 0, xs > 1
        inside = ~(left | right)
        y = xs.copy()
        if inside.any():
            y[inside] = self._eval(xs[inside], deriv)
        if (left | right).any():
            xr = np.array([self.min, self.min + self.range])
            xrs = (xr - self.min) / self.range
            if deriv == 0:
                end = self._eval(xrs, 0)
                slope = (self._eval(xrs, 1) / self.range) * self.range
                y[left] = end[0] + slope[0] * (xs[left] - 0)
                y[right] = end[1] + slope[1] * (xs[right] - 1)
            elif deriv == 1:
                slope = (self._eval(xrs, 1) / self.range) * self.range
                y[left], y[right] = slope[0], slope[1]
            else:
                y[left | right] = 0.0
        if deriv > 0:
            y = y / (self.range ** deriv)
        return y

    __call__ = predict


def smooth_spline(x, y, spar: float = 1.0) -> SmoothSpline:
    """Fit a :class:`SmoothSpline` (functional alias of the constructor)."""
    return SmoothSpline(x, y, spar=spar)
