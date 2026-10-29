"""
Streaming Peaks-Over-Threshold (SPOT) thresholding for anomaly detection.

A compact, clean-room re-implementation of the SPOT method
(Siffer et al., "Anomaly Detection in Streams with Extreme Value Theory",
KDD 2017), built on top of SciPy's Generalized Pareto Distribution fit.

This file is original work and is released under the MIT license of this
repository (see ../LICENSE). It shares no source code with the authors'
GPLv3 reference implementation; only the published algorithm is followed.

The public surface is intentionally small and matches the callers in
``lib/evaluate.py``::

    s = SPOT(q)
    s.fit(init_data, data)                 # 1-D numpy arrays
    s.initialize(level=0.99, min_extrema=False)
    ret = s.run(dynamic=True)              # -> {"thresholds": [...], "alarms": [...]}
    thr = s.extreme_quantile               # float, in the original data space
"""

import numpy as np

try:  # SciPy (BSD-licensed) provides the GPD MLE fit
    from scipy.stats import genpareto
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - graceful fallback if SciPy is missing
    genpareto = None
    _HAS_SCIPY = False


class SPOT:
    """Streaming Peaks-Over-Threshold estimator for a univariate score stream.

    Parameters
    ----------
    q : float
        Target probability of a value exceeding the extreme quantile
        (the detection risk level), e.g. ``1e-3``.
    """

    def __init__(self, q=1e-4, **kwargs):
        self.q = float(q)
        # extra kwargs (e.g. a legacy ``device`` argument) are accepted and ignored
        self.init_data = None
        self.data = None
        self.n = 0                  # number of (non-anomalous) observations seen
        self.init_threshold = None  # base threshold t (a high empirical quantile)
        self.peaks = None           # excesses above t used to fit the GPD tail
        self.extreme_quantile = None
        self._gamma = 0.0           # GPD shape parameter
        self._sigma = 1.0           # GPD scale parameter
        self._sign = 1.0            # +1 upper tail, -1 lower tail (min_extrema)

    # ------------------------------------------------------------------ data
    def fit(self, init_data, data=None):
        """Provide the calibration window ``init_data`` and the stream ``data``."""
        self.init_data = np.asarray(init_data, dtype=float).ravel()
        self.data = None if data is None else np.asarray(data, dtype=float).ravel()
        return self

    def add(self, data):
        """Append more points to the stream (kept for API parity)."""
        d = np.asarray(data, dtype=float).ravel()
        self.data = d if self.data is None else np.concatenate([self.data, d])
        return self

    # ------------------------------------------------------------- GPD model
    def _fit_gpd(self):
        peaks = self.peaks
        if peaks is None or peaks.size < 2 or np.allclose(peaks, 0.0):
            # too few excesses to estimate a tail: fall back to an exponential tail
            mean = float(peaks.mean()) if (peaks is not None and peaks.size) else 1e-8
            self._gamma, self._sigma = 0.0, max(mean, 1e-8)
            return
        if _HAS_SCIPY:
            try:
                c, _loc, scale = genpareto.fit(peaks, floc=0.0)
                self._gamma = float(c)
                self._sigma = float(scale) if scale > 0 else 1e-8
                return
            except Exception:
                pass  # fall through to the moment-based fallback below
        # method-of-moments fallback (also used when SciPy is unavailable)
        m = float(peaks.mean())
        v = float(peaks.var())
        if v <= 0:
            self._gamma, self._sigma = 0.0, max(m, 1e-8)
        else:
            self._gamma = 0.5 * (1.0 - (m * m) / v)
            self._sigma = max(0.5 * m * ((m * m) / v + 1.0), 1e-8)

    def _calc_quantile(self):
        t = self.init_threshold
        Nt = max(self.peaks.size if self.peaks is not None else 0, 1)
        n = max(self.n, 1)
        r = max(self.q * n / Nt, 1e-12)
        g, s = self._gamma, self._sigma
        if abs(g) < 1e-8:
            return t - s * np.log(r)
        return t + (s / g) * (pow(r, -g) - 1.0)

    # --------------------------------------------------------------- fitting
    def initialize(self, level=0.98, min_extrema=False, verbose=False):
        """Set the base threshold and fit the initial tail on ``init_data``."""
        if self.init_data is None:
            raise ValueError("call fit(init_data, data) before initialize()")
        self._sign = -1.0 if min_extrema else 1.0
        x = self._sign * self.init_data
        self.n = int(x.size)
        self.init_threshold = float(np.quantile(x, level))
        self.peaks = x[x > self.init_threshold] - self.init_threshold
        self._fit_gpd()
        self.extreme_quantile = self._sign * self._calc_quantile()
        if verbose:
            print(f"[SPOT] t={self.init_threshold:.6g} Nt={self.peaks.size} "
                  f"gamma={self._gamma:.4g} sigma={self._sigma:.4g} "
                  f"zq={self.extreme_quantile:.6g}")
        return self

    # ----------------------------------------------------------- streaming run
    def run(self, dynamic=True, with_alarm=True):
        """Stream over ``data`` and return per-step thresholds and alarm indices.

        Returns a dict with ``"thresholds"`` (one value per input point, in the
        original data space) and ``"alarms"`` (indices flagged as anomalies).
        """
        if self.extreme_quantile is None:
            raise ValueError("call initialize() before run()")
        if self.data is None:
            return {"thresholds": [self.extreme_quantile], "alarms": []}

        x = self._sign * self.data
        zq = self._sign * self.extreme_quantile  # work in the signed space
        thresholds, alarms = [], []
        for i, xi in enumerate(x):
            if xi > zq:
                if with_alarm:
                    alarms.append(i)
                # anomaly: excluded from the normal-regime tail model
            elif xi > self.init_threshold:
                self.peaks = np.append(self.peaks, xi - self.init_threshold)
                self.n += 1
                if dynamic:
                    self._fit_gpd()
                    zq = self._calc_quantile()
            else:
                self.n += 1
            thresholds.append(self._sign * zq)

        self.extreme_quantile = self._sign * zq
        return {"thresholds": thresholds, "alarms": alarms}
