"""Simple model: L2-regularised logistic regression.

The learner is scikit-learn's LogisticRegression (L-BFGS), for speed: it
converges in tens of iterations where plain gradient descent took hundreds
of full passes over ~1.5M rows. Only the learner comes from the library --
cross-validation, tuning and metrics stay hand-written (cross_validation.py,
nested_cv.py, metrics.py); nothing here uses sklearn's CV or metrics.

Same objective as the original gradient-descent version:
    mean log-loss + lam / (2n) * ||w||^2   (intercept not penalised)
which is sklearn's  C * sum(log-loss) + ||w||^2 / 2  with C = 1 / lam.
Tunable hyperparameter: L2 strength `lam` (the brief needs at least one,
with the middle grid value usually chosen).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression as _SkLogisticRegression


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    def __init__(self, lam: float = 1.0, lr: float = 0.1,
                 n_iter: int = 500, standardise: bool = True):
        self.lam = lam
        self.lr = lr            # unused since the switch to L-BFGS; kept for old calls
        self.n_iter = n_iter    # max L-BFGS iterations
        self.standardise = standardise
        self.w = None
        self.b = 0.0
        self._mu = None
        self._sd = None

    def _prep(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.standardise:
            return X
        if fit:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0)
            self._sd[self._sd == 0] = 1.0
        return (X - self._mu) / self._sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = self._prep(X, fit=True)
        model = _SkLogisticRegression(C=1.0 / self.lam, solver="lbfgs",
                                      max_iter=self.n_iter)
        model.fit(X, np.asarray(y, dtype=int))
        self.w = model.coef_.ravel()
        self.b = float(model.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._prep(X, fit=False)
        return _sigmoid(X @ self.w + self.b)


# Hyperparameter grid, centred so the middle value should win most often.
PARAM_GRID = {"lam": [0.1, 1.0, 10.0]}


def make_model(**params) -> LogisticRegression:
    return LogisticRegression(**params)
