"""Simple model: L2-regularised logistic regression via gradient descent.

Implemented from scratch (no sklearn) so it can stand as the clearly-documented
simple baseline. Tunable hyperparameter: L2 strength `lam` (the brief needs at
least one, with the middle grid value usually chosen).
"""
from __future__ import annotations

import numpy as np


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
        self.lr = lr
        self.n_iter = n_iter
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
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        y = np.asarray(y, dtype=float)
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            p = _sigmoid(z)
            err = p - y
            grad_w = X.T @ err / n + self.lam * self.w / n
            grad_b = float(np.mean(err))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._prep(X, fit=False)
        return _sigmoid(X @ self.w + self.b)


# Hyperparameter grid, centred so the middle value should win most often.
PARAM_GRID = {"lam": [0.1, 1.0, 10.0]}


def make_model(**params) -> LogisticRegression:
    return LogisticRegression(**params)
