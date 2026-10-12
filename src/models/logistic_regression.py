"""Simple model: L2-regularised logistic regression.

The learner is scikit-learn's LogisticRegression (L-BFGS), for speed: it
converges in tens of iterations where plain gradient descent took hundreds
of full passes over ~1.5M rows. Only the learner comes from the library --
cross-validation, tuning and metrics stay hand-written (cross_validation.py,
nested_cv.py, metrics.py); nothing here uses sklearn's CV or metrics.

Objective (on standardised features, intercept not penalised):
    mean log-loss + lam / 2 * ||w||^2
which is sklearn's  C * sum(log-loss) + ||w||^2 / 2  with C = 1 / (lam * n).
The penalty is per mean loss, so lam means the same thing at any n. (The
original version divided it by n, so with ~1e5-1e6 rows every lam on the grid
gave the same, effectively unregularised, model.)

Tunable hyperparameter: L2 strength `lam` (the brief needs at least one,
with the middle grid value usually chosen). Grid chosen from a scan on 4 of
exp1's per-tour outer folds: held-out log-loss is best at lam 1e-3 to 1e-2
and clearly worse at 0.1.
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
    def __init__(self, lam: float = 0.01, lr: float = 0.1,
                 n_iter: int = 500, standardise: bool = True):
        self.lam = lam
        self.lr = lr            # unused since the switch to L-BFGS; kept for old calls
        self.n_iter = n_iter    # max L-BFGS iterations
        self.standardise = standardise
        self.w = None
        self.b = 0.0
        self._mu = None
        self._sd = None
        self._const = None

    def _prep(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Standardise with the training mean/sd.

        A feature that is constant in the training data -- all values equal
        to within floating-point rounding, not just sd == 0 exactly -- is set
        to 0 in training AND prediction. Otherwise rounding noise (sd ~1e-12)
        gets divided into, and a feature that never varied in training can
        produce standardised test values of ~1e12 (it happened for the rally
        features in a training block from 2012-2015, when no rally length
        was recorded).
        """
        X = np.asarray(X, dtype=float)
        if not self.standardise:
            return X
        if fit:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0)
            spread = X.max(axis=0) - X.min(axis=0)
            self._const = spread <= 1e-9 * np.maximum(1.0, np.abs(self._mu))
            self._sd[self._const] = 1.0
        Z = (X - self._mu) / self._sd
        Z[:, self._const] = 0.0
        return Z

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = self._prep(X, fit=True)
        model = _SkLogisticRegression(C=1.0 / (self.lam * len(X)), solver="lbfgs",
                                      max_iter=self.n_iter)
        model.fit(X, np.asarray(y, dtype=int))
        self.w = model.coef_.ravel()
        self.b = float(model.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._prep(X, fit=False)
        return _sigmoid(X @ self.w + self.b)


# Hyperparameter grid, centred so the middle value should win most often.
PARAM_GRID = {"lam": [0.001, 0.01, 0.1]}

# Per-tour grids for experiments that fit men and women separately (exp1).
# In exp1 the men's models chose the middle lam (0.01) in 8/10 outer folds,
# the women's chose the top of the grid (0.1) in 10/10: women's points are
# harder to predict and want more regularisation, so their grid is centred
# on 0.1.
PARAM_GRID_BY_TOUR = {
    "men": {"lam": [0.001, 0.01, 0.1]},
    "women": {"lam": [0.03, 0.1, 0.3]},
}


def make_model(**params) -> LogisticRegression:
    return LogisticRegression(**params)
