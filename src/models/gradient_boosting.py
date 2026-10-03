"""Medium model: gradient-boosted decision trees.

Boosting is medium model-class complexity: more expressive than logistic
regression, less than a sequence network. It reads the same hand-crafted
feature matrix as logistic regression, so differences in momentum signal are
attributable to the model class, not the inputs.

Interface matches the others: fit / predict_proba.

NOTE ON THE FROM-SCRATCH RULE: the brief forbids third-party libraries for
cross-validation, tuning, nested CV, and metric computation — which we honour.
It does NOT forbid a library for the learner itself. The team should still
confirm this reading with Jean; if he wants the learner from scratch too, swap
in a small from-scratch boosting implementation (regression stumps on the
negative-gradient of log-loss). A minimal from-scratch skeleton is sketched
below so the switch is cheap.

Tunable hyperparameter: n_estimators (centred grid). Depth / learning rate can
be added, but keep <=3 per the brief's advice.
"""
from __future__ import annotations

import numpy as np


class GradientBoosting:
    def __init__(self, n_estimators: int = 200, max_depth: int = 3,
                 learning_rate: float = 0.1, seed: int = 51):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self._impl = None

    def fit(self, X, y):
        # TODO(team): choose ONE of:
        # (A) library learner (confirm with Jean):
        #     from sklearn.ensemble import GradientBoostingClassifier
        #     self._impl = GradientBoostingClassifier(
        #         n_estimators=self.n_estimators, max_depth=self.max_depth,
        #         learning_rate=self.learning_rate, random_state=self.seed)
        #     self._impl.fit(X, y)
        # (B) from-scratch stumps/trees on the log-loss negative gradient
        #     (see _fit_from_scratch skeleton below).
        raise NotImplementedError(
            "Pick a boosting backend in gradient_boosting.py (see TODO)."
        )

    def predict_proba(self, X) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("Model not fitted.")
        # For (A): return self._impl.predict_proba(X)[:, 1]
        raise NotImplementedError

    # --- Optional: from-scratch skeleton, if the learner must be from scratch ---
    def _fit_from_scratch(self, X, y):
        """Functional-gradient boosting on binary log-loss.

        F_0 = log(p/(1-p)); at each round fit a shallow regression tree to the
        residual r = y - sigmoid(F), then F += lr * tree(X). Store trees and
        the init. A depth-limited CART regressor is the only extra piece.
        """
        raise NotImplementedError("Implement if from-scratch learner required.")


PARAM_GRID = {"n_estimators": [100, 200, 400]}


def make_model(**params) -> GradientBoosting:
    return GradientBoosting(**params)
