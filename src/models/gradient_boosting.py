"""Medium model: gradient-boosted decision trees (LightGBM).

Boosting is medium model-class complexity: more expressive than logistic
regression, less than a sequence network. It reads the same hand-crafted
feature matrix as logistic regression, so differences in momentum signal are
attributable to the model class, not the inputs.

Interface matches the others: fit / predict_proba.

FROM-SCRATCH RULE: the brief forbids third-party libraries for
cross-validation, tuning, nested CV, and metric computation -- which we
honour (cross_validation.py, nested_cv.py, metrics.py). Only the learner
comes from LightGBM; none of its CV, early stopping or metrics are used.

Tunable hyperparameter: n_estimators (centred grid). Depth / learning rate can
be added, but keep <=3 per the brief's advice.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier


class GradientBoosting:
    def __init__(self, n_estimators: int = 40, max_depth: int = 3,
                 learning_rate: float = 0.1, seed: int = 51):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self._impl = None

    def fit(self, X, y):
        self._impl = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=2 ** self.max_depth,   # a full tree of that depth
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbose=-1,
        )
        self._impl.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("Model not fitted.")
        return self._impl.predict_proba(np.asarray(X, dtype=float))[:, 1]


# Centred on a scan of 4 exp1 outer folds per tour (lr 0.1, depth 3): held-out
# log-loss is best at ~50 trees for men and ~25 for women, and clearly worse
# at 200+; the old grid [100, 200, 400] picked its lowest value in every fold.
PARAM_GRID = {"n_estimators": [20, 40, 80]}


def make_model(**params) -> GradientBoosting:
    return GradientBoosting(**params)
