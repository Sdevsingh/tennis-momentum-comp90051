"""Nested cross-validation + hyperparameter tuning, from scratch (Section 2.f).

Outer loop (cross_validation.py) estimates generalisation. Inside each outer
training fold, this inner loop picks the best hyperparameter by its own CV.
No third-party tuning (GridSearchCV etc.) is permitted.

The brief wants, per hyperparameter: >=3 candidate values, and the grid
centred so the MIDDLE value is usually selected. If the extreme value keeps
winning, widen the grid in that direction and re-run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cross_validation import kfold_match_splits, expanding_window_splits, iter_fold_indices
from .config import N_INNER_FOLDS
from . import metrics


def _fit_rows(mask: np.ndarray, match_ids: np.ndarray, ordered_match_ids=None) -> np.ndarray:
    """Row indices of `mask`, oldest match first when `ordered_match_ids` is given.

    Models get their training rows in time order, so anything a model holds
    out internally from the END of its rows (the RNN's early stopping) is a
    later block, like the inner CV. Rows within a match keep playing order.
    """
    idx = np.flatnonzero(mask)
    if ordered_match_ids is not None:
        pos = {m: i for i, m in enumerate(ordered_match_ids)}
        uniq, inv = np.unique(match_ids[idx], return_inverse=True)
        rank = np.array([pos[m] for m in uniq])[inv]
        idx = idx[np.argsort(rank, kind="stable")]
    return idx


@dataclass
class TuningResult:
    best_params: dict
    selection_counts: Counter  # how often each value won (sanity-check centring)
    inner_scores: dict


def grid_search(make_model: Callable,
                param_grid: dict[str, list],
                X: np.ndarray,
                y: np.ndarray,
                match_ids: np.ndarray,
                score_name: str = "log_loss",
                k_inner: int = N_INNER_FOLDS,
                seed: int = 51,
                ordered_match_ids=None) -> TuningResult:
    """Exhaustive grid search by inner CV over matches.

    With `ordered_match_ids` (the training matches in time order), the inner
    folds are forward-chaining: the matches are cut into k_inner + 1 time
    blocks and fold r trains on blocks 1..r and validates on block r + 1, so
    tuning never validates on matches played before the ones it trained on.
    Without it (e.g. simulated data with no time axis), plain random k-fold.

    make_model(**params) must return an object with .fit(X, y) and
    .predict_proba(X) -> P(class 1). Lower-is-better for log_loss/brier;
    AUC/accuracy are handled by sign.

    Keep grids small (<=3 hyperparameters, 3-5 values each) or inner CV gets
    expensive — the brief warns about exactly this.
    """
    lower_is_better = score_name in ("log_loss", "brier")
    score_fn = metrics.ALL_METRICS[score_name]

    if ordered_match_ids is not None:
        inner_splits = expanding_window_splits(list(ordered_match_ids), n_repeats=k_inner)
    else:
        unique_matches = list(dict.fromkeys(match_ids.tolist()))
        inner_splits = kfold_match_splits(unique_matches, k=k_inner, seed=seed)

    # Cartesian product of the grid.
    names = list(param_grid)
    from itertools import product
    combos = [dict(zip(names, vals)) for vals in product(*param_grid.values())]

    combo_scores = {tuple(c.items()): [] for c in combos}
    for split in inner_splits:
        tr, te = iter_fold_indices(match_ids, split)
        tr = _fit_rows(tr, match_ids, ordered_match_ids)
        for c in combos:
            model = make_model(**c)
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[te])
            s = score_fn(y[te], p)
            combo_scores[tuple(c.items())].append(s)

    # Mean inner score per combo; pick best.
    mean_scores = {k: np.nanmean(v) for k, v in combo_scores.items()}
    best_key = (min if lower_is_better else max)(mean_scores, key=mean_scores.get)
    best_params = dict(best_key)

    return TuningResult(
        best_params=best_params,
        selection_counts=Counter({best_key: 1}),
        inner_scores={k: float(v) for k, v in mean_scores.items()},
    )


def nested_cv(make_model: Callable,
              param_grid: dict[str, list],
              X: np.ndarray,
              y: np.ndarray,
              match_ids: np.ndarray,
              outer_splits,
              score_names=("log_loss", "brier", "auc"),
              seed: int = 51,
              temporal_inner: bool = True):
    """Full nested CV.

    temporal_inner (default): the outer splits list each training fold's
    matches in time order (temporal_match_splits / expanding_window_splits
    do), and the inner tuning folds are forward-chaining in that order. Pass
    False when the outer splits have no time order (random k-fold, e.g. simulated
    data); the inner folds are then random k-fold.

    For each outer split: tune on the training fold, refit the best model on
    all of that fold, evaluate on the held-out fold. Returns per-split metric
    values (so metrics.summarise gives you means + error bars) and the record
    of which hyperparameter value won each time (to verify the middle value is
    usually chosen).
    """
    per_split_metrics = {m: [] for m in score_names}
    selected_params = []

    for split in outer_splits:
        tr, te = iter_fold_indices(match_ids, split)
        ordered = split.train_match_ids if temporal_inner else None
        tr = _fit_rows(tr, match_ids, ordered)
        tuning = grid_search(make_model, param_grid, X[tr], y[tr],
                             match_ids[tr], seed=seed,
                             ordered_match_ids=ordered)
        selected_params.append(tuning.best_params)

        model = make_model(**tuning.best_params)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])
        for m in score_names:
            per_split_metrics[m].append(metrics.ALL_METRICS[m](y[te], p))

    # Tally selected values per hyperparameter — the brief's centring check.
    centring = {}
    if selected_params:
        for name in param_grid:
            centring[name] = Counter(sp[name] for sp in selected_params)

    summary = {m: metrics.summarise(v) for m, v in per_split_metrics.items()}
    return {
        "summary": summary,
        "per_split": per_split_metrics,
        "selected_params": selected_params,
        "centring": centring,
    }
