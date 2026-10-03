"""Experiment 1 — No-momentum shuffle control (the key test for Section 2.a).

Idea: destroy temporal ORDER while preserving BASE RATES, then see whether the
models lose their edge. We permute point outcomes WITHIN each match, but only
among points that share the same server (and optionally a coarse score bucket),
so serve-conditional win rates are untouched and only the sequence is scrambled.

    real data performance  -  shuffled data performance  =  momentum signal

If the gap is within error bars, momentum carries no predictive information at
this resolution — a valid, publishable null. The shuffle also inherits any
streak-selection bias (Miller & Sanjurjo), so comparing real vs shuffled
cancels that bias instead of being fooled by it.

This runner is functional end-to-end for the logistic-regression model so the
team can see numbers immediately; the other two models slot into MODELS once
their fit/predict are implemented.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RESULTS_DIR, SEED, N_OUTER_REPEATS
from ..features import build_features, FEATURE_COLUMNS
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression
# from ..models import gradient_boosting, relational_rnn  # enable when ready


MODELS = {
    "logreg": (logistic_regression.make_model, logistic_regression.PARAM_GRID),
    # "gboost": (gradient_boosting.make_model, gradient_boosting.PARAM_GRID),
    # "rmc":    (relational_rnn.make_model, relational_rnn.PARAM_GRID),
}


def shuffle_within_match(points: pd.DataFrame, target: str = "p1_won",
                         by_server: bool = True, seed: int = SEED) -> pd.DataFrame:
    """Return a copy with `target` permuted within each match.

    Permuting within server groups preserves serve-conditional base rates and
    scrambles only the order — the null the control needs.
    """
    rng = np.random.default_rng(seed)
    out = points.copy()
    group_cols = ["match_id"] + (["point_server"] if by_server else [])
    vals = out[target].to_numpy().copy()
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx = np.array(list(idx))
        perm = rng.permutation(len(idx))
        vals[idx] = vals[idx][perm]
    out[target] = vals
    return out


def _run_one(points: pd.DataFrame, label: str) -> dict:
    feats = build_features(points, player_won_col="p1_won")
    X = feats[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()

    # Time order matches by first appearance (file names sort chronologically-ish;
    # TODO(team): sort by real tournament date parsed from match_id).
    ordered = list(dict.fromkeys(match_ids.tolist()))
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)

    results = {}
    for name, (make, grid) in MODELS.items():
        res = nested_cv(make, grid, X, y, match_ids, splits)
        results[name] = res["summary"]
        print(f"  [{label}/{name}] " +
              ", ".join(f"{m}={s['mean']:.4f}±{s['se']:.4f}"
                        for m, s in res["summary"].items()))
    return results


def main() -> None:
    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")

    print("REAL data:")
    real = _run_one(pts, "real")

    print("SHUFFLED (no-momentum) data:")
    shuffled = _run_one(shuffle_within_match(pts), "shuffled")

    out = {"real": real, "shuffled": shuffled}
    (RESULTS_DIR / "exp1_shuffle_control.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'exp1_shuffle_control.json'}")
    print("Momentum signal = real - shuffled, per metric (see report).")


if __name__ == "__main__":
    main()
