"""Experiment 3 — Base vs full: what does knowing how the match is going add?

Each model is trained twice on the same folds:

  base  MATCH + POINT_STRUCTURE features only: pre-match strength, ratings,
        expectations, rest, h2h, and who is serving. No information about
        how the match is going (no score, no earlier points).
  full  every feature (exp1's X): base plus all POINT_MEMORY features.

The paired per-fold difference (full minus base) is what in-match information
adds over a stock pre-match + serve model. It is an upper bound on momentum:
memory features also carry the score and within-match form, which can help
without any momentum (exp1's shuffle control separates those).

Same evaluation as exp1: per tour (men and women get separate models and
exp1's per-tour grids) unless --pooled, temporal outer folds, forward-chaining
inner CV, and identical splits for base and full so the comparison is paired.

Models: logistic regression and gradient boosting (LightGBM). With --rnn the
Relational RNN runs too: full = its normal input (static MATCH + POINT_STRUCTURE
plus the raw point sequence), base = the same with the sequence blanked (all
zeros, as at the first point of a match), so it sees only the static context.
The RNN takes hours on a laptop GPU.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import (PROCESSED_DIR, RESULTS_DIR, N_OUTER_REPEATS,
                      RNN_HISTORY, RNN_STEP_FEATURES)
from ..features import build_features, FeatureScope, feature_columns, FEATURE_COLUMNS
from ..sequences import build_sequence_matrix, STATIC_FEATURES
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from .. import metrics
from ..models import logistic_regression, gradient_boosting
from ..predictions import point_keys, prediction_frame, save_predictions
from .exp1_shuffle_control import TOUR_GRIDS, matches_in_time_order, tour_of


MODELS = {
    "logreg": (logistic_regression.make_model, logistic_regression.PARAM_GRID),
    "gboost": (gradient_boosting.make_model, gradient_boosting.PARAM_GRID),
}
SEQUENCE_MODELS = {"rmc"}

BASE_FEATURES = [c for c in FEATURE_COLUMNS
                 if c in set(feature_columns(FeatureScope.MATCH)
                             + feature_columns(FeatureScope.POINT_STRUCTURE))]

def paired_gain(base: dict, full: dict) -> dict:
    """Per-fold gain of full over base (+ = full better); see metrics.paired_gain."""
    out = metrics.paired_gain(base, full)
    for v in out.values():
        v["full_better"] = v.pop("better")
    return out


def _cv(make, grid, X, y, match_ids, splits, frames=None, keys=None, **labels) -> dict:
    res = nested_cv(make, grid, X, y, match_ids, splits)
    if frames is not None and keys is not None:
        frames.append(prediction_frame(res, keys, **labels))
    return {"summary": res["summary"], "per_split": res["per_split"],
            "centring": {k: {str(v): n for v, n in c.items()}
                         for k, c in res["centring"].items()}}


def _run(X_base, X_full, X_seq, y, match_ids, label: str, tour: str | None,
         models: dict, keys: pd.DataFrame | None = None, frames: list | None = None) -> dict:
    ordered = matches_in_time_order(match_ids.tolist())
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)
    out = {}
    for name, (make, grid) in models.items():
        grid = TOUR_GRIDS.get(name, {}).get(tour, grid)
        if name in SEQUENCE_MODELS:
            # full first, then blank the point memory IN PLACE for base, so
            # no second copy of the (large) sequence matrix is made
            full = _cv(make, grid, X_seq, y, match_ids, splits, frames, keys,
                       tour=tour or "pooled", model=name, variant="full")
            X_seq[:, :RNN_HISTORY * RNN_STEP_FEATURES] = 0.0
            base = _cv(make, grid, X_seq, y, match_ids, splits, frames, keys,
                       tour=tour or "pooled", model=name, variant="base")
        else:
            base = _cv(make, grid, X_base, y, match_ids, splits, frames, keys,
                       tour=tour or "pooled", model=name, variant="base")
            full = _cv(make, grid, X_full, y, match_ids, splits, frames, keys,
                       tour=tour or "pooled", model=name, variant="full")
        gain = paired_gain(base["per_split"], full["per_split"])
        out[name] = {"grid": grid, "base": base, "full": full, "gain_full_over_base": gain}
        print(f"  [{label}/{name}] base log_loss={base['summary']['log_loss']['mean']:.4f}  "
              f"full log_loss={full['summary']['log_loss']['mean']:.4f}  "
              f"gain={gain['log_loss']['mean']:+.4f}±{gain['log_loss']['se']:.4f} "
              f"(full better {gain['log_loss']['full_better']}/{gain['log_loss']['n_folds']}); "
              f"auc gain={gain['auc']['mean']:+.4f}", flush=True)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Exp 3: base (pre-match + serve) vs full model.")
    ap.add_argument("--pooled", action="store_true",
                    help="one model for men and women together (default: separate models per tour)")
    ap.add_argument("--rnn", action="store_true",
                    help="also run the Relational RNN (hours on a laptop GPU)")
    args = ap.parse_args()

    models = dict(MODELS)
    if args.rnn:
        from ..models import relational_rnn
        models["rmc"] = (relational_rnn.make_model, relational_rnn.PARAM_GRID)

    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")
    feats = build_features(pts, player_won_col="p1_won")
    del pts
    X_full = feats[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    X_base = feats[BASE_FEATURES].fillna(0.0).to_numpy()
    # the RNN's sequence matrix is built per tour, from just the columns it reads
    seq_src = (feats[["match_id", "point_server", "point_winner", "rally"]
                     + list(STATIC_FEATURES)].copy() if args.rnn else None)
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()
    tours = tour_of(feats["match_id"]).to_numpy()
    keys = point_keys(feats)
    frames: list = []
    del feats

    results = {"by_tour": not args.pooled, "base_features": BASE_FEATURES,
               "full_features": FEATURE_COLUMNS}
    if args.pooled:
        X_seq = build_sequence_matrix(seq_src) if args.rnn else None
        results["pooled"] = _run(X_base, X_full, X_seq, y, match_ids, "pooled", None, models,
                                 keys, frames)
    else:
        for tour in ("men", "women"):
            m = tours == tour
            X_seq = build_sequence_matrix(seq_src[m]) if args.rnn else None
            results[tour] = _run(X_base[m], X_full[m], X_seq,
                                 y[m], match_ids[m], tour, tour, models, keys[m], frames)
            del X_seq

    suffix = ("_pooled" if args.pooled else "") + ("_rnn" if args.rnn else "")
    dest = RESULTS_DIR / f"exp3_base_vs_full{suffix}.json"
    dest.write_text(json.dumps(results, indent=2))
    save_predictions(frames, dest.stem)
    print(f"\nWrote {dest}")
    print("gain_full_over_base: + means the full model is better (log-loss/Brier "
          "lower, AUC higher), paired over the same folds.")


if __name__ == "__main__":
    main()
