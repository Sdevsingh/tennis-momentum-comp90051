"""Experiment 1 — No-momentum shuffle control (the key test for Section 2.a).

Idea: destroy temporal ORDER while preserving BASE RATES, then see whether the
models lose their edge. The null is the real matches with their order shuffled
at two levels (src/null_shuffle.py): points within each game, and each
player's service games within each set. Every point/game result, game score,
set score and serve record is kept, serve still alternates, and every shuffled
match is legal tennis -- only the order of what happened is scrambled.

    real data performance  -  shuffled data performance  =  momentum signal

If the gap is within error bars, momentum carries no predictive information at
this resolution — a valid, publishable null. The shuffle also inherits any
streak-selection bias (Miller & Sanjurjo), so comparing real vs shuffled
cancels that bias instead of being fooled by it.

--null points|games|both picks the levels (default both); comparing them shows
whether any order effect lives inside games or across games. --null replay
instead re-forms each match from per-server pools of its real points (games
and sets end wherever the scoring rules say), which removes the within-game
"drawing without replacement" structure the other modes keep; see
src/null_shuffle.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RESULTS_DIR, SEED, N_OUTER_REPEATS
from ..features import build_features, FEATURE_COLUMNS
from ..sequences import build_sequence_matrix
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression, gradient_boosting, relational_rnn
from ..null_shuffle import shuffle_null, MODES


# Per-tour hyperparameter grids (used when men and women get separate models);
# a model without one uses its MODELS grid for both tours.
TOUR_GRIDS = {
    "logreg": logistic_regression.PARAM_GRID_BY_TOUR,
    "gboost": gradient_boosting.PARAM_GRID_BY_TOUR,
}

MODELS = {
    "logreg": (logistic_regression.make_model, logistic_regression.PARAM_GRID),
    "gboost": (gradient_boosting.make_model, gradient_boosting.PARAM_GRID),
    "rmc":    (relational_rnn.make_model, relational_rnn.PARAM_GRID),
}

# Models whose input is the raw point sequence rather than the tabular features.
SEQUENCE_MODELS = {"rmc"}


SLAM_ORDER = {"ausopen": 0, "frenchopen": 1, "wimbledon": 2, "usopen": 3}


def matches_in_time_order(match_ids) -> list[str]:
    """Unique match ids in playing order: year, then AO, RG, Wimbledon, USO.

    Within a tournament, the source's own order is kept.
    """
    uniq = list(dict.fromkeys(match_ids))
    return sorted(uniq, key=lambda m: (int(m[:4]), SLAM_ORDER.get(m.split("-")[1], 9)))


def tour_of(match_ids: pd.Series) -> pd.Series:
    """'men' / 'women' from the match number (1xxx/M... men, 2xxx/W... women)."""
    first = match_ids.str.split("-").str[-1].str[0]
    return first.map({"1": "men", "M": "men", "2": "women", "W": "women"})


def _nested_cv_all_models(X_tab, X_seq, y, match_ids, label: str, tour: str | None = None) -> dict:
    ordered = matches_in_time_order(match_ids.tolist())
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)
    results = {}
    for name, (make, grid) in MODELS.items():
        X = X_seq if name in SEQUENCE_MODELS else X_tab
        grid = TOUR_GRIDS.get(name, {}).get(tour, grid)
        res = nested_cv(make, grid, X, y, match_ids, splits)
        results[name] = {
            "grid": grid,
            "summary": res["summary"],
            "per_split": res["per_split"],
            "centring": {k: {str(v): n for v, n in c.items()}
                         for k, c in res["centring"].items()},
        }
        print(f"  [{label}/{name}] " +
              ", ".join(f"{m}={s['mean']:.4f}±{s['se']:.4f}"
                        for m, s in res["summary"].items()), flush=True)
    return results


def _run_one(points: pd.DataFrame, label: str, by_tour: bool = True) -> dict:
    """Nested CV for every model; one set of models per tour unless pooled."""
    feats = build_features(points, player_won_col="p1_won")
    X = feats[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    X_seq = build_sequence_matrix(feats)              # raw sequence for the RNN
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()
    if not by_tour:
        return _nested_cv_all_models(X, X_seq, y, match_ids, label)

    tours = tour_of(feats["match_id"]).to_numpy()
    results = {}
    for tour in ("men", "women"):
        m = tours == tour
        results[tour] = _nested_cv_all_models(
            X[m], X_seq[m], y[m], match_ids[m], f"{label}/{tour}", tour)
    return results


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Exp 1: real vs shuffled (no-momentum) data.")
    ap.add_argument("--pooled", action="store_true",
                    help="one model for men and women together (default: separate models per tour)")
    ap.add_argument("--null", choices=MODES, default="both",
                    help="points within games, games within sets, both, or replay "
                         "(re-form each match from per-server pools of its points)")
    ap.add_argument("--on-exhaust", choices=("recycle", "stop"), default="recycle",
                    help="replay only: when a server's pool runs out, reuse a fresh "
                         "shuffled copy (recycle) or end the match there (stop)")
    args = ap.parse_args()
    by_tour = not args.pooled
    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")

    print("REAL data:")
    real = _run_one(pts, "real", by_tour)

    stats: dict = {}
    null = shuffle_null(pts, mode=args.null, stats=stats, on_exhaust=args.on_exhaust)
    print(f"SHUFFLED ({args.null}) no-momentum data:  {stats}")
    shuffled = _run_one(null, "shuffled", by_tour)

    out = {"by_tour": by_tour, "null": args.null, "on_exhaust": args.on_exhaust,
           "shuffle_stats": stats,
           "real": real, "shuffled": shuffled}
    suffix = ("" if by_tour else "_pooled") + ("" if args.null == "both" else f"_{args.null}")
    dest = RESULTS_DIR / f"exp1_shuffle_control{suffix}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")
    print("Momentum signal = real - shuffled, per metric (see report).")


if __name__ == "__main__":
    main()
