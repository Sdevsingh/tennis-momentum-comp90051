"""Experiment 2 — Memory horizon.

Give each model increasing amounts of in-match history (config.HISTORY_LENGTHS)
and track predictive skill, with error bars. Where the curve flattens is the
"momentum horizon": the number of past points beyond which no extra predictive
information is gained.

Every history length H gets the same base: all MATCH features (pre-match
ratings, expectations, rest, h2h, ...) and all POINT_STRUCTURE features (who
serves). So the curve measures what in-match MEMORY adds on top of player
strength and serve, never strength being re-learned from the match so far.

On top of the base, H adds every POINT_MEMORY feature whose lookback is at most
H points (MEMORY_LOOKBACK below). The last option, H = "match", is the entire
match so far: every memory feature, i.e. exactly exp1's feature set, so its
score should match exp1's real-data result.

Lookbacks (points before the one being predicted):
  - fixed windows: their size (winrate_w10 -> 10, rally_len_avg10 -> 10);
  - EWMAs: the effective memory 1/(1-decay) (ewma_d0.85 -> 6.7);
  - current-game state (game score, game/break point, court side, the
    server's rally ratio this game): 10, which covers 94% of games (mean 6.35,
    p90 10, p99 16 points);
  - "last k service games" features: ~2k games of ~6.5 points (k=2 -> 26,
    k=4 -> 52); the server's last service game -> 13;
  - anything that looks back to the start of the set or match (set/match
    score, tiebreak flag, streaks, match aggregates, Bayesian serve estimate,
    breaks, match rally stats): the whole match (inf), so only H = "match".

Each H is run per tour (men and women get separate models and exp1's per-tour
grids) unless --pooled, with the same temporal outer folds and forward-chaining
inner CV as exp1.

Runs logistic regression and gradient boosting (LightGBM). For the Relational
RNN, H would be its input sequence length; it is not wired in here.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RESULTS_DIR, HISTORY_LENGTHS, N_OUTER_REPEATS
from ..features import build_features, FeatureScope, feature_columns, FEATURE_COLUMNS
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression, gradient_boosting
from .exp1_shuffle_control import TOUR_GRIDS, matches_in_time_order, tour_of


MODELS = {
    "logreg": (logistic_regression.make_model, logistic_regression.PARAM_GRID),
    "gboost": (gradient_boosting.make_model, gradient_boosting.PARAM_GRID),
}

_GAME = 10                 # current-game state, see module docstring
_WHOLE_MATCH = math.inf

# Lookback in points of every POINT_MEMORY feature (see module docstring).
MEMORY_LOOKBACK: dict[str, float] = {
    **{f"winrate_w{w}": w for w in (1, 3, 5, 10, 20)},
    **{f"ewma_d{d}": 1.0 / (1.0 - d) for d in (0.5, 0.7, 0.85, 0.95)},
    "surprise_ewma": 1.0 / 0.15,                       # ewm(alpha=0.15)
    "streak_signed": _WHOLE_MATCH,                     # runs back to the last change
    **{f"{who}_serve_win_g{k}": 13 * k for k in (2, 4) for who in ("p1", "p2", "server")},
    **{c: _WHOLE_MATCH for c in ("p1_serve_win_match", "p2_serve_win_match",
                                 "p1_serve_points_match", "p2_serve_points_match",
                                 "p1_serve_bayes_map", "p2_serve_bayes_map",
                                 "p1_serve_bayes_sd", "p2_serve_bayes_sd",
                                 "p1_breaks_won", "p1_breaks_conceded")},
    "prev_rally_len": 1,
    "rally_len_avg10": 10,
    **{c: _WHOLE_MATCH for c in ("match_rally_avg", "p1_long_rally_win_pct",
                                 "rally_data_share", "server_rally_ratio_match")},
    "server_rally_ratio_last_game": 13,
    "server_rally_ratio_cur_game": _GAME,
    **{c: _GAME for c in ("is_break_point", "is_game_point", "points_in_game",
                          "server_pts_game", "receiver_pts_game", "serve_from_ad_court")},
    **{c: _WHOLE_MATCH for c in ("is_set_point", "is_tiebreak", "set_diff",
                                 "point_idx_match", "p1_games_set", "p2_games_set",
                                 "p1_sets_won", "p2_sets_won")},
}

_missing = set(feature_columns(FeatureScope.POINT_MEMORY)) - set(MEMORY_LOOKBACK)
assert not _missing, f"give these memory features a lookback in exp2: {sorted(_missing)}"

BASE_FEATURES = (feature_columns(FeatureScope.MATCH)
                 + feature_columns(FeatureScope.POINT_STRUCTURE))


def feature_cols_for_history(h: int | str) -> list[str]:
    """MATCH + POINT_STRUCTURE features, plus the memory features whose lookback
    is at most h points. h = "match" means the entire match: all of them."""
    limit = _WHOLE_MATCH if h == "match" else h
    keep = set(BASE_FEATURES) | {c for c in feature_columns(FeatureScope.POINT_MEMORY)
                                 if MEMORY_LOOKBACK[c] <= limit}
    # exp1's column order, so H = "match" is exactly exp1's X
    return [c for c in FEATURE_COLUMNS if c in keep]


def horizon_curve(feats: pd.DataFrame, y: np.ndarray, match_ids: np.ndarray,
                  splits, make, grid, label: str) -> dict:
    """Nested-CV score at each history length for one model."""
    curve = {}
    for h in HISTORY_LENGTHS:
        cols = feature_cols_for_history(h)
        X = feats[cols].fillna(0.0).to_numpy()
        res = nested_cv(make, grid, X, y, match_ids, splits)
        s = res["summary"]
        curve[str(h)] = {
            "n_features": len(cols),
            "n_memory_features": len(cols) - len(BASE_FEATURES),
            "summary": s,
            "per_split": res["per_split"],
            "centring": {k: {str(v): n for v, n in c.items()}
                         for k, c in res["centring"].items()},
        }
        print(f"  [{label}] H={str(h):>5} ({len(cols) - len(BASE_FEATURES):2d} memory): "
              f"log_loss={s['log_loss']['mean']:.4f}±{s['log_loss']['se']:.4f}  "
              f"auc={s['auc']['mean']:.4f}", flush=True)
    return curve


def _run(feats: pd.DataFrame, label: str, tour: str | None) -> dict:
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()
    ordered = matches_in_time_order(match_ids.tolist())   # year, then AO, RG, Wimbledon, USO
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)
    out = {}
    for name, (make, grid) in MODELS.items():
        grid = TOUR_GRIDS.get(name, {}).get(tour, grid)
        out[name] = {"grid": grid,
                     "curve": horizon_curve(feats, y, match_ids, splits, make, grid,
                                            f"{label}/{name}")}
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Exp 2: memory horizon.")
    ap.add_argument("--pooled", action="store_true",
                    help="one model for men and women together (default: separate models per tour)")
    args = ap.parse_args()

    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")
    feats = build_features(pts, player_won_col="p1_won")
    del pts
    feats = feats[["match_id", "p1_won"] + feature_cols_for_history("match")]

    results = {"by_tour": not args.pooled,
               "base_features": BASE_FEATURES,
               "memory_features_by_h": {
                   str(h): [c for c in feature_cols_for_history(h) if c not in BASE_FEATURES]
                   for h in HISTORY_LENGTHS},
               "memory_lookback": {c: (None if math.isinf(v) else v)
                                   for c, v in MEMORY_LOOKBACK.items()}}
    if args.pooled:
        results["pooled"] = _run(feats, "pooled", None)
    else:
        tours = tour_of(feats["match_id"]).to_numpy()
        for tour in ("men", "women"):
            results[tour] = _run(feats[tours == tour], tour, tour)

    dest = RESULTS_DIR / f"exp2_memory_horizon{'_pooled' if args.pooled else ''}.json"
    dest.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {dest}")
    print("Plot metric vs H with error bars, per model and tour; mark where it "
          "flattens (the horizon). H='match' should reproduce exp1's real result.")


if __name__ == "__main__":
    main()
