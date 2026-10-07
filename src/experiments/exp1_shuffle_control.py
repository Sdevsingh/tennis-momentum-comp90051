"""Experiment 1 — No-momentum control (the key test for Section 2.a).

Idea: remove any dependence between points while preserving BASE RATES, then
see whether the models lose their edge. Each real match is replayed as a
simulated match in which every point is independent: each player wins a point
on serve with their actual serve win rate in that match, and the match is
played under the event's real scoring rules (src/tennis_rules.py). Serve
rates ("form on the day") are kept; only order-dependence is removed.

    real data performance  -  simulated data performance  =  momentum signal

If the gap is within error bars, momentum carries no predictive information at
this resolution — a valid, publishable null. The null also inherits any
streak-selection bias (Miller & Sanjurjo), so comparing real vs simulated
cancels that bias instead of being fooled by it.

Why a simulation and not a shuffle: permuting point winners inside the real
game/set boundaries produced impossible tennis (only ~26% of games ended on a
legal score, breaks rose from 25% to 39%), so every score-based feature was
meaningless on the null side. The simulation is the standard independent-
points null in the tennis momentum literature (e.g. Klaassen & Magnus).

Runs logistic regression and gradient boosting (LightGBM); the Relational RNN
slots into MODELS once its fit/predict are implemented.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RESULTS_DIR, SEED, N_OUTER_REPEATS
from ..features import build_features, FEATURE_COLUMNS
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression, gradient_boosting
from ..tennis_rules import sets_to_win, tiebreak_rule
# from ..models import relational_rnn  # enable when ready


MODELS = {
    "logreg": (logistic_regression.make_model, logistic_regression.PARAM_GRID),
    "gboost": (gradient_boosting.make_model, gradient_boosting.PARAM_GRID),
    # "rmc":    (relational_rnn.make_model, relational_rnn.PARAM_GRID),
}


def _simulate_match(match_id: str, p_serve: dict, first_server: int, rng) -> list[tuple]:
    """One match with independent points under the event's scoring rules.

    Returns (set_no, game_no, server, winner) per point. Serve alternates
    every game (a tiebreak counts as one game); inside a tiebreak the first
    server serves one point, then players alternate every two points.
    """
    rows = []
    sets = {1: 0, 2: 0}
    need = sets_to_win(match_id)
    server = first_server
    set_no = 0
    while max(sets.values()) < need:
        set_no += 1
        games = {1: 0, 2: 0}
        tb_at, tb_target = tiebreak_rule(match_id, set_no)
        game_no = 0
        while True:
            game_no += 1
            other = 3 - server
            if tb_at is not None and games[1] == games[2] == tb_at:
                pts = {1: 0, 2: 0}
                j = 0
                while not (max(pts.values()) >= tb_target and abs(pts[1] - pts[2]) >= 2):
                    srv = server if ((j + 1) // 2) % 2 == 0 else other
                    win = srv if rng.random() < p_serve[srv] else 3 - srv
                    rows.append((set_no, game_no, srv, win))
                    pts[win] += 1
                    j += 1
                games[1 if pts[1] > pts[2] else 2] += 1
                server = other
                break                                    # a tiebreak ends the set
            pts = {1: 0, 2: 0}
            while not (max(pts.values()) >= 4 and abs(pts[1] - pts[2]) >= 2):
                win = server if rng.random() < p_serve[server] else other
                rows.append((set_no, game_no, server, win))
                pts[win] += 1
            games[1 if pts[1] > pts[2] else 2] += 1
            server = other
            if max(games.values()) >= 6 and abs(games[1] - games[2]) >= 2:
                break
        sets[1 if games[1] > games[2] else 2] += 1
    return rows


def simulate_no_momentum(points: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Replay every match with independent points (the no-momentum null).

    Per match, each player wins a point on serve with their real serve win
    rate in that match (clipped to [0.05, 0.95]); the first server is the
    real one. The simulated match keeps its match_id, so it gets the same
    pre-match (MATCH) features; its length and result follow from the
    simulation.
    """
    rng = np.random.default_rng(seed)
    out = []
    for mid, m in points.groupby("match_id", sort=False):
        p_serve = {}
        for p in (1, 2):
            served = m["point_server"] == p
            rate = (m.loc[served, "point_winner"] == p).mean() if served.any() else 0.6
            p_serve[p] = float(np.clip(rate, 0.05, 0.95))
        rows = _simulate_match(mid, p_serve, int(m["point_server"].iloc[0]), rng)
        sim = pd.DataFrame(rows, columns=["set_no", "game_no", "point_server", "point_winner"])
        sim.insert(0, "match_id", mid)
        sim["point_no"] = (np.arange(len(sim)) + 1).astype(str)
        out.append(sim)
    sim = pd.concat(out, ignore_index=True)
    sim["p1_won"] = (sim["point_winner"] == 1).astype(int)
    sim["server_won"] = (sim["point_winner"] == sim["point_server"]).astype(int)
    return sim


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


def _nested_cv_all_models(X, y, match_ids, label: str) -> dict:
    ordered = matches_in_time_order(match_ids.tolist())
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)
    results = {}
    for name, (make, grid) in MODELS.items():
        res = nested_cv(make, grid, X, y, match_ids, splits)
        results[name] = {
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
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()
    if not by_tour:
        return _nested_cv_all_models(X, y, match_ids, label)

    tours = tour_of(feats["match_id"]).to_numpy()
    results = {}
    for tour in ("men", "women"):
        m = tours == tour
        results[tour] = _nested_cv_all_models(X[m], y[m], match_ids[m], f"{label}/{tour}")
    return results


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Exp 1: real vs simulated no-momentum data.")
    ap.add_argument("--pooled", action="store_true",
                    help="one model for men and women together (default: separate models per tour)")
    by_tour = not ap.parse_args().pooled
    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")

    print("REAL data:")
    real = _run_one(pts, "real", by_tour)

    print("SIMULATED no-momentum data:")
    simulated = _run_one(simulate_no_momentum(pts), "simulated", by_tour)

    out = {"by_tour": by_tour, "real": real, "simulated": simulated}
    dest = RESULTS_DIR / ("exp1_shuffle_control.json" if by_tour
                          else "exp1_shuffle_control_pooled.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")
    print("Momentum signal = real - simulated, per metric (see report).")


if __name__ == "__main__":
    main()
