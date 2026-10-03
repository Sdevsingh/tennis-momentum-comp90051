"""Experiment 3 — Planted-momentum positive control.

Simulate matches where the next-point win probability gets a KNOWN bump delta
after a streak. Run the full pipeline on them. This measures the smallest
momentum effect the methods can detect, so a null result on real data means
"no detectable momentum", not "methods too weak to find any".

Pairs with exp1: the shuffle control shows the pipeline does not INVENT
momentum; this control shows it can FIND momentum when it is present.

Generator model (simple, documented, defensible):
  For each simulated match, server alternates by game as in real tennis. Base
  per-point win prob for the server is p_serve (e.g. 0.64, a realistic hold
  rate). After a player wins `streak_k` points in a row, their win prob is
  nudged by +delta (capped to [0,1]). delta=0 is the no-momentum null.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR, PLANTED_DELTAS, SEED, N_OUTER_REPEATS
from ..features import build_features, FEATURE_COLUMNS
from ..cross_validation import kfold_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression


def simulate_matches(n_matches: int = 400, points_per_match: int = 150,
                     p_serve: float = 0.64, delta: float = 0.0,
                     streak_k: int = 3, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic point table with planted momentum of size `delta`."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        server = 1
        game_pts = 0
        recent = []  # recent point winners (1/2) for streak detection
        set_no, game_no = 1, 1
        for t in range(points_per_match):
            # base: server wins with p_serve
            p1_is_server = server == 1
            base_p1 = p_serve if p1_is_server else (1 - p_serve)

            # momentum bump: if the about-to-act leader is on a streak_k streak
            bump = 0.0
            if len(recent) >= streak_k and len(set(recent[-streak_k:])) == 1:
                leader = recent[-1]
                if leader == 1:
                    bump = delta
                else:
                    bump = -delta
            p1 = min(max(base_p1 + bump, 0.0), 1.0)

            winner = 1 if rng.random() < p1 else 2
            rows.append({
                "match_id": f"sim{m:04d}", "set_no": set_no, "game_no": game_no,
                "point_no": t, "point_server": server,
                "point_winner": winner, "p1_won": int(winner == 1),
                "server_won": int(winner == server),
            })
            recent.append(winner)
            game_pts += 1
            if game_pts >= 4:  # crude "game" length; alternate server
                server = 2 if server == 1 else 1
                game_no += 1
                game_pts = 0
    return pd.DataFrame(rows)


def main() -> None:
    sensitivity = {}
    for delta in PLANTED_DELTAS:
        sim = simulate_matches(delta=delta)
        feats = build_features(sim, player_won_col="p1_won").fillna(0.0)
        X = feats[FEATURE_COLUMNS].to_numpy()
        y = feats["p1_won"].to_numpy()
        match_ids = feats["match_id"].to_numpy()
        # Synthetic data has no real time axis -> plain k-fold over matches.
        splits = kfold_match_splits(list(dict.fromkeys(match_ids.tolist())),
                                    k=max(10, N_OUTER_REPEATS))
        res = nested_cv(logistic_regression.make_model,
                        logistic_regression.PARAM_GRID,
                        X, y, match_ids, splits,
                        score_names=("log_loss", "auc"))
        sensitivity[delta] = res["summary"]
        s = res["summary"]
        print(f"  delta={delta:.2f}: auc={s['auc']['mean']:.4f}±{s['auc']['se']:.4f}")

    (RESULTS_DIR / "exp3_positive_control.json").write_text(
        json.dumps(sensitivity, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'exp3_positive_control.json'}")
    print("The smallest delta whose AUC clears the delta=0 error bar is your "
          "detection floor — report it next to the real-data effect size.")


if __name__ == "__main__":
    main()
