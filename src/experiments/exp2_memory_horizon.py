"""Experiment 2 — Memory horizon.

Give each model increasing amounts of history (config.HISTORY_LENGTHS) and plot
predictive gain over the zero-history baseline, with error bars. Where the curve
flattens is the "momentum horizon": the number of past points beyond which no
extra predictive information is gained.

For the hand-crafted-feature models, "history length H" means: include only the
window/EWMA features whose effective memory is <= H (plus the H=0 baseline of
server+score only). For the Relational RNN, H is literally the input sequence
length. Keep H consistent across models when comparing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RESULTS_DIR, HISTORY_LENGTHS, N_OUTER_REPEATS
from ..features import build_features
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression


def feature_cols_for_history(h: int) -> list[str]:
    """Return the feature subset whose memory does not exceed h points.

    TODO(team): decide the exact mapping. Suggested:
      h=0  -> baseline only: ['server_is_p1', 'is_break_point', 'score_diff']
      h>=w -> include winrate_w{w}; include ewma_d{d} if its effective memory <= h
    Effective memory of an EWMA with decay d is ~1/(1-d) points.
    """
    baseline = ["server_is_p1", "is_break_point", "score_diff"]
    if h == 0:
        return baseline
    cols = list(baseline)
    for w in (1, 3, 5, 10, 20):
        if w <= h:
            cols.append(f"winrate_w{w}")
    for d in (0.5, 0.7, 0.85, 0.95):
        if 1.0 / (1.0 - d) <= h:
            cols.append(f"ewma_d{d}")
    cols += ["streak_signed", "surprise_ewma"]
    return cols


def main() -> None:
    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")
    feats = build_features(pts, player_won_col="p1_won").fillna(0.0)
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()
    ordered = list(dict.fromkeys(match_ids.tolist()))
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)

    curve = {}
    for h in HISTORY_LENGTHS:
        cols = feature_cols_for_history(h)
        X = feats[cols].to_numpy()
        res = nested_cv(logistic_regression.make_model,
                        logistic_regression.PARAM_GRID,
                        X, y, match_ids, splits,
                        score_names=("log_loss", "auc"))
        curve[h] = res["summary"]
        s = res["summary"]
        print(f"  H={h:>2}: log_loss={s['log_loss']['mean']:.4f}"
              f"±{s['log_loss']['se']:.4f}  auc={s['auc']['mean']:.4f}")

    (RESULTS_DIR / "exp2_memory_horizon.json").write_text(json.dumps(curve, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'exp2_memory_horizon.json'}")
    print("Plot metric vs H with error bars; mark where it flattens (the horizon).")


if __name__ == "__main__":
    main()
