"""Out-of-fold model predictions, saved so results can be analysed without
rerunning the experiments.

nested_cv returns, for every outer fold, the predicted probability for each
held-out point from the model refit on that fold's training data (the same
predictions the reported metrics are computed from). Each experiment turns
these into one long table and writes it to

    results/predictions/<experiment file stem>.parquet

(next to the JSON of the same name; *.parquet is gitignored, so these stay
local). One row per (configuration, test point):

    labels      the experiment's configuration columns, e.g. dataset
                (real/shuffled), tour, model, H, variant
    fold        outer fold number (0 = first test block)
    match_id, point_idx (0-based position in the match), set_no, game_no,
    server_is_p1
    y           1 if player 1 won the point
    p           predicted P(player 1 wins the point)
    param, param_value   the hyperparameter chosen for that fold

Every test point appears once per configuration, so two configurations can be
compared point by point by joining on (match_id, point_idx), e.g. full vs base
in exp3 or real vs shuffled per fold in exp1 (shuffled matches have the same
ids, but their points are re-ordered, so compare those at fold level).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

PRED_DIR = RESULTS_DIR / "predictions"


def point_keys(feats: pd.DataFrame) -> pd.DataFrame:
    """Identifying columns for each row of a feature table (rows aligned)."""
    return pd.DataFrame({
        "match_id": feats["match_id"].to_numpy(),
        "point_idx": feats.groupby("match_id", sort=False).cumcount().to_numpy(np.int32),
        "set_no": feats["set_no"].to_numpy(np.int8),
        "game_no": feats["game_no"].to_numpy(np.int16),
        "server_is_p1": (feats["point_server"].to_numpy() == 1).astype(np.int8),
        "y": feats["p1_won"].to_numpy(np.int8),
    })


def prediction_frame(res: dict, keys: pd.DataFrame, **labels) -> pd.DataFrame:
    """Long table of one nested_cv result's out-of-fold predictions.

    `keys` must be row-aligned with the X passed to nested_cv; `labels` become
    constant columns (configuration)."""
    pred = res["predictions"]
    rows = pred["row"]
    out = keys.iloc[rows].reset_index(drop=True)
    out.insert(0, "fold", pred["fold"])
    for k, v in reversed(labels.items()):
        out.insert(0, k, str(v))
    out["p"] = pred["p"]
    params = res["selected_params"]
    if params and len(params[0]) == 1:
        (name,) = params[0]
        out["param"] = name
        out["param_value"] = np.asarray([params[f][name] for f in pred["fold"]], dtype=float)
    return out


def save_predictions(frames: list[pd.DataFrame], name: str) -> Path | None:
    """Concatenate prediction tables and write results/predictions/<name>.parquet."""
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRED_DIR / f"{name}.parquet"
    df.to_parquet(dest, index=False)
    print(f"Wrote {dest} ({len(df):,} predictions)")
    return dest
