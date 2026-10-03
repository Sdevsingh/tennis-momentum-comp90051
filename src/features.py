"""Feature construction (brief Section 2.c).

The brief forbids relying on simple preprocessing alone. These features turn a
raw point stream into a momentum-aware design matrix. The non-obvious one is
the *surprise* feature: momentum measured as departure from a serve+score
expectation, not as raw recent wins.

All features for a point use ONLY points strictly before it in the same match,
so nothing leaks the label. Across a game/set the server changes, so we always
carry the server indicator to let models separate "momentum" from "serving".

Each function appends columns to the per-match frame and returns it. Compose
them in `build_features`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WINDOW_SIZES, DECAY_RATES
from .player_features import add_player_features, PLAYER_FEATURES


def _prev_only(series: pd.Series) -> pd.Series:
    """Shift by one so a point never sees its own outcome."""
    return series.shift(1)


def rolling_winrate(match: pd.DataFrame, player_won_col: str) -> pd.DataFrame:
    """Rolling win rate over the last W points, for several W.

    Computed from the point of view of player 1 (symmetric for player 2).
    """
    won = _prev_only(match[player_won_col])
    for w in WINDOW_SIZES:
        match[f"winrate_w{w}"] = (
            won.rolling(window=w, min_periods=1).mean()
        )
    return match


def ewma_winrate(match: pd.DataFrame, player_won_col: str) -> pd.DataFrame:
    """Exponentially-weighted win rate at several decay rates.

    Higher decay = longer memory. Giving the models several decay rates lets
    the memory-horizon question be answered partly through feature selection
    too, not only through the sequence models.
    """
    won = _prev_only(match[player_won_col])
    for d in DECAY_RATES:
        # alpha relates to decay: larger d -> slower decay -> longer memory.
        match[f"ewma_d{d}"] = won.ewm(alpha=1 - d, min_periods=1).mean()
    return match


def streak_length(match: pd.DataFrame, player_won_col: str) -> pd.DataFrame:
    """Signed current streak: +k if player 1 just won k in a row, -k if lost."""
    won = _prev_only(match[player_won_col]).fillna(0).astype(int).values
    streak = np.zeros(len(won), dtype=int)
    run = 0
    prev = None
    for i, w in enumerate(won):
        if prev is None:
            run = 0
        elif w == prev:
            run += 1
        else:
            run = 1
        streak[i] = run if w == 1 else -run
        prev = w
    match["streak_signed"] = streak
    return match


def pressure_features(match: pd.DataFrame) -> pd.DataFrame:
    """Score-context / pressure indicators.

    TODO(team): derive break point, set point, deciding set, and game/set
    score differential from the score columns in the Sackmann file. These are
    control variables: an apparent momentum effect that vanishes once pressure
    is included was really a pressure effect.
    """
    # Placeholder columns so downstream shape is stable; replace with real logic.
    match["is_break_point"] = 0
    match["is_set_point"] = 0
    match["score_diff"] = 0
    return match


def surprise_feature(match: pd.DataFrame, player_won_col: str,
                     expected_col: str = "expected_p1_win") -> pd.DataFrame:
    """Momentum as *departure from expectation*.

    surprise_t = EWMA over previous points of (outcome - expected_outcome),
    where expected_outcome is the pre-match serve/return expectation for
    this server against this returner (src/player_features.py). It uses
    only matches played before this one, so it is CV-safe without fitting.

    This separates "won more than expected, given who is serving to whom"
    from "won because they were serving". It is the feature most aligned
    with the research question.

    TODO(team): the expectation ignores the score. A score-aware baseline
    (fitted on train folds only) could replace it.
    """
    if expected_col not in match.columns:
        match["surprise_ewma"] = 0.0
        return match
    resid = _prev_only(match[player_won_col] - match[expected_col])
    match["surprise_ewma"] = resid.ewm(alpha=0.15, min_periods=1).mean()
    return match


def build_features(points: pd.DataFrame, player_won_col: str = "p1_won") -> pd.DataFrame:
    """Apply all feature builders per match and return the full design matrix.

    Grouping by match_id guarantees history never crosses match boundaries.
    """
    # Pre-match player features (strength, serve/return, workload, ...),
    # frozen for the whole match. First, because the surprise feature
    # measures against their serve/return expectation.
    points = add_player_features(points)
    expected_col = "p_serve_expected" if player_won_col == "server_won" else "expected_p1_win"

    parts = []
    for _, match in points.groupby("match_id", sort=False):
        match = match.copy()
        match = rolling_winrate(match, player_won_col)
        match = ewma_winrate(match, player_won_col)
        match = streak_length(match, player_won_col)
        match = pressure_features(match)
        match = surprise_feature(match, player_won_col, expected_col)
        parts.append(match)
    out = pd.concat(parts, ignore_index=True)

    # Interaction features (windows x server) push the feature count over 100
    # as the tabular requirement needs. TODO(team): add the interactions you
    # actually use and document the final count in the report.
    out["server_is_p1"] = (out["point_server"] == 1).astype(int)
    return out


FEATURE_COLUMNS = (
    [f"winrate_w{w}" for w in WINDOW_SIZES]
    + [f"ewma_d{d}" for d in DECAY_RATES]
    + ["streak_signed", "is_break_point", "is_set_point", "score_diff",
       "surprise_ewma", "server_is_p1"]
    + PLAYER_FEATURES
)
