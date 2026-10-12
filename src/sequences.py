"""Input for the Relational RNN (complex model): raw point memory + static context.

The feature models read the engineered table, including the POINT_MEMORY
features (rolling win rates, streaks, serve estimates, rally ratios, ...) that
summarise earlier points. The RNN gets none of those summaries. Instead it gets
the RAW memory of the match, the earlier points themselves, and has to learn
whatever order-dependent structure there is on its own. That is the point of
the complex model for the research question.

For each point i the input is:
  - a sequence of the RNN_HISTORY points before i in the same match (oldest
    first, zero-padded at the start of a match, never crossing a match
    boundary). Each step is RNN_STEP_FEATURES numbers, see STEP_FEATURES:
      valid        1 for a real point, 0 for padding
      p1_won       player 1 won that point
      p1_served    player 1 served it
      rally_known  1 if the rally length was recorded (2012-2015 Slams mostly)
      rally        log1p(shots) / log1p(RNN_RALLY_SCALE), 0 when unknown;
                   a log scale with no cap (rally 10 -> 1.0, 40 -> 1.55)
  - static context: every MATCH feature (pre-match ratings, expectations,
    rest, h2h, ...) and every POINT_STRUCTURE feature (who serves point i),
    i.e. feature_columns(MATCH) + feature_columns(POINT_STRUCTURE). These are
    left raw here (NaN kept); RelationalRNN.fit standardises them with
    training-fold statistics only.

The sequence only uses points strictly before i, and the static block only
holds pre-match information and the current server, which is known before the
point is played. So there is no label leakage.

build_sequence_matrix returns one flat 2-D array (n_points, RNN_HISTORY *
RNN_STEP_FEATURES + n_static) so it drops into the nested CV like any other X;
the RNN reshapes it back into (batch, time, features) internally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RNN_HISTORY, RNN_STEP_FEATURES, RNN_RALLY_SCALE
from .features import FeatureScope, feature_columns

STEP_FEATURES = ["valid", "p1_won", "p1_served", "rally_known", "rally"]
assert len(STEP_FEATURES) == RNN_STEP_FEATURES, "keep config.RNN_STEP_FEATURES in sync"

STATIC_FEATURES = (feature_columns(FeatureScope.MATCH)
                   + feature_columns(FeatureScope.POINT_STRUCTURE))


def _point_steps(feats: pd.DataFrame) -> np.ndarray:
    """(n, RNN_STEP_FEATURES) description of each point as it happened."""
    n = len(feats)
    winner = feats["point_winner"].to_numpy()
    server = feats["point_server"].to_numpy()
    rally = (pd.to_numeric(feats["rally"], errors="coerce").to_numpy(float)
             if "rally" in feats else np.full(n, np.nan))
    known = ~np.isnan(rally)
    scaled = np.where(known, np.log1p(np.where(known, rally, 0.0)), 0.0) / np.log1p(RNN_RALLY_SCALE)
    return np.stack([np.ones(n), winner == 1, server == 1, known, scaled],
                    axis=1).astype(np.float32)


def build_sequence_matrix(feats: pd.DataFrame, history: int = RNN_HISTORY) -> np.ndarray:
    """Flat (n_points, history*RNN_STEP_FEATURES + n_static) matrix.

    Row i holds the `history` points before point i in its match (oldest
    first, zero rows where the match has not gone that far), flattened,
    followed by the static context for point i. `feats` must keep each
    match's points contiguous and in playing order (build_features does).
    """
    d = RNN_STEP_FEATURES
    n = len(feats)
    steps = _point_steps(feats)
    # position of each point inside its match
    mid = feats["match_id"].to_numpy()
    new_match = np.r_[True, mid[1:] != mid[:-1]]
    if pd.Series(mid[new_match]).duplicated().any():
        raise ValueError("each match's points must be contiguous")
    start = np.maximum.accumulate(np.where(new_match, np.arange(n), 0))
    pos = np.arange(n) - start

    static = feats.reindex(columns=STATIC_FEATURES).apply(pd.to_numeric, errors="coerce")
    out = np.zeros((n, history * d + len(STATIC_FEATURES)), dtype=np.float32)
    # lag k = 1 (the point just before) goes in the last slot
    for k in range(1, history + 1):
        ok = pos >= k
        slot = history - k
        out[ok, slot * d:(slot + 1) * d] = steps[np.flatnonzero(ok) - k]
    out[:, history * d:] = static.to_numpy(np.float32)
    return out


def n_static_features() -> int:
    return len(STATIC_FEATURES)
