"""Raw point-sequence input for the Relational RNN (complex model).

The other two models read Benjamin's engineered feature table. The RNN instead
reads the RAW sequence of recent points and learns any momentum structure
itself, which is the whole reason it is the complex model and why it ties to the
research question: if momentum is some order-dependent pattern, this is the model
that can find it without being told what to look for.

For each point we build:
  - a sequence of the last RNN_HISTORY points (zero-padded at the start of a
    match, never crossing a match boundary), each described by RNN_STEP_FEATURES
    numbers (who won, who served, court side, break point, server won);
  - a small static context (who serves the CURRENT point, the pre-match
    expectation that player 1 wins a point) so the model is not blind to player
    strength and the comparison with the feature models is fair.

Everything uses only points strictly BEFORE the one being predicted (the
sequence is shifted by one), plus the current point's server/court, which are
known before the outcome. So there is no label leakage.

build_sequence_matrix returns one flat 2-D array (n_points, RNN_HISTORY *
RNN_STEP_FEATURES + n_static), so it drops straight into the from-scratch
nested CV exactly like the other models' X. The RNN wrapper reshapes it back
into (batch, time, features) internally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RNN_HISTORY, RNN_STEP_FEATURES

# Static context columns appended after the flattened sequence. The first is
# always the current point's server; the rest are pre-match, if present.
_STATIC_OPTIONAL = ["expected_p1_win"]


def _match_steps(match: pd.DataFrame) -> np.ndarray:
    """Per-point step vectors (n, RNN_STEP_FEATURES) for one match, describing
    each point as it happened (not shifted yet)."""
    server = match["point_server"].to_numpy()
    winner = match["point_winner"].to_numpy()
    p1_won = (winner == 1).astype(np.float32)
    server_is_p1 = (server == 1).astype(np.float32)
    ad_court = (match["serve_from_ad_court"].to_numpy()
                if "serve_from_ad_court" in match else np.zeros(len(match)))
    bp = (match["is_break_point"].to_numpy()
          if "is_break_point" in match else np.zeros(len(match)))
    server_won = (winner == server).astype(np.float32)
    return np.stack([p1_won, server_is_p1, ad_court.astype(np.float32),
                     bp.astype(np.float32), server_won], axis=1)


def build_sequence_matrix(feats: pd.DataFrame, history: int = RNN_HISTORY) -> np.ndarray:
    """Flat (n_points, history*RNN_STEP_FEATURES + n_static) sequence matrix.

    Row i holds the `history` points before point i (within its match, zero
    padded), flattened, followed by the static context for point i.
    """
    d = RNN_STEP_FEATURES
    static_cols = [c for c in _STATIC_OPTIONAL if c in feats.columns]
    n_static = 1 + len(static_cols)          # +1 for current server_is_p1
    out = np.zeros((len(feats), history * d + n_static), dtype=np.float32)

    server_is_p1_now = (feats["point_server"].to_numpy() == 1).astype(np.float32)

    pos = 0
    for _, match in feats.groupby("match_id", sort=False):
        steps = _match_steps(match)                      # (m, d)
        m = len(match)
        # sliding window of the PREVIOUS `history` points (shifted by one)
        for j in range(m):
            lo = max(0, j - history)
            past = steps[lo:j]                            # up to history rows
            if len(past):
                out[pos + j, (history - len(past)) * d: history * d] = past.ravel()
        # static context
        out[pos:pos + m, history * d] = server_is_p1_now[pos:pos + m]
        for s, c in enumerate(static_cols, start=1):
            vals = pd.to_numeric(match[c], errors="coerce").fillna(0.5).to_numpy()
            out[pos:pos + m, history * d + s] = vals.astype(np.float32)
        pos += m
    return out


def n_static_features(feats: pd.DataFrame) -> int:
    return 1 + len([c for c in _STATIC_OPTIONAL if c in feats.columns])
