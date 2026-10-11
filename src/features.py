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

Every feature has a FeatureScope: MATCH (built only from information before
the match) or POINT (updated during the match). `feature_columns(scope)`
returns one group; FEATURE_COLUMNS is all of them.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from .config import (WINDOW_SIZES, DECAY_RATES, SERVE_GAME_WINDOWS, SERVE_PRIOR_POINTS,
                     SERVE_BAYES_PRIOR_VAR_PCT, SERVE_BAYES_PRIOR_N_OFFSET,
                     RALLY_NEUTRAL_LEN, RALLY_WINDOW, LONG_RALLY_SHOTS)
from .player_features import (add_player_features, PLAYER_MATCH_FEATURES,
                              PLAYER_SERVER_VIEW_FEATURES)
from .tennis_rules import is_tiebreak as _is_tiebreak


def _prev_only(series: pd.Series) -> pd.Series:
    """Shift by one so a point never sees its own outcome."""
    return series.shift(1)


def rolling_winrate(match: pd.DataFrame, player_won_col: str) -> pd.DataFrame:
    """Rolling win rate over the last W points, for several W.

    Computed from the point of view of player 1 (symmetric for player 2).
    0.5 on a match's first point (no history yet), not NaN: the experiments
    fillna(0), which would read as "lost every point".
    """
    won = _prev_only(match[player_won_col])
    for w in WINDOW_SIZES:
        match[f"winrate_w{w}"] = (
            won.rolling(window=w, min_periods=1).mean().fillna(0.5)
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
        match[f"ewma_d{d}"] = won.ewm(alpha=1 - d, min_periods=1).mean().fillna(0.5)
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


def rolling_serve_winpct(match: pd.DataFrame) -> pd.DataFrame:
    """Each player's serve win % over their last k service games, blended
    with their pre-match expectation.

        serve_win = (won + m * prior) / (served + m)

    where won/served are the player's serve points in their last k service
    games, prior is their pre-match expected serve % against this opponent
    (p{i}_serve_expected, src/player_features.py) and m = SERVE_PRIOR_POINTS.
    So the value starts at the pre-match expectation and moves toward recent
    form as serve points accumulate; with a short window it stays a mix.

    Uses only points before the current one. A service game in progress
    counts as one of the k (with the points played so far). Tiebreak points
    count toward whoever served them.

    Adds p1_/p2_serve_win_g{k} and server_serve_win_g{k} (whoever is serving
    this point).
    """
    servers = match["point_server"].to_numpy()
    winners = match["point_winner"].to_numpy()
    games = list(zip(match["set_no"].to_numpy(), match["game_no"].to_numpy()))
    n, kmax = len(match), max(SERVE_GAME_WINDOWS)

    # per player: list of [game, points won, points served], newest last
    history = {1: [], 2: []}
    prior = {p: match[f"p{p}_serve_expected"].to_numpy(dtype=float) for p in (1, 2)}
    m = SERVE_PRIOR_POINTS
    pct = {(p, k): np.empty(n) for p in (1, 2) for k in SERVE_GAME_WINDOWS}
    for i in range(n):
        for p in (1, 2):
            h = history[p]
            for k in SERVE_GAME_WINDOWS:
                window = h[-k:]
                won = sum(g[1] for g in window)
                served = sum(g[2] for g in window)
                pct[(p, k)][i] = (won + m * prior[p][i]) / (served + m)
        s = servers[i]
        if s in history:
            h = history[s]
            if not h or h[-1][0] != games[i]:
                h.append([games[i], 0, 0])
                del h[:-kmax]
            h[-1][1] += int(winners[i] == s)
            h[-1][2] += 1

    for p in (1, 2):
        for k in SERVE_GAME_WINDOWS:
            match[f"p{p}_serve_win_g{k}"] = pct[(p, k)]
    p1_serving = servers == 1
    for k in SERVE_GAME_WINDOWS:
        match[f"server_serve_win_g{k}"] = np.where(
            p1_serving, match[f"p1_serve_win_g{k}"], match[f"p2_serve_win_g{k}"])
    return match


def serve_winpct_match(match: pd.DataFrame) -> pd.DataFrame:
    """Each player's serve win % over all their serve points so far in this
    match, with no pre-match information.

    Uses only points before the current one. Before a player has served,
    the value is 0.5 (no information) rather than NaN, which the
    experiments would fillna(0) into "won no serve points". The point
    counts say how much each average is based on.

    Adds p1_/p2_serve_win_match and p1_/p2_serve_points_match.
    """
    servers = match["point_server"].to_numpy()
    winners = match["point_winner"].to_numpy()
    for p in (1, 2):
        served = (servers == p).astype(int)
        won = (served == 1) & (winners == p)
        n_before = np.cumsum(served) - served           # exclude this point
        won_before = np.cumsum(won) - won
        match[f"p{p}_serve_points_match"] = n_before
        match[f"p{p}_serve_win_match"] = np.where(
            n_before > 0, won_before / np.maximum(n_before, 1), 0.5)
    return match


def _serve_posterior(k, m, mu0, tau0):
    """MAP and sd of theta for: theta ~ N(mu0, 1/tau0) on [0, 1], then k wins
    in m Bernoulli(theta) serve points. All arguments are arrays.

    The log posterior  -tau0 (theta - mu0)^2 / 2 + k log theta
    + (m - k) log(1 - theta)  is strictly concave, so its derivative has a
    single root: found by bisection. The sd is the Laplace approximation,
    1 / sqrt(curvature at the MAP), capped at the sd of a uniform (0.289).
    """
    lo, hi = np.full(k.shape, 1e-6), np.full(k.shape, 1 - 1e-6)
    for _ in range(50):
        mid = (lo + hi) / 2
        grad = -tau0 * (mid - mu0) + k / mid - (m - k) / (1 - mid)
        up = grad > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    theta = (lo + hi) / 2
    flat = (tau0 == 0) & (m == 0)                  # no information at all
    theta = np.where(flat, 0.5, theta)
    curv = tau0 + k / theta ** 2 + (m - k) / (1 - theta) ** 2
    sd = np.minimum(1 / np.sqrt(np.maximum(curv, 1e-12)), np.sqrt(1 / 12))
    return theta, sd


def serve_bayes(match: pd.DataFrame) -> pd.DataFrame:
    """Bayesian estimate of each player's chance of winning a point on serve.

    Prior: theta ~ Normal(career serve %, 10000 / (n + 200)) on the PERCENT
    scale (SERVE_BAYES_PRIOR_VAR_PCT, SERVE_BAYES_PRIOR_N_OFFSET), i.e.
    sd = sqrt(10000 / (n + 200)) percentage points: 7.1 for a player with
    no history, 4.5 at n = 300. Career serve % and n are over the player's
    earlier matches with serve stats (src/player_features.py). Truncated
    to [0, 1].
    Likelihood: each of the player's serve points so far in this match
    (before the current point) is Bernoulli(theta).

    Adds p{i}_serve_bayes_map (posterior mode) and p{i}_serve_bayes_sd
    (posterior sd, Laplace approximation).
    """
    servers = match["point_server"].to_numpy()
    winners = match["point_winner"].to_numpy()
    for p in (1, 2):
        served = (servers == p).astype(float)
        won = served * (winners == p)
        m = np.cumsum(served) - served               # exclude this point
        k = np.cumsum(won) - won
        n = match[f"p{p}_stat_matches"].to_numpy(dtype=float)
        mu0 = match[f"p{p}_career_serve_win_pct"].to_numpy(dtype=float)
        # var = VAR_PCT / (n + OFFSET) in percent^2 -> precision on the 0-1 scale
        tau0 = (n + SERVE_BAYES_PRIOR_N_OFFSET) * 10_000 / SERVE_BAYES_PRIOR_VAR_PCT
        mu0 = np.where(np.isfinite(mu0), mu0, 0.5)
        theta, sd = _serve_posterior(k, m, mu0, tau0)
        match[f"p{p}_serve_bayes_map"] = theta
        match[f"p{p}_serve_bayes_sd"] = sd
    return match


def court_side(match: pd.DataFrame) -> pd.DataFrame:
    """Which side of the court this point is served from.

    Fixed by the rules: an even number of points already played in the game
    (0, 2, 4, ...) means the deuce court, odd means the ad court. Tiebreaks
    follow the same rule (first point from the deuce court, then two per
    server starting from the ad court). Uses no point outcomes.

    Adds serve_from_ad_court (1 = ad court, 0 = deuce court).
    """
    k = match.groupby(["set_no", "game_no"], sort=False).cumcount().to_numpy()
    match["serve_from_ad_court"] = k % 2
    return match


def rally_features(match: pd.DataFrame) -> pd.DataFrame:
    """Rally-length features, from points BEFORE the current one only.

    A point's own rally length is only known once it has been played, so it
    is never a feature for that point. Rally length is missing for ~35% of
    points (no Slam recorded it in 2012-2015, and some matches are
    untracked); missing rallies are skipped in the averages, and a value
    with nothing to average is RALLY_NEUTRAL_LEN (or 0.5 for the win %).

    Adds:
      prev_rally_len         shots in the previous point's rally
      rally_len_avg10        mean rally length over the last RALLY_WINDOW points
      match_rally_avg        mean rally length so far in the match
      p1_long_rally_win_pct  player 1's share of long rallies (>= LONG_RALLY_SHOTS
                             shots) won so far, smoothed: (wins + 1) / (long + 2)
      rally_data_share       share of earlier points with a known rally length
    """
    rally = match["rally"] if "rally" in match.columns else pd.Series(np.nan, index=match.index)
    prev = _prev_only(rally.astype(float))
    match["prev_rally_len"] = prev.fillna(RALLY_NEUTRAL_LEN)
    match["rally_len_avg10"] = prev.rolling(RALLY_WINDOW, min_periods=1).mean().fillna(RALLY_NEUTRAL_LEN)
    match["match_rally_avg"] = prev.expanding(min_periods=1).mean().fillna(RALLY_NEUTRAL_LEN)
    long = (prev >= LONG_RALLY_SHOTS).to_numpy()
    p1_won_prev = (_prev_only(match["point_winner"]) == 1).to_numpy()
    wins, n = np.cumsum(long & p1_won_prev), np.cumsum(long)
    match["p1_long_rally_win_pct"] = (wins + 1) / (n + 2)
    n_earlier = np.arange(len(match))
    match["rally_data_share"] = np.cumsum(prev.notna().to_numpy()) / np.maximum(n_earlier, 1)
    return match


def _match_round(match_id: str) -> int:
    """Round from the match number: its hundreds digit (1205 / MS212 -> round 2)."""
    return int(match_id.split("-")[-1][-3])


def slam_rally_baseline(points: pd.DataFrame) -> pd.Series:
    """Per match: average rally length at the same Slam in EARLIER matches.

    Earlier = all previous years of that Slam, plus earlier rounds of the same
    year (matches in the same round are concurrent, so they are excluded).
    Only points with a known rally length count. RALLY_NEUTRAL_LEN where there
    is no history yet (2011 round 1). Built from the matches in `points`, so a
    subset gives a baseline from that subset only.
    """
    known = points[points["rally"].notna()] if "rally" in points.columns else points.iloc[:0]
    all_mids = pd.Series(points["match_id"].unique())
    # Only real Slam ids ("2019-wimbledon-1301") have a slam/year/round;
    # anything else (e.g. exp3's simulated "sim0000") gets the neutral value.
    slam_id = all_mids.str.match(r"^\d{4}-[a-z]+-[A-Za-z]*\d{3,}$")
    neutral = pd.Series(RALLY_NEUTRAL_LEN, index=all_mids[~slam_id].values, dtype=float)
    mids = all_mids[slam_id].reset_index(drop=True)
    if mids.empty:
        return neutral
    key = pd.DataFrame({"match_id": mids,
                        "slam": mids.str.split("-").str[1],
                        "year": mids.str[:4].astype(int),
                        "round": [_match_round(m) for m in mids]})
    sums = known.groupby("match_id")["rally"].agg(["sum", "count"])
    key = key.join(sums, on="match_id").fillna({"sum": 0.0, "count": 0})
    blocks = key.groupby(["slam", "year", "round"])[["sum", "count"]].sum().sort_index()
    # cumulative over earlier blocks only (shift excludes the match's own round)
    cum = blocks.groupby(level="slam").cumsum().groupby(level="slam").shift(1).fillna(0)
    base = (cum["sum"] / cum["count"].where(cum["count"] > 0)).fillna(RALLY_NEUTRAL_LEN)
    out = key.join(base.rename("baseline"), on=["slam", "year", "round"]).set_index("match_id")["baseline"]
    return pd.concat([out, neutral])


def server_rally_ratios(match: pd.DataFrame, baseline: float) -> pd.DataFrame:
    """The current server's rally length on their own serve, relative to the Slam.

    Averages over the server's SERVICE points only, from points before the
    current one, each divided by `baseline` (slam_rally_baseline):
      server_rally_ratio_last_game  their last completed service game
      server_rally_ratio_cur_game   the current service game so far
      server_rally_ratio_match      all their service points so far this match
    1.0 (typical) where there is nothing to average: no known rally yet, the
    first point of a service game, or a match without rally data. Tiebreak
    points count toward the match average and, for the current game, only the
    current server's own points; a tiebreak is never a "last service game".
    """
    servers = match["point_server"].to_numpy()
    rally = (match["rally"] if "rally" in match.columns
             else pd.Series(np.nan, index=match.index)).to_numpy(dtype=float)
    keys = list(zip(match["set_no"].to_numpy(), match["game_no"].to_numpy()))
    n = len(match)
    out = {c: np.ones(n) for c in ("server_rally_ratio_last_game",
                                   "server_rally_ratio_cur_game",
                                   "server_rally_ratio_match")}
    tot = {1: [0.0, 0], 2: [0.0, 0]}            # this match, own serve: [sum, count]
    last = {1: None, 2: None}                    # last completed service game mean
    cur = {1: [0.0, 0], 2: [0.0, 0]}             # current game, per server
    game_servers: set = set()

    def ratio(total, count):
        return total / count / baseline if count else 1.0

    for i in range(n):
        if i and keys[i] != keys[i - 1]:
            if len(game_servers) == 1:           # a completed normal service game
                (g_s,) = game_servers
                if cur[g_s][1]:
                    last[g_s] = cur[g_s][0] / cur[g_s][1]
            cur = {1: [0.0, 0], 2: [0.0, 0]}
            game_servers = set()
        s = int(servers[i])
        out["server_rally_ratio_match"][i] = ratio(*tot[s])
        out["server_rally_ratio_cur_game"][i] = ratio(*cur[s])
        if last[s] is not None:
            out["server_rally_ratio_last_game"][i] = last[s] / baseline
        game_servers.add(s)
        if not np.isnan(rally[i]):
            tot[s][0] += rally[i]; tot[s][1] += 1
            cur[s][0] += rally[i]; cur[s][1] += 1
    for c, v in out.items():
        match[c] = v
    return match


def breaks_so_far(match: pd.DataFrame) -> pd.DataFrame:
    """Breaks of serve won and conceded so far in the match.

    A break is a completed service game won by the receiver; the game
    winner is the winner of its last point. A game only counts once it is
    over, so the game in progress never does. Tiebreaks (both players
    serve) are not breaks. Counts run over the whole match, not per set.

    Adds p1_breaks_won and p1_breaks_conceded (p2's are the same two
    swapped). No server-view versions: models get server_is_p1.
    """
    servers = match["point_server"].to_numpy()
    winners = match["point_winner"].to_numpy()
    games = list(zip(match["set_no"].to_numpy(), match["game_no"].to_numpy()))
    n = len(match)

    broken = {1: 0, 2: 0}            # service games lost, per player
    won = np.zeros(n, dtype=int)     # p1 breaks won (= p2 broken)
    conceded = np.zeros(n, dtype=int)
    game_servers: set = set()
    for i in range(n):
        if i and games[i] != games[i - 1]:
            # previous game just finished: was it a break?
            if len(game_servers) == 1:
                (s,) = game_servers
                if s in broken and winners[i - 1] != s:
                    broken[s] += 1
            game_servers = set()
        won[i], conceded[i] = broken[2], broken[1]
        game_servers.add(servers[i])

    match["p1_breaks_won"] = won
    match["p1_breaks_conceded"] = conceded
    return match


def score_context(match: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the running score before each point, and derive the
    pressure / game-situation features from it.

    One forward pass over the match. Everything recorded for a point uses only
    points strictly BEFORE it (plus the current point's server/set/game, which
    are known before the outcome), so nothing leaks the label.

    Game/set bookkeeping:
      - a game is won by the winner of its last point; it is credited when the
        (set_no, game_no) key changes, so the game in progress never counts;
      - a set is won by whoever has more games when set_no changes (a set is
        always 6-x, 7-5 or 7-6, so the winner always leads on games);
      - a tiebreak is decided from the games score and the event's rules
        (src/tennis_rules.py), so it is known before its first point.

    Within-game points give standard tennis pressure flags (deuce/ad handled by
    the "win by 2" test):
      - is_break_point  the receiver can take the server's game on this point;
      - is_game_point   the server can hold on this point;
      - is_set_point    either of those, when that game also decides the set
        (approx: a leader on >=5 games winning by >=1; ignores 7-6 set points).

    Adds the pressure flags (is_break_point, is_set_point, score_diff) plus a
    set of score counters, all POINT-scope.
    """
    servers = match["point_server"].to_numpy()
    winners = match["point_winner"].to_numpy()
    set_no = match["set_no"].to_numpy()
    game_no = match["game_no"].to_numpy()
    n = len(match)

    # columns to fill
    cols = {c: np.zeros(n, dtype=int) for c in (
        "is_break_point", "is_set_point", "is_game_point", "is_tiebreak",
        "score_diff", "set_diff", "point_idx_match", "cur_set_no",
        "games_done_in_set", "points_in_game", "p1_games_set", "p2_games_set",
        "p1_sets_won", "p2_sets_won", "server_pts_game", "receiver_pts_game")}

    pts = {1: 0, 2: 0}                 # points won in the current game
    games = {1: 0, 2: 0}              # games won in the current set
    sets = {1: 0, 2: 0}              # sets won in the match
    match_id = str(match["match_id"].iloc[0])

    for i in range(n):
        if i:
            if (set_no[i], game_no[i]) != (set_no[i - 1], game_no[i - 1]):
                games[int(winners[i - 1])] += 1       # credit completed game
                pts = {1: 0, 2: 0}
            if set_no[i] != set_no[i - 1]:
                set_winner = 1 if games[1] > games[2] else 2
                sets[set_winner] += 1                  # credit completed set
                games = {1: 0, 2: 0}

        s = int(servers[i])
        r = 2 if s == 1 else 1
        sp, rp = pts[s], pts[r]
        tiebreak = _is_tiebreak(match_id, set_no[i], games[1], games[2])

        # one-point-away tests (win by 2, min 4 points), skipped in a tiebreak
        recv_wins = (rp + 1 >= 4) and (rp + 1 - sp >= 2)
        serv_wins = (sp + 1 >= 4) and (sp + 1 - rp >= 2)
        bp = int((not tiebreak) and recv_wins)
        gp = int((not tiebreak) and serv_wins)
        sp_server = gp and games[s] >= 5 and games[s] - games[r] >= 1
        sp_recv = bp and games[r] >= 5 and games[r] - games[s] >= 1

        cols["is_break_point"][i] = bp
        cols["is_game_point"][i] = gp
        cols["is_set_point"][i] = int(bool(sp_server or sp_recv))
        cols["is_tiebreak"][i] = int(tiebreak)
        cols["score_diff"][i] = games[1] - games[2]       # p1 - p2 games, this set
        cols["set_diff"][i] = sets[1] - sets[2]
        cols["point_idx_match"][i] = i
        cols["cur_set_no"][i] = int(set_no[i])
        cols["games_done_in_set"][i] = games[1] + games[2]
        cols["points_in_game"][i] = sp + rp
        cols["p1_games_set"][i] = games[1]
        cols["p2_games_set"][i] = games[2]
        cols["p1_sets_won"][i] = sets[1]
        cols["p2_sets_won"][i] = sets[2]
        cols["server_pts_game"][i] = sp
        cols["receiver_pts_game"][i] = rp

        pts[int(winners[i])] += 1

    for c, v in cols.items():
        match[c] = v
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
    match["surprise_ewma"] = resid.ewm(alpha=0.15, min_periods=1).mean().fillna(0.0)
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
    rally_base = slam_rally_baseline(points)

    parts = []
    for mid, match in points.groupby("match_id", sort=False):
        match = match.copy()
        match = rolling_winrate(match, player_won_col)
        match = ewma_winrate(match, player_won_col)
        match = streak_length(match, player_won_col)
        match = rolling_serve_winpct(match)
        match = serve_winpct_match(match)
        match = serve_bayes(match)
        match = breaks_so_far(match)
        match = court_side(match)
        match = rally_features(match)
        match = server_rally_ratios(match, float(rally_base[mid]))
        match = score_context(match)
        match = surprise_feature(match, player_won_col, expected_col)
        parts.append(match)
    out = pd.concat(parts, ignore_index=True)

    # Interaction features (windows x server) push the feature count over 100
    # as the tabular requirement needs. TODO(team): add the interactions you
    # actually use and document the final count in the report.
    out["server_is_p1"] = (out["point_server"] == 1).astype(int)
    return out


class FeatureScope(str, Enum):
    """What information a feature is built from.

    MATCH  only information from before the match starts (player history).
           Fixed for the match; the server-view ones (elo_diff_server,
           p_serve_expected, expected_p1_win, ...) are the same pre-match
           values seen from whoever is serving, so they flip with the serve
           but are never updated by what happens in the match.
    POINT  updated during the match: built from points already played, or
           the state of the current point (who is serving, the score).
    """
    MATCH = "match"
    POINT = "point"


# Computed (other features are built from them) but not given to the models:
# each is an exact sum/difference of features the models already get, so it
# adds no information and makes the feature matrix singular (VIF = infinite,
# notebooks/03_multicollinearity.ipynb).
EXCLUDED_FEATURES = {
    "cur_set_no",          # = p1_sets_won + p2_sets_won + 1
    "score_diff",          # = p1_games_set - p2_games_set
    "games_done_in_set",   # = p1_games_set + p2_games_set
    "elo_diff",            # = p1_elo - p2_elo
    "elo_surface_diff",    # = p1_elo_surface - p2_elo_surface
}

MATCH_FEATURES = [c for c in PLAYER_MATCH_FEATURES + PLAYER_SERVER_VIEW_FEATURES
                  if c not in EXCLUDED_FEATURES]

POINT_FEATURES = (
    [f"winrate_w{w}" for w in WINDOW_SIZES]
    + [f"ewma_d{d}" for d in DECAY_RATES]
    + ["streak_signed", "surprise_ewma"]
    + [f"{who}_serve_win_g{k}" for k in SERVE_GAME_WINDOWS
       for who in ("p1", "p2", "server")]
    + ["p1_serve_win_match", "p2_serve_win_match",
       "p1_serve_points_match", "p2_serve_points_match"]
    + ["p1_serve_bayes_map", "p2_serve_bayes_map",
       "p1_serve_bayes_sd", "p2_serve_bayes_sd"]
    + ["p1_breaks_won", "p1_breaks_conceded"]
    + ["is_break_point", "is_set_point", "score_diff", "server_is_p1",
       "serve_from_ad_court"]
    + ["prev_rally_len", "rally_len_avg10", "match_rally_avg",
       "p1_long_rally_win_pct", "rally_data_share"]
    + ["server_rally_ratio_last_game", "server_rally_ratio_cur_game",
       "server_rally_ratio_match"]
    # score-context counters (score_context): running game/set score before
    # each point, plus game-situation pressure flags.
    + ["is_game_point", "is_tiebreak", "set_diff", "point_idx_match",
       "cur_set_no", "games_done_in_set", "points_in_game",
       "p1_games_set", "p2_games_set", "p1_sets_won", "p2_sets_won",
       "server_pts_game", "receiver_pts_game"]
)
POINT_FEATURES = [c for c in POINT_FEATURES if c not in EXCLUDED_FEATURES]

FEATURE_SCOPE: dict[str, FeatureScope] = {
    **{c: FeatureScope.MATCH for c in MATCH_FEATURES},
    **{c: FeatureScope.POINT for c in POINT_FEATURES},
}

FEATURE_COLUMNS = MATCH_FEATURES + POINT_FEATURES


def feature_columns(scope: FeatureScope | None = None) -> list[str]:
    """All feature columns, or only those of one scope."""
    if scope is None:
        return list(FEATURE_COLUMNS)
    return [c for c in FEATURE_COLUMNS if FEATURE_SCOPE[c] is scope]
