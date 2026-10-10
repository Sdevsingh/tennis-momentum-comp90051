"""No-momentum null for exp1: a two-level shuffle of the real points.

Rearranges the order of what actually happened in each match, never inventing
outcomes, and keeps every shuffled match legal tennis:

  "points"  shuffle the points WITHIN each game. A normal game's points all
            have the same server, so serve records are untouched. Only orders
            in which the game is decided on its last point are allowed (a game
            won 4-1 cannot become W W W W L), drawn uniformly at random from
            all such orders. In a tiebreak each server's points are shuffled
            among that server's own slots, again only into legal orders.
  "games"   shuffle the games WITHIN each set: each player's service games are
            permuted among their own service slots, so serve still alternates
            exactly as it did; a tiebreak stays as the set's last game; only
            orders in which the set is decided on its last game are allowed.
  "both"    games first, then points within each game.

What is kept exactly: every point and game result, each game's final score,
each player's serve record, set scores and winners, the match result, serve
alternation. What is scrambled: the order of points within games and of games
within sets -- i.e. any dependence of an outcome on what came just before it.

Whole rows move, so a point keeps its serve speed, rally length, etc.; set_no
and game_no stay with the slot (position), point_no is renumbered.

Where the real log is not legal tennis (a mislabelled server, a missing point,
a retirement mid-set), that game or set is left in its real order; the counts
are returned by `shuffle_stats`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import SEED
from .tennis_rules import tiebreak_rule

MODES = ("points", "games", "both")


# ---------------------------------------------------------------- legality

def _decided(a: int, b: int, target: int) -> bool:
    """Has a game/tiebreak to `target` points (win by 2) been won at a-b?"""
    return max(a, b) >= target and abs(a - b) >= 2


def _legal_points(winners: np.ndarray, target: int) -> bool:
    """Does this sequence of point winners (1/2) decide the game on its last point?"""
    a = b = 0
    for k, w in enumerate(winners):
        if w == 1:
            a += 1
        else:
            b += 1
        if _decided(a, b, target):
            return k == len(winners) - 1
    return False


def _set_decided(g1: int, g2: int, tb_at) -> bool:
    if tb_at is not None and g1 == g2 == tb_at:
        return False                               # goes to a tiebreak
    return max(g1, g2) >= 6 and abs(g1 - g2) >= 2


def _legal_games(game_winners, tb_at, has_tiebreak: bool) -> bool:
    """Is the set (non-tiebreak games, in order) decided exactly at its end?

    With a tiebreak, the set must reach the tiebreak score without being
    decided earlier; without one, the last game must decide it.
    """
    g = {1: 0, 2: 0}
    n = len(game_winners)
    for k, w in enumerate(game_winners):
        g[w] += 1
        if _set_decided(g[1], g[2], tb_at):
            return (not has_tiebreak) and k == n - 1
    return has_tiebreak and tb_at is not None and g[1] == g[2] == tb_at


# ------------------------------------------- uniform legal point orders (DP)

@lru_cache(maxsize=None)
def _n_paths(a: int, b: int, W: int, L: int, target: int) -> int:
    """Legal ways to go from a-b (winner-loser points) to W-L, decided at the end."""
    if (a, b) == (W, L):
        return 1
    if _decided(a, b, target):
        return 0                                   # decided early: not allowed
    n = 0
    if a < W:
        n += _n_paths(a + 1, b, W, L, target)
    if b < L:
        n += _n_paths(a, b + 1, W, L, target)
    return n


def _sample_order(W: int, L: int, target: int, rng) -> list[bool] | None:
    """A uniformly random legal order: True = point won by the game's winner."""
    if _n_paths(0, 0, W, L, target) == 0:
        return None
    a = b = 0
    out = []
    while (a, b) != (W, L):
        nw = _n_paths(a + 1, b, W, L, target) if a < W else 0
        nl = _n_paths(a, b + 1, W, L, target) if b < L else 0
        if rng.random() < nw / (nw + nl):
            out.append(True); a += 1
        else:
            out.append(False); b += 1
    return out


# ------------------------------------------------------------------ shuffles
# Everything below works on arrays of row positions; the table is reordered
# once at the end.

def _shuffle_game_points(idx: np.ndarray, win_all, srv_all, target: int, rng, stats) -> np.ndarray:
    """New row order for one game (idx = its rows, in playing order)."""
    win, srv = win_all[idx], srv_all[idx]
    is_tb = len(set(srv.tolist())) > 1
    if not _legal_points(win, target):
        stats["games_left_unshuffled"] += 1        # real log is not legal tennis
        return idx
    if not is_tb:
        gw = win[-1]
        order = _sample_order(int((win == gw).sum()), int((win != gw).sum()), target, rng)
        if order is None:
            stats["games_left_unshuffled"] += 1
            return idx
        won_rows = list(rng.permutation(np.flatnonzero(win == gw)))
        lost_rows = list(rng.permutation(np.flatnonzero(win != gw)))
        return idx[[won_rows.pop() if o else lost_rows.pop() for o in order]]
    # tiebreak: each server's points stay in that server's slots
    slots = {p: np.flatnonzero(srv == p) for p in (1, 2)}
    for _ in range(2000):
        local = np.arange(len(idx))
        for p, sl in slots.items():
            local[sl] = rng.permutation(sl)
        if _legal_points(win[local], target):
            return idx[local]
    stats["games_left_unshuffled"] += 1
    return idx


def _shuffle_set_games(games: list[np.ndarray], win_all, srv_all, tb_at, rng, stats) -> list[np.ndarray]:
    """Permute each player's service games among their own slots (tiebreak stays last)."""
    has_tb = len(set(srv_all[games[-1]].tolist())) > 1
    body = games[:-1] if has_tb else games
    servers = [srv_all[g[0]] for g in body]
    winners = [win_all[g[-1]] for g in body]
    clean = all(len(set(srv_all[g].tolist())) == 1 for g in body)
    if not clean or not _legal_games(winners, tb_at, has_tb):
        stats["sets_left_unshuffled"] += 1
        return games
    slots = {p: [k for k, sv in enumerate(servers) if sv == p] for p in (1, 2)}
    for _ in range(2000):
        order = list(range(len(body)))
        for p, sl in slots.items():
            for k, j in zip(sl, rng.permutation(sl)):
                order[k] = j
        if _legal_games([winners[j] for j in order], tb_at, has_tb):
            return [body[j] for j in order] + (games[-1:] if has_tb else [])
    stats["sets_left_unshuffled"] += 1
    return games


def _runs(keys: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of consecutive runs of equal keys."""
    cut = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    bounds = np.concatenate([[0], cut, [len(keys)]])
    return list(zip(bounds[:-1], bounds[1:]))


def shuffle_null(points: pd.DataFrame, mode: str = "both", seed: int = SEED,
                 stats: dict | None = None) -> pd.DataFrame:
    """The exp1 null: shuffle points within games and/or games within sets.

    `points` is the loader's point table (one row per point, playing order).
    Pass a dict as `stats` to get back how many games/sets were shuffled or
    left in their real order.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = np.random.default_rng(seed)
    st = {"games": 0, "games_left_unshuffled": 0, "sets": 0, "sets_left_unshuffled": 0}
    pts = points.reset_index(drop=True)
    mid = pts["match_id"].to_numpy()
    set_no = pts["set_no"].to_numpy()
    game_no = pts["game_no"].to_numpy()
    win_all = pts["point_winner"].to_numpy()
    srv_all = pts["point_server"].to_numpy()
    # integer keys for runs: match, (match, set), (match, set, game)
    m_key = pd.factorize(mid)[0]
    s_key = pd.factorize(pd.MultiIndex.from_arrays([m_key, set_no]))[0]
    g_key = pd.factorize(pd.MultiIndex.from_arrays([m_key, set_no, game_no]))[0]

    perm = np.empty(len(pts), dtype=np.int64)
    new_game_no = np.empty(len(pts), dtype=game_no.dtype)
    pos = 0
    for s0, s1 in _runs(s_key):
        match_id, sn = mid[s0], set_no[s0]
        tb_at, tb_target = tiebreak_rule(match_id, sn)
        games = [np.arange(a, b) for a, b in _runs(g_key[s0:s1])]
        games = [g + s0 for g in games]
        slot_game_no = [game_no[g[0]] for g in games]
        st["sets"] += 1
        if mode in ("games", "both"):
            games = _shuffle_set_games(games, win_all, srv_all, tb_at, rng, st)
        for k, g in enumerate(games):
            st["games"] += 1
            if mode in ("points", "both"):
                is_tb = len(set(srv_all[g].tolist())) > 1
                g = _shuffle_game_points(g, win_all, srv_all, tb_target if is_tb else 4, rng, st)
            perm[pos:pos + len(g)] = g
            new_game_no[pos:pos + len(g)] = slot_game_no[k]
            pos += len(g)
    res = pts.iloc[perm].reset_index(drop=True)
    res["game_no"] = new_game_no
    res["point_no"] = (res.groupby("match_id", sort=False).cumcount() + 1).astype(str)
    res["p1_won"] = (res["point_winner"] == 1).astype(int)
    res["server_won"] = (res["point_winner"] == res["point_server"]).astype(int)
    if stats is not None:
        stats.update(st)
    return res
