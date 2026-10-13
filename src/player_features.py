"""Pre-match player features for every Grand Slam point-by-point match.

The momentum question is whether point order matters BEYOND serve, score and
player strength, so the strength side has to be strong. Everything here is
known before the match starts and is frozen for all of its points -- nothing
is updated point by point.

One forward pass over every ATP/WTA tour match since 1968 (main tour +
challengers/qualifying/ITF; men and women are separate pools) keeps a state
per player and records it BEFORE each match:

  Elo (chess/FIDE)        start 1500; E = 1/(1+10^((Rb-Ra)/400));
                          R += K(S-E), K = 40 for the first 30 matches,
                          then 20, 10 once the player has reached 2400.
  TrueSkill (mu, sigma)   Herbrich et al. 2007 two-player update, with
                          TrueSkill 2's skill dynamics (Minka et al. 2018):
                          sigma^2 += tau^2 * days away (eq. 3) before a
                          match and += gamma^2 after it (eq. 2). sigma is
                          the rating's uncertainty; it grows with layoffs.
  Serve / return skill    S = serve points won above the tour average,
                          R = return points won above average. After each
                          match with stats, the residual of
                          spw = avg + S_server - R_returner updates both
                          players (TrueSkill 2 section 8: individual stats
                          as performance relative to the opponent). Step
                          size 1/(n+ETA0) shrinks new players to average.
                          Gives the expected chance the SERVER wins a point.
  Workload / rest         days since the previous event, matches in events
                          starting in the 30 days before this one, matches
                          and games already played in this event (incl.
                          qualifying), games in the previous match.
                          Dates are tournament START dates (always a
                          Monday), so rest is week-resolution; within an
                          event, workload is counted in games because
                          `minutes` is missing for much of the WTA data.
  Experience              rated matches played, prior Slam main-draw
                          matches, age.
  Career serve            share of serve points won over all earlier matches
                          with stats, and how many matches that is (the
                          prior for the in-match Bayesian serve estimate).
  Retirements             retired/withdrew in the last match, did the last
                          opponent retire, retirements+walkovers given in
                          the last 90 days.
  Head-to-head            wins against this opponent in all earlier
                          meetings (any level); retirements count, walkovers
                          don't.
  Surface ratings         a separate Elo and TrueSkill per surface (Hard,
                          Clay, Grass, Carpet), updated only by matches on
                          that surface. A player's first rating on a surface
                          starts from their overall rating at that moment
                          (TrueSkill 2 section 11: borrow across modes), with
                          extra TrueSkill uncertainty; Elo then uses the
                          new-player K for their first 30 matches there.
                          Surface sigma grows with days since the player's
                          last match ON THAT SURFACE.

Walkovers update only the retirement/workload state (no tennis was played),
never the ratings. Matches are ordered by tournament start date, then round
(Q1..F), then match number.

Linking slam matches to the tour files: the slam files have no shared player
id and spell names inconsistently ("M. Zverev", "Pm Tig"), so each slam match
is matched to a tour match in the same draw (tour, year, slam) by both
players' names, strict to loose. Ambiguous candidates are left unlinked.

Run:  python -m src.player_features  [--archive PATH]
  -> data/processed/match_player_features.parquet (one row per slam match_id)
"""
from __future__ import annotations

import argparse
import math
import re
import unicodedata
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ARCHIVE_DIR, PROCESSED_DIR, ELO_START, ELO_K_NEW,
                     ELO_K_DEFAULT, ELO_K_ELITE, ELO_NEW_GAMES, ELO_ELITE_RATING,
                     TS_MU0, TS_SIGMA0, TS_BETA, TS_GAMMA, TS_TAU_PER_DAY,
                     SERVE_ETA0, SERVE_ETA_MIN, SERVE_AVG_ALPHA,
                     SURFACE_INIT_EXTRA_SIGMA)

TOURS = {"atp": ["atp_matches_[0-9]*.csv", "atp_matches_qual_chall_*.csv"],
         "wta": ["wta_matches_[0-9]*.csv", "wta_matches_qual_itf_*.csv"]}

ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "R128": 10, "R64": 11,
               "R32": 12, "R16": 13, "RR": 14, "QF": 15, "SF": 16, "BR": 17,
               "F": 18}

SLAM_NAMES = {"ausopen": "australian open", "frenchopen": "roland garros",
              "wimbledon": "wimbledon", "usopen": "us open"}

USECOLS = ["tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date",
           "match_num", "round", "score", "winner_id", "winner_name",
           "winner_age", "loser_id", "loser_name", "loser_age",
           "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]

# Per-player pre-match values recorded for every match (prefix w_/l_ in the
# tour table, p1_/p2_ once linked to a slam match).
PLAYER_COLS = ["elo", "ts_mu", "ts_sigma", "serve_skill", "return_skill",
               "stat_matches", "days_since_prev_event", "matches_prev_30d",
               "matches_this_event", "games_this_event", "prev_match_games",
               "career_matches", "slam_matches", "age", "retired_last_match",
               "opp_retired_last_match", "retirements_90d", "h2h_wins",
               "elo_surface", "ts_surface_mu", "ts_surface_sigma", "surface_matches",
               "career_serve_win_pct"]

# Every slam is played on one surface, so the slam in match_id gives it.
SLAM_SURFACE = {"ausopen": "Hard", "frenchopen": "Clay", "wimbledon": "Grass",
                "usopen": "Hard"}


# ------------------------------------------------------------------ helpers

def expected_score(r_a: float, r_b: float) -> float:
    """Probability that A beats B under Elo."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def k_factor(n_games: int, reached_elite: bool) -> float:
    """FIDE K schedule."""
    if reached_elite:
        return ELO_K_ELITE
    if n_games < ELO_NEW_GAMES:
        return ELO_K_NEW
    return ELO_K_DEFAULT


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def trueskill_update(mu_w, s2_w, mu_l, s2_l):
    """Two-player, no-draw TrueSkill update (Herbrich et al. 2007)."""
    c2 = 2 * TS_BETA ** 2 + s2_w + s2_l
    c = math.sqrt(c2)
    t = (mu_w - mu_l) / c
    v = _phi(t) / max(_Phi(t), 1e-12)
    w = v * (v + t)
    return (mu_w + s2_w / c * v, s2_w * (1 - s2_w / c2 * w),
            mu_l - s2_l / c * v, s2_l * (1 - s2_l / c2 * w))


_erf_vec = np.vectorize(math.erf, otypes=[float])


def ts_win_prob(mu_a, sig_a, mu_b, sig_b):
    """P(A beats B) given both ratings AND their uncertainty."""
    return 0.5 * (1 + _erf_vec((mu_a - mu_b) /
                               np.sqrt(2 * (2 * TS_BETA ** 2 + sig_a ** 2 + sig_b ** 2))))


def _games(score: pd.Series) -> np.ndarray:
    """Total games in a score string like '6-4 3-6 7-6(5)' (0 for W/O)."""
    sets = score.fillna("").astype(str).str.findall(r"(\d+)-(\d+)")
    return np.array([sum(int(a) + int(b) for a, b in s) for s in sets])


def _days(date_int: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(pd.Series(date_int).astype("int64").astype(str), format="%Y%m%d")
    return (d - pd.Timestamp("1960-01-01")).dt.days.to_numpy()


class _Player:
    __slots__ = ("elo", "n", "elite", "mu", "s2", "S", "R", "n_stat",
                 "last_day", "event", "event_matches", "event_games",
                 "prev_games", "slam", "recent", "quits", "last_status", "surf",
                 "sp_won", "sp_total")

    def __init__(self):
        self.elo, self.n, self.elite = float(ELO_START), 0, False
        self.mu, self.s2 = TS_MU0, TS_SIGMA0 ** 2
        self.S = self.R = 0.0
        self.n_stat = 0
        self.sp_won = self.sp_total = 0.0  # career serve points won / played
        self.last_day = None
        self.event, self.event_matches, self.event_games = None, 0, 0
        self.prev_games = 0
        self.slam = 0
        self.recent = deque()      # event start days of matches played
        self.quits = deque()       # days of retirements / walkovers given
        self.last_status = 0       # 1 retired/withdrew, 2 opponent did
        self.surf = {}             # surface -> _Surface


class _Surface:
    """One player's rating on one surface."""
    __slots__ = ("elo", "n", "elite", "mu", "s2", "last_day")

    def __init__(self, p: _Player, day: int):
        # Start from the player's overall rating right now, less certain.
        self.elo, self.n, self.elite = p.elo, 0, p.elite
        self.mu = p.mu
        self.s2 = min(p.s2 + SURFACE_INIT_EXTRA_SIGMA ** 2, TS_SIGMA0 ** 2)
        self.last_day = day


def _surface_state(p: _Player, surface, day: int):
    """The player's rating on `surface`, drifted to `day` (None if unknown)."""
    if not isinstance(surface, str):
        return None
    s = p.surf.get(surface)
    if s is None:
        s = p.surf[surface] = _Surface(p, day)
    elif day > s.last_day:
        s.s2 = min(s.s2 + TS_TAU_PER_DAY ** 2 * (day - s.last_day), TS_SIGMA0 ** 2)
        s.last_day = day
    return s


def _career_spw(p: _Player) -> float:
    """Share of serve points won over all earlier matches with stats."""
    return p.sp_won / p.sp_total if p.sp_total > 0 else np.nan


def _surface_snapshot(s) -> list:
    if s is None:
        return [np.nan] * 4
    return [s.elo, s.mu, math.sqrt(s.s2), s.n]


# ------------------------------------------------------------ forward pass

def load_tour_matches(archive: Path, tour: str) -> pd.DataFrame:
    """All singles matches for one tour, in playing order."""
    files = sorted(f for pat in TOURS[tour] for f in (archive / tour).glob(pat))
    df = pd.concat([pd.read_csv(f, usecols=USECOLS, low_memory=False)
                    for f in files], ignore_index=True)
    df["tour"] = tour
    df = df[df["winner_id"].notna() & df["loser_id"].notna()
            & df["tourney_date"].notna()].copy()
    score = df["score"].fillna("").astype(str)
    df["walkover"] = score.str.contains("W/O|Walkover", case=False) | (score == "")
    df["retired"] = score.str.contains("RET", case=False)
    df["surface"] = df["surface"].str.capitalize()      # a few rows say "clay"
    df["games"] = _games(df["score"])
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(14)
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"],
                        kind="stable").reset_index(drop=True)
    df["day"] = _days(df["tourney_date"].to_numpy())
    return df


def _snapshot(p: _Player, day: int, event, age) -> list:
    """The player's pre-match feature values (order = PLAYER_COLS)."""
    while p.recent and p.recent[0] < day - 30:
        p.recent.popleft()
    while p.quits and p.quits[0] < day - 90:
        p.quits.popleft()
    same_event = p.event == event
    prev_30 = sum(1 for d in p.recent if d < day)       # other, earlier events
    since = 365 if p.last_day is None else min(day - p.last_day, 365)
    if same_event:
        since = 0
    return [p.elo, p.mu, math.sqrt(p.s2), p.S, p.R, p.n_stat, since, prev_30,
            p.event_matches if same_event else 0,
            p.event_games if same_event else 0,
            p.prev_games, p.n, p.slam, age,
            int(p.last_status == 1), int(p.last_status == 2), len(p.quits)]


def run_forward_pass(m: pd.DataFrame) -> pd.DataFrame:
    """Walk matches in order; record both players' PRE-match state."""
    players: dict = {}
    h2h: dict = {}                 # (winner_id, loser_id) -> wins of winner over loser
    w_rows, l_rows, avg_rows = [], [], []
    spw_avg = None

    cols = ["winner_id", "loser_id", "day", "tourney_id", "tourney_level",
            "round_order", "walkover", "retired", "games", "winner_age",
            "loser_age", "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt",
            "l_1stWon", "l_2ndWon", "surface"]
    for (wid, lid, day, event, level, rnd, wo, ret, games, wage, lage,
         wsv, w1, w2, lsv, l1, l2, surface) in zip(*(m[c].to_numpy() for c in cols)):
        W = players.setdefault(wid, _Player())
        L = players.setdefault(lid, _Player())

        # TrueSkill 2 eq. 3: skill drifts with time away -> sigma grows.
        for p in (W, L):
            if p.last_day is not None and day > p.last_day:
                p.s2 = min(p.s2 + TS_TAU_PER_DAY ** 2 * (day - p.last_day),
                           TS_SIGMA0 ** 2)

        Ws, Ls = _surface_state(W, surface, day), _surface_state(L, surface, day)

        w_rows.append(_snapshot(W, day, event, wage) + [h2h.get((wid, lid), 0)]
                      + _surface_snapshot(Ws) + [_career_spw(W)])
        l_rows.append(_snapshot(L, day, event, lage) + [h2h.get((lid, wid), 0)]
                      + _surface_snapshot(Ls) + [_career_spw(L)])
        avg_rows.append(np.nan if spw_avg is None else spw_avg)

        # ---- updates (after the snapshot, so nothing leaks into it)
        if not wo:
            h2h[(wid, lid)] = h2h.get((wid, lid), 0) + 1
            # Surface Elo + TrueSkill, same rules as the overall ones
            if Ws is not None:
                e_s = expected_score(Ws.elo, Ls.elo)
                kws, kls = k_factor(Ws.n, Ws.elite), k_factor(Ls.n, Ls.elite)
                Ws.elo += kws * (1.0 - e_s)
                Ls.elo -= kls * (1.0 - e_s)
                Ws.elite |= Ws.elo >= ELO_ELITE_RATING
                Ls.elite |= Ls.elo >= ELO_ELITE_RATING
                Ws.mu, Ws.s2, Ls.mu, Ls.s2 = trueskill_update(Ws.mu, Ws.s2, Ls.mu, Ls.s2)
                Ws.s2 += TS_GAMMA ** 2
                Ls.s2 += TS_GAMMA ** 2
                Ws.n += 1
                Ls.n += 1
            # Elo
            e_w = expected_score(W.elo, L.elo)
            kw, kl = k_factor(W.n, W.elite), k_factor(L.n, L.elite)
            W.elo += kw * (1.0 - e_w)
            L.elo -= kl * (1.0 - e_w)
            W.elite |= W.elo >= ELO_ELITE_RATING
            L.elite |= L.elo >= ELO_ELITE_RATING
            # TrueSkill, then eq. 2 per-match drift
            W.mu, W.s2, L.mu, L.s2 = trueskill_update(W.mu, W.s2, L.mu, L.s2)
            W.s2 += TS_GAMMA ** 2
            L.s2 += TS_GAMMA ** 2
            W.n += 1
            L.n += 1
            # Serve / return skill
            if wsv > 0 and lsv > 0 and not np.isnan(w1 + w2 + l1 + l2):
                spw_w, spw_l = (w1 + w2) / wsv, (l1 + l2) / lsv
                if spw_avg is None:
                    spw_avg = (spw_w + spw_l) / 2
                ew = max(1 / (W.n_stat + SERVE_ETA0), SERVE_ETA_MIN)
                el = max(1 / (L.n_stat + SERVE_ETA0), SERVE_ETA_MIN)
                r_w = spw_w - (spw_avg + W.S - L.R)     # W serving
                r_l = spw_l - (spw_avg + L.S - W.R)     # L serving
                W.S += ew * r_w
                L.R -= el * r_w
                L.S += el * r_l
                W.R -= ew * r_l
                W.n_stat += 1
                L.n_stat += 1
                W.sp_won += w1 + w2
                W.sp_total += wsv
                L.sp_won += l1 + l2
                L.sp_total += lsv
                spw_avg += SERVE_AVG_ALPHA * ((spw_w + spw_l) / 2 - spw_avg)
            if level == "G" and rnd >= 10:
                W.slam += 1
                L.slam += 1

        # Workload, rest and retirement state (walkovers included).
        for p, quit_, opp_quit in ((W, False, wo or ret), (L, wo or ret, False)):
            if p.event != event:
                p.event, p.event_matches, p.event_games = event, 0, 0
            if not wo:
                p.event_matches += 1
                p.event_games += games
                p.prev_games = games
                p.recent.append(day)
            p.last_day = day
            p.last_status = 1 if quit_ else (2 if opp_quit else 0)
            if quit_:
                p.quits.append(day)

    out = m.copy()
    w = pd.DataFrame(w_rows, columns=[f"w_{c}" for c in PLAYER_COLS], index=m.index)
    lo = pd.DataFrame(l_rows, columns=[f"l_{c}" for c in PLAYER_COLS], index=m.index)
    out = pd.concat([out, w, lo], axis=1)
    out["spw_avg"] = avg_rows
    return out


# ------------------------------------------------- linking slam -> tour rows

def _tokens(name) -> list[str]:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[-.]", " ", s.lower())
    s = re.sub(r"[^a-z ]", "", s)
    return [t for t in s.split() if t not in {"jr", "sr", "ii", "iii"}]


def _keys(name) -> dict[str, object]:
    t = _tokens(name)
    if not t:
        return {"full": None, "init_last": None, "last": None, "set": frozenset()}
    return {"full": " ".join(t), "init_last": (t[0][0], t[-1]), "last": t[-1],
            "set": frozenset(t)}


def _same(a: dict, b: dict, level: str) -> bool:
    if level == "loose":
        # surname-order flips ("Zhang Shuai" / "Shuai Zhang"): each name's
        # surname appears somewhere in the other.
        return bool(a["set"]) and a["last"] in b["set"] and b["last"] in a["set"]
    return a[level] is not None and a[level] == b[level]


def link_slam_matches(slam: pd.DataFrame, rated: pd.DataFrame) -> pd.DataFrame:
    """Attach both players' pre-match features to each slam match_id."""
    rated = rated[(rated["tourney_level"] == "G") & (rated["round_order"] >= 10)
                  & ~rated["walkover"]].copy()
    rated["year"] = rated["tourney_date"] // 10000
    rated["slam"] = None
    tn = rated["tourney_name"].str.lower()
    for slam_key, name in SLAM_NAMES.items():
        rated.loc[tn.str.contains(name), "slam"] = slam_key
    rated["wk"] = rated["winner_name"].map(_keys)
    rated["lk"] = rated["loser_name"].map(_keys)
    draws = {k: g for k, g in rated.groupby(["tour", "year", "slam"])}

    rows = []
    for m in slam.itertuples(index=False):
        rec = {"match_id": m.match_id, "tour": m.tour}
        draw = draws.get((m.tour, m.year, m.slam))
        k1, k2 = _keys(m.player1), _keys(m.player2)
        hit = None
        if draw is not None:
            for level in ("full", "init_last", "last", "loose"):
                cand = []
                for r in draw.itertuples(index=False):
                    if _same(k1, r.wk, level) and _same(k2, r.lk, level):
                        cand.append((r, True))     # player1 won
                    elif _same(k1, r.lk, level) and _same(k2, r.wk, level):
                        cand.append((r, False))    # player2 won
                if len(cand) == 1:
                    hit = cand[0] + (level,)
                    break
                if len(cand) > 1:
                    break                          # ambiguous: don't guess
        if hit is not None:
            r, p1_won, level = hit
            a, b = ("w", "l") if p1_won else ("l", "w")
            rec.update({
                "p1_player_id": r.winner_id if p1_won else r.loser_id,
                "p2_player_id": r.loser_id if p1_won else r.winner_id,
                "p1_won_match": int(p1_won),
                "spw_avg": r.spw_avg,
                "tourney_id": r.tourney_id,
                "link": level,
            })
            for c in PLAYER_COLS:
                rec[f"p1_{c}"] = getattr(r, f"{a}_{c}")
                rec[f"p2_{c}"] = getattr(r, f"{b}_{c}")
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["elo_diff"] = out["p1_elo"] - out["p2_elo"]
    out["p1_elo_win_prob"] = 1.0 / (1.0 + 10.0 ** (-out["elo_diff"] / 400.0))
    ok = out["p1_ts_mu"].notna()
    out["p1_ts_win_prob"] = np.nan
    out.loc[ok, "p1_ts_win_prob"] = ts_win_prob(
        out.loc[ok, "p1_ts_mu"].to_numpy(float), out.loc[ok, "p1_ts_sigma"].to_numpy(float),
        out.loc[ok, "p2_ts_mu"].to_numpy(float), out.loc[ok, "p2_ts_sigma"].to_numpy(float))
    # Expected chance the server wins a point, for each player serving.
    out["p1_serve_expected"] = (out["spw_avg"] + out["p1_serve_skill"]
                                - out["p2_return_skill"]).clip(0.05, 0.95)
    out["p2_serve_expected"] = (out["spw_avg"] + out["p2_serve_skill"]
                                - out["p1_return_skill"]).clip(0.05, 0.95)
    # Head-to-head: meetings so far, and player 1's share of them smoothed
    # toward 0.5 ((wins + 1) / (meetings + 2)), so 1-0 isn't read as 100%.
    out["h2h_matches"] = out["p1_h2h_wins"] + out["p2_h2h_wins"]
    out["p1_h2h_share"] = (out["p1_h2h_wins"] + 1) / (out["h2h_matches"] + 2)
    # Surface ratings (surface from the slam, so unlinked matches have it too).
    out["surface"] = out["match_id"].str.split("-").str[1].map(SLAM_SURFACE)
    out["elo_surface_diff"] = out["p1_elo_surface"] - out["p2_elo_surface"]
    out["p1_elo_surface_win_prob"] = 1.0 / (1.0 + 10.0 ** (-out["elo_surface_diff"] / 400.0))
    ok = out["p1_ts_surface_mu"].notna() & out["p2_ts_surface_mu"].notna()
    out["p1_ts_surface_win_prob"] = np.nan
    out.loc[ok, "p1_ts_surface_win_prob"] = ts_win_prob(
        out.loc[ok, "p1_ts_surface_mu"].to_numpy(float), out.loc[ok, "p1_ts_surface_sigma"].to_numpy(float),
        out.loc[ok, "p2_ts_surface_mu"].to_numpy(float), out.loc[ok, "p2_ts_surface_sigma"].to_numpy(float))
    return out


def load_slam_matches(archive: Path) -> pd.DataFrame:
    files = sorted((archive / "slam_pointbypoint").glob("*-matches.csv"))
    slam = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                     ignore_index=True)
    first = slam["match_num"].astype(str).str[0]
    slam["tour"] = first.map({"1": "atp", "M": "atp", "2": "wta", "W": "wta"})
    return slam[["match_id", "year", "slam", "tour", "player1", "player2"]]


# --------------------------------------------- per-point feature attachment

MATCH_FILE = PROCESSED_DIR / "match_player_features.parquet"

# Per-player columns that go into the design matrix (both p1_ and p2_).
_PER_PLAYER = ["elo", "ts_mu", "ts_sigma", "serve_skill", "return_skill",
               "days_since_prev_event", "matches_prev_30d", "matches_this_event",
               "games_this_event", "prev_match_games", "career_matches",
               "slam_matches", "age", "retired_last_match",
               "opp_retired_last_match", "retirements_90d",
               "elo_surface", "ts_surface_mu", "ts_surface_sigma", "surface_matches",
               "career_serve_win_pct", "stat_matches"]

# Long-tailed features the models get on a log scale (feature name gets a
# "log_" prefix; the raw column stays in the table). Counts and days use
# log(1 + x) because they include 0; the TrueSkill sigmas are always > 0.
# Trees are unaffected (monotone transform); it stops a few extreme players
# dominating the linear model.
LOG1P_FEATURES = ["days_since_prev_event", "games_this_event", "slam_matches"]
LOG_FEATURES = ["ts_sigma", "ts_surface_sigma"]
LOG1P_MATCH_FEATURES = ["h2h_matches"]


def _model_name(c: str) -> str:
    return f"log_{c}" if c in LOG1P_FEATURES + LOG_FEATURES else c


# Same value on every point of a match.
PLAYER_MATCH_FEATURES = (
    [f"p{i}_{_model_name(c)}" for c in _PER_PLAYER for i in (1, 2)]
    + ["elo_diff", "p1_elo_win_prob", "p1_ts_win_prob",
       "log_h2h_matches", "p1_h2h_win_pct", "p1_h2h_share",
       "elo_surface_diff", "p1_elo_surface_win_prob", "p1_ts_surface_win_prob",
       "player_features_known"]
)
# Also pre-match information, seen from the current server's side: they flip
# when the serve changes but are never updated by the match itself.
PLAYER_SERVER_VIEW_FEATURES = [
    "elo_diff_server", "ts_diff_server", "elo_surface_diff_server",
    "ts_surface_diff_server", "p_serve_expected", "expected_p1_win",
]
PLAYER_FEATURES = PLAYER_MATCH_FEATURES + PLAYER_SERVER_VIEW_FEATURES


def add_player_features(points: pd.DataFrame, path: Path = MATCH_FILE) -> pd.DataFrame:
    """Join pre-match player features onto the point table.

    Every point of a match gets the same per-player values (frozen at the
    start of the match). Server-perspective columns flip with the serve:
      elo_diff_server / ts_diff_server  server minus receiver
      p_serve_expected                  expected P(server wins this point)
      expected_p1_win                   the same, from player 1's side --
                                        the baseline the surprise feature
                                        measures momentum against.

    Matches with no link (or simulated matches) get neutral values
    (medians, equal players) and player_features_known = 0. Not NaN: the
    experiments fillna(0), and e.g. an Elo of 0 would be wildly informative.
    """
    per = [f"p{i}_{c}" for c in _PER_PLAYER for i in (1, 2)]
    keep = ["match_id"] + per + ["elo_diff", "p1_elo_win_prob", "p1_ts_win_prob",
                                 "p1_h2h_wins", "h2h_matches", "p1_h2h_share",
                                 "elo_surface_diff", "p1_elo_surface_win_prob",
                                 "p1_ts_surface_win_prob",
                                 "p1_serve_expected", "p2_serve_expected"]
    if path.exists():
        mf = pd.read_parquet(path)[keep]
    else:
        print(f"  [player_features] {path} not found; run "
              "`python -m src.player_features`. Using neutral values.")
        mf = pd.DataFrame(columns=keep)
    out = points.merge(mf, on="match_id", how="left")
    out["player_features_known"] = out["p1_elo"].notna().astype(int)

    for c in _PER_PLAYER:
        pooled = mf[[f"p1_{c}", f"p2_{c}"]].to_numpy(dtype=float)
        neutral = float(np.nanmedian(pooled)) if np.isfinite(pooled).any() else 0.0
        if c == "stat_matches":
            neutral = 0.0   # no known history -> widest serve prior in features.py
        for i in (1, 2):
            out[f"p{i}_{c}"] = out[f"p{i}_{c}"].astype(float).fillna(neutral)
    serve = mf[["p1_serve_expected", "p2_serve_expected"]].to_numpy(dtype=float)
    serve_neutral = float(np.nanmedian(serve)) if np.isfinite(serve).any() else 0.6
    out["elo_diff"] = out["elo_diff"].astype(float).fillna(0.0)
    out["p1_elo_win_prob"] = out["p1_elo_win_prob"].astype(float).fillna(0.5)
    out["p1_ts_win_prob"] = out["p1_ts_win_prob"].astype(float).fillna(0.5)
    out["h2h_matches"] = out["h2h_matches"].astype(float).fillna(0.0)
    # Head-to-head win %: player 1's wins / earlier meetings (0.5 if none;
    # p2's is 1 minus this).
    p1_h2h = out["p1_h2h_wins"].astype(float).fillna(0.0)
    out["p1_h2h_win_pct"] = np.where(out["h2h_matches"] > 0,
                                     p1_h2h / out["h2h_matches"].clip(lower=1), 0.5)
    out = out.drop(columns="p1_h2h_wins")
    out["p1_h2h_share"] = out["p1_h2h_share"].astype(float).fillna(0.5)
    out["elo_surface_diff"] = out["elo_surface_diff"].astype(float).fillna(0.0)
    out["p1_elo_surface_win_prob"] = out["p1_elo_surface_win_prob"].astype(float).fillna(0.5)
    out["p1_ts_surface_win_prob"] = out["p1_ts_surface_win_prob"].astype(float).fillna(0.5)
    # Not a model feature (yet): which surface the match is on.
    out["surface"] = out["match_id"].astype(str).str.split("-").str[1].map(SLAM_SURFACE)
    for c in ("p1_serve_expected", "p2_serve_expected"):
        out[c] = out[c].astype(float).fillna(serve_neutral)

    p1_serving = (out["point_server"] == 1).to_numpy()
    sign = np.where(p1_serving, 1.0, -1.0)
    out["elo_diff_server"] = sign * out["elo_diff"]
    out["ts_diff_server"] = sign * (out["p1_ts_mu"] - out["p2_ts_mu"])
    out["elo_surface_diff_server"] = sign * out["elo_surface_diff"]
    out["ts_surface_diff_server"] = sign * (out["p1_ts_surface_mu"] - out["p2_ts_surface_mu"])
    out["p_serve_expected"] = np.where(p1_serving, out["p1_serve_expected"],
                                       out["p2_serve_expected"])
    out["expected_p1_win"] = np.where(p1_serving, out["p1_serve_expected"],
                                      1.0 - out["p2_serve_expected"])
    for c in LOG1P_FEATURES:
        for i in (1, 2):
            out[f"p{i}_log_{c}"] = np.log1p(out[f"p{i}_{c}"].astype(float).clip(lower=0))
    for c in LOG_FEATURES:
        for i in (1, 2):
            out[f"p{i}_log_{c}"] = np.log(out[f"p{i}_{c}"].astype(float))
    for c in LOG1P_MATCH_FEATURES:
        out[f"log_{c}"] = np.log1p(out[c].astype(float).clip(lower=0))
    # p1_/p2_serve_expected are kept (not model features): they seed the
    # rolling serve win % before a player has served (features.py).
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", type=Path, default=ARCHIVE_DIR,
                    help="folder holding atp/, wta/ and slam_pointbypoint/")
    archive = ap.parse_args().archive

    rated = pd.concat([run_forward_pass(load_tour_matches(archive, t)) for t in TOURS],
                      ignore_index=True)
    print(f"Processed {len(rated):,} tour matches.")
    out = link_slam_matches(load_slam_matches(archive), rated)

    out.to_parquet(MATCH_FILE)
    linked = out["p1_elo"].notna()
    print(f"Wrote {MATCH_FILE}: {len(out):,} slam matches, {linked.mean():.1%} linked.")
    print(out.groupby("tour")["p1_elo"].apply(lambda s: s.notna().mean()).round(3).to_string())
    print(out["link"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
