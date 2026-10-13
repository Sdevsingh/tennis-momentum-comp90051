"""Load and clean Sackmann Grand Slam point-by-point data.

Dataset: https://github.com/JeffSackmann/tennis_slam_pointbypoint
Coverage: most main-draw singles Grand Slam matches since 2011.

IMPORTANT (from the dataset README): column availability varies a lot by
tournament and year. Serve speed, 1st/2nd-serve indicators and rally length
are missing for many matches. This loader therefore:
  - discovers columns at runtime rather than hard-coding a fixed schema,
  - normalises the handful of columns we actually depend on,
  - records, per match, which optional fields are usable (so features.py can
    decide whether the "lucky vs earned point" experiment is possible).

The columns we rely on (present in essentially all '-points.csv' files):
  match_id, PointServer (1/2), PointWinner (1/2), SetNo, GameNo, PointNumber,
  plus score columns. Optional-but-useful: Rally, Speed_KMH, P1Ace, P2Ace,
  P1Winner, P2Winner, P1UnfErr, P2UnfErr.

This module is intentionally defensive: it should run and tell you what it
found, rather than crashing on a tournament with an odd schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR
from .tennis_rules import sets_to_win, tiebreak_rule

# Canonical names we use downstream -> list of possible source spellings.
COLUMN_ALIASES = {
    "point_server": ["PointServer", "Server"],
    "point_winner": ["PointWinner"],
    "set_no": ["SetNo", "set"],
    "game_no": ["GameNo", "game"],
    "point_no": ["PointNumber", "Pt"],
    "speed_kmh": ["Speed_KMH", "ServeSpeed"],
    "p1_ace": ["P1Ace"],
    "p2_ace": ["P2Ace"],
    "p1_winner": ["P1Winner"],
    "p2_winner": ["P2Winner"],
    "p1_unferr": ["P1UnfErr"],
    "p2_unferr": ["P2UnfErr"],
}

REQUIRED = ["point_server", "point_winner", "set_no", "game_no", "point_no"]


def _rally_length(df: pd.DataFrame) -> pd.Series:
    """Shots in each point's rally (serve = shot 1), NaN where not recorded.

    The source has two columns: `Rally` (2011) and `RallyCount` (2016+); no
    Slam recorded rally length in 2012-2015. Both count the serve as shot 1
    (an odd count goes with the server winning the point), but their zeros
    mix three things:
      - double fault -> a genuine 0 (no ball in play); kept as 0;
      - ace -> should be 1; AO and RG 2018-2021 write 0, recoded to 1;
      - anything else -> not recorded (untracked matches are all zeros,
        and tracking gaps come in runs); set to NaN.
    """
    def col(name):
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    raw = col("RallyCount") if "RallyCount" in df.columns else col("Rally")
    ace = (col("P1Ace").fillna(0) + col("P2Ace").fillna(0)) > 0
    double_fault = (col("P1DoubleFault").fillna(0) + col("P2DoubleFault").fillna(0)) > 0
    rally = raw.where(raw > 0)                       # positive counts are real
    rally = rally.mask((raw == 0) & double_fault, 0.0)
    rally = rally.mask((raw == 0) & ace & ~double_fault, 1.0)
    return rally


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to whatever this file actually calls them."""
    found = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        for c in candidates:
            if c in df.columns:
                found[canonical] = c
                break
    return found


def load_points_file(path: Path) -> pd.DataFrame | None:
    """Load one '-points.csv', rename to canonical columns, tag match_id.

    Returns None (with a warning) if the file lacks a required column, so one
    bad tournament doesn't sink the whole load.
    """
    df = pd.read_csv(path)
    colmap = _resolve_columns(df)
    missing = [c for c in REQUIRED if c not in colmap]
    if missing:
        print(f"  [skip] {path.name}: missing {missing}", file=sys.stderr)
        return None

    df["rally"] = _rally_length(df)
    out = df.rename(columns={v: k for k, v in colmap.items()})
    keep = ["match_id"] + [c for c in COLUMN_ALIASES if c in out.columns] + ["rally"]
    # Each file is a whole tournament; the source's own match_id
    # (e.g. 2023-usopen-1701) identifies the match.
    out = out[keep].copy()
    # PointServer / PointWinner are 1/2 in the source. Drop rows where winner
    # is 0 (unplayed / data gaps).
    out = out[out["point_winner"].isin([1, 2])]
    out["point_server"] = out["point_server"].astype("Int64")
    return out.reset_index(drop=True)


def _game_done(a: int, b: int, target: int) -> bool:
    """Is a game (or tiebreak) to `target` points, win by two, over at a-b?"""
    return max(a, b) >= target and abs(a - b) >= 2


def _set_done(g1: int, g2: int, tb_at: int | None) -> bool:
    hi, lo = max(g1, g2), min(g1, g2)
    return (hi >= 6 and hi - lo >= 2) or (tb_at is not None and hi == tb_at + 1 and lo == tb_at)


def _match_reason(mid: str, set_no, game_no, server, winner, point_no) -> str | None:
    """Replay one match under the rules; the first way its log breaks them."""
    if (np.diff(set_no) < 0).any():
        return "set number goes backwards"
    key = set_no * 10_000 + game_no
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    if len(np.unique(key)) != len(starts):
        return "a game appears twice"
    if point_no is not None and (np.diff(point_no) != 1).any():
        return "point numbers skip (missing points)"

    sets, games, cur_set, prev_first = {1: 0, 2: 0}, {1: 0, 2: 0}, set_no[0], None
    for i, (a, b) in enumerate(zip(starts, ends)):
        last_game = i == len(starts) - 1
        if set_no[a] != cur_set:
            if not _set_done(games[1], games[2], tiebreak_rule(mid, cur_set)[0]):
                return "set ends before it is won"
            sets[1 if games[1] > games[2] else 2] += 1
            if max(sets.values()) >= sets_to_win(mid):
                return "points after the match is over"
            games, cur_set = {1: 0, 2: 0}, set_no[a]
        tb_at, tb_target = tiebreak_rule(mid, cur_set)
        rule_tb = tb_at is not None and games[1] == games[2] == tb_at
        srv = server[a:b]
        first = srv[0]
        if prev_first is not None and first == prev_first:
            return "server does not alternate"
        if rule_tb:
            want = np.array([first if ((j + 1) // 2) % 2 == 0 else 3 - first
                             for j in range(b - a)])
            if not (srv == want).all():
                return "tiebreak serve pattern / tiebreak not where the rules put it"
        elif (srv != first).any():
            return "two servers in a normal game"
        prev_first = first
        x = y = 0
        for k in range(a, b):
            if winner[k] == 1:
                x += 1
            else:
                y += 1
            if _game_done(x, y, tb_target if rule_tb else 4) and k != b - 1:
                return "game continues after it is won"
        if not _game_done(x, y, tb_target if rule_tb else 4) and not last_game:
            return "game ends before it is won"
        games[int(winner[b - 1])] += 1
    return None


def corrupt_match_reasons(table: pd.DataFrame) -> dict[str, str]:
    """Matches whose point log cannot be legal tennis -> the first reason found.

    Every match is replayed from point_winner / point_server under the rules
    (src/tennis_rules.py: 4-point games won by two, tiebreaks at the set's
    tiebreak score to the set's target, serve alternating by game and in the
    tiebreak pattern, sets won at 6 by two or by the tiebreak, best of 5 / 3).
    A match is corrupt if its log breaks them anywhere: the set number goes
    backwards, a game appears twice, point numbers skip, a set ends before it
    is won, points continue after the match is over, the server does not
    alternate, a normal game has two servers or a tiebreak the wrong pattern
    (or sits where the rules have no tiebreak), a game continues after it is
    won, or a game other than the match's last ends before it is won.

    Only the very last game may be unfinished (retirements, logs cut short),
    so incomplete matches are kept. Every score-based feature would be wrong
    for the corrupt ones, and the null shuffle would keep their broken games
    and sets in real order, so they are dropped.
    """
    reasons = {}
    pn = pd.to_numeric(table["point_no"], errors="coerce") if "point_no" in table else None
    for mid, idx in table.groupby("match_id", sort=False).indices.items():
        m = table.iloc[idx]
        p = None
        if pn is not None:
            p = pn.iloc[idx].to_numpy()
            if np.isnan(p).any():
                p = None                 # text point numbers ('0X'): skip the gap check
        why = _match_reason(mid, m["set_no"].to_numpy(int), m["game_no"].to_numpy(int),
                            m["point_server"].to_numpy(int), m["point_winner"].to_numpy(int), p)
        if why:
            reasons[mid] = why
    return reasons


def build_point_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate every '-points.csv' under raw_dir into one tidy table.

    One row per point, in match order. Adds:
      - server_won: 1 if the server won the point (the natural, serve-adjusted
        target; momentum is tested as information *beyond* this),
      - p1_won: 1 if player 1 won the point (raw target).
    """
    files = sorted(raw_dir.glob("*-points.csv"))
    if not files:
        raise FileNotFoundError(
            f"No '*-points.csv' found in {raw_dir}. See data/README.md for how "
            "to fetch the Sackmann data. (Data is intentionally not committed.)"
        )

    frames = []
    for p in files:
        df = load_points_file(p)
        if df is not None:
            frames.append(df)

    table = pd.concat(frames, ignore_index=True)
    # point_no is read as int in some files and text in others ('0X'/'0Y'
    # rows); keep it text so the table can be saved to parquet.
    table["point_no"] = table["point_no"].astype(str)
    table["p1_won"] = (table["point_winner"] == 1).astype(int)
    table["server_won"] = (table["point_winner"] == table["point_server"]).astype(int)

    # Drop matches whose log is not legal tennis (see corrupt_match_reasons).
    bad = corrupt_match_reasons(table)
    if bad:
        n_pts = int(table["match_id"].isin(bad).sum())
        table = table[~table["match_id"].isin(bad)].reset_index(drop=True)
        counts = pd.Series(bad).value_counts().to_dict()
        print(f"Dropped {len(bad)} corrupt matches ({n_pts:,} points): {counts}")

    # Source rows are already in playing order within each match. Don't sort
    # on point_no: it is read as text in files with '0X'/'0Y' rows, so
    # "10" would sort before "2".

    print(
        f"Loaded {len(frames)} tournament files, "
        f"{table['match_id'].nunique()} matches, {len(table):,} points."
    )
    return table


def main() -> None:
    table = build_point_table()
    out = PROCESSED_DIR / "points.parquet"
    table.to_parquet(out)
    print(f"Wrote {out}  ({len(table):,} rows)")
    # Report optional-field coverage — useful for the report's data section.
    for col in ("rally", "speed_kmh", "p1_ace", "p1_winner", "p1_unferr"):
        if col in table.columns:
            frac = table[col].notna().mean()
            print(f"  coverage[{col}] = {frac:.1%}")


if __name__ == "__main__":
    main()
