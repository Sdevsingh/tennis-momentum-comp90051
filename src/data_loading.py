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

# Canonical names we use downstream -> list of possible source spellings.
COLUMN_ALIASES = {
    "point_server": ["PointServer", "Server"],
    "point_winner": ["PointWinner"],
    "set_no": ["SetNo", "set"],
    "game_no": ["GameNo", "game"],
    "point_no": ["PointNumber", "Pt"],
    "rally": ["Rally", "RallyCount"],
    "speed_kmh": ["Speed_KMH", "ServeSpeed"],
    "p1_ace": ["P1Ace"],
    "p2_ace": ["P2Ace"],
    "p1_winner": ["P1Winner"],
    "p2_winner": ["P2Winner"],
    "p1_unferr": ["P1UnfErr"],
    "p2_unferr": ["P2UnfErr"],
}

REQUIRED = ["point_server", "point_winner", "set_no", "game_no", "point_no"]


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

    out = df.rename(columns={v: k for k, v in colmap.items()})
    keep = ["match_id"] + [c for c in COLUMN_ALIASES if c in out.columns]
    # Each file is a whole tournament; the source's own match_id
    # (e.g. 2023-usopen-1701) identifies the match.
    out = out[keep].copy()
    # PointServer / PointWinner are 1/2 in the source. Drop rows where winner
    # is 0 (unplayed / data gaps).
    out = out[out["point_winner"].isin([1, 2])]
    out["point_server"] = out["point_server"].astype("Int64")
    return out.reset_index(drop=True)


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
