"""Grand Slam scoring rules needed by the features and the no-momentum null.

Only what differs between sets, events and eras: how many sets win the match,
and when (and to how many points) a set goes to a tiebreak.

Every set except the final one: 7-point tiebreak at 6-6.
Final set (5th for men, 3rd for women):
  US Open          7-point tiebreak at 6-6 to 2021, 10-point from 2022
  Australian Open  advantage (no tiebreak) to 2018, 10-point at 6-6 from 2019
  Roland Garros    advantage to 2021, 10-point at 6-6 from 2022
  Wimbledon        advantage to 2018, 7-point at 12-12 for 2019-2021,
                   10-point at 6-6 from 2022
"""
from __future__ import annotations


def tour(match_id: str) -> str:
    """'men' or 'women', from the match number (1xxx / M... men, 2xxx / W... women)."""
    return "men" if match_id.split("-")[-1][0] in ("1", "M") else "women"


def sets_to_win(match_id: str) -> int:
    return 3 if tour(match_id) == "men" else 2


def tiebreak_rule(match_id: str, set_no: int) -> tuple[int | None, int]:
    """(games each at which the set goes to a tiebreak, tiebreak target points).

    None means an advantage set: no tiebreak, play on until two games clear.
    """
    final = int(set_no) == 2 * sets_to_win(match_id) - 1
    if not final:
        return 6, 7
    year, slam = int(match_id[:4]), match_id.split("-")[1]
    if year >= 2022:
        return 6, 10
    if slam == "usopen":
        return 6, 7
    if slam == "ausopen":
        return (6, 10) if year >= 2019 else (None, 7)
    if slam == "wimbledon":
        return (12, 7) if year >= 2019 else (None, 7)
    return None, 7                                   # frenchopen to 2021


def is_tiebreak(match_id: str, set_no: int, games1: int, games2: int) -> bool:
    """Is the next game a tiebreak, given the games score in this set?"""
    at, _ = tiebreak_rule(match_id, set_no)
    return at is not None and games1 == games2 == at
