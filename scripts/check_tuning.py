"""Check the 2.f "middle value chosen most often" requirement from a results file.

After running an experiment that records the tuning tally (exp1 stores it under
each model's "centring"), run:

    python scripts/check_tuning.py results/exp1_shuffle_control.json

For every model and every tuned hyperparameter it prints how often each value
was chosen across the outer folds, and flags any case where an EXTREME value won
most often -- Jean asks that the MIDDLE value win most often, otherwise the grid
should be shifted (e.g. if the smallest value keeps winning, add smaller values).

It walks the JSON generically, so it works whatever the nesting (per tour, real
vs shuffled, etc.). It only needs a "centring" dict to be present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _find_centrings(node, trail=""):
    """Yield (path, centring_dict) for every 'centring' found in the JSON."""
    if isinstance(node, dict):
        if "centring" in node and isinstance(node["centring"], dict):
            yield trail, node["centring"]
        for k, v in node.items():
            yield from _find_centrings(v, f"{trail}/{k}" if trail else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _find_centrings(v, f"{trail}[{i}]")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def check(path: Path) -> int:
    data = json.loads(path.read_text())
    found = list(_find_centrings(data))
    if not found:
        print(f"No tuning tally ('centring') found in {path.name}.")
        print("Run an experiment that records it (e.g. exp1_shuffle_control).")
        return 1

    problems = 0
    for trail, centring in found:
        for hp, counts in centring.items():
            # counts: {value(str): times chosen}
            values = sorted(counts, key=lambda v: _num(v))
            mid = values[len(values) // 2]
            winner = max(counts, key=lambda v: counts[v])
            tally = ", ".join(f"{v}:{counts[v]}" for v in values)
            flag = ""
            if winner != mid:
                flag = "  <-- EXTREME won; consider shifting the grid"
                problems += 1
            print(f"[{trail}] {hp}: {tally}  (middle={mid}, winner={winner}){flag}")

    print()
    if problems:
        print(f"{problems} hyperparameter(s) where the middle did NOT win most "
              f"often. Shift those grids and re-run (see 2.f).")
    else:
        print("All tuned hyperparameters: the middle value won most often. (2.f OK)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/check_tuning.py <results/*.json>")
        raise SystemExit(2)
    raise SystemExit(check(Path(sys.argv[1])))
