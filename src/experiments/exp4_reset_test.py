"""Experiment 4 (optional) — Reset test.

Does momentum survive game changeovers and set breaks? If momentum is
psychological, cutting the history at these boundaries should REDUCE predictive
gain. If the apparent effect is really serve or player-quality structure,
cutting at boundaries should not matter.

Method: build two feature sets —
  (a) history allowed to cross game/set boundaries (as in exp1/exp2),
  (b) history reset to empty at the start of each game (and separately, set),
then compare momentum signal (real - simulated gap, or horizon gain) between them.

This is a strong piece of critical analysis for the discussion, but optional
given the 4-page limit and the Oct 16 deadline. Implement only if time allows.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Optional experiment. See module docstring for the method. Reuse "
        "features.build_features but reset rolling/EWMA state per game/set "
        "(add a `reset_on` argument), then rerun the exp1/exp2 comparison."
    )


if __name__ == "__main__":
    main()
