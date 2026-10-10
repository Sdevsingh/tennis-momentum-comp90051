"""Experiment runners.

Each experiment is what makes the project answer the *research question*
(brief Section 2.a) rather than a plain classification task:

  exp1_shuffle_control  — negative control: does order beat a shuffled (game-aware) null?
  exp2_memory_horizon   — how far back does predictive memory reach?
  exp3_positive_control — planted-momentum sanity floor: smallest detectable effect
  exp4_reset_test       — (optional) does momentum survive game/set breaks?

Run them as modules, e.g.  python -m src.experiments.exp1_shuffle_control
Results (tables/arrays) are written to results/ for the report's charts.
"""
