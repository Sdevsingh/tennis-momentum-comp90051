"""Central configuration: paths and experiment constants.

Keeping these in one place means every experiment uses the same splits,
seeds, and horizons, which the brief requires for a fair comparison.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Sackmann CSVs go here (not committed)
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

for _d in (PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
SEED = 51  # group number, why not

# --- Experiment constants ---
# Memory horizon: how many previous points of history each model is given.
HISTORY_LENGTHS = [0, 1, 3, 5, 10, 20, 50]

# Exponential decay rates for weighted win-rate features.
DECAY_RATES = [0.5, 0.7, 0.85, 0.95]

# Window sizes for rolling win-rate features.
WINDOW_SIZES = [1, 3, 5, 10, 20]

# Outer CV: temporal repetitions (predict future from past).
N_OUTER_REPEATS = 10
# Inner CV for nested hyperparameter tuning.
N_INNER_FOLDS = 3

# Positive-control planted momentum sizes (bump to next-point win prob after a streak).
PLANTED_DELTAS = [0.0, 0.02, 0.05, 0.10]
