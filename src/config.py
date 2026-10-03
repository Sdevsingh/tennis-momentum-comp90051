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
# Full Sackmann archive (atp/, wta/, slam_pointbypoint/), used for Elo.
ARCHIVE_DIR = DATA_DIR / "sackmann"
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

# Elo (chess/FIDE): start rating and K schedule (see src/player_features.py).
ELO_START = 1500
ELO_K_NEW = 40          # first ELO_NEW_GAMES rated matches
ELO_NEW_GAMES = 30
ELO_K_DEFAULT = 20
ELO_K_ELITE = 10        # once a player has reached ELO_ELITE_RATING
ELO_ELITE_RATING = 2400

# TrueSkill (Herbrich et al. 2007 defaults) + TrueSkill 2 skill dynamics.
TS_MU0 = 25.0
TS_SIGMA0 = 25.0 / 3
TS_BETA = 25.0 / 6        # performance noise per match
TS_GAMMA = 25.0 / 300     # skill drift per match played (TS2 eq. 2)
TS_TAU_PER_DAY = 0.2      # skill drift per day away (TS2 eq. 3); a year off
                          # adds ~3.8 sd. Not tuned.

# Serve/return skill: step size max(1/(n + ETA0), ETA_MIN) after each match
# with stats; ETA0 acts as a prior of ETA0 average matches.
SERVE_ETA0 = 10
SERVE_ETA_MIN = 0.05
SERVE_AVG_ALPHA = 0.0005  # running tour-average serve points won

# Positive-control planted momentum sizes (bump to next-point win prob after a streak).
PLANTED_DELTAS = [0.0, 0.02, 0.05, 0.10]
