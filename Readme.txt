COMP90051 Group 51 — Tennis Momentum
How to run (the brief asks for a short Readme.txt like this)

1. Install dependencies:
     pip install -r requirements.txt

2. Get the data (NOT included in this repo; the brief says do not submit data):
     See data/README.md. In short, clone Jeff Sackmann's Grand Slam
     point-by-point repo and put the '*-points.csv' files in data/raw/.

3. Build the cleaned point table:
     python -m src.data_loading
   -> writes data/processed/points.parquet and prints coverage stats.

4. Run the experiments:
     python -m src.experiments.exp1_shuffle_control    # real vs no-momentum null
     python -m src.experiments.exp2_memory_horizon     # how far back memory reaches
     python -m src.experiments.exp3_base_vs_full       # pre-match + serve only vs full model
   -> each writes a JSON of results to results/, and the out-of-fold model
      predictions (one row per test point and configuration) to
      results/predictions/<same name>.parquet (local only; see src/predictions.py).

5. Figures for the report are plotted from the results/*.json files
   (plotting libraries are allowed; metric computation is from scratch).

Notes:
- Cross-validation, nested CV, hyperparameter tuning and all metrics are
  implemented from scratch in src/ (no third-party libraries for these),
  as required.
- The logistic-regression model and both control experiments run as-is.
  Gradient boosting and the Relational RNN have a shared interface and are
  stubbed with method docstrings for the team to complete.
