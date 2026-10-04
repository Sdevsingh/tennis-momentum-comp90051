# Tennis Momentum — COMP90051 Group 51

**Does momentum have a real, predictive effect on the next point in a tennis match — and if so, how far back does it reach?**

Statistical Machine Learning (COMP90051), Semester 2 2026. Group project.

| | |
|---|---|
| **Group** | 51 |
| **Members** | Benjamin Morrison, Dev Karan Singh, Faisal Abdullah Khan |
| **Report due** | Friday 16 October 2026, 11:59pm (Canvas) |
| **Dataset** | Jeff Sackmann, Grand Slam point-by-point — https://github.com/JeffSackmann/tennis_slam_pointbypoint |

---

## 1. Research question

The obvious task on this dataset is "predict who wins the next point". That is *not* our research question, and the brief (Section 2.a) explicitly rules out anything that reduces to "which features are good for classification".

Instead we ask a **falsifiable scientific question**:

> Does the **order** of the preceding points carry information about the next point's winner, **beyond** what is explained by who is serving, the score/pressure context, and player strength? If it does, **how many points back** does that memory extend?

This is answerable only by *experiment design*, not by a feature-importance table:

1. **No-momentum control (negative control).** We build a null dataset by permuting point outcomes *within each match*, conditioning on the server (and a coarse score bucket) so that base rates are preserved but temporal order is destroyed. If our models do no better on real data than on shuffled data, momentum carries no predictive signal. A null result here is a valid, reportable finding.
2. **Memory horizon.** We give each model 0, 1, 3, 5, 10, 20 and 50 previous points of history and plot predictive gain over the no-history baseline, with error bars. Where the curve flattens is the "momentum horizon".
3. **Planted-momentum positive control.** We simulate matches with a *known* momentum bump δ after streaks and run the whole pipeline on them, to show the smallest effect our methods could detect. This makes any null result on real data meaningful rather than an artefact of weak methods.

**Why this is not trivial.** Serving alternates each game and stronger players win more points, so *apparent* streaks arise with no momentum at all. "Won the last 3 points" frequently just means "I am serving this game" or "I am the better player". The server/strength baseline plus the shuffle control are what stop us reporting a spurious effect — and they are the source of the critical analysis the rubric rewards.

See `report/report_skeleton.md` for how each experiment maps to the rubric.

## 2. Repository structure

```
tennis-momentum-comp90051/
├── README.md                  ← this file
├── Readme.txt                 ← short "how to run" file required by the brief
├── requirements.txt
├── .gitignore                 ← excludes data/ (brief says: do NOT submit data)
├── data/                      ← dataset lives here locally, NOT committed
│   └── README.md              ← how to fetch the Sackmann data
├── src/
│   ├── config.py              ← paths, experiment constants
│   ├── data_loading.py        ← load + clean Sackmann point-by-point CSVs
│   ├── features.py            ← momentum / history / pressure feature construction
│   ├── cross_validation.py    ← k-fold + temporal CV, FROM SCRATCH
│   ├── nested_cv.py           ← nested CV + hyperparameter tuning, FROM SCRATCH
│   ├── metrics.py             ← log-loss, Brier, AUC, calibration, FROM SCRATCH
│   ├── models/
│   │   ├── logistic_regression.py   ← simple (sklearn learner, own CV/tuning/metrics)
│   │   ├── gradient_boosting.py     ← medium (LightGBM learner, own CV/tuning/metrics)
│   │   └── relational_rnn.py        ← complex (Santoro et al. 2018, NeurIPS)
│   └── experiments/
│       ├── exp1_shuffle_control.py
│       ├── exp2_memory_horizon.py
│       ├── exp3_positive_control.py
│       └── exp4_reset_test.py        ← optional (game/set-break reset)
├── results/                   ← generated tables & figures (committed)
├── report/
│   └── report_skeleton.md     ← 4-page report plan, rubric-aligned
├── docs/
│   ├── meeting_minutes_template.md
│   └── group_agreement_notes.md
└── notebooks/
    └── 00_exploration.ipynb   ← scratch EDA (optional)
```

## 3. Three algorithms (Section 2.d)

| Complexity | Algorithm | In class? | Input |
|---|---|---|---|
| Simple | Regularised logistic regression (own gradient descent) | Yes | Hand-crafted momentum/pressure features |
| Medium | Gradient-boosted trees | Yes | Same hand-crafted features |
| Complex | **Relational Recurrent Neural Network** (Santoro et al., NeurIPS 2018) | **No** | Raw sequence of recent points |

The Relational RNN is the required post-2017, not-taught-in-class algorithm. Its self-attention "memory slots" let us ask whether it rediscovers momentum structure the hand-crafted features miss — tying the paper directly to the research question. The literature review must explain *why it was not covered in class* (per Jean's annotation).

## 4. From-scratch requirements (Sections 2.e–2.g)

The brief forbids third-party libraries for cross-validation, hyperparameter tuning, nested CV, and metric computation. These are implemented by hand in `cross_validation.py`, `nested_cv.py`, and `metrics.py`. Plotting libraries are permitted for charting already-computed results.

Because the data is temporal, CV **predicts the future from the past**: outer folds are time-ordered splits (train on earlier matches, test on later), not random shuffles across points within a match.

## 5. How to run

See `Readme.txt`. In short:

```bash
pip install -r requirements.txt
python -m src.data_loading          # builds the cleaned point table
python -m src.experiments.exp1_shuffle_control
python -m src.experiments.exp2_memory_horizon
python -m src.experiments.exp3_positive_control
```

## 6. Status

This is the **initial scaffold** committed for the team to divide work from. Model internals and experiment bodies are stubbed with clear TODOs and docstrings describing the intended method. Nothing here is a final result.
