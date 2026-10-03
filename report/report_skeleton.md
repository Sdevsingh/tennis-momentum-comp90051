# Report Skeleton — Tennis Momentum (COMP90051 Group 51)

**This is a planning document, not the report itself.** It lays out the 4-page structure, the word budget per section, exactly what each section must contain to hit the rubric, and the specific risks Jean has already flagged. Write the final report as a PDF (LaTeX recommended) following the format rules below.

---

## Hard format rules (from the brief, Section 3)

- PDF, **≤ 4 pages**, single column, A4. Anything past page 4 is ignored.
- Font **10pt**; margins **1.5–2cm** all sides.
- No cover page. References **and appendices count** toward the 4 pages.
- Student IDs + names required.
- APA citation style encouraged (be consistent).

> **LaTeX starting point:** `\documentclass[10pt,a4paper]{article}` with `\usepackage[margin=1.8cm]{geometry}`. Two-column is *not* required and single-column is specified. A compact `\titlespacing` and `\setlength{\parskip}` help fit 4 pages.

## Rubric → where it is earned

| Rubric criterion (max) | Where in report | What the marker looks for |
|---|---|---|
| Completeness & correctness (12) | Methods + Results + code | From-scratch CV/tuning/metrics actually implemented and correct; all pipeline stages present |
| Clarity & structure (6) | Whole report | A postgrad can read it with no difficulty; clean figures with error bars |
| Critical analysis (12) | Intro motivation + Results discussion | *Why* methods work/not; advantages/disadvantages; alternatives considered and rejected with reasons — **not** "0.6 vs 0.7 so I picked B" |

**Half the marks are critical analysis.** Budget the most words to the discussion.

---

## Page budget (target ~2,400–2,800 words total at 10pt single column)

| Section | Target | Page |
|---|---|---|
| Title + IDs/names (1–2 lines, no cover page) | 30 w | p1 top |
| 1. Introduction (research question + dataset + URL) | 300 w | p1 |
| 2. Literature review | 350 w | p1–2 |
| 3. Methods | 850 w | p2–3 |
| 4. Results & Discussion | 900 w | p3–4 |
| 5. Conclusion | 150 w | p4 |
| 6. Bibliography | ~8 refs | p4 |

Figures/tables eat page space — plan for **3–4 figures max**, each earning its place.

---

## 1. Introduction  (~300 words)

**Must contain:** the research question stated precisely; dataset name + **URL**; one or two sentences of motivation.

Draft content:
- Open with the confound, not the hype: serving alternates and stronger players win more, so streaks appear without momentum. State that the question is whether *order* adds predictive information **beyond** server, score/pressure, and player strength, and how far back it reaches.
- Name the dataset: Sackmann Grand Slam point-by-point, singles since 2011, URL. One line on size after cleaning (fill in real numbers once `data_loading.py` runs).
- State the three-pronged design in one sentence (shuffle control, memory horizon, planted-momentum control).

> **Jean's objection to pre-empt here:** make it unmistakable in the first paragraph that this is *not* "which features classify points well". Frame it as hypothesis-testing with a control.

## 2. Literature review  (~350 words)

**Required:** dataset reference + **at least two** additional papers; **one** must relate to an algorithm you implement.

Planned citations (verify every one before submitting):
- **Sackmann** dataset (cite the repository / data note).
- **Santoro et al., 2018, *Relational Recurrent Neural Networks*, NeurIPS** — the algorithm paper. **Explain why it was not taught in class** (Jean's explicit instruction): the subject covers standard RNNs/LSTMs but not the Relational Memory Core, which augments memory with multi-head self-attention *between* memory slots so stored items can interact — a mechanism outside the COMP90051 syllabus. Summarise, don't reproduce the paper.
- **Klaassen & Magnus, 2001, JASA** — are tennis points i.i.d.? Classic evidence the effect is small but non-zero; motivates a careful control.
- **Gilovich, Vallone & Tversky, 1985** — the original "hot hand" paper; frames momentum as a cognitive-bias question.
- *(Optional 5th if space)* **Miller & Sanjurjo, 2018, Econometrica** — streak-selection bias. Note our within-match shuffle control inherits the same bias, so comparing real vs shuffled cancels it.

Keep this tight: two-to-three sentences per source, all paraphrased.

## 3. Methods  (~850 words) — Sections 2.c–2.f

Four subsections:

**3.1 Feature construction (2.c).** This must be more than one-hot/normalisation or the brief penalises it.
- History-window win rates at multiple sizes (last 1/3/5/10/20).
- Exponentially-weighted win rate at several decay rates.
- Current streak length (signed).
- Pressure indicators: break point, set point, game/set score differential, deciding set.
- **"Surprise" momentum feature:** weighted mean of (outcome − expected win prob from a server+score-only baseline). This is the key non-obvious feature — momentum as *departure from expectation*, not raw wins.
- Server and a coarse player-strength proxy (pre-match or rolling) as baseline controls.
- State how you reach ≥100 features (windows × decay rates × pressure interactions). Note: the tabular size requirement (≥10k instances, ≥100 features) is satisfied at the *point* level.

**3.2 Three algorithms (2.d).** One short paragraph each: logistic regression (own GD), gradient boosting (medium), Relational RNN (complex, reads the raw point sequence; summarise the memory-slot + self-attention mechanism, cite Santoro). Say *which inputs* each sees (hand-crafted vs raw sequence) and why.

**3.3 Cross-validation (2.e), from scratch.** Temporal outer CV — train on earlier matches, test on later (the 11-part scheme in the brief, or random-from-first-half / random-from-second-half, ≥10 repetitions). Same splits for all three models. Justify why random point-level shuffling would **leak** the very temporal signal under study.

**3.4 Hyperparameter tuning + nested CV (2.f), from scratch.** One tunable hyperparameter per model minimum (e.g. L2 strength; number of trees / depth; number of memory slots or memory size). ≥3 values each, centred so the middle is usually chosen. Inner CV ≥3 folds. State the grid.

## 4. Results & Discussion  (~900 words) — Sections 2.g + rubric's critical analysis

**Results (≥3, with error bars — computed from scratch):**
- **R1 — Shuffle control:** real vs shuffled log-loss/Brier/AUC per model, with error bars. The gap = momentum signal.
- **R2 — Memory horizon curve:** metric vs history length, error bars; mark where it flattens.
- **R3 — Positive control:** detection vs planted δ, establishing the sensitivity floor.
- **R4 (optional):** reset test, or per-subgroup (server/returner, favourite/underdog) breakdown.
- Learning curves (vs #training samples) are acceptable; "loss vs epochs" is **explicitly not** an acceptable result.

**Discussion — this is where 12 marks live:**
- *Why* each model does/doesn't gain from history. (e.g. logistic reg can't model order beyond engineered windows; RNN can but may overfit on a weak signal.)
- Advantages/disadvantages of each, in the context of the question, grounded in subject material (bias–variance, model-class complexity, regularisation).
- Alternatives considered and rejected **with reasons** (e.g. Neural Hawkes Process as a self-exciting model; plain LSTM; survival models) — not just a metric table.
- Interpret honestly: if the real-vs-shuffle gap is within error bars, say momentum is not predictively detectable at this horizon, and lean on the positive control to argue the method *could* have found it.

## 5. Conclusion  (~150 words)

What you now know about momentum in tennis — a crisp, defensible claim with its caveats. Not a summary of what you did; a statement of what you learned.

## 6. Bibliography

APA, consistent. ~6–8 entries. Counts toward the page limit.

---

## Pre-submission checklist

- [ ] ≤ 4 pages, 10pt, 1.5–2cm margins, single column, PDF.
- [ ] Research question is clearly *not* "predict the next point / which features classify".
- [ ] Literature review explains why the Relational RNN was not taught in class.
- [ ] ≥2 extra papers, one tied to an implemented algorithm.
- [ ] CV, nested CV, tuning, and all metrics implemented from scratch (no sklearn for these).
- [ ] Same CV splits across all three models.
- [ ] Middle hyperparameter value chosen most often (adjust grid if not).
- [ ] ≥3 results, all with error bars.
- [ ] No "loss vs epochs" presented as a learning curve.
- [ ] Discussion goes beyond "A got 0.6, B got 0.7".
- [ ] Every citation verified; APA consistent.
- [ ] ZIP of code + Readme.txt submitted; **data not submitted**.
