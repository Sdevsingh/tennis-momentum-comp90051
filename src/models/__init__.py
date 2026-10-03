"""Model implementations.

All models share a minimal interface so nested_cv.py can treat them uniformly:

    model = Model(**hyperparams)
    model.fit(X, y)            # X: (n, d) float, y: (n,) in {0, 1}
    p = model.predict_proba(X) # -> (n,) P(class 1)

- logistic_regression: SIMPLE   (from scratch)
- gradient_boosting:   MEDIUM   (own tuning; tree library permitted for the
                                 learner itself per team's reading of the rule)
- relational_rnn:      COMPLEX  (Santoro et al., NeurIPS 2018; not taught in class)
"""
