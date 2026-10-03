"""Complex model: Relational Recurrent Neural Network (Santoro et al., 2018).

Paper: Santoro, Faulkner, Raposo, Rae, Chrzanowski, Weber, Wierstra,
Vinyals, Pascanu, Lillicrap. "Relational Recurrent Neural Networks."
NeurIPS 2018.  https://arxiv.org/abs/1806.01822

WHY IT FITS THE PROJECT
-----------------------
A standard LSTM compresses the past into a single hidden vector. The Relational
Memory Core (RMC) instead keeps a *matrix* of memory slots and lets them
interact through multi-head self-attention at every step, so the model can
represent relations between separate past events. For tennis momentum this is
exactly the hypothesis class of interest: "does what happened several points
ago interact with the current state to shift the next outcome?" Its memory
slots also give an interpretability handle (how attention weights depend on how
far back a point is), which we use in the discussion.

WHY IT WAS NOT TAUGHT IN CLASS (state this in the literature review)
-------------------------------------------------------------------
COMP90051 covers standard recurrent architectures (RNN/LSTM) but not the
Relational Memory Core, whose defining component — self-attention *among memory
slots* — is a post-2017 development outside the syllabus.

INPUT
-----
Unlike the other two models, the RMC reads the RAW SEQUENCE of recent points
(e.g. one-hot of {who won, who served, coarse score} for the last H points),
not the hand-crafted features. This lets us ask whether it recovers momentum
structure the hand-crafted features miss.

Tunable hyperparameter: memory size / number of slots (centred grid). Others
(heads, learning rate) can be added but keep the count small.

IMPLEMENTATION NOTES
--------------------
Implement in PyTorch. The RMC can be written directly, or a reference
implementation adapted (cite it). Keep the model small: the momentum signal is
weak and the dataset, while large, has a low-information target, so a big
network will overfit. Wrap training so .fit / .predict_proba match the shared
interface used by nested_cv.py. Early stopping on an inner validation slice is
fine; "loss vs epochs" must NOT be presented as a result in the report.
"""
from __future__ import annotations

import numpy as np


class RelationalRNN:
    def __init__(self, mem_slots: int = 4, head_size: int = 16, n_heads: int = 2,
                 n_epochs: int = 20, lr: float = 1e-3, history: int = 20,
                 seed: int = 51):
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.lr = lr
        self.history = history
        self.seed = seed
        self._net = None

    def fit(self, X_seq, y):
        """X_seq: (n, history, feat) sequence tensor; y: (n,) in {0,1}.

        TODO(team):
          1. Build the RMC (memory matrix + multi-head self-attention update +
             gating), then a linear head to a single logit.
          2. Train with BCE-with-logits, Adam, early stopping on an inner split.
          3. Store the trained net on self._net.
        Note: experiments pass sequence tensors to this model, and the flat
        feature matrix to the other two. exp runners handle the difference.
        """
        raise NotImplementedError("Implement the RMC (see docstring + paper).")

    def predict_proba(self, X_seq) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        raise NotImplementedError


# Centred so the middle memory size should win most often.
PARAM_GRID = {"mem_slots": [2, 4, 8]}


def make_model(**params) -> RelationalRNN:
    return RelationalRNN(**params)
