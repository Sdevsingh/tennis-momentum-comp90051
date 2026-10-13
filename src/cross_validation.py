"""Cross-validation from scratch (brief Section 2.e).

Third-party CV splitters (e.g. sklearn.model_selection) are NOT allowed, so
these are implemented by hand. Because the data is temporal, the default here
is a TEMPORAL scheme: always train on earlier matches and test on later ones,
following the brief's "divide into 11 ordered parts" idea. Random point-level
splits would leak the temporal signal we are studying and are deliberately
avoided for the outer loop.

Splits operate at the MATCH level, not the point level: an entire match is in
train or test, never split across both. This prevents a point in the test set
from seeing its own match's earlier points through a model trained on them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train_match_ids: np.ndarray
    test_match_ids: np.ndarray


def temporal_match_splits(match_ids_in_time_order: list[str],
                          n_parts: int = 11) -> list[Split]:
    """The brief's scheme: cut the time-ordered matches into n_parts blocks;
    repetition r trains on block r and tests on block r+1.

    With n_parts=11 this yields 10 repetitions. Expanding-window variants are
    also reasonable; document whichever you use.
    """
    ids = np.array(match_ids_in_time_order)
    blocks = np.array_split(ids, n_parts)
    splits = []
    for r in range(len(blocks) - 1):
        splits.append(Split(train_match_ids=blocks[r],
                            test_match_ids=blocks[r + 1]))
    return splits


def expanding_window_splits(match_ids_in_time_order: list[str],
                            n_repeats: int = 10) -> list[Split]:
    """Alternative: train on everything up to time t, test on the next block.

    Often more realistic than single-block training. Provided so the team can
    compare; pick ONE scheme and use it for all three models.
    """
    ids = np.array(match_ids_in_time_order)
    blocks = np.array_split(ids, n_repeats + 1)
    splits = []
    for r in range(1, len(blocks)):
        train = np.concatenate(blocks[:r])
        test = blocks[r]
        splits.append(Split(train_match_ids=train, test_match_ids=test))
    return splits


def kfold_match_splits(match_ids: list[str], k: int = 10,
                       seed: int = 51) -> list[Split]:
    """Plain k-fold over matches (non-temporal).

    Only for data with no real time axis to respect (e.g. simulated
    matches). k >= 10 per the brief.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(match_ids)
    rng.shuffle(ids)
    folds = np.array_split(ids, k)
    splits = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append(Split(train_match_ids=train, test_match_ids=test))
    return splits


def iter_fold_indices(match_id_array: np.ndarray, split: Split):
    """Translate a match-level split into boolean row masks for the point table."""
    train_mask = np.isin(match_id_array, split.train_match_ids)
    test_mask = np.isin(match_id_array, split.test_match_ids)
    return train_mask, test_mask
