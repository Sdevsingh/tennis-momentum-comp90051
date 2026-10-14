"""Experiment 1 — No-momentum shuffle control (the key test for Section 2.a).

Idea: destroy temporal ORDER while preserving BASE RATES, then see whether the
models lose their edge. The null is the real matches with their order shuffled
at two levels (src/null_shuffle.py): points within each game, and each
player's service games within each set. Every point/game result, game score,
set score and serve record is kept, serve still alternates, and every shuffled
match is legal tennis -- only the order of what happened is scrambled.

    real data performance  -  shuffled data performance  =  momentum signal

If the gap is within error bars, momentum carries no predictive information at
this resolution — a valid, publishable null. The shuffle also inherits any
streak-selection bias (Miller & Sanjurjo), so comparing real vs shuffled
cancels that bias instead of being fooled by it.

--null points|games|both picks the levels (default both); comparing them shows
whether any order effect lives inside games or across games. --null replay
instead re-forms each match from per-server pools of its real points (games
and sets end wherever the scoring rules say), which removes the within-game
"drawing without replacement" structure the other modes keep; see
src/null_shuffle.py.

--seeds runs the null with several shuffle seeds (default 51 52 53): one
shuffle is a single random draw, and on a subset the real-vs-null gap moved by
as much as its fold standard error between seeds. The shuffled per-fold
metrics are averaged over seeds before pairing.

The headline is PAIRED: real and every null use identical match splits, so
"paired" in the JSON holds, per tour and model, the per-fold gain of real over
the seed-averaged null (+ = real predicts better = momentum signal), its
standard error over folds, and the per-seed gains with their spread.

Checkpoints: every finished (dataset, tour) block is written to
results/checkpoints/<output name>/<dataset>/<tour>.json (+ .parquet with its
predictions), and a rerun with the same arguments loads finished blocks
instead of recomputing them, so a crash or reboot costs at most one block.
--null-first runs the shuffled data before the real data (same results).

--models picks which models run (default all three). The RNN's sequence input
(~2.7 GB for all points) is built only when "rmc" is run, one tour at a time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RESULTS_DIR, SEED, N_OUTER_REPEATS
from ..features import build_features, FEATURE_COLUMNS
from ..cross_validation import temporal_match_splits
from ..nested_cv import nested_cv
from ..models import logistic_regression, gradient_boosting
from ..null_shuffle import shuffle_null, MODES
from ..sequences import build_sequence_matrix, STATIC_FEATURES
from .. import metrics
from ..models import relational_rnn
from ..predictions import point_keys, prediction_frame, save_predictions


# Per-tour hyperparameter grids (used when men and women get separate models);
# a model without one uses its MODELS grid for both tours.
TOUR_GRIDS = {
    "logreg": logistic_regression.PARAM_GRID_BY_TOUR,
    "gboost": gradient_boosting.PARAM_GRID_BY_TOUR,
}

MODELS = {
    "logreg": (logistic_regression.make_model, logistic_regression.PARAM_GRID),
    "gboost": (gradient_boosting.make_model, gradient_boosting.PARAM_GRID),
    "rmc":    (relational_rnn.make_model, relational_rnn.PARAM_GRID),
}

# Models whose input is the raw point sequence rather than the tabular features.
SEQUENCE_MODELS = {"rmc"}


SLAM_ORDER = {"ausopen": 0, "frenchopen": 1, "wimbledon": 2, "usopen": 3}


def matches_in_time_order(match_ids) -> list[str]:
    """Unique match ids in playing order: year, then AO, RG, Wimbledon, USO.

    Within a tournament, the source's own order is kept.
    """
    uniq = list(dict.fromkeys(match_ids))
    return sorted(uniq, key=lambda m: (int(m[:4]), SLAM_ORDER.get(m.split("-")[1], 9)))


def tour_of(match_ids: pd.Series) -> pd.Series:
    """'men' / 'women' from the match number (1xxx/M... men, 2xxx/W... women)."""
    first = match_ids.str.split("-").str[-1].str[0]
    return first.map({"1": "men", "M": "men", "2": "women", "W": "women"})


def _nested_cv_all_models(X, X_seq, y, match_ids, label: str,
                          tour: str | None = None, keys: pd.DataFrame | None = None,
                          frames: list | None = None, dataset: str | None = None,
                          models: list[str] | None = None,
                          extra_labels: dict | None = None) -> dict:
    """Nested CV for every model in `models` (default all). If `frames` is
    given, each model's out-of-fold predictions (rows identified by `keys`,
    aligned with X) are appended to it as a table (src/predictions.py)."""
    ordered = matches_in_time_order(match_ids.tolist())
    splits = temporal_match_splits(ordered, n_parts=N_OUTER_REPEATS + 1)
    results = {}
    for name in (models or list(MODELS)):
        make, grid = MODELS[name]
        grid = TOUR_GRIDS.get(name, {}).get(tour, grid)
        Xm = X_seq if name in SEQUENCE_MODELS else X   # RNN reads the raw sequence
        res = nested_cv(make, grid, Xm, y, match_ids, splits)
        if frames is not None and keys is not None:
            frames.append(prediction_frame(res, keys, dataset=dataset or label,
                                           **(extra_labels or {}),
                                           tour=tour or "pooled", model=name))
        results[name] = {
            "grid": grid,
            "summary": res["summary"],
            "per_split": res["per_split"],
            "centring": {k: {str(v): n for v, n in c.items()}
                         for k, c in res["centring"].items()},
        }
        print(f"  [{label}/{name}] " +
              ", ".join(f"{m}={s['mean']:.4f}±{s['se']:.4f}"
                        for m, s in res["summary"].items()), flush=True)
    return results


def _run_one(points: pd.DataFrame, label: str, by_tour: bool = True,
             frames: list | None = None, models: list[str] | None = None,
             extra_labels: dict | None = None, ckpt: Path | None = None) -> dict:
    """Nested CV for every model; one set of models per tour unless pooled.
    Out-of-fold predictions are appended to `frames` when it is given.

    With `ckpt` (a directory), each finished tour is saved there as
    <tour>.json (+ <tour>.parquet predictions) and tours already saved are
    loaded instead of recomputed; `points` may then be a callable that builds
    the data only if some tour still has to run."""
    models = models or list(MODELS)
    names = ["men", "women"] if by_tour else ["pooled"]
    results = {}
    if ckpt is not None:
        for group in names:
            f = ckpt / f"{group}.json"
            if f.exists():
                results[group] = json.loads(f.read_text())
                if frames is not None and f.with_suffix(".parquet").exists():
                    frames.append(pd.read_parquet(f.with_suffix(".parquet")))
                print(f"  [{label}/{group}] loaded from checkpoint {f}", flush=True)
        if len(results) == len(names):
            return results
    if callable(points):
        points = points()
    feats = build_features(points, player_won_col="p1_won")
    X = feats[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    y = feats["p1_won"].to_numpy()
    match_ids = feats["match_id"].to_numpy()
    keys = point_keys(feats)
    if by_tour:
        tours = tour_of(feats["match_id"]).to_numpy()
        groups = [(t, tours == t) for t in ("men", "women")]
    else:
        groups = [("pooled", np.ones(len(feats), dtype=bool))]
    # The RNN's sequence matrix is built one group at a time from just the
    # columns it reads, so the full ~2.7 GB matrix never exists.
    need_seq = any(m in SEQUENCE_MODELS for m in models)
    seq_src = (feats[["match_id", "point_server", "point_winner", "rally"]
                     + list(STATIC_FEATURES)].copy() if need_seq else None)
    del feats
    for group, m in groups:
        if group in results:
            continue                                   # loaded from a checkpoint
        X_seq = build_sequence_matrix(seq_src[m]) if need_seq else None
        group_frames: list = []
        results[group] = _nested_cv_all_models(
            X[m], X_seq, y[m], match_ids[m], f"{label}/{group}",
            None if group == "pooled" else group, keys=keys[m], frames=group_frames,
            dataset=label, models=models, extra_labels=extra_labels)
        del X_seq
        if frames is not None:
            frames.extend(group_frames)
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            if group_frames:
                pd.concat(group_frames, ignore_index=True).to_parquet(ckpt / f"{group}.parquet")
            (ckpt / f"{group}.json").write_text(json.dumps(results[group], indent=2))
            print(f"  [{label}/{group}] checkpoint saved to {ckpt / (group + '.json')}", flush=True)
    return {g: results[g] for g in names}


def paired_real_vs_null(real: dict, shuffled: dict) -> dict:
    """Per group and model: gain of real over the seed-averaged null, paired by
    fold (+ = real better), plus each seed's gain and the spread over seeds."""
    seeds = list(shuffled)
    out = {}
    for group, models in real.items():
        out[group] = {}
        for name, r in models.items():
            per_seed = {sd: metrics.paired_gain(shuffled[sd][group][name]["per_split"],
                                                r["per_split"]) for sd in seeds}
            avg = {m: np.mean([shuffled[sd][group][name]["per_split"][m] for sd in seeds], axis=0)
                   for m in r["per_split"]}
            gain = metrics.paired_gain(avg, r["per_split"])
            for m, g in gain.items():
                means = [per_seed[sd][m]["mean"] for sd in seeds]
                g["seed_sd"] = float(np.std(means, ddof=1)) if len(seeds) > 1 else float("nan")
                g["per_seed"] = {sd: per_seed[sd][m]["mean"] for sd in seeds}
            out[group][name] = gain
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Exp 1: real vs shuffled (no-momentum) data.")
    ap.add_argument("--pooled", action="store_true",
                    help="one model for men and women together (default: separate models per tour)")
    ap.add_argument("--null", choices=MODES, default="both",
                    help="points within games, games within sets, both, or replay "
                         "(re-form each match from per-server pools of its points)")
    ap.add_argument("--on-exhaust", choices=("recycle", "stop"), default="recycle",
                    help="replay only: when a server's pool runs out, reuse a fresh "
                         "shuffled copy (recycle) or end the match there (stop)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 1, SEED + 2],
                    help="shuffle seeds for the null (default 51 52 53); the shuffled "
                         "per-fold metrics are averaged over them")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                    help="models to run (default all; the RNN takes hours on a GPU)")
    ap.add_argument("--null-first", action="store_true",
                    help="run the shuffled data before the real data (same results)")
    args = ap.parse_args()
    by_tour = not args.pooled
    suffix = (("" if by_tour else "_pooled") + ("" if args.null == "both" else f"_{args.null}")
              + ("" if args.models == list(MODELS) else "_" + "-".join(args.models)))
    dest = RESULTS_DIR / f"exp1_shuffle_control{suffix}.json"
    ckpt_root = RESULTS_DIR / "checkpoints" / dest.stem
    pts = pd.read_parquet(PROCESSED_DIR / "points.parquet")
    frames: list = []
    real, shuffled, stats = None, {}, {}

    def run_real():
        print("REAL data:")
        return _run_one(pts, "real", by_tour, frames, args.models, {"null_seed": "none"},
                        ckpt=ckpt_root / "real")

    if not args.null_first:
        real = run_real()
    for seed in args.seeds:
        st: dict = {}

        def make_null(seed=seed, st=st):
            null = shuffle_null(pts, mode=args.null, seed=seed, stats=st,
                                on_exhaust=args.on_exhaust)
            print(f"SHUFFLED ({args.null}, seed {seed}) no-momentum data:  {st}")
            return null

        print(f"SHUFFLED ({args.null}, seed {seed}):")
        shuffled[str(seed)] = _run_one(make_null, "shuffled", by_tour, frames, args.models,
                                       {"null_seed": str(seed)},
                                       ckpt=ckpt_root / f"null_{args.null}_seed{seed}")
        stats[str(seed)] = st
    if args.null_first:
        real = run_real()
    del pts

    paired = paired_real_vs_null(real, shuffled)
    for group, models in paired.items():
        for name, g in models.items():
            ll = g["log_loss"]
            print(f"  [paired {group}/{name}] real better by {ll['mean']:+.5f} ± {ll['se']:.5f} "
                  f"log-loss (t {ll['t']:+.1f}, {ll['better']}/{ll['n_folds']} folds; "
                  f"seed sd {ll['seed_sd']:.5f})", flush=True)

    out = {"by_tour": by_tour, "null": args.null, "on_exhaust": args.on_exhaust,
           "seeds": args.seeds, "models": args.models, "shuffle_stats": stats,
           "paired": paired, "real": real, "shuffled": shuffled}
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")
    save_predictions(frames, dest.stem)
    print("Momentum signal = paired gain of real over the null ('paired' in the JSON).")


if __name__ == "__main__":
    main()
