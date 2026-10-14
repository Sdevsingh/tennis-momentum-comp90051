"""Generate the report figures from results/*.json (all plotting in Python).

Run the experiments first so the JSON files exist, then:
    python report/make_figures.py
Every figure is written to report/figures/ three ways: <name>.pdf (what
report.tex includes), <name>.png (300 dpi, for slides or a quick look) and
<name>.csv (exactly the values plotted, with their error bars, so a figure can
be redrawn or restyled without rerunning anything).

Plotting libraries are allowed by the brief; only metric computation must be
from scratch, and that lives in src/metrics.py. The error bars here are the
standard error of per-fold paired differences already stored in the JSON.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

_GROUPS = ("men", "women", "pooled")


def _save(fig, stem: str, rows: list[dict]) -> None:
    """Write the figure as PDF and PNG, and the plotted values as CSV."""
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=300)
    with open(FIGDIR / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plt.close(fig)
    print(f"  wrote {FIGDIR / stem}.pdf / .png / .csv")


def _paired(base: list[float], other: list[float]) -> tuple[float, float]:
    """Mean and standard error of base - other over folds (+ = other is better
    for log-loss)."""
    d = np.asarray(base) - np.asarray(other)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def memory_horizon_figure(name: str = "exp2_memory_horizon"):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp2 first.")
        return
    data = json.loads(path.read_text())
    groups = [g for g in _GROUPS if g in data]

    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 2.6),
                             squeeze=False)
    rows = []
    for ax, group in zip(axes[0], groups):
        for model, res in data[group].items():
            curve = res["curve"]
            hs = list(curve)                       # 0, 1, ..., 50, "match"
            h0 = curve[hs[0]]["per_split"]["log_loss"]
            gains = [_paired(h0, curve[h]["per_split"]["log_loss"]) for h in hs]
            rows += [{"tour": group, "model": model, "H": h,
                      "n_memory_features": curve[h].get("n_memory_features"),
                      "log_loss_gain_vs_H0": g[0], "se": g[1]}
                     for h, g in zip(hs, gains)]
            ax.errorbar(range(len(hs)), [g[0] for g in gains], yerr=[g[1] for g in gains],
                        marker="o", capsize=3, label=model)
        ax.set_xticks(range(len(hs)), hs)
        ax.tick_params(axis="x", labelsize=7)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("History length H (points)")
        ax.set_title(group)
    axes[0][0].set_ylabel("Log-loss gain vs. H=0")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Memory horizon" + (" (truncated EWMAs)" if "truncated" in name else ""),
                 fontsize=9)
    fig.tight_layout()
    _save(fig, name.replace("exp2_", ""), rows)


def base_vs_full_figure():
    path = RESULTS / "exp3_base_vs_full.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp3 first.")
        return
    data = json.loads(path.read_text())
    groups = [g for g in _GROUPS if g in data]
    models = list(dict.fromkeys(m for grp in groups for m in data[grp]))

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    width = 0.8 / len(models)
    nan = {"mean": np.nan, "se": np.nan}
    rows = []
    for i, model in enumerate(models):
        g = [data[grp][model]["gain_full_over_base"]["log_loss"] if model in data[grp] else nan
             for grp in groups]
        rows += [{"tour": grp, "model": model, "log_loss_gain_full_over_base": v["mean"],
                  "se": v["se"], "full_better_folds": v.get("full_better")}
                 for grp, v in zip(groups, g) if v is not nan]
        ax.bar(np.arange(len(groups)) + i * width, [x["mean"] for x in g], width,
               yerr=[x["se"] for x in g], capsize=3, label=model)
    ax.set_xticks(np.arange(len(groups)) + width * (len(models) - 1) / 2, groups)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Log-loss gain, full vs. base")
    ax.set_title("What in-match information adds")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "base_vs_full", rows)


def shuffle_control_figure(name: str = "exp1_shuffle_control"):
    """exp1: paired log-loss gain of real over the shuffled null (+ = order
    helps), per tour and model; bars = seed-averaged null with fold SE, dots =
    each shuffle seed."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp1 first.")
        return
    data = json.loads(path.read_text())
    if "paired" not in data:
        print(f"  [skip] {path.name} has no paired results — rerun exp1.")
        return
    paired = data["paired"]
    groups = [g for g in _GROUPS if g in paired]
    models = list(dict.fromkeys(m for grp in groups for m in paired[grp]))

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    width = 0.8 / len(models)
    rows = []
    for i, model in enumerate(models):
        x = np.arange(len(groups)) + i * width
        g = [paired[grp][model]["log_loss"] if model in paired[grp] else None for grp in groups]
        rows += [{"tour": grp, "model": model, "log_loss_gain_real_over_null": v["mean"],
                  "se": v["se"], "t": v["t"], "real_better_folds": v["better"],
                  "seed_sd": v["seed_sd"],
                  **{f"seed_{sd}": val for sd, val in v["per_seed"].items()}}
                 for grp, v in zip(groups, g) if v]
        ax.bar(x, [v["mean"] if v else np.nan for v in g], width,
               yerr=[v["se"] if v else np.nan for v in g], capsize=3, label=model)
        for xi, v in zip(x, g):
            if v:
                seeds = list(v["per_seed"].values())
                ax.plot([xi] * len(seeds), seeds, "k.", ms=3, zorder=3)
    ax.set_xticks(np.arange(len(groups)) + width * (len(models) - 1) / 2, groups)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Log-loss gain, real vs. null")
    ax.set_title(f"Shuffle control ({data.get('null', 'both')} null)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, name.replace("exp1_", ""), rows)


def calibration_figure(name: str = "exp1_shuffle_control_logreg-gboost",
                       n_bins: int = 20, min_count: int = 1000):
    """Reliability curves of the full models on real data, from the saved
    out-of-fold predictions (results/predictions/<name>.parquet, dataset =
    real): mean predicted vs observed P(player 1 wins the point) per bin, one
    panel per tour, with the share of points per bin underneath. Bins come from
    src/metrics.calibration_bins (from scratch); bins with < min_count points
    are dropped. ECE = point-weighted mean |observed - predicted|."""
    import sys
    sys.path.insert(0, str(ROOT))
    import pyarrow.parquet as pq
    from src.metrics import calibration_bins

    path = RESULTS / "predictions" / f"{name}.parquet"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp1 first.")
        return
    df = pq.read_table(path, columns=["dataset", "tour", "model", "y", "p"],
                       filters=[("dataset", "=", "real")]).to_pandas()
    groups = [g for g in _GROUPS if g in set(df["tour"].astype(str))]
    models = list(dict.fromkeys(df["model"].astype(str)))

    fig, axes = plt.subplots(2, len(groups), figsize=(3.4 * len(groups), 3.6), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    rows = []
    for j, group in enumerate(groups):
        ax, hx = axes[0][j], axes[1][j]
        lo, hi = 1.0, 0.0
        for i, model in enumerate(models):
            d = df[(df["tour"] == group) & (df["model"] == model)]
            y, p = d["y"].to_numpy(float), d["p"].to_numpy(float)
            pred, obs, cnt = calibration_bins(y, p, n_bins=n_bins)
            keep = cnt >= min_count
            ece = float(np.nansum(cnt * np.abs(obs - pred)) / cnt.sum())
            ax.plot(pred[keep], obs[keep], marker="o", ms=4, lw=1.5, color=f"C{i}",
                    label=f"{model} (ECE {ece:.4f})")
            w = 1.0 / n_bins
            hx.bar(pred[keep] + (i - 0.5) * w * 0.4, cnt[keep] / cnt.sum(), width=w * 0.4,
                   color=f"C{i}", alpha=0.8)
            lo, hi = min(lo, np.nanmin(pred[keep])), max(hi, np.nanmax(pred[keep]))
            rows += [{"tour": group, "model": model, "bin": b, "mean_pred": pred[b],
                      "obs_freq": obs[b], "count": int(cnt[b]), "ece": ece}
                     for b in range(n_bins) if cnt[b] > 0]
        pad = 0.03
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", lw=0.8, ls="--",
                label="perfect")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(group)
        ax.legend(fontsize=6.5, loc="upper left")
        hx.set_xlabel("Predicted P(player 1 wins point)")
        hx.tick_params(labelsize=7)
    axes[0][0].set_ylabel("Observed frequency")
    axes[1][0].set_ylabel("Share", fontsize=8)
    fig.suptitle("Calibration of the full models (real data, out-of-fold)", fontsize=9)
    fig.tight_layout()
    _save(fig, "calibration", rows)


if __name__ == "__main__":
    calibration_figure()
    # every exp1 results file (the default run and per-model runs such as
    # exp1_shuffle_control_logreg-gboost / _rmc)
    for f in sorted(RESULTS.glob("exp1_shuffle_control*.json")):
        shuffle_control_figure(f.stem)
    memory_horizon_figure()
    memory_horizon_figure("exp2_memory_horizon_ewma_truncated")
    base_vs_full_figure()
    print("Done.")
