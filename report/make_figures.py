"""Generate the report figures from results/*.json (all plotting in Python).

Run the experiments first so the JSON files exist, then:
    python report/make_figures.py
It writes report/figures/memory_horizon.pdf (exp2) and base_vs_full.pdf
(exp3), which report.tex includes (swap the \\fbox placeholders for the
\\includegraphics lines already written as comments in the .tex).

Plotting libraries are allowed by the brief; only metric computation must be
from scratch, and that lives in src/metrics.py. The error bars here are the
standard error of per-fold paired differences already stored in the JSON.
"""
from __future__ import annotations

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
    for ax, group in zip(axes[0], groups):
        for model, res in data[group].items():
            curve = res["curve"]
            hs = list(curve)                       # 0, 1, ..., 50, "match"
            h0 = curve[hs[0]]["per_split"]["log_loss"]
            gains = [_paired(h0, curve[h]["per_split"]["log_loss"]) for h in hs]
            ax.errorbar(range(len(hs)), [g[0] for g in gains], yerr=[g[1] for g in gains],
                        marker="o", capsize=3, label=model)
        ax.set_xticks(range(len(hs)), hs)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("History length H (points)")
        ax.set_title(group)
    axes[0][0].set_ylabel("Log-loss gain vs. H=0")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Memory horizon" + (" (truncated EWMAs)" if "truncated" in name else ""),
                 fontsize=9)
    fig.tight_layout()
    out = FIGDIR / f"{name.replace('exp2_', '')}.pdf"
    fig.savefig(out)
    print(f"  wrote {out}")


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
    for i, model in enumerate(models):
        g = [data[grp][model]["gain_full_over_base"]["log_loss"] if model in data[grp] else nan
             for grp in groups]
        ax.bar(np.arange(len(groups)) + i * width, [x["mean"] for x in g], width,
               yerr=[x["se"] for x in g], capsize=3, label=model)
    ax.set_xticks(np.arange(len(groups)) + width * (len(models) - 1) / 2, groups)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Log-loss gain, full vs. base")
    ax.set_title("What in-match information adds")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "base_vs_full.pdf")
    print(f"  wrote {FIGDIR / 'base_vs_full.pdf'}")


if __name__ == "__main__":
    memory_horizon_figure()
    memory_horizon_figure("exp2_memory_horizon_ewma_truncated")
    base_vs_full_figure()
    print("Done. Swap the \\fbox placeholders in report.tex for the "
          "\\includegraphics lines once the PDFs exist.")
