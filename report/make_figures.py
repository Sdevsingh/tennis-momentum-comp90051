"""Generate the report figures from results/*.json (all plotting in Python).

Run the experiments first so the JSON files exist, then:
    python report/make_figures.py
It writes report/figures/memory_horizon.pdf and positive_control.pdf, which
report.tex includes (swap the \\fbox placeholders for the \\includegraphics
lines already written as comments in the .tex).

Plotting libraries are allowed by the brief; only metric computation must be
from scratch, and that lives in src/metrics.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)


def memory_horizon_figure():
    path = RESULTS / "exp2_memory_horizon.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp2 first.")
        return
    data = json.loads(path.read_text())
    # New format: {model: {h: summary}}. Fall back to the old flat {h: summary}.
    curves = data if all(isinstance(v, dict) and "log_loss" not in v
                         for v in data.values()) else {"model": data}

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for name, curve in curves.items():
        hs = sorted(int(h) for h in curve)
        means = [curve[str(h)]["log_loss"]["mean"] for h in hs]
        ses = [curve[str(h)]["log_loss"]["se"] for h in hs]
        base = means[0]  # gain over the zero-history baseline
        gain = [base - m for m in means]
        ax.errorbar(hs, gain, yerr=ses, marker="o", capsize=3, label=name)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("History length (points)")
    ax.set_ylabel("Log-loss gain vs. H=0")
    ax.set_title("Memory horizon")
    if len(curves) > 1:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "memory_horizon.pdf")
    print(f"  wrote {FIGDIR / 'memory_horizon.pdf'}")


def positive_control_figure():
    path = RESULTS / "exp3_positive_control.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp3 first.")
        return
    data = json.loads(path.read_text())
    # New format: {model: {delta: summary}}. Fall back to old flat {delta: summary}.
    series = data if all(isinstance(v, dict) and "auc" not in v
                         for v in data.values()) else {"model": data}

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for name, sens in series.items():
        deltas = sorted(float(d) for d in sens)
        aucs = [sens[str(d) if str(d) in sens else f"{d}"]["auc"]["mean"] for d in deltas]
        ses = [sens[str(d) if str(d) in sens else f"{d}"]["auc"]["se"] for d in deltas]
        ax.errorbar(deltas, aucs, yerr=ses, marker="s", capsize=3, label=name)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="chance")
    ax.set_xlabel(r"Planted momentum $\delta$")
    ax.set_ylabel("AUC")
    ax.set_title("Positive control")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "positive_control.pdf")
    print(f"  wrote {FIGDIR / 'positive_control.pdf'}")


if __name__ == "__main__":
    memory_horizon_figure()
    positive_control_figure()
    print("Done. If both PDFs exist, swap the \\fbox placeholders in report.tex "
          "for the \\includegraphics lines.")
