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
    curve = json.loads(path.read_text())
    hs = sorted(int(h) for h in curve)
    means = [curve[str(h)]["log_loss"]["mean"] for h in hs]
    ses = [curve[str(h)]["log_loss"]["se"] for h in hs]
    # Gain over the zero-history baseline (baseline minus current log-loss).
    base = means[0]
    gain = [base - m for m in means]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(hs, gain, yerr=ses, marker="o", capsize=3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("History length (points)")
    ax.set_ylabel("Log-loss gain vs. H=0")
    ax.set_title("Memory horizon")
    fig.tight_layout()
    fig.savefig(FIGDIR / "memory_horizon.pdf")
    print(f"  wrote {FIGDIR / 'memory_horizon.pdf'}")


def positive_control_figure():
    path = RESULTS / "exp3_positive_control.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run exp3 first.")
        return
    sens = json.loads(path.read_text())
    deltas = sorted(float(d) for d in sens)
    aucs = [sens[str(d)]["auc"]["mean"] for d in deltas]
    ses = [sens[str(d)]["auc"]["se"] for d in deltas]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(deltas, aucs, yerr=ses, marker="s", capsize=3)
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
