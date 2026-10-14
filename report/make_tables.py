"""Word tables of the exp1 (shuffle control) and exp2 (memory horizon) results.

Run the experiments first so the JSON files exist, then:
    python report/make_tables.py
It writes report/tables/results_tables.docx (A4 landscape): exp1 real vs
null, exp2 gain over H=0 for both EWMA modes, and exp2 log-loss per H. All
differences are paired per fold; + = lower log-loss = better. Needs
python-docx (pip install python-docx).
"""
import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "report" / "tables" / "results_tables.docx"
MODEL = {"logreg": "Logistic regression", "gboost": "LightGBM"}

doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)          # A4 landscape
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(1.8))
style = doc.styles["Normal"]
style.font.name, style.font.size = "Calibri", Pt(10)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(header_rows, body, widths_cm, first_cols=2):
    t = doc.add_table(rows=0, cols=len(widths_cm))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for hr in header_rows:
        cells = t.add_row().cells
        for c, txt in zip(cells, hr):
            c.text = txt
            shade(c, "D9E2F3")
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.bold = True; r.font.size = Pt(9)
    for row in body:
        cells = t.add_row().cells
        for i, (c, txt) in enumerate(zip(cells, row)):
            c.text = txt
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i < first_cols else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(9)
    for row in t.rows:
        for c, w in zip(row.cells, widths_cm):
            c.width = Cm(w)
    return t


def caption(title, note, new_page=False):
    p = doc.add_paragraph()
    if new_page:
        p.paragraph_format.page_break_before = True
    r = p.add_run(title); r.bold = True
    if note:
        n = doc.add_paragraph(note); n.runs[0].italic = True; n.runs[0].font.size = Pt(9)


doc.add_heading("Tennis momentum — results tables", level=1)
doc.add_paragraph(
    "All differences are paired by fold over the 10 temporal outer folds (mean ± standard error), "
    "in units of 10⁻⁴ log-loss. Positive = lower log-loss = better for the first-named condition. "
    "Men and women have separate models.")

# ---- Table 1: shuffle control (exp1)
E1 = json.load(open(REPO / "results" / "exp1_shuffle_control_logreg-gboost.json"))
seeds = [str(s) for s in E1["seeds"]]
body = []
for tour in ("men", "women"):
    for m in ("logreg", "gboost"):
        P = E1["paired"][tour][m]["log_loss"]
        A = E1["paired"][tour][m]["auc"]
        real = E1["real"][tour][m]["summary"]
        null_ll = np.mean([E1["shuffled"][s][tour][m]["summary"]["log_loss"]["mean"] for s in seeds])
        null_auc = np.mean([E1["shuffled"][s][tour][m]["summary"]["auc"]["mean"] for s in seeds])
        body.append([tour.capitalize(), MODEL[m],
                     f"{real['log_loss']['mean']:.4f}", f"{null_ll:.4f}",
                     f"{P['mean']*1e4:+.1f} ± {P['se']*1e4:.1f}", f"{P['t']:+.1f}", f"{P['better']}/10",
                     f"{P['seed_sd']*1e4:.1f}",
                     *[f"{P['per_seed'][s]*1e4:+.1f}" for s in seeds],
                     f"{real['auc']['mean']:.4f}", f"{null_auc:.4f}",
                     f"{A['mean']*1e4:+.1f} ± {A['se']*1e4:.1f}"])
caption("Table 1. Shuffle control (exp1): real data vs the order-shuffled null.",
        f"Null = points within games and games within sets shuffled, {len(seeds)} seeds "
        f"({', '.join(seeds)}); null columns average over the seeds. Gain = paired per-fold "
        "improvement of real over the seed-averaged null (+ = point order helps). Seed s.d. = spread "
        "of the gain across seeds; the per-seed columns give each seed's gain. Relational RNN run in progress.")
table([["Tour", "Model", "Log-loss real", "Log-loss null", "Gain (×10⁻⁴)", "t", "Real better",
        "Seed s.d.", *[f"Seed {s}" for s in seeds], "AUC real", "AUC null", "AUC gain (×10⁻⁴)"]],
      body, [1.4, 3.2, 1.7, 1.7, 2.3, 1.2, 1.5, 1.4, 1.4, 1.4, 1.4, 1.6, 1.6, 2.3])
doc.add_paragraph()


# ---- exp2 tables
def horizon_tables(fname, label, num):
    D = json.load(open(REPO / "results" / f"{fname}.json"))
    hs = list(D["men"]["logreg"]["curve"])
    nmem = [str(D["men"]["logreg"]["curve"][h]["n_memory_features"]) for h in hs[1:]]
    gain_rows, ll_rows = [], []
    for tour in ("men", "women"):
        for m in ("logreg", "gboost"):
            c = D[tour][m]["curve"]
            h0 = np.array(c["0"]["per_split"]["log_loss"])
            cells = []
            for h in hs[1:]:
                g = (h0 - np.array(c[h]["per_split"]["log_loss"])) * 1e4
                mean = 0.0 if abs(g.mean()) < 0.05 else g.mean()   # no "-0.0"
                cells.append(f"{mean:+.1f} ± {g.std(ddof=1) / np.sqrt(len(g)):.1f}")
            gain_rows.append([tour.capitalize(), MODEL[m], *cells])
            ll_rows.append([tour.capitalize(), MODEL[m],
                            *[f"{c[h]['summary']['log_loss']['mean']:.4f}" for h in hs]])
    head = ["Tour", "Model", *[("Whole match" if h == "match" else f"H = {h}") for h in hs[1:]]]
    caption(f"Table {num}. Memory horizon (exp2, {label}): gain over the H = 0 base (×10⁻⁴ log-loss).",
            "H = 0 is the base model: pre-match and serve features only. Each column adds the in-match "
            "memory features whose lookback is at most H points; 'Whole match' adds all of them (the full "
            "model). + = the extra history helps.", new_page=(num == 3))
    table([head, ["", "Memory features", *nmem]], gain_rows, [1.4, 3.4] + [2.4] * (len(hs) - 1))
    doc.add_paragraph()
    return hs, ll_rows


hs, ll_lookback = horizon_tables("exp2_memory_horizon", "lookback EWMAs", 2)
horizon_tables("exp2_memory_horizon_ewma_truncated",
               "truncated EWMAs: every EWMA computed from only the last H points", 3)

caption("Table 4. Memory horizon (exp2, lookback EWMAs): log-loss at each history length.",
        "Mean over the 10 folds (lower is better). Unpaired fold-to-fold s.e. is about 0.0010 (men) "
        "and 0.0005 (women); the paired gains in Table 2 remove that fold-to-fold variation.")
table([["Tour", "Model", *[("Whole match" if h == "match" else f"H = {h}") for h in hs]]],
      ll_lookback, [1.4, 3.4] + [2.1] * len(hs))

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
