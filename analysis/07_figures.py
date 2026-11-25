"""
Step 7 — Figures for the revised manuscript (PLOS: TIFF, RGB, 300 dpi, LZW, width <= 7.5 in, Arial).
  Fig 1  Study design and cohort selection (flow)
  Fig 2  ROC (A) and precision–recall (B) curves, temporal validation set
  Fig 3  Calibration (deciles), temporal validation set
  Fig 4  Decision curve analysis, temporal validation set
  Fig 5  Mean |SHAP| of the final model (training sample)
Outputs to ../05_figures/. Run 03, 05 and 06 first.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parents[1] / "05_figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "legend.frameon": False})
C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"

def save(fig, name):
    fig.savefig(FIG / f"{name}.tif", dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"},
                facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=110, facecolor="white", bbox_inches="tight")   # preview only
    plt.close(fig)

c = pd.read_pickle(OUT / "cohort.pkl")
te = c[c["set"] == "test"].reset_index(drop=True)
P = pd.read_pickle(OUT / "test_predictions.pkl"); y = P["y"].values
chosen = "CatBoost"
SER = [(f"p_{chosen}", "CatBoost (final model)", C1), ("p_LR", "Logistic regression, all predictors", C2),
       ("p_LR (index VL only)", "Logistic regression, index VL only", C3)]

# ------------------------------------------------------------------ Fig 1: flow
def box(ax, x, yy, w, h, text, title=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, yy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc="white", ec=INK, lw=0.8))
    if title:
        ax.text(x, yy + h / 2 - 0.018, title, ha="center", va="top", fontsize=8, fontweight="bold")
        ax.text(x, yy - 0.014, text, ha="center", va="center", fontsize=7.5, linespacing=1.35)
    else:
        ax.text(x, yy, text, ha="center", va="center", fontsize=7.5, linespacing=1.35)
def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
fig, ax = plt.subplots(figsize=(7.5, 8.2)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.subplots_adjust(0, 0, 1, 1)
box(ax, 0.40, 0.935, 0.66, 0.09, "Patients enrolled in HIV Case-Based Surveillance: 215,881\n"
    "VL records: 1,030,829\nUnique valid VL tests, 2010–July 2025: 565,620 (198,930 patients)")
arrow(ax, 0.40, 0.89, 0.40, 0.845)
box(ax, 0.40, 0.79, 0.66, 0.11, "Candidate index tests with a next VL test 90–548 days later: 307,153\n"
    "Index test date = t0; predictors use information on or before t0\nOutcome = first VL test 90–548 days after t0")
arrow(ax, 0.40, 0.735, 0.40, 0.585)
ax.plot([0.40, 0.75], [0.66, 0.66], color=INK, lw=0.8); arrow(ax, 0.75, 0.66, 0.765, 0.66)
box(ax, 0.875, 0.66, 0.22, 0.15, "Before ART start, missing or\nimplausible ART start: 4,602\n"
    "Age <15 years at t0: 49\nOn/after recorded death\nor ART stop: 434", title="Excluded index tests")
box(ax, 0.40, 0.555, 0.50, 0.06, "Eligible index–outcome pairs: 302,068")
ax.plot([0.40, 0.40], [0.525, 0.50], color=INK, lw=0.8); ax.plot([0.25, 0.75], [0.50, 0.50], color=INK, lw=0.8)
arrow(ax, 0.25, 0.50, 0.25, 0.465); arrow(ax, 0.75, 0.50, 0.75, 0.465)
box(ax, 0.25, 0.38, 0.46, 0.15, "Outcome test 1 Jan 2020 – 31 Dec 2023\nLatest eligible index test per patient\n"
    "72,964 patients; unsuppressed 2,334 (3.2%)\nIndex suppressed: 69,456 (1.98% rebound)\n"
    "Index unsuppressed: 3,508 (27.4% persistent)", title="Training set")
box(ax, 0.75, 0.38, 0.46, 0.15, "Outcome test 1 Jan 2024 – 31 Jul 2025\nLatest eligible index test per patient\n"
    "99,611 patients; unsuppressed 2,263 (2.3%)\nIndex suppressed: 96,535 (1.59% rebound)\n"
    "Index unsuppressed: 3,076 (23.7% persistent)", title="Temporal validation set")
arrow(ax, 0.25, 0.305, 0.25, 0.24); arrow(ax, 0.75, 0.305, 0.75, 0.24)
box(ax, 0.25, 0.175, 0.46, 0.13, "5-fold cross-validation pipelines\nOptuna tuning (objective: AUPRC)\n"
    "Model choice on out-of-fold AUPRC\nPlatt recalibration; thresholds A and B", title="Model development (training data only)")
box(ax, 0.75, 0.175, 0.46, 0.13, "Single evaluation at the true prevalence\nSubsets: no training record (53,357);\n"
    "index test from 2024 (32,525);\nrebound and persistence", title="Validation (used once)")
arrow(ax, 0.48, 0.175, 0.52, 0.175)
save(fig, "Fig1")

# ------------------------------------------------------------------ Fig 2: ROC + PR
fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.5))
for col, lab, colr in SER:
    fpr, tpr, _ = roc_curve(y, P[col]); a.plot(fpr, tpr, color=colr, lw=1.6, label=f"{lab} (AUROC {roc_auc_score(y, P[col]):.3f})")
    pr, rc, _ = precision_recall_curve(y, P[col]); b.plot(rc, pr, color=colr, lw=1.6, label=f"{lab} (AUPRC {average_precision_score(y, P[col]):.3f})")
for thr, mk, lab in [(200, "o", "Rule: index VL ≥200"), (50, "s", "Rule: index VL ≥50")]:
    f = (te["vl"] >= thr).values
    sens = (f & (y == 1)).sum() / y.sum(); spec = (~f & (y == 0)).sum() / (y == 0).sum(); ppv = (f & (y == 1)).sum() / f.sum()
    a.plot(1 - spec, sens, mk, color=INK, ms=6, label=lab); b.plot(sens, ppv, mk, color=INK, ms=6, label=lab)
a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
b.axhline(y.mean(), color=MUTED, lw=0.8, ls="--"); b.text(0.98, y.mean() + 0.01, f"Prevalence {y.mean():.3f}", ha="right", va="bottom", fontsize=7, color="#52514e")
a.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1)); a.set_title("A", loc="left", fontweight="bold")
b.set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value (precision)", xlim=(0, 1), ylim=(0, 0.6)); b.set_title("B", loc="left", fontweight="bold")
a.legend(fontsize=6.5, loc="lower right"); b.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); save(fig, "Fig2")

# ------------------------------------------------------------------ Fig 3: calibration
cal = pd.read_csv(OUT / "calibration_deciles.csv")
fig, ax = plt.subplots(figsize=(4.5, 4.2))
mx = max(cal["pred"].max(), cal["obs"].max()) * 1.08
ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8, ls="--", label="Perfect calibration")
for m, lab, colr, mk in [(chosen, "CatBoost (final model)", C1, "o"), ("LR", "Logistic regression, all predictors", C2, "s")]:
    g = cal[cal["model"] == m]; ax.plot(g["pred"] * 100, g["obs"] * 100, marker=mk, ms=5, lw=1.4, color=colr, label=lab) if False else \
        ax.plot(g["pred"], g["obs"], marker=mk, ms=5, lw=1.4, color=colr, label=lab)
ax.set(xlim=(0, mx), ylim=(0, mx), xlabel="Mean predicted risk (decile)", ylabel="Observed proportion unsuppressed")
ax.text(0.02, mx * 0.93, "CatBoost: slope 0.99, intercept −0.15\nLogistic regression: slope 1.00, intercept −0.14", fontsize=7, va="top")
ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); save(fig, "Fig3")

# ------------------------------------------------------------------ Fig 4: decision curve
d = pd.read_csv(OUT / "decision_curve.csv"); d = d[d["threshold"] <= 0.30]
fig, ax = plt.subplots(figsize=(5.5, 4.0))
for m, lab, colr, ls in [(chosen, "CatBoost (final model)", C1, "-"), ("LR", "Logistic regression, all predictors", C2, "-"),
                         ("Rule: index VL >=200", "Rule: index VL ≥200", INK, "-"), ("Treat all", "Flag all patients", MUTED, ":"),
                         ("Treat none", "Flag none", MUTED, "--")]:
    g = d[d["model"] == m]; ax.plot(g["threshold"] * 100, g["net_benefit"] * 1000, color=colr, lw=1.5 if ls == "-" else 1, ls=ls, label=lab)
ax.set(ylim=(-2, 12), xlim=(1, 30), xlabel="Risk threshold for flagging (%)", ylabel="Net benefit (net true positives per 1,000 patients)")
ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); save(fig, "Fig4")

# ------------------------------------------------------------------ Fig 5: SHAP
LAB = {"index_vl_log10": "Index VL (log10)", "cd4_enrol": "CD4 count at enrolment", "prop_suppressed_hist": "Proportion of suppressed results",
       "tw_suppression_hist": "Time-weighted proportion suppressed", "index_vl_cat": "Index VL category", "age": "Age",
       "province": "Province", "supp_streak": "Current run of suppressed results", "who_stage_enrol": "WHO stage at enrolment",
       "n_uvl_hist": "Number of unsuppressed results", "months_since_last_uvl": "Months since last unsuppressed result",
       "prev_vl_cat": "Previous VL category", "any_llv_hist": "Any low-level viraemia (50–199)", "marital": "Marital status",
       "tpt_started": "TPT started", "years_on_art": "Years on ART", "days_dx_to_art": "Days from diagnosis to ART",
       "n_tests_hist": "Number of VL tests", "female": "Female sex", "months_since_prev_test": "Months since previous test",
       "vl_switches_hist": "Changes in suppression status", "stable": "Stable-patient status (follow-up form)",
       "initial_regimen_dtg": "DTG-based initial regimen", "employed": "Employed", "adherence": "Adherence assessment (follow-up form)"}
VLH = {"index_vl_log10", "prop_suppressed_hist", "tw_suppression_hist", "index_vl_cat", "supp_streak", "n_uvl_hist",
       "months_since_last_uvl", "prev_vl_cat", "any_llv_hist", "vl_switches_hist", "n_tests_hist", "months_since_prev_test"}
s = pd.read_csv(OUT / "shap_importance.csv", index_col=0).iloc[:, 0].head(15)[::-1]
fig, ax = plt.subplots(figsize=(6.0, 4.6))
cols = [C1 if k in VLH else "#9fbfe8" for k in s.index]
ax.barh([LAB.get(k, k) for k in s.index], s.values * 100, color=cols, height=0.7)
for yy, v in enumerate(s.values * 100):
    ax.text(v + 0.3, yy, f"{v:.1f}", va="center", fontsize=7, color="#52514e")
ax.set(xlabel="Share of total mean |SHAP value| (%)"); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C1, label="VL history"), Patch(color="#9fbfe8", label="Other predictors")], fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "Fig5")
print("saved", sorted(p.name for p in FIG.glob("*.tif")))
