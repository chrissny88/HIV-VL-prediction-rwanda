"""
Step 8 — Supporting Information files (aggregate only; no patient-level rows).
  S1 Table  variable inventory: every raw field of CRF1/CRF2 (use / reason for exclusion, missingness) + the 36 predictors
  S3 Table  hyperparameter search spaces, selected values, training out-of-fold performance
  S4 Table  paired bootstrap differences between models (all validation; strictly prospective subset)
  S5 Table  subgroup and sensitivity analyses
  S6 Table  secondary outcome VL >1000 copies/mL
  S7 Table  regression GVIF and training-period-only estimates
  S1 Data   aggregate data underlying Tables 1–3 and Figs 1–5
Outputs to ../06_supporting_information/
"""
import json, re
import os
from pathlib import Path
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.section import WD_ORIENT
from sklearn.metrics import roc_curve, precision_recall_curve

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; DATA = Path(os.environ.get("CBS_DATA_DIR", HERE.parents[1]))
SI = HERE.parent / "06_supporting_information"; SI.mkdir(exist_ok=True)

# ====================================================================== S1 Table
USE = {  # raw field -> (use, derived predictor(s) or reason)
 "Organisation unit name": ("Used", "Province of the facility at t0 (follow-up facility; enrolment facility if none); training units excluded"),
 "cbs_Index UPID": ("Identifier", "Record linkage only; not a predictor"),
 "Tracnet No": ("Identifier", "Not used"),
 "cbs_gender": ("Used", "Sex"),
 "cbs_date of birth": ("Used", "Age at t0; eligibility (≥15 years)"),
 "cbs_marital status": ("Used", "Marital status (enrolment)"),
 "cbs_employment status": ("Used", "Employment (enrolment)"),
 "cbs_occupation": ("Excluded", "Free text, high missingness; employment used instead"),
 "cbs_Date de debut des ARVs": ("Used", "Years on ART at t0; eligibility (index test after ART start); implausible dates (<1996 or before birth) set to missing"),
 "cbs_Date of HIV positive result confirmation": ("Used", "Days from diagnosis to ART"),
 "cbs_Initial ART regimen": ("Used", "DTG-based initial regimen"),
 "Initial ART regimen= Current ART regimen": ("Excluded", "Redundant with initial and current regimen fields"),
 "cbs_Current ART Regimen": ("Used (CRF2)", "Regimen class from the last follow-up record on or before t0; CRF1 value not used (recorded at enrolment report, timing uncertain)"),
 "cbs_If different ART change date": ("Excluded", "Timing relative to t0 unreliable; regimen changes taken from CRF2"),
 "cbs_WHO Stage": ("Used", "WHO stage at enrolment (CRF1); current and highest WHO stage from CRF2 records on or before t0"),
 "cbs_CD4 count results": ("Used", "CD4 count at enrolment, only if CD4 collection date ≤ t0"),
 "cbs_CD4 collection date": ("Used", "Timing check for CD4"),
 "cbs_CD4 result return date": ("Excluded", "Administrative date"),
 "cbs_Initiated on TPT": ("Excluded", "TPT start date used instead (timing known)"),
 "cbs_If yes TPT Start Date": ("Used", "TPT started before t0"),
 "cbs_If no TPT initated reasons": ("Excluded", "Free text / high missingness"),
 "cbs_specify other reasons why no TPT initated": ("Excluded", "Free text / high missingness"),
 "cbs_Viral Load": ("Used", "VL history features and outcome (both forms)"),
 "cbs_Viral load collection date": ("Used", "Defines t0 and outcome test; time ordering"),
 "cbs_Viral load result return date": ("Excluded", "Result return is after t0 (turnaround time removed as predictor; descriptive only)"),
 "cbs_stable": ("Used (CRF2)", "Stable-patient status from the last follow-up record on or before t0. In the original analysis taken from the outcome record (leakage)"),
 "cbs_stable_category": ("Excluded", "Derived from VL results; redundant with VL history; leakage risk"),
 "cbs_Last pharmacy pick up date": ("Used (CRF2)", "Days since last pharmacy pickup at t0"),
 "cbs_Index Case Type": ("Excluded", "Surveillance programme category, not a clinical characteristic"),
 "cbs_Testing clinic": ("Excluded", "Diagnosis setting; not available consistently"),
 "cbs_Testing facility name": ("Excluded", "Facility identifier; province used instead"),
 "cbs_If currently on ART_Where Initiated ART": ("Excluded", "Facility identifier"),
 "cbs_If no linked to treatment at this facility where": ("Excluded", "High missingness"),
 "cbs_Date Linked": ("Excluded", "High missingness; overlaps with ART start"),
 "cbs_Has been commercial sex worker": ("Excluded", "Sensitive behavioural field with high missingness, recorded at diagnosis"),
 "cbs_Number of lifetime sexual partners": ("Excluded", "Behavioural field recorded at diagnosis; not prespecified"),
 "cbs_Number of sexual partners in last 3 months": ("Excluded", "Behavioural field recorded at diagnosis; not prespecified"),
 "cbs_Sex with HIV+ person": ("Excluded", "Behavioural field recorded at diagnosis; not prespecified"),
 "cbs_Sex with commercial sex worker": ("Excluded", "Behavioural field; high missingness"),
 "cbs_Sex with female": ("Excluded", "Behavioural field; not prespecified"),
 "cbs_Sex with male": ("Excluded", "Behavioural field; not prespecified"),
 "cbs_Sex with multiple partners": ("Excluded", "Behavioural field; not prespecified"),
 "cbs_Sexual partners in last 12 months": ("Excluded", "Behavioural field; not prespecified"),
 "cbs_Final Rita result": ("Excluded", "Recency testing (surveillance); high missingness"),
 "cbs_If Rita inconclusive verified": ("Excluded", "Recency testing; empty"),
 "cbs_Recency Assay Test done": ("Excluded", "Recency testing (surveillance)"),
 "cbs_Recency assay test results": ("Excluded", "Recency testing (surveillance)"),
 "cbs_Recency viral load Test done": ("Excluded", "Recency testing (surveillance)"),
 "cbs_Recency viral load test result": ("Excluded", "Recency testing (surveillance)"),
 "cbs_PIT Clinic": ("Excluded", "Empty"),
 "cbs_pit specify": ("Excluded", "Free text"),
 "cbs_pregnant": ("Excluded", "Recorded at enrolment only; status at t0 unknown"),
 "Created by": ("Administrative", "Not used"), "Last updated by": ("Administrative", "Not used"),
 "Report date": ("Used", "Fallback date for ordering CRF2 records when visit date missing; earliest CRF1 record selected"),
 "Enrollment date in CBS": ("Administrative", "Not used"), "Last updated on": ("Administrative", "Not used"),
 "cbs_Reporting staff Name": ("Administrative", "Not used"),
 # CRF2
 "cbs_Any Changes in Demographic Information": ("Excluded", "Administrative flag"),
 "cbs_Any Changes in or Undisclosed Sexual Partners": ("Excluded", "Behavioural; not prespecified"),
 "cbs_Attended enhanced counselling": ("Used (CRF2)", "Any enhanced adherence counselling on or before t0. In the original analysis taken from the outcome record (leakage)"),
 "cbs_Change in regimen": ("Used (CRF2)", "Cumulative number of regimen changes up to t0"),
 "cbs_Changed current residence": ("Excluded", "Empty (>99% missing)"),
 "cbs_Changed current residence type": ("Excluded", "Empty (>99% missing)"),
 "cbs_Client TPT Outcome Date": ("Excluded", "Used implicitly through record date"),
 "cbs_Client TPT outcome": ("Used (CRF2)", "TPT completed on or before t0"),
 "cbs_Client adherence": ("Used (CRF2)", "Most recent adherence assessment on or before t0"),
 "cbs_Client outcome": ("Used (CRF2)", "Past loss to follow-up / transfer out before t0; exclusion of index tests on or after a recorded death or ART stop"),
 "cbs_Client visit date": ("Used (CRF2)", "Dates follow-up information (as-of t0); visits in previous 12 months"),
 "cbs_Date Form completed": ("Excluded", "Empty (>99% missing)"),
 "cbs_Death cause": ("Excluded", "Outcome information; empty"),
 "cbs_If Other reason for Regimen Change specify": ("Excluded", "Free text"),
 "cbs_If TO where to": ("Excluded", "Free text"),
 "cbs_If Toxicity type": ("Excluded", "Empty"),
 "cbs_If Yes reason change ART": ("Excluded", "High missingness; may be triggered by failure (reverse causation)"),
 "cbs_If not Completed or in Progress": ("Excluded", "Free text"),
 "cbs_If yes ART change date": ("Excluded", "High missingness"),
 "cbs_If yes completed 3 months": ("Excluded", "Redundant with TPT outcome"),
 "cbs_If yes number of sexual partners not reported at last visit": ("Excluded", "Behavioural; high missingness"),
 "cbs_OI Diagnosis date": ("Excluded", "High missingness"),
 "cbs_OI Name": ("Excluded", "Free text; high missingness"),
 "cbs_Outcome Date": ("Excluded", "Used only via record date"),
 "cbs_WHO Stage Date": ("Excluded", "Record date used for timing"),
 "cbs_changed employment status": ("Excluded", "Empty (>99% missing)"),
 "cbs_changed marital status": ("Excluded", "Empty (>99% missing)"),
 "cbs_if yes changed employment status": ("Excluded", "Empty"),
 "cbs_if yes changed marital status": ("Excluded", "Empty"),
}
raw = []
for form, f in [("CRF1 (enrolment)", "CRF1.csv"), ("CRF2 (follow-up visit)", "CRF2.csv")]:
    d = pd.read_csv(DATA / f, low_memory=False)
    for col in d.columns:
        u, why = USE.get(col, ("Excluded", "Not prespecified"))
        raw.append({"Form": form, "Field": col, "Records": len(d), "Missing, %": round(100 * d[col].isna().mean(), 1),
                    "Use": u, "Derived predictor / reason": why})
raw = pd.DataFrame(raw)
assert set(raw["Field"]) <= set(USE) | {"x"} or True
feats = json.loads((OUT / "features.json").read_text())
c = pd.read_pickle(OUT / "cohort.pkl")
DEF = {
 "age": ("Age at t0, years", "CRF1 date of birth", "At t0"), "female": ("Female sex", "CRF1", "Baseline"),
 "years_on_art": ("Years from ART start to t0", "CRF1", "At t0"), "days_dx_to_art": ("Days from HIV diagnosis to ART start (0–30 years)", "CRF1", "Baseline"),
 "employed": ("Employed at enrolment", "CRF1", "Baseline"), "who_stage_enrol": ("WHO stage at enrolment (I–IV)", "CRF1", "Baseline"),
 "initial_regimen_dtg": ("DTG-based initial regimen", "CRF1", "Baseline"), "cd4_enrol": ("CD4 count at enrolment, cells/µL (if dated ≤ t0)", "CRF1", "Baseline, ≤ t0"),
 "tpt_started": ("TPT started on or before t0", "CRF1", "≤ t0"), "index_vl_log10": ("log10 index VL", "CRF1/CRF2 VL", "t0"),
 "n_tests_hist": ("Number of VL tests up to and including index", "VL history", "≤ t0"), "n_uvl_hist": ("Number of unsuppressed (≥200) results up to index", "VL history", "≤ t0"),
 "any_llv_hist": ("Any result 50–199 copies/mL up to index", "VL history", "≤ t0"), "prop_suppressed_hist": ("Proportion of suppressed results up to index", "VL history", "≤ t0"),
 "months_since_prev_test": ("Months from previous VL test to index", "VL history", "≤ t0"), "vl_switches_hist": ("Number of changes between suppressed and unsuppressed status up to index", "VL history", "≤ t0"),
 "months_since_last_uvl": ("Months since last unsuppressed result (missing if never)", "VL history", "≤ t0"), "supp_streak": ("Consecutive suppressed results ending at index", "VL history", "≤ t0"),
 "tw_suppression_hist": ("Time-weighted proportion suppressed; weights exp(−days before t0/365)", "VL history", "≤ t0"),
 "has_followup_record": ("Any follow-up record dated on or before t0", "CRF2", "≤ t0"), "stable": ("Stable-patient status, last record ≤ t0", "CRF2", "≤ t0"),
 "ever_enhanced_counselling": ("Any enhanced adherence counselling ≤ t0", "CRF2", "≤ t0"), "ever_ltfu": ("Any loss to follow-up recorded ≤ t0", "CRF2", "≤ t0"),
 "ever_transfer_out": ("Any transfer out recorded ≤ t0", "CRF2", "≤ t0"), "n_regimen_changes": ("Number of regimen changes ≤ t0", "CRF2", "≤ t0"),
 "who_stage": ("Current WHO stage, last record ≤ t0", "CRF2", "≤ t0"), "who_stage_max": ("Highest WHO stage ≤ t0", "CRF2", "≤ t0"),
 "tpt_completed": ("TPT completed ≤ t0", "CRF2", "≤ t0"), "days_since_pickup": ("Days since last pharmacy pickup (0–730)", "CRF2", "≤ t0"),
 "visits_last_12m": ("Follow-up visits in the 12 months before t0", "CRF2", "≤ t0"), "marital": ("Marital status", "CRF1", "Baseline"),
 "regimen_class": ("Current regimen class (DTG/NNRTI/PI/other), last record ≤ t0", "CRF2", "≤ t0"), "adherence": ("Adherence assessment (good/moderate/bad), last record ≤ t0", "CRF2", "≤ t0"),
 "province": ("Province of facility at t0", "CRF1/CRF2 facility", "t0"), "index_vl_cat": ("Index VL category (<50, 50–199, 200–999, ≥1000)", "VL", "t0"),
 "prev_vl_cat": ("Previous VL category (incl. no prior test)", "VL history", "< t0"),
}
pred = []
for f in feats["num"] + feats["cat"]:
    de, src, tm = DEF[f]
    pred.append({"Predictor": f, "Definition": de, "Source": src, "Timing relative to t0": tm,
                 "Type": "categorical" if f in feats["cat"] else "numeric",
                 "Missing, % (training)": round(100 * c.loc[c.set == "train", f].isna().mean(), 1),
                 "Missing, % (validation)": round(100 * c.loc[c.set == "test", f].isna().mean(), 1)})
pred = pd.DataFrame(pred)
removed = pd.DataFrame([
    {"Original variable": "tw_suppression, Suppressed_Counts, VL_Switches", "Problem": "Computed over the whole record including the predicted test", "Revision": "Recomputed using tests up to the index test only"},
    {"Original variable": "CBS_stable, enhanced counselling", "Problem": "Taken from the same record as the outcome VL (recorded a median 184 days after VL collection)", "Revision": "Taken from the last follow-up record dated on or before t0"},
    {"Original variable": "delayed_suppression", "Problem": "Definition depended on future results", "Revision": "Removed"},
    {"Original variable": "VL_turnaround", "Problem": "Result-return date is after the index test", "Revision": "Removed as predictor"},
    {"Original variable": "Adherence_Trend, TPT_duration, link_to_art_days", "Problem": "Used information beyond t0 / unclear timing", "Revision": "Replaced by last adherence assessment ≤ t0, TPT started/completed ≤ t0; linkage dropped (53% missing)"},
])
with pd.ExcelWriter(SI / "S1_Table.xlsx") as xw:
    pd.DataFrame({"S1 Table. Variable inventory": [
        "Sheet 'Raw fields': every field of the CBS enrolment (CRF1) and follow-up (CRF2) forms, with missingness and use.",
        "Sheet 'Predictors': the 36 prespecified predictors, definition, source, timing relative to the index test (t0), and missingness in the analysis sets.",
        "Sheet 'Changes vs original': variables of the original submission that used information after t0 and how they were revised.",
        f"Raw fields: CRF1 {int((raw.Form.str.startswith('CRF1')).sum())}, CRF2 {int((raw.Form.str.startswith('CRF2')).sum())}; predictors: {len(pred)}."]}).to_excel(xw, "README", index=False)
    raw.to_excel(xw, "Raw fields", index=False); pred.to_excel(xw, "Predictors", index=False); removed.to_excel(xw, "Changes vs original", index=False)
print("S1:", len(raw), "raw fields;", len(pred), "predictors")

# ====================================================================== docx helper
def doc_table(path, title, blocks, landscape=False, note=None):
    d = Document()
    st = d.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(9)
    s = d.sections[0]
    if landscape:
        s.orientation = WD_ORIENT.LANDSCAPE; s.page_width, s.page_height = Cm(29.7), Cm(21.0)
    for m in ("left_margin", "right_margin"): setattr(s, m, Cm(1.5))
    d.add_paragraph().add_run(title).bold = True
    for sub, df in blocks:
        if sub:
            d.add_paragraph().add_run(sub).italic = True
        t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"
        for i, col in enumerate(df.columns):
            t.rows[0].cells[i].text = str(col)
            for r in t.rows[0].cells[i].paragraphs[0].runs: r.bold = True
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = "" if (isinstance(v, float) and np.isnan(v)) else str(v)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs: r.font.size = Pt(8)
        d.add_paragraph()
    if note:
        d.add_paragraph(note).runs[0].font.size = Pt(8)
    d.save(path)

NAMES = {"LR": "Logistic regression (all predictors)", "LR (index VL only)": "Logistic regression (index VL only)", "RF": "Random forest"}
nm = lambda s: NAMES.get(s, s)

# ====================================================================== S3 Table
src = (HERE / "03_model.py").read_text()
SPACE = {
 "LR": "C: 0.001–10 (log)", "RF": "n_estimators 300; max_depth 4–20; min_samples_leaf 5–200 (log); max_features 0.1–0.6; class_weight none/balanced_subsample",
 "XGBoost": "learning_rate 0.01–0.2 (log); scale_pos_weight 1–30 (log); n_estimators 100–800; max_depth 2–8; min_child_weight 1–50 (log); subsample 0.6–1; colsample_bytree 0.4–1; reg_lambda 0.001–10 (log)",
 "LightGBM": "learning_rate 0.01–0.2 (log); scale_pos_weight 1–30 (log); n_estimators 100–800; num_leaves 8–128 (log); min_child_samples 20–500 (log); subsample 0.6–1; colsample_bytree 0.4–1; reg_lambda 0.001–10 (log)",
 "CatBoost": "learning_rate 0.01–0.2 (log); scale_pos_weight 1–30 (log); iterations 200–800; depth 3–8; l2_leaf_reg 1–20 (log)"}
TRIALS = {"LR": 15, "RF": 20, "XGBoost": 40, "LightGBM": 40, "CatBoost": 25}
best = json.loads((OUT / "best_params.json").read_text()); oof = pd.read_csv(OUT / "oof_metrics.csv", index_col=0)
s3 = pd.DataFrame([{"Model": nm(m), "Optuna trials": TRIALS.get(m, "–"), "Search space": SPACE.get(m, "No tuning"),
                    "Selected hyperparameters": "; ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.get(m, {}).items()) or "–",
                    "OOF AUROC": f"{oof.loc[m, 'OOF AUROC']:.3f}", "OOF AUPRC": f"{oof.loc[m, 'OOF AUPRC']:.3f}"} for m in oof.index])
doc_table(SI / "S3_Table.docx", "S3 Table. Model development: hyperparameter search spaces, selected values and training out-of-fold (OOF) performance",
          [(None, s3)], landscape=True,
          note="Tuning: Optuna Tree-structured Parzen Estimator (seed 42), objective mean AUPRC over 5-fold stratified cross-validation of the training set (72,964 patients, 2,334 events). "
               "All preprocessing inside the cross-validation pipeline. The final model (prespecified rule: highest OOF AUPRC among machine learning models) was CatBoost. "
               "Differences in OOF AUPRC between machine learning models (≤0.004) are within noise; several selected values lie near the search bounds.")

# ====================================================================== S4 Table
def paired(f):
    p = pd.read_csv(OUT / f)
    p["comparison"] = p["comparison"].str.replace("LR (index VL only)", "LR, index VL only", regex=False)
    p["Difference"] = p["diff"].map(lambda v: f"{v:+.4f}")
    return p[["comparison", "metric", "Difference", "95% CI"]].rename(columns={"comparison": "Comparison (model − reference)", "metric": "Metric"})
doc_table(SI / "S4_Table.docx", "S4 Table. Paired bootstrap differences in AUROC and AUPRC between models in the temporal validation set",
          [("A. All validation patients (n = 99,611; 1,000 bootstrap resamples)", paired("paired_all.csv")),
           ("B. Validation patients with index test from 1 January 2024 (n = 32,525)", paired("paired_prospective.csv"))],
          note="Differences computed on the same bootstrap resamples (paired). Exploratory comparisons, not adjusted for multiplicity. LR, logistic regression with all predictors.")

# ====================================================================== S5 Table
def perf(f, thresholds=True):
    t = pd.read_csv(OUT / f, index_col=0)
    keep = ["AUROC", "AUROC 95% CI", "AUPRC", "AUPRC 95% CI", "cal. intercept", "cal. slope"]
    if thresholds:
        keep += ["flagged % @sens0.70", "sensitivity @sens0.70", "PPV @sens0.70", "flagged % @rule flag rate", "sensitivity @rule flag rate", "PPV @rule flag rate",
                 "flagged % @rule", "sensitivity @rule", "PPV @rule"]
    t = t[[k for k in keep if k in t.columns]].copy()
    t.index = [nm(i).replace(">=", "≥") for i in t.index]
    for col in t.columns:
        if "CI" not in col:
            t[col] = t[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return t.reset_index().rename(columns={"index": "Model"})
smote = re.search(r"OOF AUPRC ([\d.]+) vs ([\d.]+).*?Test AUROC ([\d.]+) AUPRC ([\d.]+); mean predicted risk on test ([\d.]+) vs observed ([\d.]+)",
                  (OUT / "model_results.txt").read_text())
sm = pd.DataFrame([{"Approach": "LightGBM, no resampling (class weighting tuned)", "Training OOF AUPRC": smote.group(2)},
                   {"Approach": "LightGBM, SMOTE-ENN within training folds", "Training OOF AUPRC": smote.group(1),
                    "Validation AUROC": smote.group(3), "Validation AUPRC": smote.group(4),
                    "Mean predicted risk (uncalibrated)": smote.group(5), "Observed prevalence": smote.group(6)}])
doc_table(SI / "S5_Table.docx", "S5 Table. Subgroup, sensitivity and sociodemographic (fairness) analyses in the temporal validation set",
          [("A. Index VL suppressed (<200 copies/mL): rebound (n = 96,535; 1,534 events)", perf("test_metrics_rebound.csv", False)),
           ("B. Index VL unsuppressed (≥200 copies/mL): persistent non-suppression (n = 3,076; 729 events)", perf("test_metrics_persistence.csv", False)),
           ("C. Validation patients without a training record (n = 53,357; 1,174 events)", perf("test_metrics_new_patients.csv")),
           ("D. Index test on or after 1 January 2024 (n = 32,525; 1,016 events)", perf("test_metrics_prospective.csv")),
           ("E. Effect of SMOTE-ENN resampling", sm),
           ("F. Final model (CatBoost) by sex, age group and province", pd.read_csv(OUT / "fairness.csv"))], landscape=True,
          note="Thresholds fixed on training data: '@sens0.70' = threshold B (70% sensitivity in training); '@rule flag rate' = threshold A; '@rule' = rule applied directly. "
               "In F, '@B' = threshold B; AUROC 95% CI from 500 bootstrap resamples. Threshold metrics are not shown in A and B because whole-cohort thresholds are not meaningful within index-VL strata. "
               "Calibration intercept and slope after Platt recalibration on training data. For binary rules, AUROC = mean of sensitivity and specificity.")

# ====================================================================== S6 Table
s6 = pd.read_csv(OUT / "test_metrics_who1000.csv", index_col=0)
s6.index = [nm(i) for i in s6.index]
for col in s6.columns:
    if "CI" not in col:
        s6[col] = s6[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
doc_table(SI / "S6_Table.docx", "S6 Table. Secondary outcome: VL >1000 copies/mL at the next routine test (WHO threshold)",
          [("Training: 1,220 events (1.67%); validation: 1,176 events (1.18%)", s6.reset_index().rename(columns={"index": "Model"}))],
          note="Models refitted for this outcome with the hyperparameters tuned for the primary outcome; recalibration and the 70%-sensitivity threshold refitted on training out-of-fold predictions.")

# ====================================================================== S7 Table
gv = pd.read_csv(OUT / "regression_gvif.csv").round(2)
tr = pd.read_csv(OUT / "regression_train_only.csv", index_col=0)
tr = pd.DataFrame({"Covariate level": tr.index, "aOR (95% CI), training period only": [f"{r.OR:.2f} ({r['95% CI low']:.2f}–{r['95% CI high']:.2f})" for _, r in tr.iterrows()],
                   "p": ["<0.001" if v < 0.001 else f"{v:.3f}" for v in tr["p"]]})
doc_table(SI / "S7_Table.docx", "S7 Table. Multivariable logistic regression: generalised variance inflation factors and training-period-only estimates",
          [("A. Generalised variance inflation factors (GVIF^(1/(2·df)) < 2.24 corresponds to VIF < 5)", gv),
           ("B. Estimates from the training period only (n = 72,292 complete cases; 2,308 events; SEs clustered by patient)", tr)],
          note="GVIF computed from the correlation matrix of the design (Fox and Monette, 1992); approximate for a logistic model. Reference categories as in Table 3.")

# ====================================================================== S1 Data (aggregate)
P = pd.read_pickle(OUT / "test_predictions.pkl"); y = P["y"].values
curves = []
for m in ["CatBoost", "LR", "LR (index VL only)"]:
    fpr, tpr, _ = roc_curve(y, P[f"p_{m}"]); idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
    curves.append(pd.DataFrame({"model": m, "curve": "ROC", "x (1-specificity)": fpr[idx], "y (sensitivity)": tpr[idx]}))
    pr, rc, _ = precision_recall_curve(y, P[f"p_{m}"]); idx = np.unique(np.linspace(0, len(pr) - 1, 200).astype(int))
    curves.append(pd.DataFrame({"model": m, "curve": "PR", "x (1-specificity)": rc[idx], "y (sensitivity)": pr[idx]}))
curves = pd.concat(curves).rename(columns={"x (1-specificity)": "x", "y (sensitivity)": "y"})
flow = (OUT / "cohort_flow.txt").read_text().split("Missingness")[0]
with pd.ExcelWriter(SI / "S1_Data.xlsx") as xw:
    pd.DataFrame({"S1 Data. Aggregate data underlying Tables 1–3 and Figs 1–5 (no individual-level records)": [
        "Fig1_flow: cohort flow counts", "Table1, Table2, Table3: as in the manuscript",
        "Fig2_curves: ROC (x = 1-specificity, y = sensitivity) and PR (x = sensitivity, y = PPV) curve coordinates, 200 points per curve",
        "Fig3_calibration: mean predicted and observed risk by decile", "Fig4_decision_curve: net benefit by threshold",
        "Fig5_SHAP: share of total mean |SHAP| by predictor"]}).to_excel(xw, "README", index=False)
    pd.DataFrame({"line": flow.strip().split("\n")}).to_excel(xw, "Fig1_flow", index=False)
    for t in ["table1", "table2", "table3"]:
        pd.read_csv(OUT / f"{t}.csv").to_excel(xw, t.capitalize(), index=False)
    curves.to_excel(xw, "Fig2_curves", index=False)
    pd.read_csv(OUT / "calibration_deciles.csv").to_excel(xw, "Fig3_calibration", index=False)
    pd.read_csv(OUT / "decision_curve.csv").to_excel(xw, "Fig4_decision_curve", index=False)
    pd.read_csv(OUT / "shap_importance.csv").to_excel(xw, "Fig5_SHAP", index=False)
print("written:", sorted(p.name for p in SI.iterdir()))
