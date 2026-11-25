"""
Step 5 — Manuscript tables from saved outputs (no refitting).
  Table 1: characteristics at the index test, training vs temporal test cohort, with uVL proportion by category
           (Reviewer #1 additional comment 3: clinical description; comment 2 on age bands 15-24).
  Table 2: test-set performance of models and benchmarks (comments 6-7).
  Table 3: adjusted associations (comment 8).
Outputs: outputs/table1.csv, table2.csv, table3.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
c = pd.read_pickle(OUT / "cohort.pkl")
c["age_group"] = pd.cut(c["age"], [15, 25, 35, 45, 55, 65, 200], right=False,
                        labels=["15–24", "25–34", "35–44", "45–54", "55–64", "≥65"])
c["sex"] = c["female"].map({1.0: "Female", 0.0: "Male"})
c["yoa"] = pd.cut(c["years_on_art"], [0, 2, 5, 10, 100], right=False, labels=["<2", "2–4", "5–9", "≥10"])
c["who"] = c["who_stage_enrol"].map({1: "I", 2: "II", 3: "III", 4: "IV"})
c["prior_uvl"] = np.where((c["n_uvl_hist"] - (c["vl"] >= 200)) > 0, "Yes", "No")
c["prior_tests"] = pd.cut(c["n_tests_hist"] - 1, [-1, 0, 2, 100], labels=["0", "1–2", "≥3"])
c["regimen"] = c["regimen_class"].fillna("Not recorded before index test")
c["prov"] = c["province"].str.title().replace({"Kigali City": "Kigali City", "East": "Eastern", "West": "Western",
                                                "South": "Southern", "North": "Northern"}).fillna("Unknown")
c["marital_s"] = c["marital"].str.capitalize().fillna("Unknown")
c["tpt"] = c["tpt_started"].map({1: "Yes", 0: "No"})
c["ivl"] = c["index_vl_cat"].replace({"<50": "<50", "50-199": "50–199", "200-999": "200–999", ">=1000": "≥1000"})
c["fu"] = c["has_followup_record"].map({1: "Yes", 0: "No"})
c["out"] = np.where(c["y"] == 1, "Unsuppressed (≥200)", "Suppressed (<200)")

def med(s): return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"

rows = []
def add(label, var=None, order=None):
    if var is None:
        rows.append({"Characteristic": label}); return
    rows.append({"Characteristic": label})
    levels = order or [l for l in c[var].dropna().unique()]
    for lv in levels:
        r = {"Characteristic": f"   {lv}"}
        for s in ["train", "test"]:
            g = c[c["set"] == s]; m = g[var] == lv
            r[f"{s} n (%)"] = f"{m.sum():,} ({100 * m.mean():.1f})"
            r[f"{s} uVL %"] = f"{100 * g.loc[m, 'y'].mean():.1f}" if m.sum() else "–"
        rows.append(r)

r = {"Characteristic": "Patients, n"}
for s in ["train", "test"]:
    g = c[c["set"] == s]; r[f"{s} n (%)"] = f"{len(g):,}"; r[f"{s} uVL %"] = f"{100 * g['y'].mean():.1f}"
rows.append(r)
for lab, v in [("Age, years, median (IQR)", "age"), ("Years on ART, median (IQR)", "years_on_art"),
               ("Months from index to outcome test, median (IQR)", None)]:
    r = {"Characteristic": lab}
    for s in ["train", "test"]:
        g = c[c["set"] == s]
        r[f"{s} n (%)"] = med(g[v]) if v else med(g["gap_days"] / 30.44)
    rows.append(r)
add("Age group, years", "age_group", ["15–24", "25–34", "35–44", "45–54", "55–64", "≥65"])
add("Sex", "sex", ["Female", "Male"])
add("Marital status", "marital_s", ["Married/cohabiting", "Single", "Divorced/separated", "Widowed", "Unknown"])
add("Province", "prov", ["Kigali City", "Eastern", "Northern", "Southern", "Western", "Unknown"])
add("WHO stage at enrolment", "who", ["I", "II", "III", "IV"])
add("Years on ART at index test", "yoa", ["<2", "2–4", "5–9", "≥10"])
add("ART regimen class (last follow-up record before index)", "regimen",
    ["DTG-based", "NNRTI-based", "PI-based", "other/unknown", "Not recorded before index test"])
add("TPT started before index test", "tpt", ["Yes", "No"])
add("Index VL, copies/mL", "ivl", ["<50", "50–199", "200–999", "≥1000"])
add("Any unsuppressed VL before the index test", "prior_uvl", ["No", "Yes"])
add("Number of VL tests before the index test", "prior_tests", ["0", "1–2", "≥3"])
add("Follow-up form recorded before index test", "fu", ["Yes", "No"])
add("Outcome at next VL test", "out", ["Suppressed (<200)", "Unsuppressed (≥200)"])
t1 = pd.DataFrame(rows).fillna("")
t1.loc[t1["Characteristic"].str.contains(r"\(<200\)|\(≥200\)"), ["train uVL %", "test uVL %"]] = ""
t1.to_csv(OUT / "table1.csv", index=False)
print(t1.to_string(index=False))

# ---------------- Table 2
t = pd.read_csv(OUT / "test_metrics_all.csv", index_col=0)
order = ["Rule: index VL >=200", "Rule: index VL >=50", "LR (index VL only)", "LR", "RF", "XGBoost", "LightGBM", "CatBoost"]
names = {"Rule: index VL >=200": "Rule: index VL ≥200", "Rule: index VL >=50": "Rule: index VL ≥50",
         "LR (index VL only)": "Logistic regression, index VL only", "LR": "Logistic regression, all predictors",
         "RF": "Random forest", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost (prespecified final model)"}
f = lambda v, d=3: "–" if pd.isna(v) else f"{v:.{d}f}"
rows = []
for m in order:
    x = t.loc[m]
    rule = m.startswith("Rule")
    rows.append({
        "Model": names[m],
        "AUROC (95% CI)": f"{x['AUROC']:.3f}" + ("†" if rule else f" {x['AUROC 95% CI']}"),
        "AUPRC (95% CI)": "–" if rule else f"{x['AUPRC']:.3f} {x['AUPRC 95% CI']}",
        "Calibration slope": "–" if rule else f(x["cal. slope"], 2),
        "Calibration intercept": "–" if rule else f(x["cal. intercept"], 2),
        "Flagged % (A)": f(x["flagged % @rule"], 1) if rule else f(x["flagged % @rule flag rate"], 1),
        "Sensitivity (A)": f(x["sensitivity @rule"], 2) if rule else f(x["sensitivity @rule flag rate"], 2),
        "PPV (A)": f(x["PPV @rule"], 2) if rule else f(x["PPV @rule flag rate"], 2),
        "Flagged % (B)": "–" if rule else f(x["flagged % @sens0.70"], 1),
        "Sensitivity (B)": "–" if rule else f(x["sensitivity @sens0.70"], 2),
        "PPV (B)": "–" if rule else f(x["PPV @sens0.70"], 3),
    })
t2 = pd.DataFrame(rows)
t2.to_csv(OUT / "table2.csv", index=False)
print("\n", t2.to_string(index=False))

# ---------------- Table 3
reg = pd.read_csv(OUT / "regression_main.csv", index_col=0)
crude = pd.read_csv(OUT / "regression_crude.csv")
REF = {"age_group": "35-44", "sex": "male", "marital": "married/cohabiting", "employed": "yes",
       "who_stage_enrol": "I", "years_on_art": ">=10", "initial_dtg": "no", "index_vl": "<50",
       "prior_uvl": "no", "n_prior_tests": ">=3", "province": "kigali city", "period": "2020-23"}
LAB = {"age_group": "Age group, years", "sex": "Sex", "marital": "Marital status", "employed": "Employed",
       "who_stage_enrol": "WHO stage at enrolment", "years_on_art": "Years on ART", "initial_dtg": "DTG-based initial regimen",
       "index_vl": "Index VL, copies/mL", "prior_uvl": "Unsuppressed VL before the index test",
       "n_prior_tests": "VL tests before the index test", "province": "Province", "period": "Period of outcome test"}
ORDER = {"age_group": ["15-24", "25-34", "35-44", "45-54", ">=55"], "sex": ["female", "male"],
         "marital": ["married/cohabiting", "single", "divorced/separated", "widowed"], "employed": ["yes", "no"],
         "who_stage_enrol": ["I", "II", "III/IV"], "years_on_art": ["<2", "2-4", "5-9", ">=10"], "initial_dtg": ["no", "yes"],
         "index_vl": ["<50", "50-199", "200-999", ">=1000"], "prior_uvl": ["no", "yes"], "n_prior_tests": ["0", "1-2", ">=3"],
         "province": ["kigali city", "east", "north", "south", "west"], "period": ["2020-23", "2024-25"]}
pretty = lambda s: s.replace("-", "–").replace(">=", "≥").replace("kigali city", "Kigali City").replace("east", "Eastern") \
    .replace("west", "Western").replace("south", "Southern").replace("north", "Northern")
rows = []
for v in ORDER:
    rows.append({"Variable": LAB[v]})
    for lv in ORDER[v]:
        cr = crude[(crude["variable"] == v) & (crude["level"].astype(str) == lv)]
        crs = f"{int(cr['uVL n'].iloc[0]):,}/{int(cr['n'].iloc[0]):,} ({cr['uVL %'].iloc[0]:.1f})" if len(cr) else ""
        if lv == REF[v]:
            rows.append({"Variable": f"   {pretty(lv).capitalize() if v not in ('province', 'who_stage_enrol') else pretty(lv)}",
                         "uVL/n (%)": crs, "aOR (95% CI)": "1 (reference)", "p": ""})
        else:
            x = reg.loc[f"{v} = {lv}"]
            p = "<0.001" if x["p"] < 0.001 else f"{x['p']:.3f}"
            rows.append({"Variable": f"   {pretty(lv).capitalize() if v not in ('province', 'who_stage_enrol') else pretty(lv)}",
                         "uVL/n (%)": crs, "aOR (95% CI)": f"{x['OR']:.2f} ({x['95% CI low']:.2f}–{x['95% CI high']:.2f})", "p": p})
t3 = pd.DataFrame(rows).fillna("")
t3.to_csv(OUT / "table3.csv", index=False)
print("\n", t3.to_string(index=False))
