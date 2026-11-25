"""
Step 4 — Explanatory logistic regression (Reviewer #1, comment 8).

  * Covariates PRESPECIFIED on clinical grounds (not selected by the ML models), all measured at or before t0
    and with <1% missingness (complete-case analysis).
  * Fitted on the original, non-resampled data. Main analysis: all landmark records (both periods),
    one record per patient per period, standard errors clustered by patient. Sensitivity: training period only.
  * Odds ratios reported per clinically interpretable unit / category (no min-max scaling).
  * Multicollinearity: generalised VIF (GVIF^(1/(2·df))) per covariate.
  * Interpreted as adjusted associations, not causal effects.
  * Deliberately NOT included: enhanced counselling, stable status, adherence counselling, visit frequency,
    because they are triggered by prior non-suppression (reverse causation) and are missing for ~54% of records.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT = Path(__file__).resolve().parent / "outputs"
lines = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

c = pd.read_pickle(OUT / "cohort.pkl")
d = pd.DataFrame({
    "y": c["y"], "upid": c["upid"], "set": c["set"],
    "age_group": pd.cut(c["age"], [15, 25, 35, 45, 55, 200], right=False,
                        labels=["15-24", "25-34", "35-44", "45-54", ">=55"]),
    "sex": np.where(c["female"] == 1, "female", np.where(c["female"] == 0, "male", None)),
    "marital": c["marital"],
    "employed": c["employed"].map({1.0: "yes", 0.0: "no"}),
    "who_stage_enrol": c["who_stage_enrol"].map({1: "I", 2: "II", 3: "III/IV", 4: "III/IV"}),
    "years_on_art": pd.cut(c["years_on_art"], [0, 2, 5, 10, 100], right=False,
                           labels=["<2", "2-4", "5-9", ">=10"]),
    "initial_dtg": c["initial_regimen_dtg"].map({1.0: "yes", 0.0: "no"}),
    "index_vl": c["index_vl_cat"],
    "prior_uvl": np.where((c["n_uvl_hist"] - (c["vl"] >= 200)) > 0, "yes", "no"),
    "n_prior_tests": pd.cut(c["n_tests_hist"] - 1, [-1, 0, 2, 100], labels=["0", "1-2", ">=3"]),
    "province": c["province"],
    "period": np.where(c["set"] == "test", "2024-25", "2020-23"),
})
REF = {"age_group": "35-44", "sex": "male", "marital": "married/cohabiting", "employed": "yes",
       "who_stage_enrol": "I", "years_on_art": ">=10", "initial_dtg": "no", "index_vl": "<50",
       "prior_uvl": "no", "n_prior_tests": ">=3", "province": "kigali city", "period": "2020-23"}
COV = list(REF)
n0 = len(d)
d = d.dropna(subset=COV).reset_index(drop=True)
log(f"Records {n0:,}; complete cases {len(d):,} ({len(d)/n0:.1%}); patients {d['upid'].nunique():,}; events {d['y'].sum():,}")

formula = "y ~ " + " + ".join(f"C({v}, Treatment(reference='{REF[v]}'))" for v in COV)

def fit(data, label):
    m = smf.logit(formula, data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(data["upid"])[0]})
    ci = m.conf_int()
    t = pd.DataFrame({"OR": np.exp(m.params), "95% CI low": np.exp(ci[0]), "95% CI high": np.exp(ci[1]),
                      "p": m.pvalues}).drop("Intercept")
    t.index = [i.split(",")[0].replace("C(", "") + " = " + i.split("[T.")[1].rstrip("]") for i in t.index]
    log(f"\n=== {label}: n={int(m.nobs):,}, events={int(data['y'].sum()):,} ===")
    log(t.round(3).to_string())
    return t

# Descriptive: crude event rate per category (to accompany ORs)
crude = []
for v in COV:
    g = d.groupby(v, observed=True)["y"].agg(["size", "sum", "mean"])
    for k, r in g.iterrows():
        crude.append({"variable": v, "level": k, "n": int(r["size"]), "uVL n": int(r["sum"]), "uVL %": 100 * r["mean"]})
crude = pd.DataFrame(crude)
log("\nCrude uVL proportion by category:")
log(crude.round(2).to_string(index=False))

main = fit(d, "MAIN: all landmark records, SE clustered by patient")
sens = fit(d[d["set"] == "train"].reset_index(drop=True), "SENSITIVITY: training period only")

# Generalised VIF (Fox & Monette 1992)
X = pd.get_dummies(d[COV], drop_first=False)
for v in COV:   # drop reference levels
    X = X.drop(columns=f"{v}_{REF[v]}")
X = X.astype(float)
R = np.corrcoef(X.values, rowvar=False)
detR = np.linalg.det(R)
gv = []
for v in COV:
    idx = [i for i, col in enumerate(X.columns) if col.startswith(v + "_")]
    oth = [i for i in range(X.shape[1]) if i not in idx]
    g = np.linalg.det(R[np.ix_(idx, idx)]) * np.linalg.det(R[np.ix_(oth, oth)]) / detR
    gv.append({"covariate": v, "df": len(idx), "GVIF": g, "GVIF^(1/(2df))": g ** (1 / (2 * len(idx)))})
gv = pd.DataFrame(gv)
log("\nGeneralised VIF (values of GVIF^(1/(2df)) < 2.24, i.e. VIF < 5, indicate no problematic collinearity):")
log(gv.round(2).to_string(index=False))

main.to_csv(OUT / "regression_main.csv"); sens.to_csv(OUT / "regression_train_only.csv")
gv.to_csv(OUT / "regression_gvif.csv", index=False); crude.to_csv(OUT / "regression_crude.csv", index=False)
(OUT / "regression_results.txt").write_text("\n".join(lines))
