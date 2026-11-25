"""
Step 3 — Model development and temporal validation (Reviewer #1, comments 4–7 and 9).

Design (prespecified):
  * Outcome: unsuppressed VL (>=200 copies/mL) at the first test 90–548 days after the index test (t0).
    Secondary outcome: VL >1000 copies/mL (WHO threshold).
  * Development: training period (outcome test 2020–2023). Validation: temporal test period
    (outcome test 2024–Jul 2025), used ONCE, at its original prevalence, never resampled.
  * Predictors: the full prespecified set in outputs/features.json (no data-driven feature selection).
  * All preprocessing (imputation, missing indicators, scaling, one-hot encoding) and any resampling are
    steps of a Pipeline, so they are fitted inside each training CV fold only.
  * Hyperparameters tuned by Optuna on 5-fold stratified CV of the training set (objective: mean AUPRC).
  * Model choice: highest training out-of-fold (OOF) AUPRC.
  * Probabilities recalibrated with logistic (Platt) recalibration fitted on training OOF predictions.
  * Operating thresholds chosen on training OOF predictions:
      (a) sensitivity 0.70; (b) same flag rate as the rule "index VL >=200".
  * Benchmarks: rule index VL >=200; rule index VL >=50 (includes low-level viremia);
    logistic regression on index log10 VL only; full logistic regression.
  * Test-set metrics: AUROC, AUPRC, Brier, calibration intercept/slope, threshold metrics,
    bootstrap 95% CIs (1000 reps) and paired bootstrap differences between models; decision curves;
    subgroups (index suppressed = rebound; index unsuppressed = persistence; new patients only).
  * Sensitivity: SMOTE-ENN resampling inside CV folds (LightGBM) to show the effect of resampling.

Run:  python 03_model.py            (reuses tuned parameters in outputs/best_params.json if present)
      python 03_model.py --retune   (re-runs Optuna)
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
N_BOOT = 1000
OUT = Path(__file__).resolve().parent / "outputs"
lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); lines.append(s)
    (OUT / "model_results.txt").write_text("\n".join(lines))

# ---------------------------------------------------------------- data
c = pd.read_pickle(OUT / "cohort.pkl")
feats = json.loads((OUT / "features.json").read_text())
NUM, CAT = feats["num"], feats["cat"]
X_COLS = NUM + CAT
c["y1000"] = (c["out_vl"] > 1000).astype(int)
tr = c[c["set"] == "train"].reset_index(drop=True)
te = c[c["set"] == "test"].reset_index(drop=True)
TRAIN_END = pd.Timestamp("2024-01-01")
Xtr, Xte = tr[X_COLS], te[X_COLS]
ytr, yte = tr["y"].values, te["y"].values
log(f"Train n={len(tr):,} events={ytr.sum():,} ({ytr.mean():.2%}); "
    f"Test n={len(te):,} events={yte.sum():,} ({yte.mean():.2%}); predictors: {len(NUM)} numeric + {len(CAT)} categorical")

# ---------------------------------------------------------------- preprocessing
def pre(kind):
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    if kind == "impute":   # LR and resampling: impute + missing indicators + scale
        num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("sc", StandardScaler())])
    else:                  # tree models handle missing values natively
        num = "passthrough"
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

NJ = -1
def build(name, p):
    p = dict(p)
    if name == "LR":
        m = LogisticRegression(C=p["C"], max_iter=3000)
        return Pipeline([("pre", pre("impute")), ("m", m)])
    if name == "RF":
        cw = p.pop("class_weight")
        m = RandomForestClassifier(n_estimators=300, class_weight=None if cw == "none" else cw,
                                   n_jobs=NJ, random_state=SEED, **p)
    elif name == "XGBoost":
        m = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=NJ, random_state=SEED, **p)
    elif name == "LightGBM":
        m = LGBMClassifier(subsample_freq=1, n_jobs=NJ, random_state=SEED, verbose=-1, **p)
    elif name == "CatBoost":
        m = CatBoostClassifier(verbose=0, thread_count=NJ, random_seed=SEED, **p)
    return Pipeline([("pre", pre("tree")), ("m", m)])

def space(name, t):
    if name == "LR":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if name == "RF":
        return {"max_depth": t.suggest_int("max_depth", 4, 20),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 200, log=True),
                "max_features": t.suggest_float("max_features", 0.1, 0.6),
                "class_weight": t.suggest_categorical("class_weight", ["none", "balanced_subsample"])}
    common = {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
              "scale_pos_weight": t.suggest_float("scale_pos_weight", 1, 30, log=True)}
    if name == "XGBoost":
        return {**common, "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth": t.suggest_int("max_depth", 2, 8),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "LightGBM":
        return {**common, "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
                "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "CatBoost":
        return {**common, "iterations": t.suggest_int("iterations", 200, 800, step=100),
                "depth": t.suggest_int("depth", 3, 8),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 20, log=True)}

N_TRIALS = {"LR": 15, "RF": 20, "XGBoost": 40, "LightGBM": 40, "CatBoost": 25}
QUICK = "--quick" in sys.argv          # smoke test: 1 trial per model, few bootstraps
if QUICK:
    N_TRIALS = {k: 1 for k in N_TRIALS}; N_BOOT = 20
MODELS = list(N_TRIALS)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(Xtr, ytr))

def oof_pred(pipe, X, y):
    """Out-of-fold predicted probabilities; every fitted step sees training-fold rows only."""
    oof = np.zeros(len(y))
    for a, b in FOLDS:
        p = clone(pipe).fit(X.iloc[a], y[a])
        oof[b] = p.predict_proba(X.iloc[b])[:, 1]
    return oof

# ---------------------------------------------------------------- tuning (training CV only)
bp_file = OUT / ("best_params_quick.json" if QUICK else "best_params.json")
best = json.loads(bp_file.read_text()) if bp_file.exists() and "--retune" not in sys.argv else {}
for name in MODELS:
    if name in best:
        continue
    t0 = time.time()
    def obj(t):
        return average_precision_score(ytr, oof_pred(build(name, space(name, t)), Xtr, ytr))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS[name])
    best[name] = st.best_params
    bp_file.write_text(json.dumps(best, indent=1))
    log(f"Tuned {name}: CV AUPRC {st.best_value:.4f} ({N_TRIALS[name]} trials, {time.time()-t0:.0f}s) {st.best_params}")

# ---------------------------------------------------------------- OOF predictions, model choice, recalibration
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def fit_recal(raw, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw).reshape(-1, 1), y)

oof, test_raw, recal, fitted = {}, {}, {}, {}
for name in MODELS:
    pipe = build(name, best[name])
    oof[name] = oof_pred(pipe, Xtr, ytr)
    fitted[name] = clone(pipe).fit(Xtr, ytr)
    test_raw[name] = fitted[name].predict_proba(Xte)[:, 1]
    recal[name] = fit_recal(oof[name], ytr)

# Benchmark: logistic regression on index log10 VL only
vl_only = Pipeline([("m", LogisticRegression(max_iter=1000))])
oof["LR (index VL only)"] = np.zeros(len(ytr))
for a, b in FOLDS:
    oof["LR (index VL only)"][b] = clone(vl_only).fit(tr.loc[a, ["index_vl_log10"]], ytr[a]).predict_proba(tr.loc[b, ["index_vl_log10"]])[:, 1]
fitted["LR (index VL only)"] = clone(vl_only).fit(tr[["index_vl_log10"]], ytr)
test_raw["LR (index VL only)"] = fitted["LR (index VL only)"].predict_proba(te[["index_vl_log10"]])[:, 1]
recal["LR (index VL only)"] = fit_recal(oof["LR (index VL only)"], ytr)

SCORES = list(oof)
log("\n=== Training out-of-fold performance (model choice uses OOF AUPRC only) ===")
oof_tab = pd.DataFrame({n: {"OOF AUROC": roc_auc_score(ytr, oof[n]), "OOF AUPRC": average_precision_score(ytr, oof[n])}
                        for n in SCORES}).T.round(4)
log(oof_tab.to_string())
ML = [m for m in MODELS if m != "LR"]
CHOSEN = max(ML, key=lambda n: oof_tab.loc[n, "OOF AUPRC"])
log(f"Selected ML model (highest OOF AUPRC among ML models): {CHOSEN}")
oof_tab.to_csv(OUT / "oof_metrics.csv")

cal = lambda n, raw: recal[n].predict_proba(logit(raw).reshape(-1, 1))[:, 1]
oof_cal = {n: cal(n, oof[n]) for n in SCORES}
P = {n: cal(n, test_raw[n]) for n in SCORES}          # recalibrated test probabilities

# Rules (binary)
RULES = {"Rule: index VL >=200": (te["vl"] >= 200).astype(int).values,
         "Rule: index VL >=50": (te["vl"] >= 50).astype(int).values}
rule_rate_tr = (tr["vl"] >= 200).mean()

# Thresholds on training OOF
def thr_for_sens(p, y, s=0.70):
    return np.sort(p[y == 1])[::-1][int(np.ceil(s * y.sum())) - 1]
THR = {n: {"sens70": thr_for_sens(oof_cal[n], ytr),
           "rule_rate": np.quantile(oof_cal[n], 1 - rule_rate_tr)} for n in SCORES}
json.dump({n: {k: float(v) for k, v in d.items()} for n, d in THR.items()}, open(OUT / "thresholds.json", "w"), indent=1)
log(f"\nThresholds fixed on training OOF: (a) sensitivity 0.70; (b) flag rate of rule 'index VL >=200' in training = {rule_rate_tr:.2%}")

# ---------------------------------------------------------------- metrics
def cal_int_slope(y, p):
    lp = logit(p)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return inter, slope

def thr_metrics(y, flag):
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    div = lambda a, b: a / b if b else np.nan
    ppv = div(tp, tp + fp)
    return {"flagged %": 100 * (tp + fp) / len(y), "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "PPV": ppv, "NPV": div(tn, tn + fn), "flagged per case": div(1, ppv) if ppv == ppv else np.nan, "TP": tp, "FP": fp}

rng = np.random.default_rng(SEED)
BOOT = [rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT)]

def boot_auc(y, scores, idx_list):
    out = {n: {"AUROC": [], "AUPRC": []} for n in scores}
    for idx in idx_list:
        yb = y[idx]
        if yb.sum() == 0: continue
        for n, s in scores.items():
            out[n]["AUROC"].append(roc_auc_score(yb, s[idx])); out[n]["AUPRC"].append(average_precision_score(yb, s[idx]))
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in out.items()}

def ci(a): return f"({np.percentile(a, 2.5):.3f}–{np.percentile(a, 97.5):.3f})"

def evaluate(mask, label, n_boot=N_BOOT, save=None, thresholds=True):
    """thresholds=False in index-VL subgroups: whole-cohort thresholds are not meaningful there (audit F6)."""
    y = yte[mask]; idx_all = np.arange(len(y))
    rs = np.random.default_rng(SEED)
    boots = [rs.integers(0, len(y), len(y)) for _ in range(n_boot)]
    sc = {n: P[n][mask] for n in SCORES}
    bs = boot_auc(y, sc, boots)
    rows = []
    for n in SCORES:
        p = sc[n]; ci_, sl = cal_int_slope(y, p)
        r = {"model": n, "AUROC": roc_auc_score(y, p), "AUROC 95% CI": ci(bs[n]["AUROC"]),
             "AUPRC": average_precision_score(y, p), "AUPRC 95% CI": ci(bs[n]["AUPRC"]),
             "Brier": brier_score_loss(y, p), "cal. intercept": ci_, "cal. slope": sl}
        for tk, tl in ([("sens70", "@sens0.70"), ("rule_rate", "@rule flag rate")] if thresholds else []):
            for k, v in thr_metrics(y, (p >= THR[n][tk]).astype(int)).items():
                if k in ("flagged %", "sensitivity", "PPV", "flagged per case"):
                    r[f"{k} {tl}"] = v
        rows.append(r)
    for n, f in (RULES.items() if thresholds else []):
        f = f[mask]; m = thr_metrics(y, f)
        rows.append({"model": n, "AUROC": roc_auc_score(y, f), "AUPRC": np.nan,
                     **{f"{k} @rule": v for k, v in m.items() if k in ("flagged %", "sensitivity", "specificity", "PPV", "flagged per case")}})
    tab = pd.DataFrame(rows).set_index("model")
    log(f"\n=== TEST SET: {label} (n={len(y):,}, events={y.sum():,}, {y.mean():.2%}) ===")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        log(tab.to_string())
    # paired differences vs selected model and vs full LR
    pr = []
    for ref in [CHOSEN, "LR"]:
        for n in SCORES:
            if n == ref: continue
            for k in ["AUROC", "AUPRC"]:
                d = bs[n][k] - bs[ref][k]
                pr.append({"comparison": f"{n} − {ref}", "metric": k,
                           "diff": (roc_auc_score if k == "AUROC" else average_precision_score)(y, sc[n]) - (roc_auc_score if k == "AUROC" else average_precision_score)(y, sc[ref]),
                           "95% CI": ci(d)})
    pr = pd.DataFrame(pr)
    log("\nPaired bootstrap differences (exploratory, unadjusted for multiplicity; CIs only):")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        log(pr.to_string(index=False))
    if save:
        tab.to_csv(OUT / f"test_metrics_{save}.csv"); pr.to_csv(OUT / f"paired_{save}.csv", index=False)
    return tab

all_mask = np.ones(len(yte), bool)
evaluate(all_mask, "ALL", save="all")
# Strictly prospective subset: index test AND outcome after the end of the training period (audit F1)
evaluate((te["t0"] >= TRAIN_END).values, "PROSPECTIVE: index test on/after 2024-01-01", save="prospective")
evaluate(te["index_suppressed"].values == 1, "index test SUPPRESSED (rebound)", n_boot=min(500, N_BOOT), save="rebound", thresholds=False)
evaluate(te["index_suppressed"].values == 0, "index test UNSUPPRESSED (persistence)", n_boot=min(500, N_BOOT), save="persistence", thresholds=False)
evaluate(te["in_train"].values == 0, "NEW patients only (no training record)", n_boot=min(500, N_BOOT), save="new_patients")

# ---------------------------------------------------------------- decision curve + calibration data
pts = np.round(np.arange(0.01, 0.41, 0.01), 2)
n = len(yte); prev = yte.mean()
dca = []
for pt in pts:
    w = pt / (1 - pt)
    dca.append({"threshold": pt, "model": "Treat all", "net_benefit": prev - (1 - prev) * w})
    dca.append({"threshold": pt, "model": "Treat none", "net_benefit": 0.0})
    for m in SCORES:
        f = P[m] >= pt
        dca.append({"threshold": pt, "model": m, "net_benefit": (f & (yte == 1)).sum() / n - (f & (yte == 0)).sum() / n * w})
    for m, f in RULES.items():
        dca.append({"threshold": pt, "model": m, "net_benefit": ((f == 1) & (yte == 1)).sum() / n - ((f == 1) & (yte == 0)).sum() / n * w})
dca = pd.DataFrame(dca); dca.to_csv(OUT / "decision_curve.csv", index=False)
log("\nNet benefit at selected thresholds (test):")
log(dca[dca.threshold.isin([0.02, 0.05, 0.10, 0.20])].pivot(index="model", columns="threshold", values="net_benefit").round(5).to_string())

calib = []
for m in SCORES:
    q = pd.qcut(P[m], 10, duplicates="drop")
    g = pd.DataFrame({"p": P[m], "y": yte, "bin": q}).groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    g["model"] = m; calib.append(g.reset_index(drop=True))
pd.concat(calib).to_csv(OUT / "calibration_deciles.csv", index=False)

# Bootstrap CI for net-benefit difference vs the rule "index VL >=200" (audit F12)
def nb(flag, y, pt):
    return ((flag & (y == 1)).sum() - (flag & (y == 0)).sum() * pt / (1 - pt)) / len(y)
rule = RULES["Rule: index VL >=200"] == 1
nbd = []
for pt in [0.02, 0.05, 0.10, 0.20]:
    for m in [CHOSEN, "LR"]:
        f = P[m] >= pt
        d = np.array([nb(f[i], yte[i], pt) - nb(rule[i], yte[i], pt) for i in BOOT])
        nbd.append({"threshold": pt, "comparison": f"{m} − rule VL>=200", "NB diff": nb(f, yte, pt) - nb(rule, yte, pt),
                    "95% CI": ci(d), "net extra true cases per 1000 tested": 1000 * (nb(f, yte, pt) - nb(rule, yte, pt))})
nbd = pd.DataFrame(nbd); nbd.to_csv(OUT / "decision_curve_diff.csv", index=False)
log("\nNet-benefit difference vs rule (bootstrap 95% CI):")
with pd.option_context("display.width", 250, "display.float_format", "{:.5f}".format):
    log(nbd.to_string(index=False))

# ---------------------------------------------------------------- sensitivity: SMOTE-ENN inside CV folds (LightGBM)
log("\n=== Sensitivity: SMOTE-ENN resampling inside training folds (LightGBM, tuned params, no class weighting) ===")
lp = {k: v for k, v in best["LightGBM"].items() if k != "scale_pos_weight"}
smote_pipe = ImbPipeline([("pre", pre("impute")), ("rs", SMOTEENN(random_state=SEED)),
                          ("m", LGBMClassifier(subsample_freq=1, n_jobs=NJ, random_state=SEED, verbose=-1, **lp))])
o = oof_pred(smote_pipe, Xtr, ytr)
p_te = clone(smote_pipe).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
log(f"OOF AUPRC {average_precision_score(ytr, o):.4f} vs {oof_tab.loc['LightGBM','OOF AUPRC']:.4f} without resampling; "
    f"Test AUROC {roc_auc_score(yte, p_te):.4f} AUPRC {average_precision_score(yte, p_te):.4f}; "
    f"mean predicted risk on test {p_te.mean():.3f} vs observed {yte.mean():.3f} (uncalibrated)")

# ---------------------------------------------------------------- secondary outcome: WHO threshold >1000
log("\n=== Secondary outcome: VL >1000 copies/mL at outcome test (same tuned hyperparameters) ===")
y1tr, y1te = tr["y1000"].values, te["y1000"].values
log(f"Train events {y1tr.sum():,} ({y1tr.mean():.2%}); test events {y1te.sum():,} ({y1te.mean():.2%})")
rows = []
for name in ["LR", CHOSEN]:
    pipe = build(name, best[name])
    o1 = oof_pred(pipe, Xtr, y1tr)
    rc = fit_recal(o1, y1tr)
    p1 = rc.predict_proba(logit(clone(pipe).fit(Xtr, y1tr).predict_proba(Xte)[:, 1]).reshape(-1, 1))[:, 1]
    bs = boot_auc(y1te, {name: p1}, BOOT[:min(500, N_BOOT)])[name]
    rows.append({"model": name, "AUROC": roc_auc_score(y1te, p1), "AUROC 95% CI": ci(bs["AUROC"]),
                 "AUPRC": average_precision_score(y1te, p1), "AUPRC 95% CI": ci(bs["AUPRC"]),
                 "sens @ thr sens0.70": thr_metrics(y1te, (p1 >= thr_for_sens(rc.predict_proba(logit(o1).reshape(-1, 1))[:, 1], y1tr)).astype(int))["sensitivity"],
                 "PPV @ thr sens0.70": thr_metrics(y1te, (p1 >= thr_for_sens(rc.predict_proba(logit(o1).reshape(-1, 1))[:, 1], y1tr)).astype(int))["PPV"]})
r1 = (te["vl"] > 1000).astype(int).values; m = thr_metrics(y1te, r1)
rows.append({"model": "Rule: index VL >1000", "AUROC": roc_auc_score(y1te, r1), "sens @ thr sens0.70": m["sensitivity"], "PPV @ thr sens0.70": m["PPV"]})
s1 = pd.DataFrame(rows).set_index("model")
with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
    log(s1.to_string())
s1.to_csv(OUT / "test_metrics_who1000.csv")

# ---------------------------------------------------------------- save
joblib.dump({"fitted": fitted, "recal": recal, "best": best, "chosen": CHOSEN, "thr": THR}, OUT / "models.joblib")
pd.DataFrame({"upid": te["upid"], "y": yte, **{f"p_{k}": v for k, v in P.items()}}).to_pickle(OUT / "test_predictions.pkl")
pd.DataFrame({"upid": tr["upid"], "y": ytr, **{f"oof_{k}": v for k, v in oof_cal.items()}}).to_pickle(OUT / "oof_predictions.pkl")
log("\nDone.")
