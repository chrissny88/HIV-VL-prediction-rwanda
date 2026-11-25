"""
Step 6 — Model interpretation (SHAP) for the prespecified final model, and precise decision-curve differences.
Descriptive only; no refitting. Outputs: outputs/shap_importance.csv, outputs/decision_curve_diff_precise.csv
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from catboost import Pool

OUT = Path(__file__).resolve().parent / "outputs"
c = pd.read_pickle(OUT / "cohort.pkl"); feats = json.loads((OUT / "features.json").read_text())
X_COLS = feats["num"] + feats["cat"]
M = joblib.load(OUT / "models.joblib"); chosen = M["chosen"]; pipe = M["fitted"][chosen]
tr = c[c["set"] == "train"].reset_index(drop=True)
Xs = tr[X_COLS].sample(20000, random_state=42)
Z = pipe.named_steps["pre"].transform(Xs)
names = pipe.named_steps["pre"].get_feature_names_out()
sv = pipe.named_steps["m"].get_feature_importance(Pool(Z), type="ShapValues")[:, :-1]
orig = [n.split("__", 1)[1] for n in names]
base = []
for n in orig:
    m = [f for f in feats["cat"] if n.startswith(f + "_")]
    base.append(m[0] if m else n)
imp = pd.DataFrame(np.abs(sv), columns=base).T.groupby(level=0).sum().T.mean().sort_values(ascending=False)
imp = (imp / imp.sum()).rename("share of total mean |SHAP|")
imp.to_csv(OUT / "shap_importance.csv")
print(imp.head(15).round(3).to_string())
pd.DataFrame(sv, columns=base).T.groupby(level=0).sum().T.assign(**{"_x_" + k: Xs[k].values for k in feats["num"]}).to_pickle(OUT / "shap_values_sample.pkl")

# precise net-benefit differences
p = pd.read_pickle(OUT / "test_predictions.pkl"); te = c[c["set"] == "test"].reset_index(drop=True)
y = p["y"].values; rule = (te["vl"] >= 200).values
rng = np.random.default_rng(42); B = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
def nb(f, y, pt): return ((f & (y == 1)).sum() - (f & (y == 0)).sum() * pt / (1 - pt)) / len(y)
rows = []
for pt in [0.02, 0.05, 0.10, 0.20]:
    for m in [chosen, "LR"]:
        f = p[f"p_{m}"].values >= pt
        d = np.array([nb(f[i], y[i], pt) - nb(rule[i], y[i], pt) for i in B]) * 1000
        rows.append({"threshold": pt, "model": m, "extra net TP per 1000": 1000 * (nb(f, y, pt) - nb(rule, y, pt)),
                     "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
                     "NB model": nb(f, y, pt), "NB rule": nb(rule, y, pt)})
r = pd.DataFrame(rows); r.to_csv(OUT / "decision_curve_diff_precise.csv", index=False); print(r.round(3).to_string(index=False))
