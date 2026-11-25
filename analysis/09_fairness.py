"""
Step 9 — Performance by sociodemographic group in the temporal validation set (TRIPOD+AI items 14, 23a).
Final model (CatBoost) and the index VL >=200 rule. Output: outputs/fairness.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
OUT = Path(__file__).resolve().parent / "outputs"
c = pd.read_pickle(OUT / "cohort.pkl"); te = c[c.set == "test"].reset_index(drop=True)
P = pd.read_pickle(OUT / "test_predictions.pkl"); assert (P.upid.values == te.upid.values).all()
p = P["p_CatBoost"].values; y = te["y"].values
import json; thrB = json.loads((OUT / "thresholds.json").read_text())["CatBoost"]["sens70"]
te["age_group"] = pd.cut(te.age, [15, 25, 35, 45, 55, 200], right=False, labels=["15–24", "25–34", "35–44", "45–54", "≥55"]).astype(str)
te["sex"] = te.female.map({1.0: "Female", 0.0: "Male"}); te["prov"] = te.province.str.title().fillna("Unknown")
rng = np.random.default_rng(42); rows = []
for var, lab in [("sex", "Sex"), ("age_group", "Age group"), ("prov", "Province")]:
    for g in sorted(te[var].dropna().unique()):
        m = (te[var] == g).values; yy, pp = y[m], p[m]
        if yy.sum() < 20: continue
        bs = []
        for _ in range(500):
            i = rng.integers(0, m.sum(), m.sum())
            if 0 < yy[i].sum() < len(i): bs.append(roc_auc_score(yy[i], pp[i]))
        f = pp >= thrB
        rows.append({"Group": lab, "Level": g, "n": int(m.sum()), "Events": int(yy.sum()), "Prevalence, %": round(100 * yy.mean(), 2),
                     "Mean predicted, %": round(100 * pp.mean(), 2), "AUROC": round(roc_auc_score(yy, pp), 3),
                     "AUROC 95% CI": f"({np.percentile(bs, 2.5):.3f}–{np.percentile(bs, 97.5):.3f})",
                     "AUPRC": round(average_precision_score(yy, pp), 3),
                     "Sensitivity @B": round((f & (yy == 1)).sum() / yy.sum(), 2), "Flagged % @B": round(100 * f.mean(), 1)})
r = pd.DataFrame(rows); r.to_csv(OUT / "fairness.csv", index=False); print(r.to_string(index=False))
