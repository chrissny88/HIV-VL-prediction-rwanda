"""
Step 1 — Data profiling to define the prediction design (Reviewer #1, comments 1–3).

Questions answered:
  * How far apart are consecutive VL tests? (can a 6-month horizon be supported?)
  * How many test-to-test transitions are rebound (S->U) vs persistent non-suppression (U->U)?
  * Are `cbs_stable` / enhanced counselling recorded at the same visit as the VL they would "predict"?
Outputs aggregate tables only (no patient-level rows) to outputs/profile_summary.txt.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("CBS_DATA_DIR", Path(__file__).resolve().parents[2]))   # folder with CRF1.csv, CRF2.csv
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)

UPID, VL, VLD = "cbs_Index UPID", "cbs_Viral Load", "cbs_Viral load collection date"
cols1 = ["Organisation unit name", UPID, VL, VLD, "cbs_stable", "cbs_stable_category",
         "cbs_Date de debut des ARVs", "Report date"]
cols2 = cols1 + ["cbs_Attended enhanced counselling", "cbs_Client outcome",
                 "cbs_Client visit date", "cbs_Viral load result return date"]

d1 = pd.read_csv(DATA / "CRF1.csv", usecols=cols1, low_memory=False); d1["src"] = "CRF1"
d2 = pd.read_csv(DATA / "CRF2.csv", usecols=cols2, low_memory=False); d2["src"] = "CRF2"
log(f"CRF1 rows {len(d1):,}  CRF2 rows {len(d2):,}")
d2 = d2[d2[UPID].isin(d1[UPID])]
df = pd.concat([d1, d2], ignore_index=True)
df = df[df[UPID].notna() & (df["Organisation unit name"].str.lower() != "training")]
log(f"Combined rows {len(df):,}; patients {df[UPID].nunique():,}")

# Patients with death/stopped outcome excluded (as in the original analysis)
out = df["cbs_Client outcome"].astype(str).str.lower()
dead = df.loc[out.isin(["death", "stopped"]), UPID].unique()
df = df[~df[UPID].isin(dead)]
log(f"Excluded {len(dead):,} patients with death/stopped outcome")

for c in [VLD, "cbs_Date de debut des ARVs", "Report date", "cbs_Client visit date",
          "cbs_Viral load result return date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df[VL] = pd.to_numeric(df[VL], errors="coerce")

# ---------------- Test-level dataset ----------------
t = df[df[VL].notna() & (df[VL] > 0) & df[VLD].notna()].copy()
log(f"\nRows with valid VL (>0) and date: {len(t):,}; patients {t[UPID].nunique():,}")
log("VL collection year distribution:")
log(t[VLD].dt.year.value_counts().sort_index().to_string())

# Same test recorded more than once (same patient, same date)
dup = t.duplicated([UPID, VLD], keep=False)
g = t[dup].groupby([UPID, VLD])[VL].nunique()
log(f"\nRows sharing patient+date: {dup.sum():,}; patient-dates with conflicting VL values: {(g > 1).sum():,}")
t = (t.sort_values([UPID, VLD, "src"])  # CRF2 (follow-up) preferred over CRF1 on ties
       .drop_duplicates([UPID, VLD], keep="last"))
t = t[(t[VLD] >= "2015-01-01") & (t[VLD] <= "2025-07-31")]
log(f"Unique tests (2015–Jul 2025): {len(t):,}; patients {t[UPID].nunique():,}")
log(f"Unique tests 2020–Jul 2025: {(t[VLD] >= '2020-01-01').sum():,}")

t["U"] = (t[VL] >= 200).astype(int)
t["LLV"] = ((t[VL] >= 50) & (t[VL] < 200)).astype(int)
log(f"Test-level uVL (>=200) prevalence: {t['U'].mean():.2%}; low-level viremia 50–199: {t['LLV'].mean():.2%}")

ntests = t.groupby(UPID).size()
log("\nTests per patient:")
log(ntests.describe(percentiles=[.25, .5, .75, .9]).round(1).to_string())
log((ntests.clip(upper=8).value_counts().sort_index()).rename(index={8: "8+"}).to_string())

# ---------------- Consecutive test pairs ----------------
t = t.sort_values([UPID, VLD])
t["prev_U"] = t.groupby(UPID)["U"].shift()
t["prev_VL"] = t.groupby(UPID)[VL].shift()
t["gap_days"] = t.groupby(UPID)[VLD].diff().dt.days
p = t[t["prev_U"].notna()].copy()
p = p[p[VLD] >= "2020-01-01"]
p["transition"] = np.select(
    [(p.prev_U == 0) & (p.U == 0), (p.prev_U == 0) & (p.U == 1),
     (p.prev_U == 1) & (p.U == 1), (p.prev_U == 1) & (p.U == 0)],
    ["S->S", "S->U (rebound)", "U->U (persistent)", "U->S (re-suppression)"])
log(f"\nConsecutive test pairs with outcome test in 2020–Jul 2025: {len(p):,} (patients {p[UPID].nunique():,})")
log(p["transition"].value_counts().to_string())
log(p["transition"].value_counts(normalize=True).mul(100).round(2).to_string())

def gap_summary(s, label):
    m = s / 30.44
    log(f"\nGap between consecutive tests (months) — {label} (n={len(s):,})")
    log(m.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    bins = [-0.01, 3, 5, 7, 9, 13, 18, 24, np.inf]
    lab = ["<=3", "3-5", "5-7", "7-9", "9-13", "13-18", "18-24", ">24"]
    log(pd.cut(m, bins, labels=lab).value_counts(normalize=True).sort_index().mul(100).round(1).to_string())

gap_summary(p["gap_days"], "all pairs")
gap_summary(p.loc[p.prev_U == 0, "gap_days"], "previous test suppressed")
gap_summary(p.loc[p.prev_U == 1, "gap_days"], "previous test unsuppressed")

log("\nOutcome by gap window, previous test SUPPRESSED (rebound risk):")
p["gap_win"] = pd.cut(p["gap_days"] / 30.44, [0, 4, 9, 15, 24, np.inf],
                      labels=["<4m", "4-9m", "9-15m", "15-24m", ">24m"])
log(p[p.prev_U == 0].groupby("gap_win", observed=True)["U"].agg(["size", "mean"]).round(4).to_string())
log("\nOutcome by gap window, previous test UNSUPPRESSED:")
log(p[p.prev_U == 1].groupby("gap_win", observed=True)["U"].agg(["size", "mean"]).round(4).to_string())

log("\nPrevious VL category vs next uVL (prev suppressed range split by LLV):")
p["prev_cat"] = pd.cut(p["prev_VL"], [0, 50, 200, 1000, np.inf], right=False,
                       labels=["<50", "50-199", "200-999", ">=1000"])
log(p.groupby("prev_cat", observed=True)["U"].agg(["size", "mean"]).round(4).to_string())

# ---------------- Patient-level: last pair (the original design) ----------------
last = p.groupby(UPID).tail(1)
log(f"\nPatient-level, LAST pair per patient (original design): n={len(last):,}")
log(last["transition"].value_counts().to_string())
gap_summary(last["gap_days"], "last pair per patient")

# ---------------- Same-visit leakage check ----------------
log("\n=== Leakage check: variables recorded on the SAME row as the VL result ===")
r = df[df[VL].notna() & (df[VL] > 0)].copy()
r["U"] = (r[VL] >= 200).astype(int)
r["stable"] = pd.to_numeric(r["cbs_stable"], errors="coerce")
log("uVL rate by cbs_stable on the same row:")
log(r.groupby("stable")["U"].agg(["size", "mean"]).round(4).to_string())
log("\nstable_category x same-row uVL:")
log(pd.crosstab(r["cbs_stable_category"].fillna("NA"), r["U"]).to_string())
ec = pd.to_numeric(r["cbs_Attended enhanced counselling"], errors="coerce")
log("\nuVL rate by enhanced counselling (same row, CRF2):")
log(r.assign(ec=ec).groupby("ec", dropna=False)["U"].agg(["size", "mean"]).round(4).to_string())

# Stable status vs PREVIOUS test (would show stable reflects past VL, i.e. usable if timed right)
t2 = t.copy()
t2["stable"] = pd.to_numeric(t2["cbs_stable"], errors="coerce")
log("\nuVL rate of CURRENT test by stable recorded on that test row, split by PREVIOUS test status:")
log(t2[t2.prev_U.notna()].groupby(["prev_U", "stable"])["U"].agg(["size", "mean"]).round(4).to_string())

# Report date vs VL date (how far after the test the row was recorded)
lag = (r["Report date"] - r[VLD]).dt.days
log("\nDays from VL collection to row report date:")
log(lag.describe(percentiles=[.1, .5, .9]).round(0).to_string())
tat = (r["cbs_Viral load result return date"] - r[VLD]).dt.days
log("\nVL turnaround (result return - collection), days:")
log(tat.describe(percentiles=[.1, .5, .9]).round(0).to_string())

(OUT / "profile_summary.txt").write_text("\n".join(lines))
print(f"\nSaved {OUT / 'profile_summary.txt'}")
