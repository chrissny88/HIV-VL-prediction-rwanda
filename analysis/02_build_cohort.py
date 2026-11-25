"""
Step 2 — Build the landmark prediction cohort (revised design).

Design (agreed 2026-11-25):
  * Prediction time point (t0): date of an index VL test. The prediction is made once the
    index result is available.
  * Predictors: only information dated <= t0 (VL tests by collection date; CRF2 follow-up
    records by visit date, falling back to report date; CRF1 baseline items by their own dates).
  * Outcome: unsuppressed VL (>=200 copies/mL) at the first VL test 90–548 days after t0
    (i.e. the next routine test; tests <90 days after t0 are treated as confirmatory repeats).
  * Unit: one record per patient per period (latest eligible index test in that period).
  * Temporal split: TRAIN = outcome test in 2020-01-01..2023-12-31;
                    TEST  = outcome test in 2024-01-01..2025-07-31.
Writes outputs/cohort.pkl (patient-level, git-ignored) and outputs/cohort_flow.txt (aggregate).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CBS_DATA_DIR", Path(__file__).resolve().parents[2]))   # folder with CRF1.csv, CRF2.csv
FACILITY = Path(os.environ.get("CBS_FACILITY_FILE", ROOT.parents[1] / "CBS Analysis" / "cbs_Facility .xlsx"))  # facility -> province
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

DATA_END = pd.Timestamp("2025-07-31")
HIST_START = pd.Timestamp("2010-01-01")
TRAIN = (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31"))
TEST = (pd.Timestamp("2024-01-01"), DATA_END)
MIN_GAP, MAX_GAP = 90, 548
U = "upid"

flow = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); flow.append(s)

def dt(s):
    d = pd.to_datetime(s, errors="coerce")
    return d.where((d >= "1920-01-01") & (d <= DATA_END))

# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------
c1 = ["Organisation unit name", "cbs_Index UPID", "cbs_Viral Load", "cbs_Viral load collection date",
      "cbs_gender", "cbs_date of birth", "cbs_Date de debut des ARVs",
      "cbs_Date of HIV positive result confirmation", "cbs_Initial ART regimen", "cbs_WHO Stage",
      "cbs_marital status", "cbs_employment status", "cbs_CD4 count results", "cbs_CD4 collection date",
      "cbs_If yes TPT Start Date", "Report date"]
c2 = ["Organisation unit name", "cbs_Index UPID", "cbs_Viral Load", "cbs_Viral load collection date",
      "cbs_Viral load result return date", "cbs_Client visit date", "Report date", "cbs_stable",
      "cbs_Attended enhanced counselling", "cbs_Client adherence", "cbs_Client outcome",
      "cbs_Current ART Regimen", "cbs_Change in regimen", "cbs_WHO Stage", "cbs_Client TPT outcome",
      "cbs_Last pharmacy pick up date"]
d1 = pd.read_csv(ROOT / "CRF1.csv", usecols=c1, low_memory=False)
d2 = pd.read_csv(ROOT / "CRF2.csv", usecols=c2, low_memory=False)
for d in (d1, d2):
    d.rename(columns={"cbs_Index UPID": U}, inplace=True)
    d["Organisation unit name"] = d["Organisation unit name"].astype(str).str.lower().str.strip()
d1 = d1[d1[U].notna() & (d1["Organisation unit name"] != "training")]
d2 = d2[d2[U].isin(d1[U]) & (d2["Organisation unit name"] != "training")]
log(f"Enrolled patients (CRF1, excl. training units): {d1[U].nunique():,}")
log(f"Follow-up records (CRF2) for enrolled patients: {len(d2):,}")

# --------------------------------------------------------------------------------------
# Baseline (CRF1): one row per patient (earliest report)
# --------------------------------------------------------------------------------------
d1["Report date"] = dt(d1["Report date"])
base = d1.sort_values([U, "Report date"]).drop_duplicates(U, keep="first").copy()
base["female"] = (base["cbs_gender"].str.lower() == "female").astype(float)
base.loc[base["cbs_gender"].isna(), "female"] = np.nan
base["dob"] = dt(base["cbs_date of birth"])
base["art_start"] = dt(base["cbs_Date de debut des ARVs"])
# implausible ART start (before combination ART became available, 1996, or before birth) -> missing
bad_art = (base["art_start"] < "1996-01-01") | (base["art_start"] < base["dob"])
log(f"Implausible ART start date set to missing (before 1996 or before birth): {bad_art.sum():,} patients")
base.loc[bad_art, "art_start"] = pd.NaT
base["dx_date"] = dt(base["cbs_Date of HIV positive result confirmation"])
dx_art = (base["art_start"] - base["dx_date"]).dt.days
base["days_dx_to_art"] = dx_art.where((dx_art >= 0) & (dx_art <= 365 * 30))
mar = base["cbs_marital status"].str.lower().replace({
    "marie": "married/cohabiting", "cohabite": "married/cohabiting", "co-habiting": "married/cohabiting",
    "married": "married/cohabiting", "celibataire": "single", "veuf": "widowed", "widow": "widowed",
    "divorce ou separe": "divorced/separated", "divorced": "divorced/separated"})
base["marital"] = mar
base["employed"] = base["cbs_employment status"].str.lower().map(
    {"employe": 1.0, "employed": 1.0, "sans emploi": 0.0, "unemployed": 0.0})
base["who_stage_enrol"] = base["cbs_WHO Stage"].str.lower().map({"i": 1, "ii": 2, "iii": 3, "iv": 4})
ir = base["cbs_Initial ART regimen"].astype(str).str.lower()
base["initial_regimen_dtg"] = ir.str.contains("dtg|dolutegravir").astype(float)
base.loc[base["cbs_Initial ART regimen"].isna(), "initial_regimen_dtg"] = np.nan
base["cd4_enrol"] = pd.to_numeric(base["cbs_CD4 count results"], errors="coerce").where(lambda x: x.between(0, 3000))
base["cd4_date"] = dt(base["cbs_CD4 collection date"])
base["tpt_start"] = dt(base["cbs_If yes TPT Start Date"])
base["facility_enrol"] = base["Organisation unit name"]
base = base[[U, "female", "dob", "art_start", "days_dx_to_art", "marital", "employed", "who_stage_enrol",
             "initial_regimen_dtg", "cd4_enrol", "cd4_date", "tpt_start", "facility_enrol"]]

# --------------------------------------------------------------------------------------
# VL tests: unique (patient, collection date); conflicting values -> highest (conservative)
# --------------------------------------------------------------------------------------
vl = pd.concat([d1[[U, "cbs_Viral Load", "cbs_Viral load collection date"]],
                d2[[U, "cbs_Viral Load", "cbs_Viral load collection date"]]])
vl.columns = [U, "vl", "date"]
vl["vl"] = pd.to_numeric(vl["vl"], errors="coerce")
vl["date"] = dt(vl["date"])
n_raw = len(vl.dropna(subset=["vl"]))
vl = vl[vl["vl"].notna() & (vl["vl"] > 0) & (vl["date"] >= HIST_START)]
vl = vl.groupby([U, "date"], as_index=False)["vl"].max().sort_values([U, "date"])
log(f"VL records with a result: {n_raw:,} -> unique valid tests 2010–Jul 2025: {len(vl):,} "
    f"({vl[U].nunique():,} patients)")

# Turnaround time per test (from CRF2 rows carrying that test). Descriptive only: NOT a predictor,
# because the result-return date is after t0 (audit finding F2).
tat = d2[[U, "cbs_Viral load collection date", "cbs_Viral load result return date"]].copy()
tat.columns = [U, "date", "ret"]
tat["date"], tat["ret"] = dt(tat["date"]), dt(tat["ret"])
tat["tat_days"] = (tat["ret"] - tat["date"]).dt.days
tat = tat[tat["tat_days"].between(0, 180)].groupby([U, "date"], as_index=False)["tat_days"].min()
vl = vl.merge(tat, on=[U, "date"], how="left")

# Cumulative VL-history features, evaluated AT each test (that test included)
g = vl.groupby(U)
vl["uvl"] = (vl["vl"] >= 200).astype(int)
vl["llv"] = ((vl["vl"] >= 50) & (vl["vl"] < 200)).astype(int)
vl["n_tests_hist"] = g.cumcount() + 1
vl["n_uvl_hist"] = g["uvl"].cumsum()
vl["any_llv_hist"] = (g["llv"].cumsum() > 0).astype(int)
vl["prop_suppressed_hist"] = 1 - vl["n_uvl_hist"] / vl["n_tests_hist"]
vl["prev_vl"] = g["vl"].shift()
vl["months_since_prev_test"] = g["date"].diff().dt.days / 30.44
sw = (vl["uvl"] != g["uvl"].shift()) & g["uvl"].shift().notna()
vl["vl_switches_hist"] = sw.astype(int).groupby(vl[U]).cumsum()
last_uvl_date = vl["date"].where(vl["uvl"] == 1).groupby(vl[U]).ffill()
vl["months_since_last_uvl"] = (vl["date"] - last_uvl_date).dt.days / 30.44   # NaN = never unsuppressed
# consecutive suppressed tests ending at this test
blk = (vl["uvl"] == 1).astype(int).groupby(vl[U]).cumsum()
vl["supp_streak"] = (vl["uvl"] == 0).astype(int).groupby([vl[U], blk]).cumsum()
# time-weighted suppression relative to the current test (half-weight ~ 1 year)
def tw(gr):
    d = gr["date"].values.astype("datetime64[D]").astype(float)
    s = 1 - gr["uvl"].values
    out = np.empty(len(d))
    for i in range(len(d)):
        w = np.exp(-(d[i] - d[: i + 1]) / 365.0)
        out[i] = (w * s[: i + 1]).sum() / w.sum()
    return pd.Series(out, index=gr.index)
vl["tw_suppression_hist"] = vl.groupby(U, group_keys=False)[["date", "uvl"]].apply(tw)

# --------------------------------------------------------------------------------------
# Candidate (index, outcome) pairs
# --------------------------------------------------------------------------------------
dates = vl[[U, "date", "vl"]].rename(columns={"date": "out_date", "vl": "out_vl"})
idx = vl.rename(columns={"date": "t0"}).copy()
idx["target_date"] = idx["t0"] + pd.Timedelta(days=MIN_GAP)
idx = idx.sort_values("target_date")
dates = dates.sort_values("out_date")
pairs = pd.merge_asof(idx, dates, left_on="target_date", right_on="out_date", by=U,
                      direction="forward")
pairs["gap_days"] = (pairs["out_date"] - pairs["t0"]).dt.days
pairs = pairs[pairs["gap_days"].between(MIN_GAP, MAX_GAP)].copy()
pairs["y"] = (pairs["out_vl"] >= 200).astype(int)
log(f"Index tests with a next test 90–548 days later: {len(pairs):,}")

# baseline + eligibility
pairs = pairs.merge(base, on=U, how="inner")
pairs["age"] = (pairs["t0"] - pairs["dob"]).dt.days / 365.25
pairs["years_on_art"] = (pairs["t0"] - pairs["art_start"]).dt.days / 365.25
n0 = len(pairs)
pairs = pairs[pairs["art_start"].notna() & (pairs["t0"] >= pairs["art_start"])]
log(f"  excluded index test before ART start / missing ART start: {n0 - len(pairs):,}")
n0 = len(pairs)
pairs = pairs[pairs["age"].between(15, 100)]
log(f"  excluded age at index <15 or implausible: {n0 - len(pairs):,}")

# baseline items only if dated before t0
pairs.loc[~(pairs["cd4_date"] <= pairs["t0"]), "cd4_enrol"] = np.nan
pairs["tpt_started"] = (pairs["tpt_start"] <= pairs["t0"]).astype(int)

# --------------------------------------------------------------------------------------
# Follow-up (CRF2) features as of t0: records with visit date <= t0
# --------------------------------------------------------------------------------------
f = d2.copy()
f["anchor"] = dt(f["cbs_Client visit date"]).fillna(dt(f["Report date"]))
f = f[f["anchor"].notna() & (f["anchor"] >= HIST_START)].sort_values([U, "anchor"])
gf = f.groupby(U)
f["stable"] = pd.to_numeric(f["cbs_stable"], errors="coerce")
f["ec"] = pd.to_numeric(f["cbs_Attended enhanced counselling"], errors="coerce").fillna(0)
f["ever_enhanced_counselling"] = gf["ec"].cummax()
oc = f["cbs_Client outcome"].astype(str).str.lower()
f["ever_ltfu"] = (oc == "ltfu").astype(int).groupby(f[U]).cummax()
f["ever_transfer_out"] = oc.str.startswith("transfer").astype(int).groupby(f[U]).cummax()
rc = pd.to_numeric(f["cbs_Change in regimen"], errors="coerce").fillna(0)
f["n_regimen_changes"] = rc.groupby(f[U]).cumsum()
reg = f["cbs_Current ART Regimen"].astype(str).str.lower()
f["regimen_class"] = np.select(
    [reg.str.contains("dtg|dolutegravir"), reg.str.contains("atazanavir|lopinavir|lpv|kaletra|darunavir|atv"),
     reg.str.contains("efv|efavirenz|nvp|nevirapine")],
    ["DTG-based", "PI-based", "NNRTI-based"], default="other/unknown")
f["who_stage"] = f["cbs_WHO Stage"].astype(str).str.lower().map({"i": 1, "ii": 2, "iii": 3, "iv": 4})
f["who_stage_max"] = f.groupby(U)["who_stage"].cummax()
f["adherence"] = f["cbs_Client adherence"].str.lower()
f["tpt_completed"] = (f["cbs_Client TPT outcome"].astype(str).str.lower() == "completed tpt") \
    .astype(int).groupby(f[U]).cummax()
f["pickup"] = dt(f["cbs_Last pharmacy pick up date"])
f["n_visits"] = gf.cumcount() + 1
f["facility"] = f["Organisation unit name"]
# ffill "latest known" values within patient so the as-of lookup returns the last non-missing value
for c in ["stable", "who_stage", "who_stage_max", "adherence", "pickup"]:
    f[c] = f.groupby(U)[c].ffill()
fcols = ["stable", "ever_enhanced_counselling", "ever_ltfu", "ever_transfer_out", "n_regimen_changes",
         "regimen_class", "who_stage", "who_stage_max", "adherence", "tpt_completed", "pickup",
         "n_visits", "facility"]
asof = pd.merge_asof(pairs.sort_values("t0"), f[[U, "anchor"] + fcols].sort_values("anchor"),
                     left_on="t0", right_on="anchor", by=U, direction="backward")
asof["has_followup_record"] = asof["anchor"].notna().astype(int)

# Records after a recorded terminal event (death / stopped ART dated on or before t0) are data-integrity
# problems -> excluded. Later deaths are NOT used for exclusion (that would select on the future).
term = f.loc[oc.isin(["death", "stopped"])].groupby(U)["anchor"].min().rename("terminal_date")
asof = asof.merge(term, on=U, how="left")
n0 = len(asof)
asof = asof[~(asof["terminal_date"] <= asof["t0"])].drop(columns="terminal_date")
log(f"  excluded index tests on/after a recorded death or ART stop: {n0 - len(asof):,}")
asof["days_since_pickup"] = (asof["t0"] - asof["pickup"]).dt.days.where(lambda x: x.between(0, 730))

# visits in the 12 months before t0 (n_visits as of t0 minus as of t0-365)
prior = pd.merge_asof(asof[[U, "t0"]].assign(t_1y=asof["t0"] - pd.Timedelta(days=365)).sort_values("t_1y"),
                      f[[U, "anchor", "n_visits"]].sort_values("anchor"),
                      left_on="t_1y", right_on="anchor", by=U, direction="backward")
asof = asof.merge(prior[[U, "t0", "n_visits"]].rename(columns={"n_visits": "nv_1y"}), on=[U, "t0"], how="left")
asof["visits_last_12m"] = asof["n_visits"].fillna(0) - asof["nv_1y"].fillna(0)

# Province from the facility at t0 (fallback: enrolment facility)
fac = pd.read_excel(FACILITY)
fac["Facility_name"] = fac["Facility_name"].astype(str).str.lower().str.strip()
fac = fac.drop_duplicates("Facility_name").set_index("Facility_name")["Province"].str.lower()
asof["province"] = asof["facility"].fillna(asof["facility_enrol"]).map(fac)

# --------------------------------------------------------------------------------------
# Final per-period selection: latest eligible index test per patient
# --------------------------------------------------------------------------------------
asof["index_vl_log10"] = np.log10(asof["vl"])
asof["index_vl_cat"] = pd.cut(asof["vl"], [0, 50, 200, 1000, np.inf], right=False,
                              labels=["<50", "50-199", "200-999", ">=1000"]).astype(str)
asof["prev_vl_cat"] = pd.cut(asof["prev_vl"], [0, 50, 200, 1000, np.inf], right=False,
                             labels=["<50", "50-199", "200-999", ">=1000"]).astype(str).replace("nan", "no prior test")
asof["index_suppressed"] = (asof["vl"] < 200).astype(int)
asof["ever_uvl_hist"] = (asof["n_uvl_hist"] > 0).astype(int)

def pick(lo, hi, name):
    s = asof[(asof["out_date"] >= lo) & (asof["out_date"] <= hi)]
    s = s.sort_values("t0").drop_duplicates(U, keep="last").copy()
    s["set"] = name
    log(f"\n{name}: outcome test {lo.date()}–{hi.date()}: {len(s):,} patients; "
        f"uVL {s['y'].sum():,} ({s['y'].mean():.2%})")
    log(f"  index suppressed (rebound risk): {s['index_suppressed'].sum():,}, "
        f"uVL {s.loc[s.index_suppressed == 1, 'y'].mean():.2%} | index unsuppressed (persistence): "
        f"{(1 - s['index_suppressed']).sum():,}, uVL {s.loc[s.index_suppressed == 0, 'y'].mean():.2%}")
    log(f"  median gap index->outcome: {s['gap_days'].median() / 30.44:.1f} months "
        f"(IQR {s['gap_days'].quantile(.25) / 30.44:.1f}–{s['gap_days'].quantile(.75) / 30.44:.1f})")
    return s

tr = pick(*TRAIN, "train")
te = pick(*TEST, "test")
te["in_train"] = te[U].isin(tr[U]).astype(int)
log(f"  test patients also contributing a (earlier) training record: {te['in_train'].sum():,} "
    f"({te['in_train'].mean():.1%}); new patients in test: {(1 - te['in_train']).sum():,}")
cohort = pd.concat([tr, te], ignore_index=True)

FEATURES_NUM = ["age", "female", "years_on_art", "days_dx_to_art", "employed", "who_stage_enrol",
                "initial_regimen_dtg", "cd4_enrol", "tpt_started",
                "index_vl_log10", "n_tests_hist", "n_uvl_hist", "any_llv_hist", "prop_suppressed_hist",
                "months_since_prev_test", "vl_switches_hist", "months_since_last_uvl", "supp_streak",
                "tw_suppression_hist",
                "has_followup_record", "stable", "ever_enhanced_counselling", "ever_ltfu",
                "ever_transfer_out", "n_regimen_changes", "who_stage", "who_stage_max", "tpt_completed",
                "days_since_pickup", "visits_last_12m"]
FEATURES_CAT = ["marital", "regimen_class", "adherence", "province", "index_vl_cat", "prev_vl_cat"]
keep = [U, "set", "in_train", "t0", "out_date", "gap_days", "vl", "out_vl", "y", "index_suppressed",
        "ever_uvl_hist"] + FEATURES_NUM + FEATURES_CAT
cohort = cohort[keep]
cohort.to_pickle(OUT / "cohort.pkl")
pd.Series({"num": FEATURES_NUM, "cat": FEATURES_CAT}).to_json(OUT / "features.json")

log("\nMissingness of predictors (%), train / test:")
miss = pd.DataFrame({"train": tr[FEATURES_NUM + FEATURES_CAT].isna().mean() * 100,
                     "test": te[FEATURES_NUM + FEATURES_CAT].isna().mean() * 100}).round(1)
log(miss.to_string())
(OUT / "cohort_flow.txt").write_text("\n".join(flow))
print(f"\nSaved cohort ({len(cohort):,} rows) to {OUT / 'cohort.pkl'}")
