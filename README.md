# Predicting unsuppressed HIV viral load at the next routine test in Rwanda

Analysis code for the manuscript:

**Predicting unsuppressed HIV viral load at the next routine test among people living with HIV on antiretroviral therapy in Rwanda: development and temporal validation of machine learning models using national surveillance data, 2020–2025** (PLOS Global Public Health, revised submission PGPH-D-26-02821).

The code reproduces every result in the revised manuscript, from the source data to the tables, figures and supporting information.

## Study design (summary)

* **Landmark design.** The prediction time point (t0) is the collection date of an index viral load (VL) test. Only information recorded on or before t0 is used as predictors.
* **Outcome.** Unsuppressed VL (≥200 copies/mL) at the first VL test 90–548 days after t0. Secondary outcome: >1000 copies/mL (WHO).
* **Temporal validation.** Training uses outcome tests from 2020 to 2023 (72,964 patients). Validation uses outcome tests from 2024 to July 2025 (99,611 patients). Each patient contributes one record per period. The validation set is used once.
* **Models.** Logistic regression, random forest, XGBoost, LightGBM and CatBoost, all using 36 prespecified predictors.
  * All preprocessing is inside cross-validation pipelines, and the main analysis uses no resampling.
  * Tuning, model choice, recalibration and thresholds use training data only.
* **Evaluation.**
  * Discrimination: AUROC and AUPRC with bootstrap CIs and paired comparisons.
  * Calibration and decision curve analysis.
  * Benchmarks: rules based on the current VL.
  * Subgroups, sensitivity analyses and fairness analyses.
* **Regression.** Logistic regression with prespecified covariates and patient-clustered standard errors.

## Scripts (`analysis/`, run in order)

| Script | Purpose | Main outputs |
|---|---|---|
| `01_profile_data.py` | Test intervals, transitions, and a leakage check of the original variables | `outputs/profile_summary.txt` |
| `02_build_cohort.py` | Cleaning, landmark index–outcome pairs, predictors as of t0, and the temporal split | `outputs/cohort.pkl`, `features.json`, `cohort_flow.txt` |
| `03_model.py` | Tuning (Optuna, 5-fold CV), out-of-fold model choice, recalibration, thresholds, the single validation, bootstrap CIs, decision curves, subgroups, SMOTE-ENN and WHO >1000 sensitivity analyses | `outputs/model_results.txt`, `test_metrics_*.csv`, `paired_*.csv` |
| `04_regression.py` | Prespecified multivariable logistic regression and GVIF | `outputs/regression_*.csv` |
| `05_tables.py` | Tables 1–3 | `outputs/table1-3.csv` |
| `06_interpret.py` | SHAP summary of the final model and decision-curve differences | `outputs/shap_importance.csv` |
| `07_figures.py` | Figs 1–5 (TIFF, 300 dpi) | `../05_figures/` |
| `08_supporting_info.py` | S1 and S3–S7 Tables, and S1 Data (aggregate only) | `../06_supporting_information/` |
| `09_fairness.py` | Performance by sex, age group and province | `outputs/fairness.csv` |
| `10_tripod_checklist.py` | S2 TRIPOD+AI checklist | `../06_supporting_information/` |

Full run time is about 1 hour, mostly hyperparameter tuning in `03_model.py`. Seeds are fixed at 42.

## Running the code

```bash
pip install -r requirements.txt          # Python 3.10
export CBS_DATA_DIR=/path/to/data        # folder containing CRF1.csv and CRF2.csv
export CBS_FACILITY_FILE=/path/to/cbs_Facility.xlsx   # facility -> province lookup
cd analysis
python 01_profile_data.py && python 02_build_cohort.py && python 03_model.py && python 04_regression.py \
  && python 05_tables.py && python 06_interpret.py && python 07_figures.py && python 09_fairness.py \
  && python 08_supporting_info.py && python 10_tripod_checklist.py
```

`03_model.py` reuses tuned hyperparameters from `outputs/best_params.json` if present. Add `--retune` to tune again, or `--quick` for a fast smoke test.

## Data availability

The data underlying the results are available to researchers who meet set criteria, upon request and approval by the Rwanda Biomedical Centre (RBC). Send data access requests to info@rbc.gov.rw.

Patient-level Case-Based Surveillance data are not included in this repository, and the scripts never write patient-level records to shareable files. All aggregate data underlying the published tables and figures are provided as S1 Data with the article.

## Original submission

The notebook submitted with the original manuscript is kept unchanged in `archive/original_submission/` for transparency. The revised analysis replaces it, following peer review. The main changes:

* an explicit prediction time point and outcome window;
* removal of predictors that used information recorded after the predicted test;
* one record per patient per period;
* temporal validation;
* no use of the test set for tuning or for model or feature selection;
* no evaluation on resampled data.
