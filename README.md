# Machine learning prediction of unsuppressed HIV viral load in Rwanda

This repository contains cleaned analysis code for the manuscript:

**Using machine learning and longitudinal data to predict unsuppressed HIV viral load and rebound among people living with HIV on antiretroviral therapy in Rwanda, 2020–2025**

The code supports the analysis of national longitudinal HIV Case-Based Surveillance data from Rwanda to develop and evaluate machine learning models for predicting unsuppressed HIV viral load and viral rebound among people living with HIV on antiretroviral therapy.

## Repository contents

* `notebooks/analysis_clean.ipynb`: cleaned analysis notebook used for data preprocessing, feature engineering, class imbalance handling, model training, hyperparameter tuning, model evaluation, feature selection, and multivariable logistic regression.
* `requirements.txt`: Python package versions used in the analysis.
* `.gitignore`: file specifying data files and sensitive files that should not be uploaded to GitHub.

## Data availability

Patient-level Case-Based Surveillance data are not included in this repository because they contain sensitive national HIV surveillance and clinical information. Public sharing of these data would be inconsistent with ethical approvals, national data protection requirements, and confidentiality obligations for routine HIV surveillance data.

De-identified data may be made available to qualified researchers upon reasonable request, subject to approval by the Rwanda Biomedical Centre, the Rwanda National Ethics Committee, and any applicable institutional data access or data-sharing agreements.

## Software

Analyses were conducted using Python 3.10.15 with pandas 2.1.4, NumPy 1.26.4, scikit-learn 1.4.2, imbalanced-learn 0.14.1, Optuna 4.0.0, LightGBM 4.5.0, XGBoost 2.1.2, CatBoost 1.2.7, SciPy 1.11.4, and statsmodels 0.14.4.

## Notes

The notebook has been cleaned to remove patient-level data, confidential file paths, private outputs, and any information that could compromise the confidentiality of Rwanda’s national HIV Case-Based Surveillance data.
