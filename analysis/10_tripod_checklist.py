"""S2 Checklist: TRIPOD+AI (Collins et al., BMJ 2024;385:e078378; checklist version 11-January-2024) with locations."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.section import WD_ORIENT
SI = Path(__file__).resolve().parent.parent / "06_supporting_information"
I = [
("Title","1","D;E","Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted","Title"),
("Abstract","2","D;E","See TRIPOD+AI for Abstracts checklist","Abstract"),
("Background","3a","D;E","Explain the healthcare context (including whether diagnostic or prognostic) and rationale for developing or evaluating the prediction model, including references to existing models","Introduction, paragraphs 4–10 (prognostic; existing models [23,24])"),
("","3b","D;E","Describe the target population and the intended purpose of the prediction model in the context of the care pathway, including its intended users (e.g., healthcare professionals, patients, public)","Introduction, last two paragraphs; Methods, Study design; Discussion, Implications"),
("","3c","D;E","Describe any known health inequalities between sociodemographic groups","Introduction, paragraph 2 (children and young adults); Discussion, Predictors and associated factors"),
("Objectives","4","D;E","Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both)","Introduction, last paragraph (development and temporal validation)"),
("Data","5a","D;E","Describe the sources of data separately for the development and evaluation datasets (e.g., randomised trial, cohort, routine care or registry data), the rationale for using these data, and representativeness of the data","Methods, Data source and ethical considerations; Cohort selection and temporal validation (both sets from national CBS; ~94% coverage)"),
("","5b","D;E","Specify the dates of the collected participant data, including start and end of participant accrual; and, if applicable, end of follow-up","Methods, Data source; Cohort selection and temporal validation (training outcome tests 2020–2023; validation 2024–July 2025; VL history from 2010)"),
("Participants","6a","D;E","Specify key elements of the study setting (e.g., primary care, secondary care, general population) including the number and location of centres","Methods, Data source (all health facilities nationwide); Table 1 (province)"),
("","6b","D;E","Describe the eligibility criteria for study participants","Methods, Cohort selection and temporal validation; Fig 1"),
("","6c","D;E","Give details of any treatments received, and how they were handled during model development or evaluation, if relevant","Methods, Predictors (initial and current regimen class, regimen changes, TPT); Table 1"),
("Data preparation","7","D;E","Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups","Methods, Data source (de-duplication, conflicting values, implausible dates); Cohort selection (ART date rule); Model development (pipeline preprocessing); applied identically to all groups"),
("Outcome","8a","D;E","Clearly define the outcome that is being predicted and the time horizon, including how and when assessed, the rationale for choosing this outcome, and whether the method of outcome assessment is consistent across sociodemographic groups","Methods, Study design; Outcome variable (≥200 copies/mL at first test 90–548 days after t0; national laboratory VL, same for all groups)"),
("","8b","D;E","If outcome assessment requires subjective interpretation, describe the qualifications and demographic characteristics of the outcome assessors","Not applicable (laboratory VL result)"),
("","8c","D;E","Report any actions to blind assessment of the outcome to be predicted","Not applicable (objective laboratory measurement; routine data)"),
("Predictors","9a","D","Describe the choice of initial predictors (e.g., literature, previous models, all available predictors) and any pre-selection of predictors before model building","Methods, Predictor variables; S1 Table (all 107 form fields, use or reason for exclusion)"),
("","9b","D;E","Clearly define all predictors, including how and when they were measured (and any actions to blind assessment of predictors for the outcome and other predictors)","Methods, Predictor variables; S1 Table, sheet 'Predictors' (definition, source, timing relative to t0)"),
("","9c","D;E","If predictor measurement requires subjective interpretation, describe the qualifications and demographic characteristics of the predictor assessors","Routine clinical records by health facility staff (Methods, Data collection and management); adherence assessment and WHO stage are clinician-assessed"),
("Sample size","10","D;E","Explain how the study size was arrived at (separately for development and evaluation), and justify that the study size was sufficient to answer the research question. Include details of any sample size calculation","Methods, Cohort selection and temporal validation (all available national data; 2,334 and 2,263 events)"),
("Missing data","11","D;E","Describe how missing data were handled. Provide reasons for omitting any data","Methods, Predictor variables (indicator for no follow-up record) and Model development (imputation with indicators for LR; native handling for trees); Regression (complete case, <1% missing); S1 Table (missingness)"),
("Analytical methods","12a","D","Describe how the data were used (e.g., for development and evaluation of model performance) in the analysis, including whether the data were partitioned, considering any sample size requirements","Methods, Cohort selection and temporal validation (temporal split); Model development"),
("","12b","D","Depending on the type of model, describe how predictors were handled in the analyses (functional form, rescaling, transformation, or any standardisation).","Methods, Model development (standardisation and one-hot encoding within pipelines; log10 VL); Regression (categories)"),
("","12c","D","Specify the type of model, rationale, all model-building steps, including any hyperparameter tuning, and method for internal validation","Methods, Model development; Class imbalance handling; S3 Table"),
("","12d","D;E","Describe if and how any heterogeneity in estimates of model parameter values and model performance was handled and quantified across clusters (e.g., hospitals, countries)","Performance by province (Results, Subgroup and sensitivity analyses; S5 Table F); regression SEs clustered by patient"),
("","12e","D;E","Specify all measures and plots used (and their rationale) to evaluate model performance (e.g., discrimination, calibration, clinical utility) and, if relevant, to compare multiple models","Methods, Model performance evaluation; Benchmarks"),
("","12f","E","Describe any model updating (e.g., recalibration) arising from the model evaluation, either overall or for particular sociodemographic groups or settings","No updating after evaluation; Platt recalibration was fitted on training data before evaluation (Methods, Model development)"),
("","12g","E","For model evaluation, describe how the model predictions were calculated (e.g., formula, code, object, application programming interface)","Methods, Software, code, and reproducibility (code repository)"),
("Class imbalance","13","D;E","If class imbalance methods were used, state why and how this was done, and any subsequent methods to recalibrate the model or the model predictions","Methods, Class imbalance handling (no resampling; class weighting tuned; Platt recalibration); SMOTE-ENN sensitivity analysis (S5 Table E)"),
("Fairness","14","D;E","Describe any approaches that were used to address model fairness and their rationale","Methods, Model performance evaluation (performance by sex, age group and province); Results; S5 Table F"),
("Model output","15","D","Specify the output of the prediction model (e.g., probabilities, classification). Provide details and rationale for any classification and how the thresholds were identified","Methods, Model development (recalibrated probabilities) and Model performance evaluation (thresholds A and B fixed on training data)"),
("Training versus evaluation","16","D;E","Identify any differences between the development and evaluation data in healthcare setting, eligibility criteria, outcome, and predictors","Methods, Cohort selection (same eligibility, outcome, predictors; later period); Results, Study population and Table 1 (case-mix and prevalence differences)"),
("Ethical approval","17","D;E","Name the institutional research board or ethics committee that approved the study and describe the participant-informed consent or the ethics committee waiver of informed consent","Methods, Data source and ethical considerations (RNEC No. 907/RNEC/2019; verbal consent)"),
("Funding","18a","D;E","Give the source of funding and the role of the funders for the present study","Submission form (Funding statement)"),
("Conflicts of interest","18b","D;E","Declare any conflicts of interest and financial disclosures for all authors","Submission form (Competing interests)"),
("Protocol","18c","D;E","Indicate where the study protocol can be accessed or state that a protocol was not prepared","Methods, Data source and ethical considerations (no published protocol)"),
("Registration","18d","D;E","Provide registration information for the study, including register name and registration number, or state that the study was not registered","Methods, Data source and ethical considerations (not registered)"),
("Data sharing","18e","D;E","Provide details of the availability of the study data","Data Availability Statement; S1 Data (aggregate data)"),
("Code sharing","18f","D;E","Provide details of the availability of the analytical code","Methods, Software, code, and reproducibility"),
("Patient & Public Involvement","19","D;E","Provide details of any patient and public involvement during the design, conduct, reporting, interpretation, or dissemination of the study or state no involvement.","Methods, Data source and ethical considerations (no involvement)"),
("Participants","20a","D;E","Describe the flow of participants through the study, including the number of participants with and without the outcome and, if applicable, a summary of the follow-up time. A diagram may be helpful.","Results, Study population characteristics; Fig 1"),
("","20b","D;E","Report the characteristics overall and, where applicable, for each data source or setting, including the key dates, key predictors (including demographics), treatments received, sample size, number of outcome events, follow-up time, and amount of missing data. A table may be helpful. Report any differences across key demographic groups.","Table 1; Results, Study population characteristics; S1 Table (missingness)"),
("","20c","E","For model evaluation, show a comparison with the development data of the distribution of important predictors (demographics, predictors, and outcome).","Table 1 (training vs validation)"),
("Model development","21","D;E","Specify the number of participants and outcome events in each analysis (e.g., for model development, hyperparameter tuning, model evaluation)","Results; Fig 1; Tables 2–3; S3, S5, S6 Tables"),
("Model specification","22","D","Provide details of the full prediction model (e.g., formula, code, object, application programming interface) to allow predictions in new individuals and to enable third-party evaluation and implementation, including any restrictions to access or re-use (e.g., freely available, proprietary)","Code repository (model code and hyperparameters, S3 Table); fitted model object available on request through RBC because it is derived from restricted data"),
("Model performance","23a","D;E","Report model performance estimates with confidence intervals, including for any key subgroups (e.g., sociodemographic). Consider plots to aid presentation.","Table 2; Figs 2–4; S4, S5 Tables"),
("","23b","D;E","If examined, report results of any heterogeneity in model performance across clusters.","S5 Table F (province)"),
("Model updating","24","E","Report the results from any model updating, including the updated model and subsequent performance","Not applicable (no updating after evaluation)"),
("Interpretation","25","D;E","Give an overall interpretation of the main results, including issues of fairness in the context of the objectives and previous studies","Discussion, Principal findings; Model performance and clinical usefulness; Comparison with previous studies"),
("Limitations","26","D;E","Discuss any limitations of the study (such as a non-representative sample, sample size, overfitting, missing data) and their effects on any biases, statistical uncertainty, and generalizability","Discussion, Limitations"),
("Usability of the model in the context of current care","27a","D","Describe how poor quality or unavailable input data (e.g., predictor values) should be assessed and handled when implementing the prediction model","Discussion, Implications"),
("","27b","D","Specify whether users will be required to interact in the handling of the input data or use of the model, and what level of expertise is required of users","Discussion, Implications (automatic generation from CBS; no additional data entry)"),
("","27c","D;E","Discuss any next steps for future research, with a specific view to applicability and generalizability of the model","Discussion, Limitations and Implications (prospective impact evaluation, external validation, recalibration)"),
]
d = Document(); st = d.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(8)
s = d.sections[0]; s.orientation = WD_ORIENT.LANDSCAPE; s.page_width, s.page_height = Cm(29.7), Cm(21.0)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(s, m, Cm(1.5))
d.add_paragraph().add_run("S2 Checklist. TRIPOD+AI checklist").bold = True
d.add_paragraph("Checklist items reproduced from Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378 "
                "(checklist version 11 January 2024). D, development; E, evaluation. Locations refer to section headings of the revised manuscript.")
t = d.add_table(rows=1, cols=5); t.style = "Table Grid"
for i, h in enumerate(["Section/Topic", "Item", "D/E", "Checklist item", "Where reported"]):
    t.rows[0].cells[i].text = h; t.rows[0].cells[i].paragraphs[0].runs[0].bold = True
for row in I:
    cells = t.add_row().cells
    for i, v in enumerate(row): cells[i].text = v
for row in t.rows:
    for i, w in enumerate([Cm(3.2), Cm(1.0), Cm(1.1), Cm(11.5), Cm(9.5)]):
        row.cells[i].width = w
        for p in row.cells[i].paragraphs:
            for r in p.runs: r.font.size = Pt(8)
d.save(SI / "S2_Checklist.docx"); print("S2 items:", len(I))
