# Construction Cycle Time Prediction

**Predicting residential construction duration with machine learning and time-aware evaluation**

How much can a machine learning model improve construction-duration estimates compared with the traditional practice of using the average duration for each floor plan?

This project analyzes historical residential construction records, develops regression models for construction cycle time, and evaluates Gradient Boosting using chronological training and validation. Particular attention is paid to when information becomes available: a model intended for use at construction start must not learn from outcomes that were unknown at the prediction date.

## Key results

- 20.2% lower MAE in 202 and 21.6% lower MAE in 2024 compared with a floor-plan-average baseline, using nested time-based validation.
- **Nested Gradient Boosting** achieved 28.26 days MAE (2023) and 29.21 days MAE (2024).
- The nested procedure independently selected hyperparameters from earlier historical periods for each evaluation year, rather than choosing them using that year's outcomes.
- Performance was much weaker in the early 2022 evaluation, when only 426 completed projects were available for training. This highlights the importance of training-data availability and period-specific evaluation.

## Business problem

Construction cycle time is the number of days between the start and end of construction:

*Cycle time (days) = construction end date − construction start date*

Cycle-time estimates inform trade scheduling, expected completion dates, customer commitments, and project planning. The original operating benchmark estimated duration using the historical mean for each floor plan. This project investigates whether a model that combines building specifications, location, sales information, and permitting history can produce more accurate estimates.

The prediction scenario assumes an estimate is needed **on the day construction starts**. This is an essential constraint on feature selection and validation.

## Dataset and confidentiality

The original construction-management-system export contains 4,684 records, including completed and in-progress builds. After the existing cleaning and eligibility rules, 3,762 completed-build records were used for the historical modeling analysis.

The original dataset contains confidential residential construction records and is not publicly distributed. All source CSV files are maintained locally and excluded from version control.

To protect individual identities, site-manager names are replaced with anonymous identifiers (e.g., Manager_001, Manager_002) before modeling and analysis. The mapping between actual names and identifiers is not published.

The public analysis focuses on model performance, aggregated statistics, and anonymized operational comparisons rather than individual construction records.

Because the original dataset is unavailable publicly, reproducing the reported results requires authorized access to the source data.

## Data preparation and feature engineering

The notebook investigates missing values, inconsistent entries, date validity, and the distribution of construction cycle times. The modeling target is calculated only for completed projects with valid construction dates.

Examples of model inputs include:

- *Home specifications:* floor plan, product line, size, bedrooms, bathrooms, stories, garage, basement, and lot characteristics.
- *Location and operations:* community, metropolitan area, permitting authority, and site manager.
- *Sales and permitting:* spec-home indicator, selected-options value, permit-processing and permit-to-start durations, and sale-to-start duration.
- *Construction-start timing:* start year and month.

A `sale_date_missing` indicator was added to retain information carried by missing sale dates. Among the 3,762 modeled records, 1,266 have no sale date: 862 spec homes and 404 non-spec homes.

To limit target leakage, fields that become known after construction starts—such as the construction end date, final inspection date, and total trade invoices—are excluded from prediction features. Recorded dates were also checked for sales and permits occurring after construction start. As with any retrospective operational export, features that might be updated later (such as selected-options value or site-manager assignment) would require point-in-time verification before deployment.

Preprocessing, including missing-value handling and categorical encoding, is fitted using training data within each evaluation split.

## Modeling approach

The existing analysis compares simple prediction baselines and regression models, including **Ridge Regression**, **Random Forest**, and **Gradient Boosting**. The principal business benchmark is the **floor-plan mean**: predict each project's duration from completed training projects sharing its floor plan, falling back to the training-set global mean for unseen plans.

Model quality is evaluated using:

- **MAE (days):** average absolute difference between predicted and actual construction duration; the primary selection metric.
- **RMSE (days):** gives more weight to large prediction errors and helps reveal the effect of unusually inaccurate forecasts.

The chronological evaluations account for *outcome availability*: an observation can enter a training set only after the project has completed. The later evaluation periods are scored retrospectively once their actual durations are known.

## 5.5 — Gradient Boosting hyperparameter tuning

**Purpose:** Find a promising Gradient Boosting configuration by minimizing MAE on a designated validation period.

The model is fitted on 2,306 projects completed before January 2024 and assessed on 923 projects starting in the 2024 validation window. Six candidate combinations of `n_estimators`, `learning_rate`, and `max_depth` are evaluated with the same training and validation sets.

| Trees (`n_estimators`) | Learning rate | Max depth | Validation MAE (days) | Validation RMSE (days) |
|---:|---:|---:|---:|---:|
| 100 | 0.10 | 3 | 29.75 | 36.36 |
| 200 | 0.05 | 3 | 29.21 | 35.74 |
| 300 | 0.03 | 3 | 29.44 | 35.98 |
| 150 | 0.05 | 2 | 32.60 | 39.48 |
| 200 | 0.05 | 2 | 31.76 | 38.70 |
| **200** | **0.05** | **4** | **28.20** | **34.75** |

**Outcome:** The lowest validation MAE was **28.20 days**, obtained with 200 boosting stages, learning rate 0.05, and maximum tree depth 4.

**Interpretation:** This is a *hyperparameter-selection result*, not an independent test estimate. The 2024 validation outcomes were used to choose the winning configuration, so 28.20 days should not be presented as an unbiased estimate of future performance.

## 5.6 — Expanding-window validation of fixed configurations

**Purpose:** Assess whether two **fixed** Gradient Boosting configurations behave similarly across historical evaluation periods as the available training set grows. This is a stability diagnostic, not hyperparameter tuning.

The training window expands from projects completed before 2022, to those completed before 2023, and finally to those completed before 2024. Models are refitted from scratch for each period and evaluated on projects starting in the next validation period.

The two configurations are:

- *Original:* 300 estimators, learning rate 0.10, maximum depth 3.
- *2024-tuned:* 200 estimators, learning rate 0.05, maximum depth 4 (selected in Section 5.5).

| Evaluation year | Training rows | Validation rows | Original MAE | Original RMSE | 2024-tuned MAE | 2024-tuned RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 426 | 978 | 76.58 | 92.44 | 76.99 | 92.89 |
| 2023 | 1,176 | 1,039 | 28.72 | 37.08 | 28.38 | 37.15 |
| 2024 | 2,306 | 923 | 29.39 | 36.34 | 28.20 | 34.75 |

**Outcome:** Both configurations have substantially higher errors in 2022 than in 2023–2024. Across the latter two periods, their MAEs are relatively stable at roughly 28–29 days. The 2024-tuned configuration performs better by MAE in 2023 and 2024 but not in 2022.

**Important limitation:** The second configuration was chosen using 2024 validation outcomes. Applying it retrospectively to 2022 and 2023 is useful for comparison, but it does **not** simulate hyperparameter choices that could genuinely have been made in those earlier years. This motivates the nested evaluation in Section 5.7.

## 5.7 — Nested expanding-window validation

**Purpose:** Evaluate the *entire selection-and-training procedure* without using the outer evaluation year's outcomes to select hyperparameters.

For each outer evaluation year, the procedure:

1. Uses only earlier completed projects to fit candidate Gradient Boosting models.
2. Evaluates the same **six candidate configurations** from Section 5.5 on an **inner historical validation period**. Only projects completed before the outer-year cutoff can provide labels for this tuning step.
3. Selects the configuration with the lowest inner-validation MAE.
4. Refits that configuration on **all eligible projects completed before the outer evaluation year**.
5. Evaluates the refitted model on projects starting during that outer period and compares it with the floor-plan baseline trained on the same available historical data.

The 2022 outer fold is omitted because there is limited earlier information for a meaningful inner tuning period. The two evaluated outer periods are **2023 and 2024**.

### Model selection and outer-fold performance

| Outer year | Inner train | Inner validation | Outer train | Outer validation | Selected trees | Learning rate | Max depth | Inner MAE | Outer MAE | Outer RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 426 | 357 | 1,176 | 1,039 | 100 | 0.10 | 3 | 41.12 | **28.26** | **36.85** |
| 2024 | 1,176 | 514 | 2,306 | 923 | 200 | 0.05 | 3 | 24.66 | **29.21** | **35.74** |

The selected hyperparameters differ by period because each fold uses the information available at that point in time. **Inner MAE** is used to choose parameters; **outer MAE and RMSE** are the backtesting results for the selected models. The inner validation set includes only projects that had finished before the outer cutoff, which can favor shorter-duration projects; the inner and outer errors should not be interpreted as directly comparable.

### Comparison with the traditional floor-plan baseline

| Evaluation year | Floor-plan MAE | Nested Gradient Boosting MAE | **MAE reduction** | Floor-plan RMSE | Nested Gradient Boosting RMSE |
|---|---:|---:|---:|---:|---:|
| **2023** | 35.42 | **28.26** | **20.2%** | 47.28 | **36.85** |
| **2024** | 37.27 | **29.21** | **21.6%** | 45.23 | **35.74** |

**Main finding:** Nested Gradient Boosting reduced average absolute prediction error relative to the company's floor-plan-average approach in both evaluation periods. The reductions were **7.16 days (20.2%)** in 2023 and **8.06 days (21.6%)** in 2024. RMSE also decreased in both periods.

These results suggest that project-level characteristics add useful predictive information beyond floor-plan averages. They do not establish that the model will maintain the same improvements in future construction environments.

## Interpretation and operational implications

The notebook also investigates feature importance and differences across site managers. These analyses can help identify *associations* between project characteristics and observed duration, and inform questions about planning or operational variation. They should not be treated as causal evidence that a particular manager or feature changes construction time. Differences in project mix, location, construction conditions, and unmeasured factors may explain apparent relationships.

## Repository and reproducibility

-[`construction_cycle_time_prediction.ipynb`](construction_cycle_time_prediction.ipynb) — exploratory analysis, feature engineering, modeling, hyperparameter tuning, historical validation, and interpretation.

The notebook expects the confidential `build_history.csv` to be available **locally** in its working directory. The CSV is intentionally excluded from Git using `.gitignore` and should not be committed or redistributed.
