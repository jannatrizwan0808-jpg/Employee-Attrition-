<h1 align="center">Employee Attrition Prediction</h1>
<p align="center"><b>Why do employees leave, who is likely to go, and what can a company do about it?</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-pipeline-orange" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Task-binary%20classification-informational" alt="Task">
  <img src="https://img.shields.io/badge/Best%20ROC--AUC-0.851-success" alt="ROC-AUC">
  <img src="https://img.shields.io/badge/Rows-74%2C498-lightgrey" alt="Rows">
</p>

---

## Contents
1. [Aim](#-aim)
2. [Why it matters](#-why-it-matters)
3. [Dataset](#-dataset)
4. [Why employees leave and why they stay](#-why-employees-leave-and-why-they-stay)
5. [Approach](#-approach)
6. [Model results](#-model-results)
7. [What a company can do](#-what-a-company-can-do)
8. [Project files and how to run](#-project-files-and-how-to-run)
9. [Limitations](#-limitations)

---

## 🎯 Aim

1. **Understand** what separates employees who **left** from those who **stayed**, using a full exploratory data analysis.
2. **Predict** whether an employee will leave (`Attrition` = `Left`) with a tested machine learning pipeline.
3. **Recommend** where a company should look first to keep its people.

## 💼 Why it matters

Replacing an employee costs time, money and know-how. If a model can flag the groups most likely to leave, HR can act **before** the resignation instead of after it. Here, **47.5%** of the employees in the data left, so the problem is large, and the classes are balanced enough to model directly.

## 📊 Dataset

| | |
|---|---|
| Rows | 74,498 (59,598 train + 14,900 test, stratified 80/20 split) |
| Columns | 24 (23 features and target, plus Employee ID) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target | `Attrition`: **Left** (47.5%) or **Stayed** (52.5%) |

| Feature type | Count | Columns |
|---|---|---|
| Numerical | 7 | Age, Years at Company, Monthly Income, Number of Promotions, Distance from Home, Number of Dependents, Company Tenure |
| Ordinal | 8 | Work-Life Balance, Job Satisfaction, Performance Rating, Education Level, Job Level, Company Size, Company Reputation, Employee Recognition |
| Binary | 5 | Gender, Overtime, Remote Work, Leadership Opportunities, Innovation Opportunities |
| Nominal | 2 | Job Role, Marital Status |

## 🔍 Why employees leave and why they stay

The charts and tables show the share of employees in each group who **left** (overall: 47.5%).

![Attrition by factor](images/attrition_by_factor.png)

### Reasons associated with leaving

| Factor | Group | Left | Compared with |
|---|---|---|---|
| Job Level | Entry level | **63%** | Senior: 20% |
| Marital Status | Single | **67%** | Married: 36% |
| Remote Work | No remote work | **53%** | Remote: 25% |
| Work-Life Balance | Poor or Fair | **60% / 58%** | Excellent: 36% |
| Promotions | 0 to 2 | **49%** | 3 to 4: 25% |
| Distance from Home | Over 50 | **53%** | Up to 50: 42% |
| Company Reputation | Poor | **56%** | Good: 43% |

### Reasons associated with staying

| Factor | Group | Left |
|---|---|---|
| Seniority | Senior job level | only **20%** |
| Flexibility | Remote work | only **25%** |
| Career growth | 3 or 4 promotions | only **25%** |
| Education | PhD | only **24%** (other degrees all about 49%) |
| Life situation | Married | **36%** |
| Wellbeing | Excellent work-life balance | **36%** |
| Dependents | 4 or more dependents | about **35%** (up to 3 dependents: about 50%) |

### What does not separate leavers from stayers
- **Monthly Income** (left: 7,275 vs stayed: 7,321, almost the same), **Job Role**, **Company Size**, **Employee Recognition** and **Leadership Opportunities** show little or no effect.
- Surprising result: employees with **Very High job satisfaction** left slightly more often (53%) than those with High or Medium (45%). The data does not explain why.
- Women left more often than men in this data (53% vs 43%). This should **not** be used for decisions about individual people; it is a reason to check for unfair treatment, not a rule.

### Statistical evidence
- **Chi-square and Cramér V** (strength of link with Attrition): Job Level 0.32, Marital Status 0.28, Remote Work 0.22, Work-Life Balance 0.20.
- **Left vs Stayed numeric means**: leavers live farther away (52.8 vs 47.4) and have fewer promotions, dependents and years at the company. The effect of each is small (Cohen d below 0.2), so no single number separates the two groups.
- **Multicollinearity**: none (every VIF is below 1.65).

> These are **associations, not proven causes.** For example, entry-level staff may leave for reasons that are not in the data.

## 🧪 Approach

```mermaid
flowchart LR
    A[merged.csv<br/>80/20 stratified split] --> B[EDA]
    A --> C[Preprocessing pipeline]
    C --> D[Baseline]
    C --> E[5-fold cross-validation<br/>9 models]
    E --> F[Hyperparameter tuning]
    F --> G[Final evaluation<br/>on untouched test set]
    B --> H[Insights and recommendations]
    G --> H
```

| Step | What was done |
|---|---|
| 1. EDA | Information, duplicates, outliers (IQR, z-score), variance (plus Levene, Bartlett, Fligner-Killeen), skewness, kurtosis, Shapiro, D'Agostino, Q-Q plots, Yeo-Johnson, Pearson and Spearman heatmaps, correlation network, VIF, chi-square, Cramér V, Welch t-test, Mann-Whitney, ANOVA, Kruskal-Wallis |
| 2. Train and test | `merged.csv` split 80/20 with `train_test_split` (stratified, `random_state=42`). The test set is kept untouched until the very end |
| 3. Preprocessing | One `ColumnTransformer` in the pipeline: scaling, ordered ordinal encoding, 0/1 binary, one-hot for nominal. No data leaks between cross-validation folds |
| 4. Baseline | Always predict the most common class |
| 5. Cross-validation | 5-fold stratified, 9 models: Logistic Regression, Decision Tree, KNN, Random Forest, Extra Trees, Gradient Boosting, **XGBoost, LightGBM, CatBoost** |
| 6. Tuning | `RandomizedSearchCV` (3-fold, ROC-AUC) for every model |
| 7. Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves, permutation importance. The best model is chosen by cross-validation score, not by test score |

## 🏆 Model results

![Model comparison](images/model_comparison.png)

| Model | Tuned CV ROC-AUC | Test accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (most frequent) | 0.5000 | 0.5252 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8288 | 0.7379 | 0.7160 | 0.7422 | 0.7289 | 0.8287 |
| Decision Tree | 0.8215 | 0.7335 | 0.7134 | 0.7332 | 0.7232 | 0.8230 |
| KNN | 0.8177 | 0.7374 | 0.7148 | 0.7436 | 0.7289 | 0.8225 |
| Random Forest | 0.8410 | 0.7479 | 0.7383 | 0.7263 | 0.7323 | 0.8422 |
| Extra Trees | 0.8354 | 0.7477 | 0.7317 | 0.7398 | 0.7357 | 0.8377 |
| Gradient Boosting | 0.8507 | 0.7544 | 0.7384 | 0.7475 | 0.7430 | 0.8500 |
| XGBoost | 0.8503 | 0.7536 | 0.7385 | 0.7448 | 0.7416 | 0.8492 |
| LightGBM | 0.8492 | 0.7532 | 0.7383 | 0.7440 | 0.7411 | 0.8486 |
| **CatBoost** | **0.8523** | **0.7562** | **0.7406** | **0.7489** | **0.7447** | **0.8508** |

**Winner: CatBoost.** It has the highest tuned cross-validation ROC-AUC (0.852) and the highest test accuracy (75.6%) and test ROC-AUC (0.851). It correctly classifies about 3 in 4 employees and treats both classes evenly (precision and recall about 0.74 and 0.75 for Left, 0.77 and 0.76 for Stayed). Its train accuracy (0.761) is close to its test accuracy (0.756), so it is not just memorising the training data. KNN, by contrast, scores 1.00 on the training data and only 0.74 on the test data, a clear case of overfitting.

**How big is the win?** Small. The four boosting models (CatBoost, Gradient Boosting, XGBoost, LightGBM) are within 0.003 ROC-AUC of each other, a gap that is probably within noise, and all four beat the best non-boosting model (Random Forest, 0.842). The clear result is that boosting works better than the other model families on this data; which boosting library is used matters little.

**Most important features** (permutation importance): Job Level, Marital Status, Remote Work, Work-Life Balance. This agrees with the EDA.

## ✅ What a company can do

These are suggestions based on the patterns above, not guaranteed fixes.

1. **Focus on entry-level staff and new joiners.** They leave at three times the rate of seniors.
2. **Offer remote or flexible work.** Remote employees left at half the rate of on-site ones.
3. **Improve work-life balance.** Poor and Fair groups leave far more than Good and Excellent.
4. **Build a visible promotion path.** Employees with 3 or more promotions left at about half the rate.
5. **Support employees with long commutes** (over 50 distance units) with flexible hours or relocation help.
6. **Use the model as a screening tool**, not a verdict. Use it to decide where to start conversations, never to treat individuals differently.

## 📁 Project files and how to run

| File | Purpose |
|---|---|
| `employee_attrition_eda.ipynb` | Full exploratory data analysis |
| `employee_attrition_ml_pipeline.ipynb` | Full ML pipeline (preprocessing, baseline, cross-validation, 6 models, tuning, evaluation) |
| `employee_attrition_ml_pipeline_boosting.ipynb` | Same pipeline with 3 more boosting models (XGBoost, LightGBM, CatBoost), 9 models in total |
| `employee_attrition_models.ipynb` | Short version: 3 models, no tuning |
| `eda_report.html` | EDA as a single report with charts |
| `images/` | Charts used in this README |
| `train.csv`, `test.csv`, `merged.csv` | Data |

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn networkx xgboost lightgbm catboost notebook
jupyter notebook
```

Keep the CSV files in the same folder as the notebooks. The tuning cell takes roughly 15 to 30 minutes (more with the three boosting models); lower `N_ITER` to speed it up.

## ⚠️ Limitations

- Findings show **association, not cause**.
- All models reach only about 75% accuracy, so the available features may be close to their limit. A wider tuning search is unlikely to help much.
- Default and tuned cross-validation scores are not exactly comparable (5 folds vs 3 folds).
- Number of Promotions has only values 0 to 4, so its z-score "outliers" are not real outliers.
- Job Role and Marital Status have no natural order, so they are tested with chi-square and one-hot encoded, not shown in the correlation heatmap.
