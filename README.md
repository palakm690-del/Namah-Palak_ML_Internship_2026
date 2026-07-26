# Student Placement Prediction using Machine Learning

6-Week ML & Generative AI Internship (2026) — Research Project
Team: Namah Verma & Palak Mishra

## Problem Statement

Given a student's academic performance, skills, and experience, predict whether they'll be Placed or Not Placed during campus recruitment. The idea is that if a training & placement cell can flag at-risk students early, they can actually do something about it (extra mock interviews, skill workshops, etc.) instead of finding out after the fact.

This is a binary classification problem.

## Dataset

- Source: [Student Placement Dataset](https://www.kaggle.com/datasets/sonalshinde123/student-placement-dataset) by sonalshinde123 (Kaggle)
- 45,000 rows train / 5,000 rows test, pre-split by the dataset author (we did not re-split it)
- 14 features + target (Placement_Status): Age, Gender, Degree, Branch, CGPA, Internships, Projects, Coding_Skills, Communication_Skills, Aptitude_Test_Score,Soft_Skills_Rating, Certifications, Backlogs
- **Important limitation:** this dataset is explicitly described by its author as synthetic/generated. We're upfront about this, it's why several of our models hit near-perfect accuracy, and it's a real limitation we're naming in the report rather than hiding.

We also looked at the benroshan Campus Recruitment dataset early on but dropped it — only 215 rows, not enough to work with alongside the dataset above.

## Repo structure

```
├── data/                          # train/test csv, raw and cleaned
├── notebooks/
│   ├── 01_EDA.ipynb                # exploratory data analysis
│   └── 02_baseline_models.ipynb    # preprocessing, models, tuning, SHAP
├── src/                            # empty for now, see note below
├── app/                            # empty for now, see note below
├── report/                         # empty for now, written report goes here
├── final_model.pkl                 # tuned Random Forest, saved with joblib
├── shap_summary.png
├── shap_individual_correct.png
├── shap_individual_wrong.png
├── requirements.txt
└── README.md
```

src/ and app/ are placeholders; we're not building the Streamlit deployment yet. We will build on them if we are instructed to proceed with it.

## What's done so far

- **EDA** (01_EDA.ipynb): missing value check (none), class balance (~64% Not Placed / 36% Placed; mild imbalance, not severe), distribution plots for every numeric feature, bivariate plots against Placement_Status, and a correlation heatmap.
- **Preprocessing** (02_baseline_models.ipynb): checked train/test have no overlapping Student_IDs, encoded the target, label-encoded Gender, one-hot encoded Degree/Branch, checked for outliers with the IQR method (only CGPA has any, ~0.34% of rows, and they're valid values not data errors, so we left them in).
- **Feature scaling**: StandardScaler on the continuous numeric features, used only for Logistic Regression and SVM. Tree models don't need it.
- **Multicollinearity check**: Coding_Skills, Projects, Certifications, and CGPA are all correlated with each other. We checked VIF before and  after : Projects had a VIF of ~10.2 and Coding_Skills ~9.6 on the full feature set. Since Projects had the strongest raw correlation with the target (0.50) of the four, we dropped Coding_Skills and Certifications from the Logistic Regression feature set only (VIF for Projects dropped to ~2.3 after). Tree-based models keep all features; they aren't destabilised by correlated inputs the way a linear model is.
- **Baseline models**: Logistic Regression, Decision Tree, Random Forest.
- **5-fold cross-validation** on all three (accuracy/precision/recall/F1).
- **Hyperparameter tuning**: GridSearchCV on Decision Tree and Random Forest, tuned on recall (deliberately, catching at-risk students matters more to us than raw accuracy here).
- **Extended comparison**: added SVM and XGBoost for a broader comparison against the three models above.
- **Explainability**: SHAP (TreeExplainer) on the final Random Forest; global feature ranking, summary plot, and two individual predictions (one correct, one the model's single misclassification on the test set).
- Saved the final tuned Random Forest as final_model.pkl.

## Results

Test-set performance (baseline, before tuning):

| Model                   | Accuracy | Precision | Recall | F1     |
|-------------------------|----------|-----------|--------|--------|
| Logistic Regression     | 0.8564   | 0.8012    | 0.8030 | 0.8021 |
| Decision Tree           | 1.0000   | 1.0000    | 1.0000 | 1.0000 |
| Random Forest           | 0.9998   | 1.0000    | 0.9994 | 0.9997 |

GridSearchCV (tuned on recall, 5-fold) didn't move these numbers, it showed same scores before and after tuning for both tree models, and the Random Forest misclassifies the exact same single student either way. For comparison, XGBoost also lands at a perfect 1.0000, while SVM (RBF kernel) comes in lower at 0.9612 accuracy; its curved decision boundary can't match the sharp, rectangular threshold rules the data is actually built on as exactly as the tree-based models can.

Final model: the tuned Random Forest (best params: max_depth=None, max_features='sqrt', n_estimators=100).

### A note on the near-perfect scores

Getting 99–100% accuracy is normally a red flag, so we checked for overfitting before trusting it: train vs test accuracy for both tree models match almost exactly (no gap), a Decision Tree capped at max_depth=3 already gets 93.8% using four clean threshold rules, and the learning curve gap between train and cross-validation is about 0.0013 which is nowhere near what an overfit model on noisy real-world data would show. 
Our read is that this particular dataset was generated from sharp, rule-based logic (e.g. "if Projects > 3.5 and Backlogs = 0 → Placed"), which axis-aligned models like Decision Trees, Random Forest, and XGBoost can fit almost exactly, and which a curved-boundary model like SVM can't. 
The full investigation is in 02_baseline_models.ipynb under "Investigating the near perfect scores."

### SHAP findings

Global feature ranking (by mean |SHAP value|):
Communication_Skills > Backlogs > CGPA > Coding_Skills > Projects > Aptitude_Test_Score > Certifications > Internships.

A couple of things worth flagging: 
Internships mattered less than we expected going in, and the model gives close to zero weight to Age, Gender, Degree, and Branch — placement predictions aren't leaning on demographic attributes.

## What's not done yet

- Streamlit app (app/, src/) — on hold until instructed to start it.
- Written report (report/) and the research paper.

## Reproducing this

We built this in Google Colab, uploading train.csv/test.csv directly into the Colab session rather than reading them from a relative path — so both notebooks read them as train.csv/test.csv, not ../data/train.csv. If you're running this locally, copy the CSVs from data/ into the same folder as the notebook first (or you'll get a FileNotFoundError).

```bash
pip install -r requirements.txt
```

Python 3.9+ should work fine. shap needs to be installed separately if you're on Colab ("!pip install shap --quiet") since it isn't preinstalled there.
