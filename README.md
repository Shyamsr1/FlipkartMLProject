# Flipkart Customer Service Satisfaction (CSAT) — ML Project

Predict customer satisfaction from service interactions using a clean, memory-safe scikit-learn pipeline.  
The project includes data wrangling, EDA, hypothesis testing, model training (LogReg, RandomForest, HistGradientBoosting with TruncatedSVD), evaluation, and artifact saving for inference.

---

##  Highlights
- End-to-end **scikit-learn Pipeline** with `ColumnTransformer`
- **Memory-safe**: sparse `OneHotEncoder` → **TruncatedSVD** → HGB
- **F1-driven model selection** with a simple best-model chooser
- Reproducible artifacts: `artifacts/csat_model.joblib` + metadata JSON
- Notebook first (Jupyter), but components are modular and reusable

---

##  Project Structure (suggested)
.
├─ Customer_support_data.csv
├─ FlipKartFinalMLProject1ShyamSR.ipynb 
├─ FlipKartMLProject.mp4
├─ artifacts/
│ ├─ csat_model.joblib
│ └─ csat_model.meta.json
├─ flipkart_csat_outputs/
│ └─ figs/ # plots saved from EDA
├─ README.md
└─ requirements.txt


---

---

## 🎥 Project Demo
The **video file (`FlipkartMLProject.mp4`) is included inside the folder of this repository**.  
It contains a complete walkthrough of the model pipeline, visualizations, and key insights.

If the video cannot play directly on GitHub, download it locally and open it with any media player.

---


##  Requirements

See [`requirements.txt`](./requirements.txt).  
Create and activate an environment (conda or venv), 
then install:

```bash
pip install -r requirements.txt

##  Quickstart

Use Customer_support_data.csv  (or update CSV_PATH in the notebook).

Open the notebook

Use FlipKartFinalMLProject1ShyamSR.ipynb

Run sections in order

Data Understanding → Data Wrangling → EDA → Hypothesis Tests → Preprocessing → Modeling → Evaluation → Save Artifacts

Artifacts

After training, the final pipeline is saved to:

artifacts/csat_model.joblib

artifacts/csat_model.meta.json

##  Modeling Summary

Preprocessing

- Numeric: SimpleImputer(median) + StandardScaler

- with_mean=False for sparse pipelines (LogReg/RF)

- with_mean=True for dense pipelines (HGB)

- Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")

- Date parsing + engineered parts (day/dow/month)

- Response time feature: response_minutes

Models

- Logistic Regression (saga)

- RandomForestClassifier

- HistGradientBoostingClassifier (final), preceded by:

- OneHotEncoder (sparse) → TruncatedSVD(n_components=256) → HGB

- Why SVD? => Prevents RAM blow-ups from huge one-hot matrices by compacting to a small dense feature space.


## Example: Best-model selection & comparison

res_df = pd.DataFrame([
    ["Logistic Regression", acc_log, f1_log],
    ["Random Forest", acc_rf, f1_rf],
    ["HistGradientBoosting", acc_hgb, f1_hgb]
], columns=["Model","Accuracy","F1"]).sort_values("F1", ascending=False)
print(res_df.to_string(index=False))

# Choose best by F1
scores = {"Logistic Regression": f1_log, "Random Forest": f1_rf, "HistGradientBoosting": f1_hgb}
best_name = max(scores, key=scores.get)
best_pipe = {"Logistic Regression": pipe_log, "Random Forest": pipe_rf, "HistGradientBoosting": pipe_hgb}[best_name]
print("Best model:", best_name)


## Saving & Loading the Model 

# Save
import joblib, os
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_pipe, "artifacts/csat_model.joblib")

# Load
model = joblib.load("artifacts/csat_model.joblib")

# Predict on new data (schema-safe):
import numpy as np
import pandas as pd

new_data = pd.DataFrame([{
  "Channel": "Email",
  "Response_Time": 2.4,
  "Issue_Type": "Payment",
  "Agent_Experience": "Senior"
}])

# Align columns to training schema
for col in model.feature_names_in_:
    if col not in new_data.columns:
        new_data[col] = np.nan
new_data = new_data[model.feature_names_in_]

pred = model.predict(new_data)[0]  # 0 = Dissatisfied, 1 = Satisfied
print("Prediction:", pred)

## Evaluation at a Glance (example)
- Accuracy ≈ 0.83

- F1 (class 1) ≈ 0.90

- Confusion Matrix (example):

|            | Pred 0 | Pred 1 |
| ---------- | ------ | ------ |
| **True 0** | 193    | 2821   |
| **True 1** | 111    | 14057  |

- Strong for class 1 (Satisfied); weaker recall for class 0 → consider class_weight, threshold tuning, or sampling strategies.

## Hypothesis Tests (examples)

ANOVA: CSAT differs by Agent Shift — significant (p < 0.05)

Chi-square: Proportion of high CSAT differs by Channel — significant (p < 0.05)

Spearman: CSAT vs response_minutes — negative relationship (faster response → higher CSAT)

## License

Choose the license that suits your needs (e.g., MIT).

## Acknowledgements

scikit-learn, pandas, numpy, matplotlib, SciPy

Project author: Shyam SR

---

### 🔹 `requirements.txt`
```text
# Core
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
scipy>=1.10

# Notebook & display
jupyterlab>=3.6
ipykernel>=6.20
jupytext>=1.14

# Plotting
matplotlib>=3.7

# Persistence / utils
joblib>=1.2

# (Optional) pretty tables & themes
tabulate>=0.9
