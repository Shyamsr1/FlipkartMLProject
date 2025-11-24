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

## Project Structure

```text
FlipkartMLProject/
├── Customer_support_data.csv
├── FlipKartFinalMLProject1ShyamSR.ipynb
├── FlipkartMLProject.mp4          # Video walkthrough of the project
├── artifacts/                     # Created when the notebook is run
│   ├── csat_model.joblib          # Saved model pipeline
│   └── csat_model.meta.json       # Model metadata (JSON)
├── flipkart_csat_outputs/         # Created when the notebook is run
│   └── figs/                      # Plots saved from EDA / analysis
├── README.md
└── requirements.txt




---

---

## 🎥 Project VideoFile 
The **video file (`FlipkartMLProject.mp4`) is included inside the folder of this repository**.  
It contains a complete walkthrough of the model pipeline, visualizations, and key insights.

If the video cannot play directly on GitHub, download it locally and open it with any media player.

---


---

## Installation

```bash
# create & activate your environment (example shown with conda)
conda create -n flipkart-ml python=3.10 -y
conda activate flipkart-ml

# install dependencies
pip install -r requirements.txt
```

---

## Quickstart

1. Place / verify **`Customer_support_data.csv`** in the repo root
   (or update `CSV_PATH` inside the notebook).
2. Open the notebook **`FlipKartFinalMLProject1ShyamSR.ipynb`**.
3. Run sections in order:

   * **Data Understanding** → **Data Wrangling** → **EDA**
   * **Hypothesis Tests** → **Preprocessing** → **Modeling**
   * **Evaluation** → **Save Artifacts**

### Artifacts

After training, the final pipeline and metadata are saved to:

```
artifacts/csat_model.joblib
artifacts/csat_model.meta.json
```

---

## Modeling Summary

### Preprocessing

* **Numeric:** `SimpleImputer(strategy="median")` → `StandardScaler`

  * `with_mean=False` for **sparse** pipelines (LogReg / RF)
  * `with_mean=True` for **dense** pipelines (HGB)
* **Categorical:** `SimpleImputer(most_frequent)` →
  `OneHotEncoder(handle_unknown="ignore")`
* **Datetime:** parsing + engineered parts (**day / dow / month**)
* **Derived feature:** `response_minutes` (issue_responded − issue_reported)

### Models

* **LogisticRegression** (`solver="saga"`) — fast, interpretable baseline
* **RandomForestClassifier** — non-linear relationships & robustness
* **HistGradientBoostingClassifier** (**final**) — trained on a compact dense representation:

  * `OneHotEncoder (sparse)` → **`TruncatedSVD(n_components=256)`** → `HGB`

> **Why SVD?** One-hot over many categories creates huge sparse matrices.
> `TruncatedSVD` compresses them into a small dense space, **preventing RAM blow-ups** and speeding up training.

---

## Example: Best-model Selection & Comparison

```python
res_df = pd.DataFrame([
    ["Logistic Regression", acc_log, f1_log],
    ["Random Forest",       acc_rf,  f1_rf],
    ["HistGradientBoosting",acc_hgb, f1_hgb]
], columns=["Model", "Accuracy", "F1"]).sort_values("F1", ascending=False)

print(res_df.to_string(index=False))

# choose best by F1
scores = {"Logistic Regression": f1_log,
          "Random Forest":       f1_rf,
          "HistGradientBoosting":f1_hgb}
best_name = max(scores, key=scores.get)
best_pipe = {"Logistic Regression": pipe_log,
             "Random Forest":       pipe_rf,
             "HistGradientBoosting":pipe_hgb}[best_name]
print("Best model:", best_name)
```

---

## Saving & Loading the Model

```python
# Save
import os, joblib
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_pipe, "artifacts/csat_model.joblib")

# Load
model = joblib.load("artifacts/csat_model.joblib")
```

### Predict on New Data (schema-safe)

```python
import numpy as np
import pandas as pd

new_data = pd.DataFrame([{
    "Channel": "Email",
    "Response_Time": 2.4,
    "Issue_Type": "Payment",
    "Agent_Experience": "Senior"
}])

# align to training schema
for col in model.feature_names_in_:
    if col not in new_data.columns:
        new_data[col] = np.nan
new_data = new_data[model.feature_names_in_]

pred = model.predict(new_data)[0]  # 0 = Dissatisfied, 1 = Satisfied
print("Prediction:", pred)
```

---

## Evaluation at a Glance (example)

* **Accuracy:** ≈ **0.83**
* **F1 (class 1 / Satisfied):** ≈ **0.90**

**Confusion Matrix**

|            | Pred 0 | Pred 1 |
| ---------- | ------ | ------ |
| **True 0** | 193    | 2821   |
| **True 1** | 111    | 14057  |

> Strong performance for the **Satisfied** class; consider `class_weight`, threshold tuning, or sampling strategies to improve **class 0** recall.

---

## Hypothesis Tests (examples)

* **ANOVA:** CSAT differs by **Agent Shift** — *significant* (p < 0.05)
* **Chi-square:** Proportion of **high CSAT** differs by **Channel** — *significant* (p < 0.05)
* **Spearman:** **CSAT vs `response_minutes`** — negative relationship
  *(faster responses → higher CSAT)*

---

## License

Choose a license that suits your needs (e.g., **MIT**).

## Acknowledgements

scikit-learn, pandas, numpy, matplotlib, SciPy
Project author: **Shyam SR**

---

### `requirements.txt`

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
```
