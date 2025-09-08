# ML 101 – Pima Indians Diabetes (EDA + Helpers)

Lightweight repo for exploring the Pima Indians Diabetes dataset and demonstrating baseline modeling utilities in Python. Includes an exploratory notebook and a few helper functions for logistic-style predictions.

## Quick Start

1) Create a virtual environment (Python 3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies (pick one):

- With uv (recommended, lockfile provided):
  ```bash
  pip install uv  # if you don't have it yet
  uv sync
  ```
- With pip directly:
  ```bash
  pip install -U ipykernel numpy pandas scikit-learn seaborn
  ```


## Project Structure

- `eda.ipynb`: exploratory data analysis on the diabetes dataset.
- `data/diabetes.csv`: Pima Indians Diabetes dataset (binary target).
- `utils.py`: small helpers for coefficients, odds ratios, and thresholded predictions.
- `pyproject.toml`: project metadata and dependencies.
- `uv.lock`: lockfile for reproducible installs using `uv`.

## Dataset

File: `data/diabetes.csv`

Columns (common names used for this dataset):

- `preg`: number of pregnancies
- `plas`: plasma glucose concentration
- `pres`: diastolic blood pressure (mm Hg)
- `skin`: triceps skin fold thickness (mm)
- `insu`: 2-Hour serum insulin (mu U/ml)
- `mass`: body mass index (BMI)
- `pedi`: diabetes pedigree function
- `age`: age in years
- `class`: label (`tested_positive` or `tested_negative`)

## Utilities

All utilities live in `utils.py`.

- `print_coefficients(feature_names, coefficients)` → `pd.DataFrame`
  - Builds a table with `feature`, `coefficient`, and `odds_ratio` (`exp(coef)`).
- `coefficients_to_prediction(intercept, coefficients, X)` → `np.ndarray`
  - Applies logistic transformation: `p = 1 / (1 + exp(-(b0 + X @ beta)))`.
- `predict_with_threshold(model, data, threshold=0.5)` → `list[int]`
  - Converts `model.predict_proba(..., class=1)` into hard labels using a cutoff.

Example snippet:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import print_coefficients, predict_with_threshold

X = df[["preg","plas","pres","skin","insu","mass","pedi","age"]]
y = (df["class"] == "tested_positive").astype(int)

model = LogisticRegression(max_iter=1000).fit(X, y)

coef_df = print_coefficients(X.columns, model.coef_)
preds = predict_with_threshold(model, X, threshold=0.5)
```

## Notes

- Python version is pinned to `3.13` (`.python-version`).
- If using VS Code, select the `.venv` interpreter for the `eda.ipynb` kernel.
- Feel free to extend the notebook with modeling baselines and calibration.
