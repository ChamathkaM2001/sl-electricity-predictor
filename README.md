# ⚡ Sri Lanka Household Electricity Consumption Predictor

Predicts a Sri Lankan household's **average monthly grid electricity consumption (kWh)** from a one-time survey of the house, its occupants and their appliances, with a **calibrated uncertainty range** and **SHAP explanations** for every prediction.

**Live app:** <!-- paste your Streamlit Cloud link here -->

## Approach

| | |
|---|---|
| **Data** | RECON-SL (IEEE DataPort): 25 months of meter readings (Oct 2022 – Oct 2024) joined to the Wave 1 household survey; 3,999 households, 58 engineered features |
| **Target** | Average monthly kWh per household, modelled as log(1 + kWh) |
| **Model** | **NGBoost** (Natural Gradient Boosting) — predicts a full probability distribution for each household, not just a single number |
| **Compared with** | Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost (taught benchmarks) and Gaussian Process Regression (second new algorithm) |
| **Validation** | 80/20 stratified train/test split; 5-fold cross-validation on the training set for tuning and model selection |
| **Explainability** | SHAP (global, dependence and per-household, including what drives the model's *uncertainty*) and LIME |

## Results (test set, 800 unseen households)

| Metric | NGBoost |
|---|---|
| Mean absolute error | 45.7 kWh (28% lower than a naive baseline) |
| Median absolute error | 24.0 kWh |
| Predicted within ±25% | 50% of households |
| 80% range contains the true value | 84% of households (after calibration) |

## Repository contents

| File | Purpose |
|---|---|
| `electricity.ipynb` | Full pipeline: data preparation, EDA, modelling, tuning, evaluation, explainability, model export |
| `app.py` | Streamlit web application |
| `ngb_compat.py` | Small NGBoost compatibility fix so the model works inside a scikit-learn Pipeline and loads in the app |
| `electricity_model.pkl` | Trained pipeline plus everything the app needs |
| `requirements.txt` | Pinned library versions |

## Running locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Getting the data

The dataset is not included in this repository. Download it from IEEE DataPort (free account required):

Algama, C. et al. (2025). *Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL)*. IEEE DataPort. https://doi.org/10.21227/n1dk-q860

Place `monthly_consumption.csv` and the eight `w1_*.csv` survey files in a `data/` folder, then run the notebook.
