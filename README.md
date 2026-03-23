# Loan Default Prediction

Predicting the probability a borrower will pay back their loan.  
Kaggle Playground Series Season 5, Episode 11

## Results

| Model | Mean CV AUC |
|---|---|
| LightGBM Baseline | 0.9214 |
| XGBoost Baseline | 0.9208 |
| ANN Baseline | 0.9122 |
| Logistic Regression | 0.9101 |
| XGBoost (Optuna Tuned) | 0.9218 |
| ANN (Optuna Tuned) | 0.9122 |
| **LightGBM (Optuna Tuned)** | **0.9227 ✅ Best** |

## Tech Stack
- **ML Models:** LightGBM, XGBoost, ANN, Logistic Regression
- **Hyperparameter Tuning:** Optuna
- **Explainability:** SHAP
- **Backend:** FastAPI
- **Frontend:** Streamlit

## How to Run
pip install -r requirements.txt
streamlit run streamlit_app.py

## Dataset
Can be found from [Kaggle](https://kaggle.com/competitions/playground-series-s5e11)
and place in Dataset/ folder.