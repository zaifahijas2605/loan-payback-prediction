
import os, sys, json
sys.path.append(os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import optuna
from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import xgboost as xgb

from utils import (
    load_data,
    apply_ordinal_encodings,
    get_feature_groups,
    get_tree_preprocessor,
    get_cv
)

RANDOM_STATE = 42
VERSION = "v1"   


# Load & Prepare Data

train, test = load_data()
train = apply_ordinal_encodings(train)
test  = apply_ordinal_encodings(test)

test_ids = test["id"].copy()

y = train["loan_paid_back"].astype(int)
X = train.drop(["loan_paid_back", "id"], axis=1)
X_test = test.drop(["id"], axis=1)

num_cols, cat_cols = get_feature_groups()
tree_preprocessor = get_tree_preprocessor(cat_cols)


cv = get_cv(n_splits=5)


# Output folders

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(BASE_DIR, "results")
subs_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(subs_dir, exist_ok=True)


# Optuna Objective

def objective(trial: optuna.Trial) -> float:
    params = {
        # core
        "n_estimators": trial.suggest_int("n_estimators", 400, 1800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.07, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),

        # regularization / split control
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),

        # sampling
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),

        # fixed
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }

    model = xgb.XGBClassifier(**params)

    pipe = Pipeline([
        ("prep", tree_preprocessor),
        ("model", model)
    ])

   
    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )
    return float(scores.mean())

# Run Study

N_TRIALS = 15  

print("\n===== OPTUNA TUNING: XGBOOST =====")
print(f"Trials: {N_TRIALS} | CV: 5-fold ROC-AUC | VERSION: {VERSION}\n")

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nBest CV AUC:", study.best_value)
print("Best Params:", study.best_params)

# Save best params
best_params_path = os.path.join(results_dir, f"xgb_optuna_best_params-{VERSION}.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)

# Save trials
trials_df = study.trials_dataframe()
trials_path = os.path.join(results_dir, f"xgb_optuna_trials-{VERSION}.csv")
trials_df.to_csv(trials_path, index=False)

print("Saved best params:", best_params_path)
print("Saved trials log:", trials_path)


# Train best model on full data + submission

best_model = xgb.XGBClassifier(
    **study.best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist",
)

best_pipe = Pipeline([
    ("prep", tree_preprocessor),
    ("model", best_model)
])

print("\nTraining best XGBoost on full data and generating submission...")
best_pipe.fit(X, y)
test_pred = best_pipe.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"id": test_ids, "loan_paid_back": test_pred})
sub_path = os.path.join(subs_dir, f"sub_xgboost_optuna_tuned-{VERSION}.csv")
sub.to_csv(sub_path, index=False)
print("Saved submission:", sub_path)


# Log to results

summary_row = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": f"xgboost_optuna_tuned-{VERSION}",
    "mean_cv_auc": study.best_value,
    "std_cv_auc": np.nan,
    "parameters": str(study.best_params)
}])

results_path = os.path.join(results_dir, "baseline_results.csv")
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    summary_row = pd.concat([existing, summary_row], ignore_index=True)

summary_row.to_csv(results_path, index=False)
print("Logged to:", results_path)
