
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

import lightgbm as lgb

from utils import (
    load_data,
    apply_ordinal_encodings,
    get_feature_groups,
    get_tree_preprocessor,
    get_cv
)

RANDOM_STATE = 42


# Load & Prepare Data

train, test = load_data()
train = apply_ordinal_encodings(train)
test = apply_ordinal_encodings(test)

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
        "n_estimators": trial.suggest_int("n_estimators", 600, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,

    }

    model = lgb.LGBMClassifier(**params)

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

print("\n===== OPTUNA TUNING: LIGHTGBM =====")
print(f"Trials: {N_TRIALS} | CV: 5-fold ROC-AUC\n")

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nBest CV AUC:", study.best_value)
print("Best Params:", study.best_params)

# Save best params
best_params_path = os.path.join(results_dir, "lgbm_optuna_best_params-v2.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)

# Save trials
trials_df = study.trials_dataframe()
trials_path = os.path.join(results_dir, "lgbm_optuna_trials_2.csv")
trials_df.to_csv(trials_path, index=False)

print("Saved best params:", best_params_path)
print("Saved trials log:", trials_path)


# Train best model on full data + submission

best_model = lgb.LGBMClassifier(
    **study.best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

best_pipe = Pipeline([
    ("prep", tree_preprocessor),
    ("model", best_model)
])

print("\nTraining best LightGBM on full data and generating submission...")
best_pipe.fit(X, y)
test_pred = best_pipe.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"id": test_ids, "loan_paid_back": test_pred})
sub_path = os.path.join(subs_dir, "sub_lightgbm_optuna_tuned-v2.csv")
sub.to_csv(sub_path, index=False)
print("Saved submission:", sub_path)

# Append summary to baseline_results.csv
summary_row = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "lightgbm_optuna_tuned-v2",
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
