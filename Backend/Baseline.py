
import os, sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import lightgbm as lgb
import xgboost as xgb

from utils import (
    load_data,
    apply_ordinal_encodings,
    get_feature_groups,
    get_tree_preprocessor,
    get_cv
)

#Loading and Preparing data
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

# Output Folders
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(BASE_DIR, "results")
subs_dir = os.path.join(BASE_DIR, "submissions")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(subs_dir, exist_ok=True)

#Model
models = {
    "lightgbm_baseline": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    ),
    "xgboost_baseline": xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist"
    )
}

results = []

print("\n===== BASELINE CV START =====\n")

for name, model in models.items():
    pipe = Pipeline([
        ("prep", tree_preprocessor),
        ("model", model)
    ])

    # CV evaluation
    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    mean_auc = float(scores.mean())
    std_auc = float(scores.std())

    print(f"{name} | Mean CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": name,
        "mean_cv_auc": mean_auc,
        "std_cv_auc": std_auc,
        "parameters": str(model.get_params())
    })

    #Training on FULL data and create submission 
    pipe.fit(X, y)
    test_pred = pipe.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({"id": test_ids, "loan_paid_back": test_pred})
    sub_path = os.path.join(subs_dir, f"sub_{name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}\n")

#Logging the results

results_df = pd.DataFrame(results)

results_path = os.path.join(results_dir, "baseline_results.csv")
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    results_df = pd.concat([existing, results_df], ignore_index=True)

results_df.to_csv(results_path, index=False)
print("Results saved to:", results_path)


