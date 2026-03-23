
import os, sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from utils import (
    load_data,
    apply_ordinal_encodings,
    get_feature_groups,
    get_linear_preprocessor,
    get_cv,
    RANDOM_STATE,
)

# Loading and Preparing the data
train, test = load_data()
train = apply_ordinal_encodings(train)
test = apply_ordinal_encodings(test)

test_ids = test["id"].copy()

y = train["loan_paid_back"].astype(int)
X = train.drop(["loan_paid_back", "id"], axis=1)
X_test = test.drop(["id"], axis=1)

num_cols, cat_cols = get_feature_groups()
linear_preprocessor = get_linear_preprocessor(num_cols, cat_cols)
cv = get_cv(n_splits=5)

#Output folders
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(BASE_DIR, "results")
subs_dir = os.path.join(BASE_DIR, "submissions")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(subs_dir, exist_ok=True)

# Baseline model
name = "logreg_baseline"
model = LogisticRegression(
    solver="liblinear",    
    max_iter=2000,
    random_state=RANDOM_STATE
)

print("\n===== LOGREG BASELINE CV START =====\n")

pipe = Pipeline([
    ("prep", linear_preprocessor),
    ("model", model)
])

#CV evaluation
scores = cross_val_score(
    pipe, X, y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

mean_auc = float(scores.mean())
std_auc = float(scores.std())

print(f"{name} | Mean CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

#Logging results
results_df = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": name,
    "mean_cv_auc": mean_auc,
    "std_cv_auc": std_auc,
    "parameters": str(model.get_params())
}])

results_path = os.path.join(results_dir, "baseline_results.csv")
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    results_df = pd.concat([existing, results_df], ignore_index=True)

results_df.to_csv(results_path, index=False)
print("Results saved to:", results_path)

#Training the model on FULL data and creating submission
pipe.fit(X, y)
test_pred = pipe.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"id": test_ids, "loan_paid_back": test_pred})
sub_path = os.path.join(subs_dir, f"sub_{name}.csv")
sub.to_csv(sub_path, index=False)
print(f"Saved submission: {sub_path}\n")