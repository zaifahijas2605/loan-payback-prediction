
import os, sys, json, warnings
sys.path.append(os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from utils import (
    load_data,
    apply_ordinal_encodings,
    get_feature_groups,
    get_tree_preprocessor
)

RANDOM_STATE = 42

#Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

results_dir = os.path.join(BASE_DIR, "results")
models_dir  = os.path.join(BASE_DIR, "models")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


BEST_PARAMS_PATH = os.path.join(results_dir, "lgbm_optuna_best_params -v1.json")


MODEL_OUT_PATH = os.path.join(models_dir, "final_lgbm_model.joblib")

FI_CSV_PATH = os.path.join(results_dir, "feature_importance_lgbm.csv")
FI_PNG_PATH = os.path.join(results_dir, "feature_importance_lgbm_top30.png")




# Helper: feature names from ColumnTransformer + OHE

def get_feature_names_from_preprocessor(preprocessor, num_cols, cat_cols):

    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe_feature_names = cat_transformer.get_feature_names_out(cat_cols)

    passthrough_cols = [c for c in num_cols] 

   
    all_names = list(ohe_feature_names) + passthrough_cols
    return all_names

#Loading data
train, test = load_data()
train = apply_ordinal_encodings(train)
test  = apply_ordinal_encodings(test)

test_ids = test["id"].copy()

y = train["loan_paid_back"].astype(int)
X = train.drop(["loan_paid_back", "id"], axis=1)
X_test = test.drop(["id"], axis=1)

num_cols, cat_cols = get_feature_groups()
tree_preprocessor = get_tree_preprocessor(cat_cols)

# Loading the best hyperparameters
if not os.path.exists(BEST_PARAMS_PATH):
    raise FileNotFoundError(
        f"Best params json not found at:\n{BEST_PARAMS_PATH}\n"
        f"Tip: set BEST_PARAMS_PATH to your correct file name in results/."
    )

with open(BEST_PARAMS_PATH, "r") as f:
    best_params = json.load(f)


best_params["random_state"] = RANDOM_STATE
best_params["n_jobs"] = -1
best_params["verbosity"] = -1

#Final Train Pipeline
final_model = lgb.LGBMClassifier(**best_params)

final_pipe = Pipeline([
    ("prep", tree_preprocessor),
    ("model", final_model)
])

print("\n===== TRAIN FINAL LIGHTGBM =====")
print("Using best params from:", BEST_PARAMS_PATH)

final_pipe.fit(X, y)

# Saving the pipeline (preprocessor + model together)
joblib.dump(final_pipe, MODEL_OUT_PATH)
print("Saved final model pipeline to:", MODEL_OUT_PATH)


# Feature importance Analysis
fitted_preprocessor = final_pipe.named_steps["prep"]
fitted_model = final_pipe.named_steps["model"]
feature_names = get_feature_names_from_preprocessor(fitted_preprocessor, num_cols, cat_cols)
importances = fitted_model.feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

fi.to_csv(FI_CSV_PATH, index=False)
print("Saved feature importance CSV to:", FI_CSV_PATH)


top_n = 30
fi_top = fi.head(top_n).iloc[::-1]  

plt.figure(figsize=(10, 8))
plt.barh(fi_top["feature"], fi_top["importance"])
plt.title(f"LightGBM Feature Importance (Top {top_n})")
plt.tight_layout()
plt.savefig(FI_PNG_PATH, dpi=200)
plt.close()
print("Saved feature importance plot to:", FI_PNG_PATH)

