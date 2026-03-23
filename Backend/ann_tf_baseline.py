import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras

from utils import load_data, apply_ordinal_encodings, get_feature_groups, get_cv

# Here we will load the data
train, test = load_data()
train = apply_ordinal_encodings(train)
test  = apply_ordinal_encodings(test)

test_ids = test["id"].copy()
y = train["loan_paid_back"].astype(int).values
X = train.drop(["loan_paid_back", "id"], axis=1)
X_test = test.drop(["id"], axis=1)

num_cols, cat_cols = get_feature_groups()

# Preprocessing for the  ANN model: scaling numeric + one-hot categorical  variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

cv = get_cv(n_splits=5)

def build_model(input_dim, lr=1e-3, dropout=0.2):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model

print("\n===== TF ANN BASELINE CV START =====\n")
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    X_tr_p = preprocessor.fit_transform(X_tr)
    X_va_p = preprocessor.transform(X_va)

    model = build_model(input_dim=X_tr_p.shape[1], lr=1e-3, dropout=0.2)

    es = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_tr_p, y_tr,
        validation_data=(X_va_p, y_va),
        epochs=15,
        batch_size=2048,
        verbose=0,
        callbacks=[es]
    )

    preds = model.predict(X_va_p, verbose=0).ravel()
    auc = roc_auc_score(y_va, preds)
    fold_aucs.append(auc)

    print(f"TF ANN | Fold {fold} AUC: {auc:.4f}")

mean_auc = float(np.mean(fold_aucs))
std_auc  = float(np.std(fold_aucs))
print(f"\nTF ANN | Mean CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")


print("\nTraining TF ANN on full data & generating submission...\n")
X_full = preprocessor.fit_transform(X)
X_test_p = preprocessor.transform(X_test)

final_model = build_model(input_dim=X_full.shape[1], lr=1e-3, dropout=0.2)
es = keras.callbacks.EarlyStopping(monitor="auc", mode="max", patience=2, restore_best_weights=True)

final_model.fit(X_full, y, epochs=12, batch_size=2048, verbose=0, callbacks=[es])

test_pred = final_model.predict(X_test_p, verbose=0).ravel()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
subs_dir = os.path.join(BASE_DIR, "submissions")
results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(subs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

sub_path = os.path.join(subs_dir, "sub_ann_tf_baseline.csv")
pd.DataFrame({"id": test_ids, "loan_paid_back": test_pred}).to_csv(sub_path, index=False)
print("Saved submission:", sub_path)

# logging the results
results_path = os.path.join(results_dir, "baseline_results.csv")
row = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "ann_tf_baseline",
    "mean_cv_auc": mean_auc,
    "std_cv_auc": std_auc,
    "parameters": "Keras(128-64, dropout=0.2, lr=1e-3, earlystop)"
}])

if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    row = pd.concat([existing, row], ignore_index=True)

row.to_csv(results_path, index=False)
print("Results saved to:", results_path)
