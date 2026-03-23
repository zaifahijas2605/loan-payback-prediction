
import os, sys, json, warnings
sys.path.append(os.path.dirname(__file__))


# Reduce noisy logs 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras

from utils import load_data, apply_ordinal_encodings, get_feature_groups, get_cv

# CPU performance tweaks

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


RANDOM_STATE = 42
VERSION = "v1"        
N_TRIALS = 15          


# Reproducibility

def set_seeds(seed=RANDOM_STATE):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(RANDOM_STATE)
tf.get_logger().setLevel("ERROR")  

# Load & Prepare Data

train, test = load_data()
train = apply_ordinal_encodings(train)
test  = apply_ordinal_encodings(test)

test_ids = test["id"].copy()
y = train["loan_paid_back"].astype(int).values
X = train.drop(["loan_paid_back", "id"], axis=1)
X_test = test.drop(["id"], axis=1)

num_cols, cat_cols = get_feature_groups()
cv = get_cv(n_splits=5)

# Preprocessor (dense output for Keras)

def make_preprocessor():
    
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop"
    )


# Model builder

def build_model(input_dim, n_layers, units1, units2, units3, dropout, lr, l2):
    keras.backend.clear_session()

    inputs = keras.layers.Input(shape=(input_dim,))
    x = inputs

    # Layer 1
    x = keras.layers.Dense(
        units1, activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2)
    )(x)
    x = keras.layers.Dropout(dropout)(x)

    # Layer 2 
    if n_layers >= 2:
        x = keras.layers.Dense(
            units2, activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2)
        )(x)
        x = keras.layers.Dropout(dropout)(x)

    # Layer 3 
    if n_layers >= 3:
        x = keras.layers.Dense(
            units3, activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2)
        )(x)
        x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model

# Output folders

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(BASE_DIR, "results")
subs_dir = os.path.join(BASE_DIR, "submissions")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(subs_dir, exist_ok=True)


# Optuna Objective

def objective(trial: optuna.Trial) -> float:
    set_seeds(RANDOM_STATE)

    # Hyperparameters to tune
    n_layers = trial.suggest_int("n_layers", 1, 3)

    units1 = trial.suggest_int("units1", 64, 256, step=32)
    units2 = trial.suggest_int("units2", 32, 256, step=32)
    units3 = trial.suggest_int("units3", 32, 256, step=32)

    dropout = trial.suggest_float("dropout", 0.05, 0.50)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    l2 = trial.suggest_float("l2", 1e-8, 1e-3, log=True)

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    epochs = trial.suggest_int("epochs", 10, 30)

    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        preprocessor = make_preprocessor()
        X_tr_p = preprocessor.fit_transform(X_tr)
        X_va_p = preprocessor.transform(X_va)

        model = build_model(
            input_dim=X_tr_p.shape[1],
            n_layers=n_layers,
            units1=units1, units2=units2, units3=units3,
            dropout=dropout,
            lr=lr,
            l2=l2
        )

        es = keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True
        )

        
        # tf.data pipelines (batch + prefetch) helps to train faster
       
        train_ds = tf.data.Dataset.from_tensor_slices((X_tr_p, y_tr))
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_va_p, y_va))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[es]
        )

        preds = model.predict(val_ds, verbose=0).ravel()
        auc = roc_auc_score(y_va, preds)
        fold_aucs.append(auc)

        
        trial.report(float(np.mean(fold_aucs)), step=fold)
        if trial.should_prune():
           
            del model
            keras.backend.clear_session()
            raise optuna.TrialPruned()

        del model
        keras.backend.clear_session()

    return float(np.mean(fold_aucs))

# Run Study

print("\n===== OPTUNA TUNING: TF ANN =====")
print(f"Trials: {N_TRIALS} | CV: 5-fold ROC-AUC | VERSION: {VERSION}\n")

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE),
    pruner=MedianPruner(n_startup_trials=5)   
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nBest CV AUC:", study.best_value)
print("Best Params:", study.best_params)

# Save best params
best_params_path = os.path.join(results_dir, f"ann_tf_optuna_best_params-{VERSION}.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)

# Save trials
trials_df = study.trials_dataframe()
trials_path = os.path.join(results_dir, f"ann_tf_optuna_trials-{VERSION}.csv")
trials_df.to_csv(trials_path, index=False)

print("Saved best params:", best_params_path)
print("Saved trials log:", trials_path)

# Train best model on full data + submission

print("\nTraining best TF ANN on full data and generating submission...")

best = study.best_params

preprocessor = make_preprocessor()
X_full = preprocessor.fit_transform(X)
X_test_p = preprocessor.transform(X_test)

final_model = build_model(
    input_dim=X_full.shape[1],
    n_layers=best["n_layers"],
    units1=best["units1"],
    units2=best["units2"],
    units3=best["units3"],
    dropout=best["dropout"],
    lr=best["lr"],
    l2=best["l2"],
)

es = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=4,
    restore_best_weights=True
)

# Use tf.data for full training too
train_full_ds = tf.data.Dataset.from_tensor_slices((X_full, y))
train_full_ds = train_full_ds.batch(best["batch_size"]).prefetch(tf.data.AUTOTUNE)

# validation split needs arrays, so we do a manual split for speed + tf.data
n = X_full.shape[0]
val_size = int(0.15 * n)
idx = np.random.RandomState(RANDOM_STATE).permutation(n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

X_tr_full, y_tr_full = X_full[trn_idx], y[trn_idx]
X_va_full, y_va_full = X_full[val_idx], y[val_idx]

train_ds = tf.data.Dataset.from_tensor_slices((X_tr_full, y_tr_full)).batch(best["batch_size"]).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_va_full, y_va_full)).batch(best["batch_size"]).prefetch(tf.data.AUTOTUNE)

final_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=best["epochs"],
    verbose=0,
    callbacks=[es]
)

test_ds = tf.data.Dataset.from_tensor_slices(X_test_p).batch(best["batch_size"]).prefetch(tf.data.AUTOTUNE)
test_pred = final_model.predict(test_ds, verbose=0).ravel()

sub_path = os.path.join(subs_dir, f"sub_ann_tf_optuna_tuned-{VERSION}.csv")
pd.DataFrame({"id": test_ids, "loan_paid_back": test_pred}).to_csv(sub_path, index=False)
print("Saved submission:", sub_path)

# Logging the results to baseline_results.csv
summary_row = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": f"ann_tf_optuna_tuned-{VERSION}",
    "mean_cv_auc": float(study.best_value),
    "std_cv_auc": np.nan,
    "parameters": str(study.best_params)
}])

results_path = os.path.join(results_dir, "baseline_results.csv")
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    summary_row = pd.concat([existing, summary_row], ignore_index=True)

summary_row.to_csv(results_path, index=False)
print("Logged to:", results_path)
