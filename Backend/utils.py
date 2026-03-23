
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

def load_data():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(BASE_DIR, "Dataset", "train.csv")
    test_path  = os.path.join(BASE_DIR, "Dataset", "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test
def apply_ordinal_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    edu_map = {
    
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
        "Other": 4
        
    }

    letters = ['A','B','C','D','E','F']
    numbers = ['1','2','3','4','5']
    ordered_subgrades = [l+n for l in letters for n in numbers]
    subgrade_map = {sg: i+1 for i, sg in enumerate(ordered_subgrades)}

    df["education_level"] = df["education_level"].map(edu_map)
    df["grade_subgrade"] = df["grade_subgrade"].map(subgrade_map)

    return df

def get_feature_groups():
    num_cols = [
        "annual_income",
        "debt_to_income_ratio",
        "credit_score",
        "loan_amount",
        "interest_rate",
        "education_level",
        "grade_subgrade",
    ]

    cat_cols = [
        "gender",
        "marital_status",
        "employment_status",
        "loan_purpose",
    ]

    return num_cols, cat_cols

def get_tree_preprocessor(cat_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

def get_linear_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

def get_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

