import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)
import os, sys
sys.path.append(os.path.dirname(__file__))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from utils import apply_ordinal_encodings

# Loading the model at the beginning

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_lgbm_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run Backend/train_final_lgbm.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Loan Payback Prediction API", version="1.0")


# Input schema (validations)

Gender = Literal["Male", "Female", "Other"]
Marital = Literal["Single", "Married", "Divorced", "Widowed"]
Employment = Literal["Self-employed", "Retired", "Employed", "Student", "Unemployed"]
Purpose = Literal["Car", "Business", "Debt consolidation", "Education", "Home", "Medical", "Other", "Vacation"]


Education = Literal["High School", "Bachelor's", "Master's", "PhD", "Other"]


Grade = Literal[
    "A1","A2","A3","A4","A5",
    "B1","B2","B3","B4","B5",
    "C1","C2","C3","C4","C5",
    "D1","D2","D3","D4","D5",
    "E1","E2","E3","E4","E5",
    "F1","F2","F3","F4","F5"
]

class LoanRequest(BaseModel):
    annual_income: float = Field(..., ge=0)
    debt_to_income_ratio: float = Field(..., ge=0, le=5)  
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., ge=0)
    interest_rate: float = Field(..., ge=0, le=100)

    gender: Gender
    marital_status: Marital
    education_level: Education
    employment_status: Employment
    loan_purpose: Purpose
    grade_subgrade: Grade

class LoanResponse(BaseModel):
    probability_paid_back: float
  

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=LoanResponse)
def predict(req: LoanRequest):
    row = pd.DataFrame([req.model_dump()])

    
    row = apply_ordinal_encodings(row)

    
    prob = float(model.predict_proba(row)[0, 1])

    return LoanResponse(probability_paid_back=prob)


@app.post("/predict_debug")
def predict_debug(req: LoanRequest):
    row = pd.DataFrame([req.model_dump()])
    row = apply_ordinal_encodings(row)

    X_trans = model.named_steps["prep"].transform(row)
    return {
        "input_cols": list(row.columns),
        "transformed_shape": [int(X_trans.shape[0]), int(X_trans.shape[1])],
    }