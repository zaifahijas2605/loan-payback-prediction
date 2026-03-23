import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Loan Payback Predictor",
    page_icon="💵",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Light UI polish (CSS)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}
      [data-testid="stForm"] {
        border: 1px solid rgba(49, 51, 63, 0.15);
        padding: 1.25rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.02);
      }
      .result-card {
        border: 1px solid rgba(49, 51, 63, 0.15);
        padding: 1.25rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
      }
      .muted {opacity: 0.75;}
      .tiny {font-size: 0.9rem;}
      .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(49, 51, 63, 0.18);
        margin-right: 0.4rem;
        font-size: 0.9rem;
      }

      /* Center title/caption */
      h1, [data-testid="stCaptionContainer"] {text-align: center;}

      /* ---- Center + style the HTML table ---- */
      .center-table {display: flex; justify-content: center;}
      .center-table table {
        border-collapse: collapse;
        width: 95%;
        margin: 0 auto;
      }
      .center-table th, .center-table td {
        text-align: center !important;
        vertical-align: middle !important;
        padding: 10px 8px;
        border-bottom: 1px solid rgba(49, 51, 63, 0.15);
      }
      .center-table th {
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application estimates the probability that a loan will be paid back
    using a LightGBM model trained on historical borrower data.
    """)
    st.divider()
    st.caption("How to read results:")
    st.write("- Higher probability → more likely payback")
    st.write("- “Why” section explains what pushed the score up/down (simple explanation)")

# Helpers

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def risk_band(prob: float) -> tuple[str, str]:
    if prob >= 0.80:
        return "Low Risk", "🟢"
    if prob >= 0.60:
        return "Moderate Risk", "🟠"
    return "High Risk", "🔴"

# Ordinal encoding maps 
EDU_MAP = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3, "Other": 4}
LETTERS = ["A", "B", "C", "D", "E", "F"]
NUMBERS = ["1", "2", "3", "4", "5"]
ORDERED_SUBGRADES = [l + n for l in LETTERS for n in NUMBERS]
SUBGRADE_MAP = {sg: i + 1 for i, sg in enumerate(ORDERED_SUBGRADES)}


# Load pipeline locally for explanations (cached)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "final_lgbm_model.joblib")

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

@st.cache_resource
def build_explainer(_pipeline):
    pre = _pipeline.named_steps["prep"]
    mdl = _pipeline.named_steps["model"]
    return pre, mdl, shap.TreeExplainer(mdl)


# Header

st.title("💳 Loan Payback Probability Predictor")
st.caption("Estimate the likelihood that a loan will be paid back based on borrower information.")

with st.expander("What does this score mean?", expanded=False):
    st.write(
        "The model outputs a probability between **0% and 100%**. "
        "Use this as a **decision-support tool**, not the only factor for approval."
    )


# Form

with st.form("loan_form"):
    st.subheader("💰 Financial & Loan Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        annual_income = st.number_input("Annual Income", min_value=0.0, value=30000.0, step=500.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

    with c2:
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=5000.0, step=100.0)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)

    with c3:
        debt_to_income_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            value=0.10,
            step=0.01,
            format="%.2f"
        )
        grade_subgrade = st.selectbox(
            "Grade Subgrade",
            ["A1","A2","A3","A4","A5",
             "B1","B2","B3","B4","B5",
             "C1","C2","C3","C4","C5",
             "D1","D2","D3","D4","D5",
             "E1","E2","E3","E4","E5",
             "F1","F2","F3","F4","F5"]
        )

    st.divider()
    st.subheader("👤 Borrower Profile")

    c4, c5 = st.columns(2)
    with c4:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        education_level = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's", "PhD", "Other"]
        )

    with c5:
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Self-employed", "Student", "Retired", "Unemployed"]
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Debt consolidation", "Education", "Home", "Medical", "Car", "Business", "Vacation", "Other"]
        )

    st.markdown("<div class='muted tiny'>All fields are required.</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("✨ Predict")


# Prediction result + Explanation

if submitted:
    payload = {
        "annual_income": annual_income,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": int(credit_score),
        "loan_amount": loan_amount,
        "interest_rate": float(interest_rate),
        "gender": gender,
        "marital_status": marital_status,
        "education_level": education_level,
        "employment_status": employment_status,
        "loan_purpose": loan_purpose,
        "grade_subgrade": grade_subgrade,
    }

    try:
        with st.spinner("🔍 Evaluating loan risk…"):
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            r.raise_for_status()
            prob = r.json()["probability_paid_back"]

        prob = clamp01(prob)
        label, emoji = risk_band(prob)

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Payback Probability", f"{prob:.2%}")
        col2.metric("Risk Assessment", f"{emoji} {label}")

        st.progress(prob)

        st.markdown(
            f"<span class='pill'>{emoji} {label}</span>"
            f"<span class='muted'>Higher probability indicates stronger likelihood of repayment.</span>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🧠 Why did the model give this score?", expanded=False):
            st.markdown(
                """
                **How to read this**
                - Think of this as **what pushed the score up or down** for this application.
                - **Green** = helps payback, **red** = hurts payback.
                - **Bigger bars** = mattered more for this specific prediction.
                - “Impact” is **push strength** (not a percentage).
                """.strip()
            )

            if not os.path.exists(MODEL_PATH):
                st.warning("Local model file not found for explanations. Expected: models/final_lgbm_model.joblib")
            else:
                pipe = load_pipeline(MODEL_PATH)
                pre, mdl, explainer = build_explainer(pipe)

                row_df = pd.DataFrame([payload])
                row_df["education_level"] = row_df["education_level"].map(EDU_MAP).fillna(4)
                row_df["grade_subgrade"] = row_df["grade_subgrade"].map(SUBGRADE_MAP).fillna(0)

                X_trans = pre.transform(row_df)

                try:
                    feature_names = pre.get_feature_names_out()
                except Exception:
                    feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

                # 
                shap_vals = explainer.shap_values(X_trans)

                if isinstance(shap_vals, list):
                   
                    shap_for_class1 = np.array(shap_vals[1])[0]
                else:
                   
                    shap_for_class1 = np.array(shap_vals)[0]

                feature_names_arr = np.array(feature_names, dtype=str)
                df_shap = pd.DataFrame({"raw_feature": feature_names_arr, "shap_value": shap_for_class1.astype(float)})

                df_shap["raw_feature"] = df_shap["raw_feature"].str.replace(
                    r"^(cat__|remainder__|num__|prep__)", "", regex=True
                )

                cat_bases = {"loan_purpose", "gender", "marital_status", "employment_status"}

                def get_base_feature(fname: str) -> str:
                    for base in cat_bases:
                        if fname.startswith(base + "_"):
                            return base
                    return fname

                df_shap["base_feature"] = df_shap["raw_feature"].apply(get_base_feature)
                agg = df_shap.groupby("base_feature", as_index=False)["shap_value"].sum()

                pretty_map = {
                    "annual_income": "Annual Income",
                    "debt_to_income_ratio": "Debt-to-Income Ratio",
                    "credit_score": "Credit Score",
                    "loan_amount": "Loan Amount",
                    "interest_rate": "Interest Rate (%)",
                    "education_level": "Education Level",
                    "grade_subgrade": "Credit Grade",
                    "gender": "Gender",
                    "marital_status": "Marital Status",
                    "employment_status": "Employment Status",
                    "loan_purpose": "Loan Purpose",
                }

                selected_inline = {
                    "annual_income": f"{payload['annual_income']:,.0f}",
                    "debt_to_income_ratio": f"{payload['debt_to_income_ratio']:.2f}",
                    "credit_score": str(payload["credit_score"]),
                    "loan_amount": f"{payload['loan_amount']:,.0f}",
                    "interest_rate": f"{payload['interest_rate']:.1f}%",
                    "education_level": payload["education_level"],
                    "grade_subgrade": payload["grade_subgrade"],
                    "gender": payload["gender"],
                    "marital_status": payload["marital_status"],
                    "employment_status": payload["employment_status"],
                    "loan_purpose": payload["loan_purpose"],
                }

                def factor_label_table(base: str) -> str:
                    base_label = pretty_map.get(base, base.replace("_", " ").title())
                    val = selected_inline.get(base, "")
                    return f"{base_label} — {val}" if val != "" else base_label

                def factor_label_graph(base: str) -> str:
                    base_label = pretty_map.get(base, base.replace("_", " ").title())
                    val = selected_inline.get(base, "")
                    return f"{base_label} ({val})" if val != "" else base_label

                expected_order = list(pretty_map.keys())
                agg = agg.set_index("base_feature").reindex(expected_order).reset_index()
                agg["shap_value"] = agg["shap_value"].fillna(0.0)

                def direction_html(v: float) -> str:
                    if v >= 0:
                        return "<span style='color:#16a34a; font-weight:700;'>↑ Helps payback</span>"
                    return "<span style='color:#dc2626; font-weight:700;'>↓ Hurts payback</span>"

                display_df = pd.DataFrame({
                    "Factor": [factor_label_table(b) for b in agg["base_feature"]],
                    "Impact": agg["shap_value"].astype(float),
                    "Effect": [direction_html(v) for v in agg["shap_value"].astype(float)],
                })

                #centered table wrapper + centered cells/headers
                table_html = display_df.to_html(escape=False, index=False)
                st.markdown(f"<div class='center-table'>{table_html}</div>", unsafe_allow_html=True)

                #Horizontal colored bar chart 
                plot_df = agg.copy()
                plot_df["abs"] = plot_df["shap_value"].abs()
                plot_df = plot_df.sort_values("abs", ascending=True)

                plot_labels = [factor_label_graph(b) for b in plot_df["base_feature"]]
                colors = ["green" if v >= 0 else "red" for v in plot_df["shap_value"]]

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(plot_labels, plot_df["shap_value"], color=colors)
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("Impact on the payback score (push strength)")
                ax.set_ylabel("")
                ax.set_title("What pushed the score up vs down", loc="center")
                plt.tight_layout()
                st.pyplot(fig)

    except requests.exceptions.Timeout:
        st.error("The request timed out. Please ensure the backend is running.")
    except requests.exceptions.RequestException as e:
        st.error(" Unable to retrieve prediction.")
        st.caption(str(e))