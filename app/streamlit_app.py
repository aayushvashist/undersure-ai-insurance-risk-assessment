"""
UnderSure.AI — interactive underwriting risk-assessment demo.

Run locally after training the models:
    pip install -r requirements.txt
    python -m src.train_risk_model
    python -m src.train_claims_models
    streamlit run app/streamlit_app.py
"""
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

st.set_page_config(
    page_title="UnderSure.AI — Risk Assessment Demo",
    page_icon="🚗",
    layout="centered",
)


@st.cache_resource
def load_models():
    missing = [
        p
        for p in [
            config.MODELS_DIR / "risk_classifier.joblib",
            config.MODELS_DIR / "claim_frequency_classifier.joblib",
            config.MODELS_DIR / "claim_amount_regressor.joblib",
        ]
        if not p.exists()
    ]
    if missing:
        return None, None, None, missing
    risk_model = joblib.load(config.MODELS_DIR / "risk_classifier.joblib")
    freq_model = joblib.load(config.MODELS_DIR / "claim_frequency_classifier.joblib")
    amount_model = joblib.load(config.MODELS_DIR / "claim_amount_regressor.joblib")
    return risk_model, freq_model, amount_model, []


risk_model, freq_model, amount_model, missing = load_models()

st.title("🚗 UnderSure.AI")
st.caption(
    "Machine learning-assisted motor insurance risk assessment — portfolio demo"
)

if missing:
    st.error(
        "Trained model files not found. Run the training pipeline first:\n\n"
        "```\npython -m src.train_risk_model\npython -m src.train_claims_models\n```"
    )
    st.info("Missing: " + ", ".join(str(m.name) for m in missing))
    st.stop()

with st.expander("ℹ️ About this demo — read before trusting any number here", expanded=False):
    st.markdown(
        """
This app scores a hypothetical applicant using models trained on a public
sample motor-insurance dataset (37.5k policies). Two important honesty notes:

1. **The risk tier is a documented proxy label**, built from transparent,
   weighted business rules (age, kids driving, income, vehicle age, usage
   type, etc.) — not a real underwriter's historical decision. See
   `src/risk_labeling.py` and the README for the exact formula.
2. **Claim occurrence and claim amount are modeled from real data columns**,
   and — reported honestly — barely beat a naive baseline. That's a
   legitimate finding: idiosyncratic claim risk is hard to predict from
   demographics alone, which is why real insurers lean on much richer
   telematics/claims-history data than is available here.

This is a portfolio/learning project, not a production underwriting system.
        """
    )

st.subheader("Applicant details")

col1, col2 = st.columns(2)
with col1:
    birth_year = st.number_input("Birth year", min_value=1940, max_value=2008, value=1990)
    household_income = st.number_input(
        "Annual household income", min_value=10000, max_value=500000, value=120000, step=5000
    )
    kids_driving = st.slider("Kids driving in household", 0, 4, 0)
    car_year = st.number_input("Vehicle model year", min_value=1980, max_value=2026, value=2016)
with col2:
    car_use = st.selectbox("Car use", ["Private", "Commercial"])
    coverage_zone = st.selectbox(
        "Coverage zone", ["Highly Urban", "Urban", "Suburban", "Rural", "Highly Rural"]
    )
    education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
    gender = st.selectbox("Gender", ["Male", "Female"])

col3, col4 = st.columns(2)
with col3:
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Separated"])
with col4:
    parent = st.selectbox("Parent", ["Yes", "No"])

if st.button("Assess applicant", type="primary"):
    reference_year = date.today().year
    customer_age = reference_year - birth_year
    vehicle_age = reference_year - car_year

    applicant = pd.DataFrame(
        [
            {
                "Customer_Age": customer_age,
                "Vehicle_Age": vehicle_age,
                "Car_Year": car_year,
                "Household_Income": household_income,
                "Kids_Driving": kids_driving,
                "Car_Use": car_use,
                "Coverage_Zone": coverage_zone,
                "Education": education,
                "Gender": gender,
                "Marital_Status": marital_status,
                "Parent": parent,
            }
        ]
    )

    risk_pred = risk_model.predict(applicant)[0]
    risk_proba = dict(zip(risk_model.classes_, risk_model.predict_proba(applicant)[0]))

    claim_proba = freq_model.predict_proba(applicant)[0][1]
    expected_amount = amount_model.predict(applicant)[0]

    st.subheader("Assessment result")

    risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    st.markdown(
        f"### Risk tier: :{risk_colors.get(risk_pred, 'blue')}[**{risk_pred}**]"
    )

    proba_df = pd.DataFrame(
        {"Risk tier": list(risk_proba.keys()), "Probability": list(risk_proba.values())}
    ).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Risk tier"))

    m1, m2 = st.columns(2)
    m1.metric("Estimated claim likelihood (1 yr)", f"{claim_proba*100:.1f}%")
    m2.metric("Estimated claim amount, if claimed", f"${expected_amount:,.0f}")

    st.caption(
        f"Computed features — Customer age: {customer_age}, Vehicle age: {vehicle_age} years."
    )

    st.markdown("---")
    st.caption(
        "Reminder: risk tier is a rule-based proxy for demonstration; claim "
        "estimates barely beat baseline on this dataset (see the About "
        "section above). Do not use for real underwriting decisions."
    )
