import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load all saved artefacts
@st.cache_resource
def load_artifacts():
    model         = joblib.load('loan_model.pkl')
    le_education  = joblib.load('le_education.pkl')   # LabelEncoder → Education_Level
    le_target     = joblib.load('le_target.pkl')       # LabelEncoder → Loan_Approved
    ohe           = joblib.load('ohe.pkl')             # OneHotEncoder → 6 cat cols
    feature_names = joblib.load('feature_names.pkl')   # Exact column order
    return model, le_education, le_target, ohe, feature_names

model, le_education, le_target, ohe, feature_names = load_artifacts()

# OHE was fitted on these columns in this exact order
OHE_COLS = ['Employment_Status', 'Marital_Status', 'Loan_Purpose',
            'Property_Area', 'Gender', 'Employer_Category']

# ── Header
st.title("🏦 Bank Loan Approval Prediction")
st.markdown(
    "Fill in the applicant's details below. "
    "The **XGBoost model** will predict whether the loan will be "
    "**✅ Approved** or **❌ Rejected**."
)
st.divider()

# ── Input Form
st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Details**")
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"]
    )
    age = st.number_input(
        "Age", min_value=18, max_value=80, value=30, step=1
    )
    marital_status = st.selectbox(
        "Marital Status",
        options=["Married", "Single"]
    )
    dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=10, value=0, step=1
    )
    education_level = st.selectbox(
        "Education Level",
        options=le_education.classes_.tolist()
    )

with col2:
    st.markdown("**💼 Employment & Income**")
    employment_status = st.selectbox(
        "Employment Status",
        options=["Salaried", "Self-employed", "Contract", "Unemployed"]
    )
    employer_category = st.selectbox(
        "Employer Category",
        options=["Private", "Government", "MNC", "Business", "Unemployed"]
    )
    applicant_income = st.number_input(
        "Applicant Monthly Income (Rs.)",
        min_value=0, max_value=500000, value=50000, step=1000
    )
    coapplicant_income = st.number_input(
        "Co-Applicant Monthly Income (Rs.)",
        min_value=0, max_value=300000, value=0, step=1000
    )
    savings = st.number_input(
        "Total Savings (Rs.)",
        min_value=0, max_value=2000000, value=100000, step=5000
    )

with col3:
    st.markdown("**🏦 Loan & Financial Details**")
    loan_amount = st.number_input(
        "Loan Amount (Rs.)",
        min_value=0, max_value=2000000, value=200000, step=5000
    )
    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=6, max_value=360, value=120, step=6
    )
    loan_purpose = st.selectbox(
        "Loan Purpose",
        options=["Personal", "Home", "Car", "Business", "Education"]
    )
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=700, step=1
    )
    existing_loans = st.number_input(
        "Number of Existing Loans", min_value=0, max_value=20, value=0, step=1
    )
    dti_ratio = st.number_input(
        "Debt-to-Income Ratio (0.0 – 1.0)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.2f"
    )
    collateral_value = st.number_input(
        "Collateral Value (Rs.)",
        min_value=0, max_value=5000000, value=300000, step=10000
    )
    property_area = st.selectbox(
        "Property Area",
        options=["Urban", "Semiurban", "Rural"]
    )

st.divider()

# ── Predict Button
predict_btn = st.button(
    "🔍 Predict Loan Approval",
    use_container_width=True,
    type="primary"
)

if predict_btn:

    # ── Step 1: LabelEncode Education_Level (same as training) ───────
    education_encoded = le_education.transform([education_level])[0]

    # ── Step 2: OneHotEncode the 6 categorical columns ───────────────
    # Must be in the SAME order OHE was fitted on during training
    ohe_input = pd.DataFrame([[
        employment_status,
        marital_status,
        loan_purpose,
        property_area,
        gender,
        employer_category
    ]], columns=OHE_COLS)

    ohe_encoded = ohe.transform(ohe_input)
    ohe_col_names = ohe.get_feature_names_out(OHE_COLS).tolist()

    # ── Step 3: Build numerical + label-encoded part 
    num_data = pd.DataFrame([{
        'Applicant_Income'  : applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Age'               : age,
        'Dependents'        : dependents,
        'Credit_Score'      : credit_score,
        'Existing_Loans'    : existing_loans,
        'DTI_Ratio'         : dti_ratio,
        'Savings'           : savings,
        'Collateral_Value'  : collateral_value,
        'Loan_Amount'       : loan_amount,
        'Loan_Term'         : loan_term,
        'Education_Level'   : education_encoded,
    }])

    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_col_names)

    # ── Step 4: Concat same way as training 
    # Training used: pd.concat([df.drop(columns=cols), encoded_df], axis=1)
    full_input = pd.concat(
        [num_data.reset_index(drop=True), ohe_df.reset_index(drop=True)],
        axis=1
    )

    # Align to exact feature order used during model training
    full_input = full_input.reindex(columns=feature_names, fill_value=0)

    # ── Step 5: Predict
    prediction  = model.predict(full_input)[0]
    probability = model.predict_proba(full_input)[0]

    # Decode numeric prediction → "Yes" / "No" using target_le
    prediction_label = le_target.inverse_transform([prediction])[0]

    approval_pct = probability[1] * 100
    reject_pct   = probability[0] * 100

    # ── Result Display 
    st.subheader("📊 Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction_label == "Yes":
            st.success("✅ **LOAN APPROVED**")
            st.metric("Approval Confidence", f"{approval_pct:.1f}%")
        else:
            st.error("❌ **LOAN REJECTED**")
            st.metric("Rejection Confidence", f"{reject_pct:.1f}%")

    with res_col2:
        prob_df = pd.DataFrame({
            "Outcome"       : ["Rejected ❌", "Approved ✅"],
            "Probability %" : [round(reject_pct, 2), round(approval_pct, 2)]
        })
        st.markdown("**Probability Breakdown**")
        st.bar_chart(prob_df.set_index("Outcome"))

    # ── Input Summary Table
    with st.expander("🗂️ View submitted applicant data"):
        display_data = {
            "Field": [
                "Gender", "Age", "Marital Status", "Dependents", "Education Level",
                "Employment Status", "Employer Category",
                "Applicant Income", "Co-Applicant Income", "Savings",
                "Loan Amount", "Loan Term", "Loan Purpose",
                "Credit Score", "Existing Loans", "DTI Ratio",
                "Collateral Value", "Property Area"
            ],
            "Value": [
                gender, age, marital_status, dependents, education_level,
                employment_status, employer_category,
                f"Rs. {applicant_income:,}", f"Rs. {coapplicant_income:,}",
                f"Rs. {savings:,}", f"Rs. {loan_amount:,}",
                f"{loan_term} months", loan_purpose, credit_score,
                existing_loans, dti_ratio,
                f"Rs. {collateral_value:,}", property_area
            ]
        }
        st.table(pd.DataFrame(display_data))

# ── Footer 
st.divider()
st.caption(
    "🤖 Model: XGBoost (Tuned via GridSearchCV 5-fold CV)  |  "
    "Encoding: LabelEncoder + OneHotEncoder  |  "
    "Bank Loan Approval Prediction Project"
)