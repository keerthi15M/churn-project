import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="centered",
)

# Load model + columns
model = joblib.load("models/xgb_churn_model.pkl")
training_columns = joblib.load("models/training_columns.pkl")

# Header
st.markdown("<h1 style='text-align: center; color:#4CAF50;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Fill in the customer details below to predict whether they are likely to churn.")

# Form layout
with st.container():
    st.subheader("🔍 Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        tenure = st.number_input("Tenure (months)", 0, 100)
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly = st.number_input("Monthly Charges", 0.0)
        total = st.number_input("Total Charges", 0.0)

# Prediction Button
st.markdown("---")
if st.button("🔮 Predict Churn", use_container_width=True):

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phoneservice,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in training_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[training_columns]

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Display Result
    st.markdown("### 📘 Prediction Result")
    if prediction == 1:
        st.error(f"""
    🔴 **High risk of churn!**  
    Probability: **{prob:.2f}**
    """)
    else:
        st.success(f"""
    🟢 **Customer is not likely to churn.**  
    Probability: **{prob:.2f}**
    """)



