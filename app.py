import streamlit as st
import pandas as pd
import pickle
# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load template DataFrame
template_df = pd.read_csv('template.csv')
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Use the sidebar to enter customer details and predict churn.")
st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
]) 
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])


def preprocess_input():
    # Start with a copy of the template
    input_data = template_df.copy()
    input_data.loc[0] = 0  # Fill all columns with 0

    # Numerical features
    input_data.at[0, 'SeniorCitizen'] = 1 if senior == "Yes" else 0
    input_data.at[0, 'tenure'] = tenure
    input_data.at[0, 'MonthlyCharges'] = monthly_charges
    input_data.at[0, 'TotalCharges'] = total_charges

    # Categorical features
    input_data.at[0, 'gender_Male'] = 1 if gender == "Male" else 0
    input_data.at[0, 'Partner_Yes'] = 1 if partner == "Yes" else 0
    input_data.at[0, 'Dependents_Yes'] = 1 if dependents == "Yes" else 0

    input_data.at[0, 'PhoneService_Yes'] = 1 if phone_service == "Yes" else 0

    input_data.at[0, 'MultipleLines_Yes'] = 1 if multiple_lines == "Yes" else 0
    input_data.at[0, 'MultipleLines_No phone service'] = 1 if multiple_lines == "No phone service" else 0

    input_data.at[0, 'InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
    input_data.at[0, 'InternetService_No'] = 1 if internet == "No" else 0

    input_data.at[0, 'OnlineSecurity_Yes'] = 1 if online_security == "Yes" else 0
    input_data.at[0, 'OnlineSecurity_No internet service'] = 1 if online_security == "No internet service" else 0

    input_data.at[0, 'OnlineBackup_Yes'] = 1 if online_backup == "Yes" else 0
    input_data.at[0, 'OnlineBackup_No internet service'] = 1 if online_backup == "No internet service" else 0

    input_data.at[0, 'DeviceProtection_Yes'] = 1 if device_protection == "Yes" else 0
    input_data.at[0, 'DeviceProtection_No internet service'] = 1 if device_protection == "No internet service" else 0

    input_data.at[0, 'TechSupport_Yes'] = 1 if tech_support == "Yes" else 0
    input_data.at[0, 'TechSupport_No internet service'] = 1 if tech_support == "No internet service" else 0

    input_data.at[0, 'StreamingTV_Yes'] = 1 if streaming_tv == "Yes" else 0
    input_data.at[0, 'StreamingTV_No internet service'] = 1 if streaming_tv == "No internet service" else 0

    input_data.at[0, 'StreamingMovies_Yes'] = 1 if streaming_movies == "Yes" else 0
    input_data.at[0, 'StreamingMovies_No internet service'] = 1 if streaming_movies == "No internet service" else 0

    input_data.at[0, 'Contract_One year'] = 1 if contract == "One year" else 0
    input_data.at[0, 'Contract_Two year'] = 1 if contract == "Two year" else 0

    input_data.at[0, 'PaperlessBilling_Yes'] = 1 if paperless_billing == "Yes" else 0

    input_data.at[0, 'PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
    input_data.at[0, 'PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0
    input_data.at[0, 'PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0

    return input_data

    # Assume all internet-based services are unavailable if InternetService is "No"
    if internet == "No":
        input_data.at[0, 'OnlineSecurity_No internet service'] = 1
        input_data.at[0, 'OnlineBackup_No internet service'] = 1
        input_data.at[0, 'DeviceProtection_No internet service'] = 1
        input_data.at[0, 'TechSupport_No internet service'] = 1
        input_data.at[0, 'StreamingTV_No internet service'] = 1
        input_data.at[0, 'StreamingMovies_No internet service'] = 1
    else:
        input_data.at[0, 'OnlineSecurity_Yes'] = 1
        input_data.at[0, 'OnlineBackup_Yes'] = 1
        input_data.at[0, 'DeviceProtection_Yes'] = 1
        input_data.at[0, 'TechSupport_Yes'] = 1
        input_data.at[0, 'StreamingTV_Yes'] = 1
        input_data.at[0, 'StreamingMovies_Yes'] = 1

    input_data.at[0, 'Contract_One year'] = 1 if contract == "One year" else 0
    input_data.at[0, 'Contract_Two year'] = 1 if contract == "Two year" else 0

    input_data.at[0, 'PaperlessBilling_Yes'] = 1  # Assuming paperless billing is always Yes

    input_data.at[0, 'PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0
    input_data.at[0, 'PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
    input_data.at[0, 'PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0

    return input_data

    return input_data
if st.button("Predict Churn"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability: {1 - prob:.2f})")