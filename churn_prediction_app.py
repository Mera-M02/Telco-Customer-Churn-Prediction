import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')    

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer is likely to churn or not, and see the main factors behind the prediction.")
st.header("Customer Details")

# Create input fields for customer details
tenure = st.number_input("Tenure (months with the company)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=2000.0, value=70.00)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=800.00)

gender = st.selectbox("Gender", options=["Male", "Female"])
senior = st.selectbox("Senior Citizen?", options=["No", "Yes"])
partner = st.selectbox(" Has a Partner?", options=["No", "Yes"])
dependents = st.selectbox("Has Dependents?", options=["No", "Yes"])

phone = st.selectbox("Phone Service?", options=["No", "Yes"])
multiple = st.selectbox("Multiple Lines?", options=["No", "Yes"])

internet = st.selectbox("Internet Service?", options=["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security?", options=["No", "Yes"])
online_backup = st.selectbox("Online Backup?", options=["No", "Yes"])
device_protection = st.selectbox("Device Protection?", options=["No", "Yes"])
tech_support = st.selectbox("Tech Support?", options=["No", "Yes"])
stream_tv = st.selectbox("Streaming TV?", options=["No", "Yes"])
stream_movies = st.selectbox("Streaming Movies?", options=["No", "Yes"])

contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing?", options=["No", "Yes"])
payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

