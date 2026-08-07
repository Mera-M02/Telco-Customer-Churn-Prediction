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

# turn the inputs into 30 features for the model

def build_features():
    #start every feature at 0, then set the ones that apply to 1
    row = dict.fromkeys(feature_names, 0)

    #the three numeric features
    row['Tenure Months'] = tenure
    row['Monthly Charges'] = monthly_charges
    row['Total Charges'] = total_charges

    #simplify the categorical features into binary features
    if gender == "Male": row['Gender_Male'] = 1
    if senior == "Yes": row['Senior Citizen_Yes'] = 1
    if partner == "Yes": row['Partner_Yes'] = 1
    if dependents == "Yes": row['Dependents_Yes'] = 1
    if phone == "Yes": row['Phone Service_Yes'] = 1

    #multiple lines depends on phone service
    if phone == "No":
        row['Multiple Lines_Yes'] = 1
    elif multiple == "Yes":
        row['Multiple Lines_Yes'] = 1

    #internet service type
    if internet == "Fiber optic":
        row['Internet Service_Fiber optic'] = 1
    elif internet == "No":
        row['Internet Service_No'] = 1
    
    #the six services that depend on internet service
    services = {
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': stream_tv,
        'streaming_movies': stream_movies
    }

    for name, value in services.items():
        if internet == "No":
            row[f"{name}_No internet service"] = 1
        elif value == "Yes":
            row[f"{name}_Yes"] = 1

    #contract type
    if contract == "One year":
        row['Contract_One year'] = 1
    elif contract == "Two year":
        row['Contract_Two year'] = 1 
    
    #paperless billing
    if paperless == "Yes":
        row['Paperless Billing_Yes'] = 1
    
    #payment method
    if payment_method == "credit card (automatic)":
        row['Payment Method_Credit card (automatic)'] = 1
    elif payment_method == "electronic check":
        row['Payment Method_Electronic check'] = 1
    elif payment_method == "mailed check":
        row['Payment Method_Mailed check'] = 1

    return pd.DataFrame([row])[feature_names]  # ensure the order of columns matches the training data

# Predict button
if st.button("Predict Churn"):
    # build and scale the input same as training data
    X_input = build_features()
    X_scaled = scaler.transform(X_input)

    # make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]  # probability of churn

    # Display the result
    st.header("Prediction Result")
    if prediction == 1:
        st.error(f"The customer is likely to churn."
                f"(Churn Probability: {probability:.0%})")
    else:
        st.success(f"The customer is not likely to churn."
                f"(Churn Probability: {probability:.0%})")

#show the main factors behind the prediction
st.subheader("Main Factors Behind the Prediction")
try:
    import shap
    import numpy as np
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(X_scaled))
    if shap_values.ndim == 3:  # For multi-class classification, take the values for the positive class
        shap_values = shap_values[:,:,1]

    #pair each feature with its impact and show top 5 by size
    feature_impact = pd.Series(shap_values[0], index=feature_names)
    top_features = feature_impact.reindex(feature_impact.abs().sort_values(ascending=False).index).head(5)

    st.write("These factors had the most impact on the prediction - ( positive = pushes towards churn, negative = pushes away from churn ):")
    
    for feature, value in top_features.items():
        direction = "towards churn" if value > 0 else "away from churn"
        st.write(f"{feature}: ({direction})")
except Exception as e:
    st.write("(SHAP analysis could not be performed.", e,")")
