import streamlit as st
import pandas as pd
import joblib

#the notebook tests import this same file so the encoding can't go out of sync
from churn_features import build_features, FORM_OPTIONS

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Business-derived decision threshold from Section 16B of the notebook.
try:
    THRESHOLD = joblib.load('threshold.pkl')
except Exception:
    THRESHOLD = 0.5

#SHAP needs this to explain a logistic regression, the app still runs
try:
    shap_background = joblib.load('shap_background.pkl')
except Exception:
    shap_background = None

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer is likely "
         "to churn or not, and see the main factors behind the prediction.")
st.header("Customer Details")

tenure = st.number_input("Tenure (months with the company)",
                         min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges",
                                  min_value=0.0, max_value=2000.0, value=70.00)
total_charges = st.number_input("Total Charges",
                                min_value=0.0, max_value=100000.0, value=800.00)

# Options come from the shared module so the form and the encoding stay in step.
gender = st.selectbox("Gender", options=FORM_OPTIONS["gender"])
senior = st.selectbox("Senior Citizen?", options=FORM_OPTIONS["senior"])
partner = st.selectbox("Has a Partner?", options=FORM_OPTIONS["partner"])
dependents = st.selectbox("Has Dependents?", options=FORM_OPTIONS["dependents"])

phone = st.selectbox("Phone Service?", options=FORM_OPTIONS["phone"])
multiple = st.selectbox("Multiple Lines?", options=FORM_OPTIONS["multiple"])

internet = st.selectbox("Internet Service?", options=FORM_OPTIONS["internet"])
online_security = st.selectbox("Online Security?", options=FORM_OPTIONS["online_security"])
online_backup = st.selectbox("Online Backup?", options=FORM_OPTIONS["online_backup"])
device_protection = st.selectbox("Device Protection?", options=FORM_OPTIONS["device_protection"])
tech_support = st.selectbox("Tech Support?", options=FORM_OPTIONS["tech_support"])
stream_tv = st.selectbox("Streaming TV?", options=FORM_OPTIONS["stream_tv"])
stream_movies = st.selectbox("Streaming Movies?", options=FORM_OPTIONS["stream_movies"])

contract = st.selectbox("Contract Type", options=FORM_OPTIONS["contract"])
paperless = st.selectbox("Paperless Billing?", options=FORM_OPTIONS["paperless"])
payment_method = st.selectbox("Payment Method", options=FORM_OPTIONS["payment_method"])

# Total charges should be roughly tenure multiplied by monthly charges
if tenure > 0:
    expected_total = tenure * monthly_charges
    if expected_total > 0 and not (0.5 * expected_total <= total_charges <= 1.5 * expected_total):
        st.warning(
            f"Total Charges of {total_charges:,.2f} looks inconsistent with "
            f"{tenure} months at {monthly_charges:,.2f} per month "
            f"(roughly {expected_total:,.2f} expected). The prediction may be unreliable."
        )

if st.button("Predict Churn"):
    inputs = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "gender": gender,
        "senior": senior,
        "partner": partner,
        "dependents": dependents,
        "phone": phone,
        "multiple": multiple,
        "internet": internet,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "stream_tv": stream_tv,
        "stream_movies": stream_movies,
        "contract": contract,
        "paperless": paperless,
        "payment_method": payment_method,
    }

    X_input = build_features(inputs, feature_names)
    X_scaled = scaler.transform(X_input)

    probability = model.predict_proba(X_scaled)[0][1]
    prediction = int(probability >= THRESHOLD)

    st.header("Prediction Result")
    if prediction == 1:
        st.error(f"The customer is likely to churn. "
                 f"(Churn Probability: {probability:.0%})")
    else:
        st.success(f"The customer is not likely to churn. "
                   f"(Churn Probability: {probability:.0%})")
    st.caption(f"Flagged when churn probability is {THRESHOLD:.0%} or higher. "
               f"This threshold is derived from retention cost and customer "
               f"lifetime value rather than the conventional 50%.")

    st.subheader("Main Factors Behind the Prediction")
    try:
        import shap
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
            explainer = shap.TreeExplainer(model)
        elif shap_background is not None:
            masker = shap.maskers.Independent(
                shap_background, max_samples=shap_background.shape[0])
            explainer = shap.LinearExplainer(model, masker)
        else:
            raise FileNotFoundError(
                "shap_background.pkl is required to explain a linear model.")

        shap_values = np.array(explainer.shap_values(X_scaled))
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    
        #pair each feature with its impact and show top 5 by size
        feature_impact = pd.Series(shap_values[0], index=feature_names)
        top_features = feature_impact.reindex(
            feature_impact.abs().sort_values(ascending=False).index
        ).head(5)

        st.write("These factors had the most impact on the prediction "
                 "(positive = pushes towards churn, negative = pushes away from churn):")

        for feature, value in top_features.items():
            direction = "towards churn" if value > 0 else "away from churn"
            st.write(f"**{feature}**: {value:+.3f} ({direction})")

    except Exception as e:
        st.warning(f"SHAP analysis could not be performed: {e}")