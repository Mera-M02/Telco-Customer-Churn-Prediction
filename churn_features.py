import pandas as pd

# The options offered by the application's form. Kept here so the encoding
# logic below and the widgets in the app cannot fall out of step.
FORM_OPTIONS = {
    "gender": ["Male", "Female"],
    "senior": ["No", "Yes"],
    "partner": ["No", "Yes"],
    "dependents": ["No", "Yes"],
    "phone": ["No", "Yes"],
    "multiple": ["No", "Yes"],
    "internet": ["DSL", "Fiber optic", "No"],
    "online_security": ["No", "Yes"],
    "online_backup": ["No", "Yes"],
    "device_protection": ["No", "Yes"],
    "tech_support": ["No", "Yes"],
    "stream_tv": ["No", "Yes"],
    "stream_movies": ["No", "Yes"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless": ["No", "Yes"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# The six add-on services that are only available with internet, mapped from
# the input key used in the form to the column prefix used in training.
INTERNET_SERVICES = {
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "stream_tv": "Streaming TV",
    "stream_movies": "Streaming Movies",
}


def build_features(inputs, feature_names):
    """Convert the form input into a DataFrame suitable for the trained model."""
    row = dict.fromkeys(feature_names, 0)

    # Numeric features pass through unchanged
    row["Tenure Months"] = inputs["tenure"]
    row["Monthly Charges"] = inputs["monthly_charges"]
    row["Total Charges"] = inputs["total_charges"]

    # Simple binary features. The omitted category is the reference level
    # dropped by get_dummies(drop_first=True) during training.
    if inputs["gender"] == "Male":
        row["Gender_Male"] = 1
    if inputs["senior"] == "Yes":
        row["Senior Citizen_Yes"] = 1
    if inputs["partner"] == "Yes":
        row["Partner_Yes"] = 1
    if inputs["dependents"] == "Yes":
        row["Dependents_Yes"] = 1
    if inputs["phone"] == "Yes":
        row["Phone Service_Yes"] = 1

    # Multiple lines depends on whether there is phone service at all
    if inputs["phone"] == "No":
        row["Multiple Lines_No phone service"] = 1
    elif inputs["multiple"] == "Yes":
        row["Multiple Lines_Yes"] = 1

    # Internet service type. DSL is the reference category.
    if inputs["internet"] == "Fiber optic":
        row["Internet Service_Fiber optic"] = 1
    elif inputs["internet"] == "No":
        row["Internet Service_No"] = 1

    # Add-on services. Without internet, every one takes the
    # 'No internet service' level rather than 'No'.
    for key, column in INTERNET_SERVICES.items():
        if inputs["internet"] == "No":
            row[f"{column}_No internet service"] = 1
        elif inputs[key] == "Yes":
            row[f"{column}_Yes"] = 1

    # Contract type. Month-to-month is the reference category.
    if inputs["contract"] == "One year":
        row["Contract_One year"] = 1
    elif inputs["contract"] == "Two year":
        row["Contract_Two year"] = 1

    if inputs["paperless"] == "Yes":
        row["Paperless Billing_Yes"] = 1

    # Payment method. Bank transfer (automatic) is the reference category.
    if inputs["payment_method"] == "Credit card (automatic)":
        row["Payment Method_Credit card (automatic)"] = 1
    elif inputs["payment_method"] == "Electronic check":
        row["Payment Method_Electronic check"] = 1
    elif inputs["payment_method"] == "Mailed check":
        row["Payment Method_Mailed check"] = 1

    unexpected = set(row) - set(feature_names)
    if unexpected:
        raise KeyError(
            f"Feature names do not match the trained model: {sorted(unexpected)}"
        )

    return pd.DataFrame([row])[feature_names]


def inputs_from_record(record):
    """Convert a record from the original dataset into the form input format."""
    return {
        "tenure": record["Tenure Months"],
        "monthly_charges": record["Monthly Charges"],
        "total_charges": record["Total Charges"],
        "gender": record["Gender"],
        "senior": record["Senior Citizen"],
        "partner": record["Partner"],
        "dependents": record["Dependents"],
        "phone": record["Phone Service"],
        "multiple": "Yes" if record["Multiple Lines"] == "Yes" else "No",
        "internet": record["Internet Service"],
        "online_security": "Yes" if record["Online Security"] == "Yes" else "No",
        "online_backup": "Yes" if record["Online Backup"] == "Yes" else "No",
        "device_protection": "Yes" if record["Device Protection"] == "Yes" else "No",
        "tech_support": "Yes" if record["Tech Support"] == "Yes" else "No",
        "stream_tv": "Yes" if record["Streaming TV"] == "Yes" else "No",
        "stream_movies": "Yes" if record["Streaming Movies"] == "Yes" else "No",
        "contract": record["Contract"],
        "paperless": record["Paperless Billing"],
        "payment_method": record["Payment Method"],
    }