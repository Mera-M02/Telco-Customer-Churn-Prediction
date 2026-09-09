# Customer Churn Prediction Using Machine Learning In Telecommunication Industry

MSc Computing Project - Mera (Student ID - 35030037)

**Live application:** https://telco-customer-churn-prediction-mera.streamlit.app/

---

## Overview
Telecom customers can switch provider easily, and acquiring a new customer costs more than keeping an existing one.Many churn models focus mainly on predictive performance, but this does not always explain why an individual customer is predicted to churn or help decide who should be contacted.

This project trains an interpretable churn model, explains each prediction with SHAP, and sets the decision threshold from retention cost and customer lifetime value rather than the usual 0.5. It is deployed as a web application so the predictions and their reasons can be seen by the people who would use them.

---

## Aim & Objectives
1. Compare Logistic Regression, Decision Tree and Random Forest on the IBM Telco dataset.
2. Evaluate on recall, precision, F1 and ROC-AUC rather than accuracy alone.
3. Apply SHAP to explain individual predictions and compare against the churn reasons recorded in the data.
4. Derive the decision threshold from retention cost and customer lifetime value.
5. Validate the practical usefulness of the models’ output through (UAT) with end-users and get feedbacks.
6. Develop and deploy an interactive web application to present the predictions and explanations for user-evaluation

---


## Data Set

IBM Telco Customer Churn dataset with 7,043 customers, 33 columns. Published by IBM as a Cognos Analytics sample and publicly available on Kaggle. The dataset contains no personally identifiable information about real individuals.

https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

This version was chosen over the more common 21-column one because it records Customer Lifetime Value and a Churn Reason for every customer who left. Both are needed for the cost-based threshold and for checking the SHAP explanations against what customers actually said.

Columns that reveal the outcome (Churn Score, Churn Label) were removed before modelling, leaving 30 features.

---

## Tools and Techniques

| Stage | Tools |
|-------|-------|
| **Preprocessing** | Python, pandas, NumPy — cleaning, encoding and scaling |
| **Modelling** | scikit-learn (Logistic Regression, Decision Tree, Random Forest), imbalanced-learn for SMOTE, SHAP for explanations |
| **Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices |
| **Presentation and delivery** | Matplotlib, Streamlit, joblib, Google Colab, VS Code, Git and GitHub |

---

## Results

Held-out test set of 1,409 customers, 374 of whom churned.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.745 | 0.513 | **0.775** | 0.618 | 0.848 |
| Decision Tree | 0.744 | 0.516 | 0.564 | 0.539 | 0.686 |
| Random Forest | 0.788 | 0.599 | 0.607 | 0.603 | 0.833 |

Logistic Regression was selected on recall. Applying the business-derived threshold of 0.378 instead of 0.5 raises recall to 0.866 and increases the share of at-risk customer value captured from 75.8% to 85.9%, while increasing the proportion of customers flagged for retention from 40.1% to 48.3%.

**A note on validation.** Applying SMOTE before cross-validation rather than inside each fold had inflated Random Forest recall from 0.602 to 0.875. Once corrected, the cross-validated figures agree with the held-out test results to within 0.005.

---

## Repository Contents

| File | Purpose |
|------|---------|
| `Telco_Churn_Prediction.ipynb` | Full analysis: preparation, modelling, tuning, SHAP, threshold, tests |
| `churn_features.py` | Shared preprocessing which turns form inputs into the 30 features the model expects |
| `churn_prediction_app.py` | Streamlit web application |
| `churn_model.pkl` | Trained Logistic Regression model |
| `scaler.pkl` | Fitted StandardScaler |
| `feature_names.pkl` | The 30 feature names, in training order |
| `threshold.pkl` | Business-derived decision threshold |
| `shap_background.pkl` | Reference data for SHAP explanations |
| `requirements.txt` | Dependencies |

`churn_features.py` is imported by both the application and the notebook's test cases. Keeping the encoding in one place means the application cannot end up preparing data differently from the way the model was trained.

---

## Running It Locally

```bash
git clone https://github.com/Mera-M02/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run churn_prediction_app.py
```

The application opens in your browser. Enter details for a customer and it returns a churn prediction, the probability, and the five features that most influenced the result.

---

## Testing

Twelve automated tests run in the notebook. They check that the application produces exactly the same feature encoding as the training pipeline, using five real customers chosen to cover the cases most likely to go wrong: no phone service, no internet service, and each of the three contract types. One test confirms that a feature name the model does not recognise raises an error rather than being silently ignored.

All twelve tests pass.

---

## Limitations

The recorded churn reasons show that 34.8% of customers left because of competitor offers, and the dataset holds no feature for competitor activity at all. This limits how well a model built from these features can explain competitor-related churn.

Some of the customers the model misses are high-value customers, so the prediction should not be used on its own when deciding who to target.

---
