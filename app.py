import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the model and preprocessing objects
try:
    model = tf.keras.models.load_model("model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open("le.pkl", "rb") as file:
        le = pickle.load(file)

    with open("ohe.pkl", "rb") as file:
        ohe = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading preprocessing objects: {e}")

st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", ohe.categories_[0])
gender = st.selectbox("Gender", le.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure (years)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])

# Prepare input data
input_data = {
    "CreditScore": [credit_score],
    "Gender": [le.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
}

# One-hot encode geography
geo_encoder = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoder, columns=ohe.get_feature_names_out(["Geography"])
)

# Combine input data into a DataFrame
input_data_df = pd.DataFrame(input_data)
input_data_combined = pd.concat(
    [input_data_df.reset_index(drop=True), geo_encoded_df], axis=1
)

# Scale the input data
input_data_scaled = scaler.transform(input_data_combined)

# Make prediction
try:
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
except Exception as e:
    st.error(f"Error during prediction: {e}")
    prediction_proba = None

# Display results
if prediction_proba is not None:
    st.write(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
