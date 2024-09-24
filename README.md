# Customer Churn Prediction App

This project is a **Streamlit-based web application** that predicts whether a customer is likely to churn based on various input features. The model is built using **TensorFlow** and includes preprocessing with **sklearn's StandardScaler, LabelEncoder, and OneHotEncoder**. The app allows users to enter specific details about a customer, and the model predicts the probability of churn.

## Project Structure

- `model.keras`: The trained TensorFlow model used for churn prediction.
- `le.pkl`: A saved `LabelEncoder` object for encoding categorical features like "Gender."
- `ohe.pkl`: A saved `OneHotEncoder` object for encoding the "Geography" feature.
- `scaler.pkl`: A saved `StandardScaler` object for scaling the input data.
- `app.py`: The Streamlit app where users can input customer details and get a churn prediction.

## Features

This app takes the following inputs:
- **Geography**: The customer's location (France, Spain, Germany, etc.)
- **Gender**: The customer's gender (Male or Female).
- **Age**: The age of the customer.
- **Balance**: The customer's account balance.
- **Credit Score**: The credit score of the customer.
- **Estimated Salary**: The customer's estimated salary.
- **Tenure**: The number of years the customer has been with the bank.
- **Number of Products**: The number of bank products used by the customer (1-4).
- **Has Credit Card**: Whether the customer has a credit card (Yes/No).
- **Is Active Member**: Whether the customer is an active member (Yes/No).

The app then uses these inputs to predict:
- **Churn Probability**: The probability that the customer will churn (leave the service).
- A recommendation on whether the customer is likely to churn.

## Setup and Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/MARKST-47/ann-classification-churn.git
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then, install the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the necessary model and preprocessing objects:**
   Ensure that `model.keras`, `le.pkl`, `ohe.pkl`, and `scaler.pkl` are available in the project directory.

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501` to interact with the application.

## How It Works

- **Model Loading**: The trained TensorFlow model and preprocessing objects (LabelEncoder, OneHotEncoder, and StandardScaler) are loaded when the app starts.
- **User Input**: The user provides information about the customer through dropdowns, sliders, and number inputs.
- **Prediction**: After preprocessing the input, the model predicts the probability of churn.
- **Output**: The app displays the churn probability and a recommendation based on the model's prediction.

## Dependencies

- `tensorflow`
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `pickle`

You can install these using `pip install -r requirements.txt`.

## Future Improvements

- Add more input features for better predictions.
- Incorporate advanced visualizations using Streamlit.
- Implement a feedback loop for improving model performance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
