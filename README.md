# Banking Subscription Prediction App

This project provides a predictive model and a Streamlit web application for predicting whether a customer will subscribe to a term deposit based on their profile and previous marketing interactions. 

### Overview

The project utilizes a logistic regression model trained on a banking dataset to predict customer subscriptions. The dataset includes various features such as age, job, marital status, education, previous marketing outcomes, and more. The model aims to help banks and financial institutions target potential customers more effectively.

## Features

- **Data Preprocessing**: Handling categorical and numerical features using One-Hot Encoding and Standardization.
- **Model Training**: A logistic regression model is trained to predict the likelihood of subscription.
- **Model Evaluation**: Evaluation metrics like accuracy, precision, recall, F1-score, and ROC-AUC are used to assess the model's performance.
- **Web Application**: A Streamlit app that allows users to input customer data and get subscription predictions.

### Files
- app.py: The main Streamlit application file.
- logistic_regression_model.pkl: The trained logistic regression model.
- scaler.pkl: The fitted standard scaler for numerical features.
- encoder.pkl: The fitted One-Hot Encoder for categorical features.
- requirements.txt: Lists all the dependencies required for the project.

### Model Evaluation
The model achieved the following performance metrics on the test dataset:

- Accuracy: 91.09%
- Precision: 66.08%
- Recall: 41.00%
- F1 Score: 50.61%
- ROC AUC: 0.XX (replace with actual value)
