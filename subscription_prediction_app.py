import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load the trained model and encoders
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define the Streamlit app
st.title('Banking Subscription Prediction App')

st.write("""
Enter the customer details below to predict subscription:
""")

# Define user input fields
age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                           'management', 'retired', 'self-employed', 'services',
                           'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
education = st.selectbox('Education', ['basic', 'high.school', 'illiterate', 'professional.course',
                                       'university.degree', 'unknown'])
default = st.selectbox('Default', ['no', 'yes', 'unknown'])
housing = st.selectbox('Housing Loan', ['no', 'yes', 'unknown'])
loan = st.selectbox('Personal Loan', ['no', 'yes', 'unknown'])
contact = st.selectbox('Contact', ['cellular', 'telephone'])
month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input('Duration', min_value=0, max_value=5000, value=100)
campaign = st.number_input('Campaign', min_value=1, max_value=100, value=1)
pdays = st.selectbox('Last contact date',['Recently','Not contacted'])
previous = st.number_input('Previous', min_value=0, max_value=100, value=0)
poutcome = st.selectbox('Poutcome', ['failure', 'nonexistent', 'success'])
# emp_var_rate = st.number_input('Emp Var Rate', value=1.1)
# cons_price_idx = st.number_input('Cons Price Index', value=93.994)
# cons_conf_idx = st.number_input('Cons Conf Index', value=-36.4)
# euribor3m = st.number_input('Euribor 3M', value=4.857)
# nr_employed = st.number_input('Nr Employed', value=5191)

# Prepare the input data for prediction
user_data = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
    # 'emp_var_rate': emp_var_rate,
    # 'cons_price_idx': cons_price_idx,
    # 'cons_conf_idx': cons_conf_idx,
    # 'euribor3m': euribor3m,
    # 'nr_employed': nr_employed
}

# Show predict button
if st.button('Predict'):
    user_df = pd.DataFrame(user_data, index=[0])

    # Use the feature names that were used to fit the encoder
    categorical_features = encoder.feature_names_in_

    # Apply OneHotEncoder to categorical features
    user_df_encoded = pd.DataFrame(encoder.transform(user_df[categorical_features]).toarray(),
                                   columns=encoder.get_feature_names_out())

    # Scale numerical features
    numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
                          'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']

    # user_df_scaled = pd.DataFrame(scaler.transform(user_df[numerical_features]), columns=numerical_features)

    # Combine scaled numerical features and encoded categorical features
    # user_df_final = pd.concat([user_df_scaled, user_df_encoded], axis=1)

    # Align user_df_final with model input features
    # user_df_final = user_df_final.reindex(columns=model.feature_names_in_, fill_value=0)
    user_df_final = user_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make predictions
    prediction = model.predict(user_df_final)
    prediction_proba = model.predict_proba(user_df_final)[:, 1]

    # Display the prediction result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.write(f"The customer is likely to subscribe with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.write(f"The customer is unlikely to subscribe with a probability of {1 - prediction_proba[0]:.2f}.")
