import streamlit as st
import pandas as pd
import joblib

# Load model and scaler

model = joblib.load('naive_bayes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Naive Bayes - Bank Customer Churn Prediction')

# Inputs

credit_score = st.number_input('Credit Score', 300, 900, 650)
age = st.number_input('Age', 18, 100, 30)
balance = st.number_input('Balance', 0.0, 200000.0, 50000.0)
salary = st.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)

if st.button('Predict'):


# Default values for remaining features
input_data = pd.DataFrame([[credit_score, 0, 1, age, 5, balance, 1, 1, 1, salary]],
                          columns=['CreditScore', 'Geography', 'Gender', 'Age',
                                   'Tenure', 'Balance', 'NumOfProducts',
                                   'HasCrCard', 'IsActiveMember',
                                   'EstimatedSalary'])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

if prediction[0] == 1:
    st.error('Customer will Exit')
else:
    st.success('Customer will Stay')


st.write('Developed by Achal Chaudhari')
