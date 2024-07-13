
from pydantic import BaseModel
import numpy as np
import pandas as pd
from config import config
from src.data.data_handling import pipelineHandling
import sys
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)

pl  = pipelineHandling()
model = pl.load_pipeline(config.MODEL_NAME)

# Prediction function
def prediction(input_data):
    pred = model.predict(input_data)
    return {'Status of Loan Application': pred[0]}

def main():
    st.title("Loan Application Prediction")
    id = st.text_input("ID")
    CustomerId = st.text_input("Customer ID")
    Surname = st.text_input("Surname")
    CreditScore = st.number_input("Credit Score", min_value=0, max_value=1000)
    Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=0)
    Tenure = st.number_input("Tenure", min_value=0)
    Balance = st.number_input("Balance", min_value=0.0)
    NumOfProducts = st.number_input("Number of Products", min_value=0)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0)

    input_data = pd.DataFrame({
        'id': [id],
        'CustomerId': [CustomerId],
        'Surname': [Surname],
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    if 'Unnamed: 0' in input_data.columns:
        input_data.drop(columns=['Unnamed: 0'], inplace=True)

    if st.button("Predict"):
        result = prediction(input_data)
        st.write(result)

if __name__ == "__main__":
    main()
