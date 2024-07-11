# Importing Dependencies
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
from config import config
from src.data.data_handling import pipelineHandling
from fastapi import FastAPI, HTTPException

app = FastAPI()

pl  = pipelineHandling()
model = pl.load_pipeline(config.MODEL_NAME)

#Perform parsing
class ChurnPred(BaseModel):
	id : float
	CustomerId : int
	Surname : str
	CreditScore: float
	Geography: str
	Gender: str
	Age: float
	Tenure: float
	Balance: float
	NumOfProducts: float
	HasCrCard: float
	IsActiveMember: float
	EstimatedSalary: float

@app.get('/')
def index():
	return {"Message:Welcome to bank churn prediction app"}

# @app.post("/predict")
# def predict_bank_churn(candidate_details:ChurnPred):
# 	data = candidate_details.dict()

# 	id = data['id']
# 	CustomerId = data['CustomerId']
# 	Surname = data['Surname']
# 	CreditScore = data['CreditScore']
# 	Geography= data['Geography']
# 	Gender= data['Gender']
# 	Age = data['Age']
# 	Tenure = data['Tenure']
# 	Balance = data['Balance']
# 	NumOfProducts = data['NumOfProducts']
# 	HasCrCard= data['HasCrCard']
# 	IsActiveMember= data['IsActiveMember']
# 	EstimatedSalary= data['EstimatedSalary']

# 	df = pd.DataFrame([[
#         id,CustomerId,Surname,CreditScore, Geography, Gender, Age, Tenure,
# 		Balance, NumOfProducts, HasCrCard, IsActiveMember,EstimatedSalary
#     ]], columns=[
#         'id','CustomerId','Surname','CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
# 		'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember','EstimatedSalary'
#     ])
# 	pred = model.predict(df)

# 	return {'Status of Loan Application':pred[0]}

# if __name__ == '__main__':
# 	uvicorn.run(app, host='127.0.0.1', port=8000)

@app.post("/predict")
def predict_bank_churn(candidate_details: ChurnPred):
    try:
        data = candidate_details.dict()

        df = pd.DataFrame([[
            data['id'], data['CustomerId'], data['Surname'], data['CreditScore'], data['Geography'],
            data['Gender'], data['Age'], data['Tenure'], data['Balance'], data['NumOfProducts'],
            data['HasCrCard'], data['IsActiveMember'], data['EstimatedSalary']
        ]], columns=[
            'id', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ])

        pred = model.predict(df)

        return {'Status of Loan Application': pred[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)