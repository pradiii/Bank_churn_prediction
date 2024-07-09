# Importing Dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model_name  = "Bank_churn_classification.pkl"
model = joblib.load(model_name)

#Perform parsing
class ChurnPred(BaseModel):
	CreditScore: float
	Age: float
	Tenure: float
	Balance: float
	NumOfProducts: float
	HasCrCard: float
	IsActiveMember: float
	EstimatedSalary: float
	Geography: float
	Gender: float
	