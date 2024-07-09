import os
from pathlib import Path
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = os.path.join(PACKAGE_ROOT,'data','raw')
PROCESSED_DATASET_PATH = os.path.join(PACKAGE_ROOT,'data','processed')

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_NAME = "Bank_churn_classification.pkl"
MODEL_SAVE_PATH  = os.path.join(PACKAGE_ROOT,'models')

FEATURES = ['CreditScore','Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard','IsActiveMember', 'EstimatedSalary']

# CAT_FEATURES = ['Geography','Gender']
ONEHOT_ENCODE_FEATURE = ['Geography']
LABEL_ECNCODE_FEATURE = ['Gender']

NUM_FEATURES = ['CreditScore','Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard','IsActiveMember', 'EstimatedSalary']

TARGET_FEATURE  = ['Exited']

DROP_FEATURES = ['id', 'CustomerId', 'Surname']

TRAIN_DATA_SIZE = 0.7

MODEL_PARAMS = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}