import pandas as pd
import numpy as np
import os
import sys
import mlflow

from config import config
from src.data.data_handling import dataHandling,pipelineHandling
import src.data.data_preprocessing as dp

#Need to load model
pl = pipelineHandling()
dh = dataHandling()
classification_model = pl.load_pipeline(config.MODEL_NAME)

def generate_prediction():
    data = dh.load_dataset(config.TEST_FILE)
    model_params = classification_model.get_params()

    with mlflow.start_run():

        mlflow.sklearn.log_model(classification_model,"model")
        mlflow.log_params(model_params)
        pred = classification_model.predict(data)
        # mlflow.log_param("test_file",config.TEST_FILE)
        # mlflow.log_artifact(config.TEST_FILE)
        mlflow.log_dict({"predictions":pred.tolist()},"predictions.json")
    print("predictions are ",pred)
    return pred

if __name__== "__main__":
    generate_prediction()

