import pandas as pd
import numpy as np
import os
import sys

from config import config
from src.data.data_handling import dataHandling,pipelineHandling
import src.data.data_preprocessing as dp

#Need to load model
pl = pipelineHandling()
dh = dataHandling()
classification_model = pl.load_pipeline(config.MODEL_NAME)

def generate_prediction():
    data = dh.load_dataset(config.TEST_FILE)
    print("classification model running")

    pred = classification_model.predict(data)
    print("predictions are ready")
    print(pred)
    return pred

if __name__== "__main__":
    generate_prediction()