import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config
from sklearn.pipeline import Pipeline
import src.data.data_preprocessing as dp
from src.models import model as md

classification_pipeline  = Pipeline(
    [
        ("dropColumn",dp.dropColumn(variables=config.DROP_FEATURES)),
        ("onehotEncoder" ,dp.onehotEncoder(variables=config.ONEHOT_ENCODE_FEATURE)),
        ("lableEncoder",dp.lableEncoder(variables=config.LABEL_ECNCODE_FEATURE)),
        ("minMaxScaler",dp.minmaxScaler(variables=config.NUM_FEATURES)),
        ("logistic_regression_model",md.logistic_regression_model())
    ]
)
