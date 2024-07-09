import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config
from src.data.data_handling import dataHandling,pipelineHandling
from src.pipeline import classification_pipeline

class Trainer:
    def __init__(self,config):
        self.config = config

    # def load_data(self):
    #     raw_data = dataHandling.load_dataset(self,config.TRAIN_FILE)
    #     train_X = raw_data.drop(config.TARGET_FEATURE,axis=1)
    #     train_y = raw_data[config.TARGET_FEATURE].values.ravel()

    #     return train_X,train_y


    def train_model(self):
        raw_data = dataHandling.load_dataset(self,config.TRAIN_FILE)
        train_X = raw_data.drop(config.TARGET_FEATURE,axis=1)
        train_y = raw_data[config.TARGET_FEATURE].values.ravel()
        print('Data loaded')
        classification_pipeline.fit(train_X,train_y)
        ph = pipelineHandling()
        ph.save_pipeline(classification_pipeline)


if __name__ == "__main__":
    tr = Trainer(config=config)
    tr.train_model()






