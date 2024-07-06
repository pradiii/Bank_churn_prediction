import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config
from src.data.data_handling import dataHandling
from src.data.data_handling import pipelineHandling,dataHandling
from sklearn.metrics import classification_report,accuracy_score


class modelEvaluate:
    def __init__(self,config):
        self.config = config

    def evaluate_model():
        ph = pipelineHandling()
        print('model_name',config.MODEL_NAME)
        model = ph.load_pipeline(config.MODEL_NAME)
        dh =  dataHandling()
        test_data = dh.load_preprocessed_dataset(config.TEST_FILE)
        test_X = test_data.drop(config.TARGET_FEATURE,axis=1)
        test_y = test_data[config.TARGET_FEATURE]
        test_preds =  model.predict(test_X)
        accuracy = accuracy_score(test_y,test_preds)
        class_report = classification_report(test_y,test_preds)
        print("accuracy",accuracy)
        print("classification_report",class_report)
        
        return accuracy

if __name__ =="__main__":
    modelEvaluate.evaluate_model()
