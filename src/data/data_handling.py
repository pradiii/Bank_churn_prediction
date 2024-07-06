import pandas as pd
import os
import joblib
from config import config
from sklearn.model_selection import train_test_split


class dataHandling:
    def __init__(self):
        self.config = config

    def load_dataset(self,file_name):
        print('reading path')
        file_path  = os.path.join(config.RAW_DATASET_PATH,file_name)
        _data = pd.read_csv(file_path)
        return _data

    def load_preprocessed_dataset(self,file_name):
        print('reading path')
        file_path  = os.path.join(config.PROCESSED_DATASET_PATH,file_name)
        _data = pd.read_csv(file_path)
        return _data
    
    def save_processed_data(self,train_data,processed_files_path):
        os.makedirs(os.path.dirname(processed_files_path),exist_ok=True)
        train,test = train_test_split(train_data,train_size=config.TRAIN_DATA_SIZE)
        split_data_file_path = os.path.join(processed_files_path, 'train.csv')
        train.to_csv(split_data_file_path, index=False,mode='w')
        split_data_file_path = os.path.join(processed_files_path, 'test.csv')
        test.to_csv(split_data_file_path, index=False,mode='w')
        print("Pre-processed data saved..")

class pipelineHandling:
    def __init__(self):
        self.config = config

    def save_pipeline(self,pipeline_to_save):
        save_path = os.path.join(config.MODEL_SAVE_PATH,config.MODEL_NAME)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        joblib.dump(pipeline_to_save,save_path)
        print(f"Model have been saved under name {config.MODEL_NAME}")

    def load_pipeline(self,pipeline_to_load):
        load_path  = os.path.join(config.MODEL_SAVE_PATH,pipeline_to_load)
        print('load path',load_path)
        try:
            _model_loaded = joblib.load(load_path)
            return _model_loaded
        except FileNotFoundError:
            print(f"Error: File '{load_path}' not found.")
            return None
        except PermissionError:
            print(f"Error: Permission denied for file '{load_path}'.")
            return None




