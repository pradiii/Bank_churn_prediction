import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from config import config
import os

from sklearn.model_selection import train_test_split

class oneHotEncoder:
    def __init__(self,one_hot_featues):
        self.one_hot_features =one_hot_featues
    
    def oneHot_fit_transform(self,data):
        return pd.get_dummies(data,columns=self.one_hot_features,dtype=int)
    

class labelEncoder:
    def __init__(self,label_encode_features):
        self.label_encode_features = label_encode_features
        self.label_encoders = {feature: LabelEncoder() for feature in label_encode_features}

    def label_fit_transform(self,data):
        for feature in self.label_encode_features:
            data[feature] = self.label_encoders[feature].fit_transform(data[feature])
        return data

class minMaxScaler:
    def __init__(self,min_max_scaler_features):
        self.min_max_scaler_features = min_max_scaler_features
        self.min_max_scaler = {feature: MinMaxScaler() for feature in min_max_scaler_features}

    def minmax_fit_transform(self,data):
        for feature in self.min_max_scaler_features:
            data[feature] = self.min_max_scaler[feature].fit_transform(data[feature].values.reshape(-1,1)).flatten()
        return data
    

class dataPreprocessing:
    def __init__(self,one_hot_featues,label_encode_features,min_max_scaler_features):
        self.one_hot_encode = oneHotEncoder(one_hot_featues)
        self.label_encode = labelEncoder(label_encode_features)
        self.min_max_scale = minMaxScaler(min_max_scaler_features)

    def fit_transfomm(self,data):
        data = self.one_hot_encode.fit_transform(data)  
        data = self.label_encode.fit_transform(data)
        data = self.min_max_scale.fit_transform(data)
    
        return data
    
    def drop_features(self,data):
        data = data.drop(config.DROP_FEATURES,axis=1)
        return data

    def save_processed_data(self,train_data,processed_files_path):
        os.makedirs(os.path.dirname(processed_files_path),exist_ok=True)
        train,test = train_test_split(train_data,train_size=config.TRAIN_DATA_SIZE)
        split_data_file_path = os.path.join(processed_files_path, 'train.csv')
        train.to_csv(split_data_file_path, index=False,mode='w')
        
        split_data_file_path = os.path.join(processed_files_path, 'test.csv')
        test.to_csv(split_data_file_path, index=False,mode='w')
        print("Pre-processed data saved..")
        

