from src.data.data_handling import dataHandling
from config import config
from src.data.data_preprocessing import dataPreprocessing


import src.training.train as mt

# from src.data.data_preprocessing as pp

class dataPrep:
    def __init__(self,config):
        self.config = config,
    def data_preparation():
        dh = dataHandling()
        train_data = dh.load_dataset(config.TRAIN_FILE)
        dp = dataPreprocessing(config.ONEHOT_ENCODE_FEATURE,config.LABEL_ECNCODE_FEATURE,
                                config.NUM_FEATURES)
        train_data = dp.one_hot_encode.oneHot_fit_transform(train_data)
        train_data = dp.label_encode.label_fit_transform(train_data)
        train_data = dp.min_max_scale.minmax_fit_transform(train_data)
        train_data = dp.drop_features(train_data)
        dp.save_processed_data(train_data,config.PROCESSED_DATASET_PATH)

if __name__ == '__main__':
    dataPrep.data_preparation()
    mt.modelTrain.train_model()