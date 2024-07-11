from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
from config import config

class onehotEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
        self._categories = {}

    def fit(self,X,y=None):
        for var in self.variables:
            self._categories[var] = np.unique(X[var])

        return self
    
    def transform(self,X):
        X = X.copy()

        for var in self.variables:
            #Create one hot encoded column for each category.
            for category in self._categories[var]:
                X[f"{var}_{category}"] =  (X[var]==category).astype(int)
            X.drop(columns = [var],inplace=True)

        return X

class lableEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
        self.label_dict = {}

    def fit(self,X,y=None):
        self.label_dict = {}
        
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}

        print("label dict in fit",self.label_dict)
        return self

    def transform(self,X):
        X = X.copy()
        print('label_dict in transform ',self.label_dict)
        print("label encode feature:",config.LABEL_ECNCODE_FEATURE)
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X


class minmaxScaler(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None,feature_range=(0,1)):
        self.variables = variables
        self.feature_range = feature_range

    def fit(self,X,y=None):
        self.min_ = {}
        self.max_ = {}

        for var in self.variables:
            self.min_[var] = X[var].min()
            self.max_[var] = X[var].max()
        
        return self
    
    def transform(self,X):
        X = X.copy()
        for var in self.variables:
            X[var] = (X[var]-self.min_[var]/self.max_[var]-self.min_[var])
            X[var] = X[var] * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]

        return X


class dropColumn(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables  = variables

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X = X.drop(columns = self.variables)

        return X 
