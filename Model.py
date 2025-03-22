import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

class DataPreprocessor:
    def __init__(self):
        self.ohe = None
        self.scaler = None
        self.categorical_cols = None
        self.numeric_cols = None

    def fit(self, X: pd.DataFrame):
        self.categorical_cols = X.select_dtypes(include='object').columns.tolist()
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.scaler = MinMaxScaler()
        if self.categorical_cols:
            self.ohe.fit(X[self.categorical_cols])
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_num = self.scaler.transform(X[self.numeric_cols]) if self.numeric_cols else None
        X_cat = self.ohe.transform(X[self.categorical_cols]) if self.categorical_cols else None
        if X_num is not None and X_cat is not None:
            return np.hstack([X_num, X_cat])
        elif X_num is not None:
            return X_num
        elif X_cat is not None:
            return X_cat
        else:
            return np.array([])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=0):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.classes_ = None

    def train(self, X_train: np.ndarray, y_train):
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

    def predict(self, X_new: np.ndarray):
        return self.model.predict(X_new)
    
    def predict_proba(self, X_new: np.ndarray):
        return self.model.predict_proba(X_new)
