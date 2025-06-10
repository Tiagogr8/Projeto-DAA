import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

class DataPreparation:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.categorical_cols = []
        self.numeric_cols = []

    def fit(self, train_db):
        self.categorical_cols = train_db.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_cols = [col for col in self.categorical_cols if col != "Transition"]
        train_db = train_db.drop(columns=self.categorical_cols)

        self.numeric_cols = train_db.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.scaler.fit(train_db[self.numeric_cols])
        return train_db

    def transform(self, db):
        db = db.drop(columns=self.categorical_cols, errors='ignore')
        for coluna in self.numeric_cols:
            if coluna in db.columns:
                db[coluna] = db[coluna].astype(float)
        db[self.numeric_cols] = self.scaler.transform(db[self.numeric_cols])
        return db

    def balance_data(self, train_db):
        X = train_db.drop(columns=['Transition'])
        y = train_db['Transition']
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        train_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        train_resampled['Transition'] = y_resampled
        return train_resampled

    def prep_train(self, train_db):
        train_db = self.fit(train_db)
        train_db = self.transform(train_db)
        train_db = self.balance_data(train_db)
        return train_db