import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numeric_cols = []

    def fit(self, train_db):
        self.categorical_cols = train_db.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_cols = [col for col in self.categorical_cols if col != "Transition"]
        train_db = train_db.drop(columns=self.categorical_cols)

        if 'Transition' in train_db.columns:
            train_db['Transition'] = train_db['Transition'].astype(str)

        self.numeric_cols = train_db.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.scaler.fit(train_db[self.numeric_cols])
        return train_db

    def transform(self, db):
        db = db.drop(columns=self.categorical_cols, errors='ignore')
        if 'Transition' in db.columns:
            db['Transition'] = db['Transition'].astype(str)

        for coluna in self.numeric_cols:
            if coluna in db.columns:
                db[coluna] = db[coluna].astype(float)
        db[self.numeric_cols] = self.scaler.transform(db[self.numeric_cols])
        return db

    def prep_train(self, train_db):
        train_db = self.fit(train_db)
        return self.transform(train_db)