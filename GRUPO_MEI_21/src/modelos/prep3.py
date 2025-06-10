import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreparation:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.categorical_cols = []
        self.numeric_cols = []

    def fit(self, train_db):
        # Identificar e armazenar colunas categóricas
        self.categorical_cols = train_db.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_cols = [col for col in self.categorical_cols if col != "Transition"]
        
        # Remover colunas categóricas
        train_db = train_db.drop(columns=self.categorical_cols)
        
        # Guardar colunas numéricas
        self.numeric_cols = train_db.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Ajustar MinMaxScaler nos dados de treino
        self.scaler.fit(train_db[self.numeric_cols])
        return train_db

    def transform(self, db):
        # Remover colunas categóricas
        db = db.drop(columns=self.categorical_cols, errors='ignore')

        # Garantir tipos consistentes
        for coluna in self.numeric_cols:
            if coluna in db.columns:
                db[coluna] = db[coluna].astype(float)

        # Normalizar colunas numéricas
        db[self.numeric_cols] = self.scaler.transform(db[self.numeric_cols])
        return db

    def prep_train(self, train_db):
        train_db = self.fit(train_db)
        return self.transform(train_db)