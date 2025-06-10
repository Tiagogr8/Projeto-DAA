import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreparation:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.cols_categorical_drop = [
            "ID", "Image", "Mask", "diagnostics_Image-original_Hash",
            "diagnostics_Mask-original_Hash", "diagnostics_Mask-original_BoundingBox",
            "diagnostics_Mask-original_CenterOfMassIndex", "diagnostics_Mask-original_CenterOfMass"
        ]
        self.cols_to_drop = []
        self.numeric_cols = []

    def fit(self, train_db):
        # Remover colunas com valores únicos
        cols_unique = train_db.columns[train_db.nunique() == 1].tolist()
        self.cols_to_drop.extend(cols_unique + self.cols_categorical_drop)

        # Remover colunas altamente correlacionadas
        corr_matrix = train_db.corr(numeric_only=True)
        limiar = 0.8
        cols_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > limiar:
                    cols_corr.add(corr_matrix.columns[i])
        self.cols_to_drop.extend(cols_corr)

        # Remover colunas categóricas e correlacionadas
        train_db = train_db.drop(columns=self.cols_to_drop, errors='ignore')

        # Identificar colunas numéricas
        self.numeric_cols = train_db.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Ajustar o MinMaxScaler
        self.scaler.fit(train_db[self.numeric_cols])

        return train_db

    def transform(self, db):
        # Remover colunas desnecessárias
        db = db.drop(columns=self.cols_to_drop, errors='ignore')

        # Garantir consistência de tipos
        for coluna in self.numeric_cols:
            if coluna in db.columns:
                db[coluna] = db[coluna].astype(float)

        # Normalizar colunas numéricas
        db[self.numeric_cols] = self.scaler.transform(db[self.numeric_cols])

        return db

    def prep_train(self, train_db):
        train_db = self.fit(train_db)
        return self.transform(train_db)

