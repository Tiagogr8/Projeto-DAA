import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class DataPreparation:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=0.97)
        self.categorical_cols = []
        self.numeric_cols = []
        self.feature_names = None
    
    def fit(self, train_db):
        self.categorical_cols = train_db.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_cols = [col for col in self.categorical_cols if col != "Transition"]
        
        self.has_transition = "Transition" in train_db.columns
        
        numeric_data = train_db.select_dtypes(include=['float64', 'int64'])
        if "Transition" in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=["Transition"])
        self.numeric_cols = numeric_data.columns.tolist()
        
        self.scaler.fit(numeric_data)
        
        normalized_data = self.scaler.transform(numeric_data)
        self.pca.fit(normalized_data)
        
        n_components = self.pca.n_components_
        self.feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        return train_db
    
    def transform(self, db):
        transition = None
        if "Transition" in db.columns:
            transition = db["Transition"].copy()
        
        # Remover colunas categóricas
        db = db.drop(columns=self.categorical_cols, errors='ignore')
        
        numeric_data = db[self.numeric_cols].copy()
        
        # Garantir tipos consistentes
        for coluna in self.numeric_cols:
            if coluna in numeric_data.columns:
                numeric_data[coluna] = numeric_data[coluna].astype(float)
        
        # Aplicar normalização e PCA
        normalized_data = self.scaler.transform(numeric_data)
        pca_data = self.pca.transform(normalized_data)
        
        result_df = pd.DataFrame(pca_data, columns=self.feature_names, index=db.index)
        
        if transition is not None:
            result_df["Transition"] = transition
        
        return result_df
    
    def prep_train(self, train_db):
        train_db = self.fit(train_db)
        return self.transform(train_db)
    
    def get_explained_variance_ratio(self):
        """Retorna a razão de variância explicada para cada componente."""
        return self.pca.explained_variance_ratio_
    
    def get_n_components(self):
        """Retorna o número de componentes principais selecionados."""
        return self.pca.n_components_