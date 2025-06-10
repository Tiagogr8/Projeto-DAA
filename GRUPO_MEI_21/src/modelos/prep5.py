import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast

class DataPreparation:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.categorical_cols = []
        self.numeric_cols = []
        self.colunas_para_dividir = [
            'diagnostics_Mask-original_BoundingBox',
            'diagnostics_Mask-original_CenterOfMassIndex',
            'diagnostics_Mask-original_CenterOfMass'
        ]
        
    def _expandir_colunas_array(self, df):
        df_copy = df.copy()
        
        for col in self.colunas_para_dividir:
            if col in df_copy.columns:
                try:
                    valores = df_copy[col].apply(lambda x: list(ast.literal_eval(x)))
                    col_expanded = pd.DataFrame(valores.tolist(), index=df_copy.index)
                    col_expanded.columns = [f"{col}_{i}" for i in range(col_expanded.shape[1])]
                    df_copy = pd.concat([df_copy.drop(columns=[col]), col_expanded], axis=1)
                except Exception as e:
                    print(f"Erro ao processar coluna {col}: {str(e)}")
                    continue
        
        return df_copy
    
    def _identify_columns(self, df):
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_cols = [col for col in self.categorical_cols if col != "Transition"]
        self.numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    def fit(self, train_df):
        # Expande colunas com arrays
        train_df = self._expandir_colunas_array(train_df)
        
        self._identify_columns(train_df)
        train_df = train_df.drop(columns=self.categorical_cols)
        
        if 'Transition' in train_df.columns:
            train_df['Transition'] = train_df['Transition'].astype(str)
        
        self.numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        self.scaler.fit(train_df[self.numeric_cols])
        
        return train_df
    
    def transform(self, df):
        df = self._expandir_colunas_array(df)
        
        df = df.drop(columns=self.categorical_cols, errors='ignore')
        
        if 'Transition' in df.columns:
            df['Transition'] = df['Transition'].astype(str)
        
        # Converte colunas numéricas para float
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Aplica a normalização
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        
        return df
    
    def prep_train(self, train_df):
        train_df = self.fit(train_df)
        return self.transform(train_df)
