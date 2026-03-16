# src/data/processors.py
import pandas as pd
from src.config.settings import Config

class DataFrameProcessor:
    def __init__(self, config: Config):
        self.config = config
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia colunas usando mapeamento centralizado"""
        return df.rename(columns=self.config.COLUMN_MAPPING)
    
    def clean_decimal_separators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Substitui vírgulas por pontos"""
        return df.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)
# src/data/processors.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from src.utils.logger import logger

class DataProcessor:
    """Processa e prepara dados para modelagem"""
    
    # Mapeamento centralizado de colunas
    COLUMN_MAPPING = {
        "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh": "GERACAO",
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "PRECIPITACAO",
        "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "PATM",
        "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "TEMPERATURA",
        "TEMPERATURA DO PONTO DE ORVALHO (°C)": "TEMPERATURA_ORVALHO",
        "UMIDADE RELATIVA DO AR, HORARIA (%)": "UMIDADE",
        "VENTO, VELOCIDADE HORARIA (m/s)": "VEL_VENTO",
        "NumCoordNEmpreendimento": "COORD_N",
        "NumCoordEEmpreendimento": "COORD_E",
        "radiation": "RADIACAO"
    }
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia colunas usando mapeamento padrão"""
        logger.debug("Renomeando colunas")
        return df.rename(columns=self.COLUMN_MAPPING)
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais (mês, hora)"""
        logger.debug("Adicionando features temporais")
        
        # One-hot encoding do mês
        df["Mes"] = df.index.month
        encoder = LabelBinarizer()
        months_encoded = encoder.fit_transform(df["Mes"])
        months_df = pd.DataFrame(
            months_encoded, 
            index=df.index,
            columns=[str(i) for i in range(1, 13)]
        )
        df = df.merge(months_df, left_index=True, right_index=True)
        df.drop("Mes", axis=1, inplace=True)
        
        # Transformação cosseno da hora
        df["Hour"] = df.index.hour
        df["hour_cos"] = np.cos(df["Hour"] / 24 * 2 * np.pi)
        df.drop("Hour", axis=1, inplace=True)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores faltantes"""
        logger.debug("Tratando valores ausentes")
        
        # Remove linhas sem radiação
        df = df.dropna(subset=["RADIACAO"], axis=0)
        
        # Preenche temperatura com média
        if "TEMPERATURA" in df.columns:
            mean_temp = df["TEMPERATURA"].mean()
            df["TEMPERATURA"].fillna(mean_temp, inplace=True)
        
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza os dados usando MinMaxScaler"""
        logger.debug("Normalizando dados")
        
        scaled_array = self.scaler.fit_transform(df)
        return pd.DataFrame(
            scaled_array, 
            columns=df.columns, 
            index=df.index
        )