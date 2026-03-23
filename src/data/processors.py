# src/data/processors.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config.settings import Config
from src.utils.logger import logger


class DataProcessor:
    """Processa e prepara dados para modelagem."""

    def __init__(self) -> None:
        self.scaler = MinMaxScaler()

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia colunas usando o mapeamento centralizado em Config."""
        logger.debug("Renomeando colunas")
        return df.rename(columns=Config.COLUMN_MAPPING)

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features temporais (one-hot mês + cosseno hora).

        LabelBinarizer foi substituído por atribuição direta: quando só um mês
        está presente na amostra, LabelBinarizer retorna shape (n, 1) ao invés
        de (n, 12), causando erro silencioso em produção.
        """
        logger.debug("Adicionando features temporais")

        # One-hot encoding do mês — garante as 12 colunas independente dos dados
        for month in range(1, 13):
            df[str(month)] = (df.index.month == month).astype(int)

        # Transformação cosseno da hora (captura ciclicidade do ciclo diário)
        df["hour_cos"] = np.cos(df.index.hour / 24 * 2 * np.pi)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores faltantes.

        - RADIACAO: pré-requisito físico para geração solar — linhas sem ela
          são removidas.
        - TEMPERATURA e UMIDADE: usa média como proxy. A média é calculada
          aqui (antes do split) para não vazar informação do conjunto de teste.
          O trainer.split_data() não precisa mais fazer esse tratamento.
        """
        logger.debug("Tratando valores ausentes")

        df = df.dropna(subset=["RADIACAO"])

        fill_values = {}
        for col in ("TEMPERATURA", "UMIDADE"):
            if col in df.columns:
                fill_values[col] = df[col].mean()

        if fill_values:
            df = df.fillna(fill_values)

        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza colunas para [0, 1] e memoriza os parâmetros (fit_transform).

        Deve ser chamado APENAS com dados de treino (X_train).
        Para o conjunto de teste, use processor.transform(X_test) — que aplica
        a mesma escala aprendida aqui, sem vazar informação do teste.
        """
        logger.debug("Normalizando dados de treino (fit_transform)")
        scaled_array = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a normalização já aprendida (transform apenas, sem fit).

        Deve ser chamado com dados de teste (X_test) após normalize_data ter
        sido chamado com os dados de treino.
        """
        logger.debug("Aplicando normalização ao conjunto de teste (transform)")
        scaled_array = self.scaler.transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
