# src/data/processors.py
"""
Facade de compatibilidade — use DataCleaner e DataNormalizer em código novo.

DataProcessor mantém a API original para não quebrar run_training.py durante
a transição. Internamente delega cada método para as classes especializadas:
  - DataCleaner  → rename_columns, add_temporal_features,
                   handle_missing_values, drop_irrelevant_features
  - DataNormalizer → normalize_data, transform
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.transformers.cleaning import DataCleaner
from src.data.transformers.normalization import DataNormalizer


class DataProcessor:
    """
    Facade sobre DataCleaner + DataNormalizer.

    Prefira usar as classes especializadas diretamente em código novo:

        cleaner    = DataCleaner()
        normalizer = DataNormalizer()
        df_clean   = cleaner.transform(df_raw)
        X_train    = normalizer.fit_transform(X_train)
        X_test     = normalizer.transform(X_test)
    """

    def __init__(self) -> None:
        self._cleaner = DataCleaner()
        self._normalizer = DataNormalizer()

    # ------------------------------------------------------------------
    # Limpeza — delega para DataCleaner (etapas individuais preservadas
    # para compatibilidade com run_training.py que as chama separadamente)
    # ------------------------------------------------------------------

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return DataCleaner._rename_columns(df)

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return DataCleaner._add_temporal_features(df)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return DataCleaner._handle_missing_values(df)

    def drop_irrelevant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return DataCleaner._drop_irrelevant_features(df)

    # ------------------------------------------------------------------
    # Normalização — delega para DataNormalizer
    # ------------------------------------------------------------------

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit_transform — APENAS para X_train."""
        return self._normalizer.fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform — APENAS para X_test, após normalize_data ter sido chamado."""
        return self._normalizer.transform(df)

    @property
    def scaler(self) -> MinMaxScaler:
        """Acesso ao scaler para compatibilidade com código que o inspeciona."""
        return self._normalizer.scaler
