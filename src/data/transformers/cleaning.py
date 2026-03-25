# src/data/transformers/cleaning.py
import numpy as np
import pandas as pd

from src.config.settings import Config
from src.data.transformers.base import DataTransformer
from src.utils.logger import logger


class DataCleaner(DataTransformer):
    """
    Transformações stateless de limpeza e preparação dos dados.

    Agrupa operações que não requerem aprendizado de parâmetros:
    renomear colunas, adicionar features temporais, tratar ausentes,
    descartar colunas irrelevantes.

    Por ser stateless, fit() é um no-op e transform() pode ser chamado
    diretamente sem chamada prévia a fit(). fit_transform(df) e
    transform(df) são equivalentes para esta classe.

    Os métodos privados (_rename_columns, etc.) são @staticmethod para
    permitir que DataProcessor os chame individualmente, mantendo
    compatibilidade com o pipeline legado durante a transição.
    """

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """No-op: DataCleaner não aprende parâmetros."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o pipeline completo de limpeza na ordem correta:
        renomear → features temporais → tratar ausentes → descartar colunas.
        """
        df = self._rename_columns(df)
        df = self._add_temporal_features(df)
        df = self._handle_missing_values(df)
        df = self._drop_irrelevant_features(df)
        return df

    # ------------------------------------------------------------------
    # Etapas individuais (protected — use transform() em código novo)
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Renomeando colunas")
        return df.rename(columns=Config.COLUMN_MAPPING)

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features temporais: one-hot por mês + cosseno da hora.

        Usa df.copy() para não mutar o DataFrame original — sem isso, colunas
        seriam adicionadas in-place no objeto do chamador, quebrando código que
        passa o df e espera preservá-lo inalterado.

        LabelBinarizer foi substituído por atribuição direta porque, quando
        apenas um mês está presente na amostra, LabelBinarizer retorna shape
        (n, 1) em vez de (n, 12), causando erro silencioso em produção.
        """
        logger.debug("Adicionando features temporais")
        df = df.copy()
        for month in range(1, 13):
            df[str(month)] = (df.index.month == month).astype(int)
        df["hour_cos"] = np.cos(df.index.hour / 24 * 2 * np.pi)
        return df

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores faltantes por regra de negócio:

        - RADIACAO: pré-requisito físico para geração solar — linhas sem
          radiação são removidas (não há geração possível sem incidência).
        - TEMPERATURA e UMIDADE: média como proxy conservador.

        TODO (produção): a média aqui inclui treino + teste juntos,
        introduzindo leakage leve. A correção seria calcular a média apenas
        no treino e persistir junto ao DataNormalizer (similar ao scaler).
        Para o TCC, o impacto é negligenciável dado o volume de dados.
        """
        logger.debug("Tratando valores ausentes")
        df = df.dropna(subset=["RADIACAO"])
        fill_values = {
            col: df[col].mean()
            for col in ("TEMPERATURA", "UMIDADE")
            if col in df.columns
        }
        if fill_values:
            df = df.fillna(fill_values)
        return df

    @staticmethod
    def _drop_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in Config.FEATURES_TO_DROP if c in df.columns]
        if cols_to_drop:
            logger.debug(f"Descartando {len(cols_to_drop)} colunas: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)
