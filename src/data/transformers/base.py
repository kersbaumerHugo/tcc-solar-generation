# src/data/transformers/base.py
from abc import ABC, abstractmethod

import pandas as pd


class DataTransformer(ABC):
    """
    Contrato base para transformações de dados, espelhando a API do sklearn.

    Três métodos definem o contrato:
      fit(df)           — aprende parâmetros dos dados (ex: min/max do scaler).
      transform(df)     — aplica a transformação usando os parâmetros aprendidos.
      fit_transform(df) — atalho: fit() + transform() sobre o mesmo conjunto.

    A separação fit/transform é a peça-chave para evitar data leakage:
    fit_transform() é chamado apenas com X_train (o modelo aprende os
    parâmetros só dos dados de treino), e transform() é chamado com X_test
    (aplica os mesmos parâmetros, sem reaprender com dados futuros).

    Transformações stateless (ex: DataCleaner) implementam fit() como no-op
    e podem ser usadas diretamente via transform() sem chamada prévia a fit().
    Transformações stateful (ex: DataNormalizer) precisam de fit() antes de
    transform() — tentar transform() sem fit() deve lançar NotFittedError.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "DataTransformer":
        """
        Aprende parâmetros a partir de df.

        Retorna self para permitir encadeamento:
            normalizer.fit(X_train).transform(X_test)
        """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação usando parâmetros aprendidos em fit()."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conveniência: fit() + transform() sobre o mesmo DataFrame.

        Equivalente a sklearn's fit_transform(). Use para X_train;
        use apenas transform() para X_test.
        """
        return self.fit(df).transform(df)
