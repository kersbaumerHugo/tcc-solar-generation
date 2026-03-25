# src/data/transformers/normalization.py
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from src.data.transformers.base import DataTransformer
from src.utils.logger import logger


class DataNormalizer(DataTransformer):
    """
    Normalização min-max [0, 1] com separação explícita de fit e transform.

    A separação é a proteção principal contra data leakage:
      fit_transform(X_train) — aprende min/max do treino e normaliza.
      transform(X_test)      — aplica os parâmetros do treino ao teste.

    Chamar transform() antes de fit() lança NotFittedError com mensagem
    explicativa, tornando o erro visível imediatamente em vez de silencioso
    (o que aconteceria se usássemos MinMaxScaler diretamente sem controle).

    O scaler subjacente é exposto via property para permitir serialização
    pelo ModelRegistry (o processor persistido em run_training.py carrega
    o DataNormalizer já fitado, garantindo a mesma escala na avaliação).
    """

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "DataNormalizer":
        """
        Aprende min/max de cada coluna a partir dos dados de treino.

        Deve ser chamado APENAS com X_train — nunca com X_test ou df completo.
        """
        logger.debug("Aprendendo parâmetros de normalização (fit)")
        self._scaler.fit(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica normalização usando parâmetros aprendidos em fit().

        Preserva colunas e índice do DataFrame original para que o
        resultado seja compatível com os demais passos do pipeline.
        """
        if not self._fitted:
            raise NotFittedError(
                "DataNormalizer.fit() deve ser chamado antes de transform(). "
                "Use fit_transform(X_train) para treino e transform(X_test) para teste."
            )
        logger.debug("Aplicando normalização (transform)")
        scaled = self._scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)

    @property
    def scaler(self) -> MinMaxScaler:
        """Acesso ao scaler subjacente para serialização via ModelRegistry."""
        return self._scaler
