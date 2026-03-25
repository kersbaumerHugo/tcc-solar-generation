# src/features/selection.py
from typing import List, Optional

import pandas as pd
from sklearn.exceptions import NotFittedError

from src.features.engineering import FeatureImportanceAnalyzer
from src.utils.logger import logger


class FeatureSelector:
    """
    Seleciona features com base na importância reportada por um modelo de árvore.

    Por que não herdar DataTransformer?
    DataTransformer.fit(df) recebe um DataFrame — mas a seleção de features
    baseada em importância precisa de um modelo JÁ treinado (não de dados crus).
    Forçar o encaixe na ABC criaria um contrato falso. FeatureSelector tem seu
    próprio contrato: fit(model, feature_names) → transform(df).

    Dois critérios de seleção, combináveis:
      top_n     — mantém as N features com maior importância.
      threshold — mantém apenas features com importância >= valor mínimo.

    Se ambos forem fornecidos, aplica threshold primeiro e top_n depois.
    Se nenhum for fornecido, mantém todas as features (comportamento no-op).

    Uso típico (após um primeiro treino para descobrir importâncias):
        selector = FeatureSelector(top_n=8)
        selector.fit(dt_model, feature_names)
        X_train_reduced = selector.transform(X_train)
        X_test_reduced  = selector.transform(X_test)
        # Retreina com features reduzidas para modelo mais enxuto
    """

    def __init__(
        self,
        top_n: Optional[int] = None,
        threshold: float = 0.0,
    ) -> None:
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n deve ser >= 1, recebido: {top_n}")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold deve estar em [0, 1], recebido: {threshold}")

        self.top_n = top_n
        self.threshold = threshold
        self.selected_features_: Optional[List[str]] = None
        self._importance_df: Optional[pd.DataFrame] = None

    def fit(self, model, feature_names: List[str]) -> "FeatureSelector":
        """
        Aprende quais features manter a partir das importâncias do modelo.

        Args:
            model:         Estimador sklearn com feature_importances_ (tree-based).
            feature_names: Nomes das features usadas no treinamento.

        Returns:
            self (para encadeamento com transform).
        """
        importance_df = FeatureImportanceAnalyzer().get_importance(model, feature_names)

        filtered = importance_df[importance_df["importance"] >= self.threshold]

        if self.top_n is not None:
            filtered = filtered.head(self.top_n)

        self.selected_features_ = filtered["feature"].tolist()
        self._importance_df = importance_df

        logger.info(
            f"FeatureSelector: {len(self.selected_features_)} de {len(feature_names)} "
            f"features selecionadas (top_n={self.top_n}, threshold={self.threshold})"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra o DataFrame mantendo apenas as features selecionadas.

        Features selecionadas que não existam em df são ignoradas silenciosamente
        — útil quando df de predição tem colunas ligeiramente diferentes do treino.

        Args:
            df: DataFrame com features (X_train ou X_test).

        Returns:
            DataFrame com apenas as colunas selecionadas.

        Raises:
            NotFittedError: Se fit() não foi chamado antes.
        """
        if self.selected_features_ is None:
            raise NotFittedError(
                "FeatureSelector.fit() deve ser chamado antes de transform()."
            )
        cols = [c for c in self.selected_features_ if c in df.columns]
        dropped = len(self.selected_features_) - len(cols)
        if dropped:
            logger.warning(f"FeatureSelector: {dropped} feature(s) ausente(s) no DataFrame — ignoradas.")
        return df[cols]

    def fit_transform(self, model, feature_names: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """Conveniência: fit() + transform() em uma chamada."""
        return self.fit(model, feature_names).transform(df)

    @property
    def n_features_selected(self) -> int:
        """Número de features selecionadas após fit(). 0 antes de fit()."""
        return len(self.selected_features_) if self.selected_features_ is not None else 0

    @property
    def importance_df(self) -> Optional[pd.DataFrame]:
        """DataFrame completo de importâncias (todas as features) após fit()."""
        return self._importance_df
