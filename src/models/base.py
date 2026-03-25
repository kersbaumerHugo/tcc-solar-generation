# src/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator


class ModelStrategy(ABC):
    """
    Contrato base para estratégias de treinamento de modelos (Strategy pattern).

    Por que Strategy aqui?
    ModelTrainer original tinha train_decision_tree() e train_random_forest()
    como dois métodos quase idênticos — só diferiam no estimador e no grid.
    A cada novo modelo (ex: GradientBoosting), seria necessário adicionar um
    terceiro método idêntico. Com Strategy, adicionar um novo modelo significa
    apenas criar uma nova subclasse; ModelTrainer não precisa mudar (OCP).

    Cada estratégia encapsula:
      - qual estimador sklearn usar
      - qual grid de hiperparâmetros buscar
      - o nome legível do modelo (para logs e persistência)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome legível do modelo — usado em logs e como prefixo no registry."""

    @property
    @abstractmethod
    def estimator(self) -> BaseEstimator:
        """Instância do estimador sklearn pronta para uso em GridSearchCV."""

    @property
    @abstractmethod
    def param_grid(self) -> Dict:
        """Grid de hiperparâmetros para GridSearchCV."""

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tscv,
    ) -> Tuple[BaseEstimator, Dict]:
        """
        Treina o modelo com GridSearchCV usando TimeSeriesSplit.

        Implementação comum a todas as estratégias — subclasses só precisam
        fornecer estimator e param_grid via properties abstratas.

        Args:
            X_train: features de treino (já normalizadas).
            y_train: target de treino.
            tscv:    instância de TimeSeriesSplit compartilhada com ModelTrainer.

        Returns:
            (best_estimator, cv_results_)
        """
        from sklearn.model_selection import GridSearchCV

        from src.utils.logger import logger

        logger.info(f"Treinando {self.name} com GridSearchCV...")

        cv = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=tscv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1,
        )
        cv.fit(X_train, y_train)

        logger.info(f"{self.name} — melhor score (CV): {cv.best_score_:.4f}")
        logger.info(f"{self.name} — melhores parâmetros: {cv.best_params_}")

        return cv.best_estimator_, cv.cv_results_
