# src/models/trainer.py
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from src.config.settings import Config
from src.models.base import ModelStrategy
from src.utils.logger import logger


class ModelTrainer:
    """
    Orquestra split temporal e treinamento via estratégias plugáveis.

    Por que Strategy aqui?
    A versão anterior tinha train_decision_tree() e train_random_forest()
    como dois métodos quase idênticos. Com ModelStrategy, ModelTrainer não
    precisa saber qual algoritmo está sendo treinado — recebe uma lista de
    estratégias e delega o treinamento a cada uma. Adicionar GradientBoosting,
    por exemplo, não requer nenhuma alteração nesta classe (OCP).

    Uso:
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy(), RandomForestStrategy()])
        X_train, X_test, y_train, y_test = trainer.split_data(df)
        results = trainer.train_all(X_train, y_train)
        # results: {"Decision Tree": (model, cv_results), "Random Forest": (...)}
    """

    def __init__(
        self,
        strategies: Optional[List[ModelStrategy]] = None,
        test_size: float = Config.TEST_SIZE,
        random_state: int = Config.RANDOM_STATE,
        min_samples: int = Config.MIN_DATASET_SIZE,
    ) -> None:
        self.strategies = list(strategies) if strategies is not None else []
        self.test_size = test_size
        self.random_state = random_state
        self.min_samples = min_samples
        self.tscv = TimeSeriesSplit()

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = Config.TARGET_COLUMN,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados em treino/teste preservando ordem temporal.

        shuffle=False é obrigatório em séries temporais para evitar data
        leakage — embaralhar quebraria a ordem cronológica e permitiria que
        dados futuros "vazassem" para o conjunto de treino.

        Returns:
            X_train, X_test, y_train, y_test
        """
        y = df[target_col]
        X = df.drop(target_col, axis=1)

        logger.info(f"Dividindo dados: {len(X)} amostras, test_size={self.test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            shuffle=False,  # Preserva ordem temporal — NÃO alterar
        )

        return X_train, X_test, y_train, y_test

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Tuple[BaseEstimator, Dict]]:
        """
        Treina todas as estratégias registradas e retorna os resultados.

        Returns:
            Dict mapeando strategy.name → (best_estimator, cv_results_)
        """
        if not self.strategies:
            logger.warning("ModelTrainer não tem estratégias registradas.")
            return {}

        results = {}
        for strategy in self.strategies:
            model, cv_results = strategy.train(X_train, y_train, self.tscv)
            results[strategy.name] = (model, cv_results)

        return results
