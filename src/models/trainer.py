# src/models/trainer.py
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.config.settings import Config
from src.utils.logger import logger


class ModelTrainer:
    """Responsável pelo treinamento de modelos de ML."""

    def __init__(
        self,
        test_size: float = Config.TEST_SIZE,
        random_state: int = Config.RANDOM_STATE,
        min_samples: int = Config.MIN_DATASET_SIZE,
    ) -> None:
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

        shuffle=False é obrigatório em séries temporais para evitar data leakage.

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

    def train_decision_tree(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        grid_params: Optional[Dict] = None,
    ) -> Tuple[DecisionTreeRegressor, Dict]:
        """
        Treina Decision Tree com GridSearchCV.

        Returns:
            (best_model, cv_results)
        """
        if grid_params is None:
            grid_params = Config.DT_GRID

        logger.info("Treinando Decision Tree com GridSearchCV...")

        dt_cv = GridSearchCV(
            estimator=DecisionTreeRegressor(random_state=self.random_state),
            param_grid=grid_params,
            cv=self.tscv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1,
        )
        dt_cv.fit(X_train, y_train)

        logger.info(f"Melhor score (CV): {dt_cv.best_score_:.4f}")
        logger.info(f"Melhores parâmetros: {dt_cv.best_params_}")

        return dt_cv.best_estimator_, dt_cv.cv_results_

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        grid_params: Optional[Dict] = None,
    ) -> Tuple[RandomForestRegressor, Dict]:
        """
        Treina Random Forest com GridSearchCV.

        Returns:
            (best_model, cv_results)
        """
        if grid_params is None:
            grid_params = Config.RF_GRID

        logger.info("Treinando Random Forest com GridSearchCV...")

        rf_cv = GridSearchCV(
            estimator=RandomForestRegressor(random_state=self.random_state),
            param_grid=grid_params,
            cv=self.tscv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1,
        )
        rf_cv.fit(X_train, y_train)

        logger.info(f"Melhor score (CV): {rf_cv.best_score_:.4f}")
        logger.info(f"Melhores parâmetros: {rf_cv.best_params_}")

        return rf_cv.best_estimator_, rf_cv.cv_results_
