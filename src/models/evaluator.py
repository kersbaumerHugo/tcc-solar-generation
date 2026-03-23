# src/models/evaluator.py
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import logger


class ModelEvaluator:
    """Avalia performance de modelos treinados."""

    @staticmethod
    def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calcula Root Mean Squared Error.

        Usa np.sqrt(MSE) ao invés de squared=False para compatibilidade
        com sklearn >= 1.3 e < 1.4 (onde squared=False é deprecado).
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """
        Avalia modelo e retorna dicionário com métricas.

        Returns:
            Dict com mae, mse, rmse, r2
        """
        logger.info(f"Avaliando {model_name}...")

        y_pred = model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": self.rmse(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        logger.info(f"{model_name} - MAE:  {metrics['mae']:.4f}")
        logger.info(f"{model_name} - RMSE: {metrics['rmse']:.4f}")
        logger.info(f"{model_name} - R²:   {metrics['r2']:.4f}")

        return metrics
