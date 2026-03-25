# src/models/random_forest.py
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

from src.config.settings import Config
from src.models.base import ModelStrategy


class RandomForestStrategy(ModelStrategy):
    """Estratégia de treinamento para Random Forest Regressor."""

    def __init__(self, random_state: int = Config.RANDOM_STATE) -> None:
        self._random_state = random_state

    @property
    def name(self) -> str:
        return "Random Forest"

    @property
    def estimator(self) -> BaseEstimator:
        return RandomForestRegressor(random_state=self._random_state)

    @property
    def param_grid(self) -> Dict:
        return dict(Config.RF_GRID)
