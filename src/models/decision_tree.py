# src/models/decision_tree.py
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from src.config.settings import Config
from src.models.base import ModelStrategy


class DecisionTreeStrategy(ModelStrategy):
    """Estratégia de treinamento para Decision Tree Regressor."""

    def __init__(self, random_state: int = Config.RANDOM_STATE) -> None:
        self._random_state = random_state

    @property
    def name(self) -> str:
        return "Decision Tree"

    @property
    def estimator(self) -> BaseEstimator:
        return DecisionTreeRegressor(random_state=self._random_state)

    @property
    def param_grid(self) -> Dict:
        return dict(Config.DT_GRID)
