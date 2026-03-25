# tests/test_models.py
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from src.models.base import ModelStrategy
from src.models.decision_tree import DecisionTreeStrategy
from src.models.random_forest import RandomForestStrategy
from src.models.trainer import ModelTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_data(sample_df):
    """Retorna X_train e y_train a partir do sample_df do conftest."""
    y = sample_df["GERACAO"]
    X = sample_df.drop(columns=["GERACAO"])
    return X, y


# ---------------------------------------------------------------------------
# ModelStrategy ABC
# ---------------------------------------------------------------------------


class TestModelStrategyABC:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ModelStrategy()

    def test_concrete_strategy_has_name(self):
        assert DecisionTreeStrategy().name == "Decision Tree"

    def test_concrete_strategy_has_estimator(self):
        assert isinstance(DecisionTreeStrategy().estimator, DecisionTreeRegressor)

    def test_concrete_strategy_has_param_grid(self):
        grid = DecisionTreeStrategy().param_grid
        assert isinstance(grid, dict)
        assert len(grid) > 0


# ---------------------------------------------------------------------------
# DecisionTreeStrategy
# ---------------------------------------------------------------------------


class TestDecisionTreeStrategy:
    def test_name(self):
        assert DecisionTreeStrategy().name == "Decision Tree"

    def test_estimator_type(self):
        assert isinstance(DecisionTreeStrategy().estimator, DecisionTreeRegressor)

    def test_estimator_has_random_state(self):
        strategy = DecisionTreeStrategy(random_state=7)
        assert strategy.estimator.random_state == 7

    def test_param_grid_has_max_depth(self):
        assert "max_depth" in DecisionTreeStrategy().param_grid

    def test_param_grid_is_regular_dict(self):
        # param_grid deve ser dict mutável (GridSearchCV não aceita MappingProxyType)
        grid = DecisionTreeStrategy().param_grid
        assert type(grid) is dict

    def test_train_returns_fitted_model(self, sample_df):
        X, y = _make_train_data(sample_df)
        model, cv_results = DecisionTreeStrategy().train(X, y, TimeSeriesSplit())
        assert isinstance(model, DecisionTreeRegressor)
        assert "mean_test_score" in cv_results

    def test_trained_model_can_predict(self, sample_df):
        X, y = _make_train_data(sample_df)
        model, _ = DecisionTreeStrategy().train(X, y, TimeSeriesSplit())
        preds = model.predict(X)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# RandomForestStrategy
# ---------------------------------------------------------------------------


class TestRandomForestStrategy:
    def test_name(self):
        assert RandomForestStrategy().name == "Random Forest"

    def test_estimator_type(self):
        assert isinstance(RandomForestStrategy().estimator, RandomForestRegressor)

    def test_estimator_has_random_state(self):
        strategy = RandomForestStrategy(random_state=99)
        assert strategy.estimator.random_state == 99

    def test_param_grid_has_n_estimators(self):
        assert "n_estimators" in RandomForestStrategy().param_grid

    def test_param_grid_is_regular_dict(self):
        grid = RandomForestStrategy().param_grid
        assert type(grid) is dict

    def test_train_returns_fitted_model(self, sample_df):
        X, y = _make_train_data(sample_df)
        model, _ = RandomForestStrategy().train(X, y, TimeSeriesSplit())
        assert isinstance(model, RandomForestRegressor)


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class TestModelTrainerSplitData:
    def test_split_returns_four_parts(self, sample_df):
        parts = ModelTrainer().split_data(sample_df)
        assert len(parts) == 4

    def test_split_preserves_total_rows(self, sample_df):
        X_train, X_test, y_train, y_test = ModelTrainer().split_data(sample_df)
        assert len(X_train) + len(X_test) == len(sample_df)

    def test_split_preserves_temporal_order(self, sample_df):
        """O último timestamp de treino deve ser anterior ao primeiro de teste."""
        X_train, X_test, _, _ = ModelTrainer().split_data(sample_df)
        assert X_train.index[-1] < X_test.index[0]

    def test_split_removes_target_from_X(self, sample_df):
        X_train, X_test, _, _ = ModelTrainer().split_data(sample_df)
        assert "GERACAO" not in X_train.columns
        assert "GERACAO" not in X_test.columns

    def test_y_series_name_is_target(self, sample_df):
        _, _, y_train, _ = ModelTrainer().split_data(sample_df)
        assert y_train.name == "GERACAO"


class TestModelTrainerTrainAll:
    def test_train_all_returns_all_strategies(self, sample_df):
        strategies = [DecisionTreeStrategy(), RandomForestStrategy()]
        trainer = ModelTrainer(strategies=strategies)
        X_train, _, y_train, _ = trainer.split_data(sample_df)
        results = trainer.train_all(X_train, y_train)
        assert set(results.keys()) == {"Decision Tree", "Random Forest"}

    def test_train_all_returns_fitted_models(self, sample_df):
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy()])
        X_train, X_test, y_train, _ = trainer.split_data(sample_df)
        results = trainer.train_all(X_train, y_train)
        model, _ = results["Decision Tree"]
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_train_all_empty_strategies_returns_empty_dict(self, sample_df):
        trainer = ModelTrainer(strategies=[])
        X_train, _, y_train, _ = trainer.split_data(sample_df)
        assert trainer.train_all(X_train, y_train) == {}

    def test_train_all_results_contain_cv_results(self, sample_df):
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy()])
        X_train, _, y_train, _ = trainer.split_data(sample_df)
        _, cv_results = trainer.train_all(X_train, y_train)["Decision Tree"]
        assert "mean_test_score" in cv_results
