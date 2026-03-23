# tests/test_feature_engineering.py
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.features.engineering import FeatureImportanceAnalyzer


@pytest.fixture
def trained_model_and_features():
    """Modelo treinado e nomes de features para testes."""
    rng = np.random.default_rng(42)
    feature_names = ["RADIACAO", "TEMPERATURA", "UMIDADE", "hour_cos"]
    X = rng.uniform(0, 1, (100, len(feature_names)))
    y = rng.uniform(0, 1, 100)

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    return model, feature_names


class TestFeatureImportanceAnalyzer:
    def test_get_importance_returns_dataframe(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.get_importance(model, features)
        assert isinstance(result, pd.DataFrame)

    def test_get_importance_has_correct_columns(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.get_importance(model, features)
        assert set(result.columns) == {"feature", "importance"}

    def test_get_importance_covers_all_features(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.get_importance(model, features)
        assert set(result["feature"]) == set(features)

    def test_importance_sums_to_1(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.get_importance(model, features)
        assert abs(result["importance"].sum() - 1.0) < 1e-6

    def test_result_is_sorted_descending(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.get_importance(model, features)
        assert result["importance"].is_monotonic_decreasing

    def test_top_features_returns_n_rows(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.top_features(model, features, n=2)
        assert len(result) == 2

    def test_top_features_are_highest_importance(self, trained_model_and_features):
        model, features = trained_model_and_features
        analyzer = FeatureImportanceAnalyzer()
        top2 = analyzer.top_features(model, features, n=2)
        all_imp = analyzer.get_importance(model, features)
        # Top 2 devem ter as maiores importâncias
        assert top2["importance"].min() >= all_imp["importance"].iloc[2]

    def test_raises_for_model_without_importances(self):
        """Modelos sem feature_importances_ devem lançar AttributeError claro."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit([[1], [2]], [1, 2])
        analyzer = FeatureImportanceAnalyzer()
        with pytest.raises(AttributeError, match="feature_importances_"):
            analyzer.get_importance(model, ["x"])

    def test_works_with_random_forest(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (100, 3))
        y = rng.uniform(0, 1, 100)
        model = RandomForestRegressor(n_estimators=5, random_state=0)
        model.fit(X, y)

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.get_importance(model, ["A", "B", "C"])
        assert len(result) == 3
