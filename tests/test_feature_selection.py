# tests/test_feature_selection.py
import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from src.features.selection import FeatureSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_model(sample_df):
    """DecisionTree treinado no sample_df para testar seleção de features."""
    y = sample_df["GERACAO"]
    X = sample_df.drop(columns=["GERACAO"])
    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X, y)
    return model, list(X.columns), X


# ---------------------------------------------------------------------------
# Construtor
# ---------------------------------------------------------------------------


class TestFeatureSelectorInit:
    def test_default_values(self):
        sel = FeatureSelector()
        assert sel.top_n is None
        assert sel.threshold == 0.0

    def test_raises_for_top_n_zero(self):
        with pytest.raises(ValueError, match="top_n"):
            FeatureSelector(top_n=0)

    def test_raises_for_negative_top_n(self):
        with pytest.raises(ValueError, match="top_n"):
            FeatureSelector(top_n=-1)

    def test_raises_for_threshold_above_1(self):
        with pytest.raises(ValueError, match="threshold"):
            FeatureSelector(threshold=1.5)

    def test_raises_for_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            FeatureSelector(threshold=-0.1)

    def test_selected_features_none_before_fit(self):
        assert FeatureSelector().selected_features_ is None

    def test_n_features_selected_zero_before_fit(self):
        assert FeatureSelector().n_features_selected == 0


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFeatureSelectorFit:
    def test_fit_returns_self(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector()
        assert sel.fit(model, names) is sel

    def test_fit_populates_selected_features(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector()
        sel.fit(model, names)
        assert sel.selected_features_ is not None
        assert len(sel.selected_features_) > 0

    def test_top_n_limits_selection(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector(top_n=3)
        sel.fit(model, names)
        assert sel.n_features_selected == 3

    def test_top_n_larger_than_features_keeps_all(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector(top_n=1000)
        sel.fit(model, names)
        assert sel.n_features_selected == len(names)

    def test_threshold_filters_low_importance(self, trained_model):
        model, names, _ = trained_model
        # threshold=1.0 deve manter 0 features (nenhuma tem importância == 1.0)
        sel = FeatureSelector(threshold=1.0)
        sel.fit(model, names)
        assert sel.n_features_selected == 0

    def test_threshold_zero_keeps_all(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector(threshold=0.0)
        sel.fit(model, names)
        assert sel.n_features_selected == len(names)

    def test_top_n_and_threshold_combined(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector(top_n=3, threshold=0.0)
        sel.fit(model, names)
        assert sel.n_features_selected == 3

    def test_importance_df_available_after_fit(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector()
        sel.fit(model, names)
        assert sel.importance_df is not None
        assert "feature" in sel.importance_df.columns
        assert "importance" in sel.importance_df.columns

    def test_importance_df_contains_all_features(self, trained_model):
        model, names, _ = trained_model
        sel = FeatureSelector(top_n=2)
        sel.fit(model, names)
        # importance_df deve ter TODAS as features, não só as selecionadas
        assert len(sel.importance_df) == len(names)


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------


class TestFeatureSelectorTransform:
    def test_raises_before_fit(self, trained_model):
        _, _, X = trained_model
        with pytest.raises(NotFittedError):
            FeatureSelector().transform(X)

    def test_returns_dataframe(self, trained_model):
        model, names, X = trained_model
        sel = FeatureSelector(top_n=3)
        sel.fit(model, names)
        result = sel.transform(X)
        assert isinstance(result, pd.DataFrame)

    def test_column_count_equals_top_n(self, trained_model):
        model, names, X = trained_model
        sel = FeatureSelector(top_n=3)
        sel.fit(model, names)
        result = sel.transform(X)
        assert len(result.columns) == 3

    def test_selected_columns_are_most_important(self, trained_model):
        model, names, X = trained_model
        sel = FeatureSelector(top_n=3)
        sel.fit(model, names)
        result = sel.transform(X)
        top3 = sel.importance_df.head(3)["feature"].tolist()
        assert set(result.columns) == set(top3)

    def test_preserves_index(self, trained_model):
        model, names, X = trained_model
        sel = FeatureSelector(top_n=3)
        sel.fit(model, names)
        result = sel.transform(X)
        pd.testing.assert_index_equal(result.index, X.index)

    def test_ignores_absent_columns_silently(self, trained_model):
        model, names, X = trained_model
        sel = FeatureSelector(top_n=3)
        sel.fit(model, names)
        # Remove uma das colunas selecionadas do DataFrame
        X_partial = X.drop(columns=[sel.selected_features_[0]])
        result = sel.transform(X_partial)
        assert len(result.columns) == 2


# ---------------------------------------------------------------------------
# fit_transform()
# ---------------------------------------------------------------------------


class TestFeatureSelectorFitTransform:
    def test_fit_transform_equivalent_to_fit_then_transform(self, trained_model):
        model, names, X = trained_model
        sel1 = FeatureSelector(top_n=4)
        sel1.fit(model, names)
        expected = sel1.transform(X)

        result = FeatureSelector(top_n=4).fit_transform(model, names, X)
        pd.testing.assert_frame_equal(result, expected)
