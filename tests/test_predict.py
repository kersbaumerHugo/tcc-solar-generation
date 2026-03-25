# tests/test_predict.py
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from scripts.predict import predict
from src.data.processors import DataProcessor
from src.models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_registry(tmp_path, sample_df):
    """
    Registry populado com um processor fitado e um modelo treinado,
    usando os dados do sample_df do conftest.
    """
    registry = ModelRegistry(tmp_path / "models")

    # Prepara dados e fita o processor (replica o que run_training.py faz)
    processor = DataProcessor()
    df = processor.rename_columns(sample_df)
    df = processor.add_temporal_features(df)
    df = processor.handle_missing_values(df)
    df = processor.drop_irrelevant_features(df)

    y = df["GERACAO"]
    X = df.drop(columns=["GERACAO"])
    X_train = processor.normalize_data(X)

    # Treina um modelo mínimo
    model = DecisionTreeRegressor(random_state=42, max_depth=2)
    model.fit(X_train, y)

    registry.save(processor, "TEST_processor")
    registry.save(model, "TEST_decision_tree")

    return registry


# ---------------------------------------------------------------------------
# Testes de predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_dataframe(self, fitted_registry, sample_df):
        result = predict("TEST", "decision_tree", sample_df, fitted_registry)
        assert isinstance(result, pd.DataFrame)

    def test_has_geracao_prevista_column(self, fitted_registry, sample_df):
        result = predict("TEST", "decision_tree", sample_df, fitted_registry)
        assert "GERACAO_PREVISTA" in result.columns

    def test_includes_geracao_real_when_present(self, fitted_registry, sample_df):
        """Se GERACAO está no input, deve aparecer como GERACAO_REAL no output."""
        result = predict("TEST", "decision_tree", sample_df, fitted_registry)
        assert "GERACAO_REAL" in result.columns

    def test_no_geracao_real_when_absent(self, fitted_registry, sample_df):
        """Sem GERACAO no input, GERACAO_REAL não deve aparecer."""
        df = sample_df.drop(columns=["GERACAO"])
        result = predict("TEST", "decision_tree", df, fitted_registry)
        assert "GERACAO_REAL" not in result.columns

    def test_result_has_datetime_index(self, fitted_registry, sample_df):
        result = predict("TEST", "decision_tree", sample_df, fitted_registry)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_predictions_are_numeric(self, fitted_registry, sample_df):
        result = predict("TEST", "decision_tree", sample_df, fitted_registry)
        assert pd.to_numeric(result["GERACAO_PREVISTA"], errors="coerce").notna().all()

    def test_raises_for_missing_processor(self, tmp_path, sample_df):
        empty_registry = ModelRegistry(tmp_path / "empty")
        with pytest.raises(FileNotFoundError, match="processor"):
            predict("MISSING", "decision_tree", sample_df, empty_registry)

    def test_raises_for_missing_model(self, tmp_path, sample_df):
        registry = ModelRegistry(tmp_path / "models")
        # Salva só o processor, sem o modelo
        processor = DataProcessor()
        registry.save(processor, "NOMODEL_processor")
        with pytest.raises(FileNotFoundError, match="Modelo"):
            predict("NOMODEL", "decision_tree", sample_df, registry)

    def test_result_length_le_input_length(self, fitted_registry, sample_df):
        """handle_missing_values pode remover linhas — result <= input."""
        result = predict("TEST", "decision_tree", sample_df, fitted_registry)
        assert len(result) <= len(sample_df)

    def test_uses_train_scaler_not_refitted(self, fitted_registry, sample_df):
        """
        Garante que transform() é usado, não fit_transform().
        Duas chamadas com o mesmo registry devem produzir predições idênticas.
        """
        r1 = predict("TEST", "decision_tree", sample_df, fitted_registry)
        r2 = predict("TEST", "decision_tree", sample_df, fitted_registry)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Testes de _discover_models (lógica de auto-descoberta)
# ---------------------------------------------------------------------------


class TestDiscoverModels:
    def test_returns_models_for_usina(self, tmp_path):
        from scripts.run_evaluation import _discover_models

        registry = ModelRegistry(tmp_path)
        registry.save(object(), "ANGRA_decision_tree")
        registry.save(object(), "ANGRA_random_forest")
        registry.save(object(), "ANGRA_processor")

        models = _discover_models("ANGRA", registry)
        assert set(models) == {"ANGRA_decision_tree", "ANGRA_random_forest"}

    def test_excludes_processor(self, tmp_path):
        from scripts.run_evaluation import _discover_models

        registry = ModelRegistry(tmp_path)
        registry.save(object(), "USINA_processor")

        models = _discover_models("USINA", registry)
        assert models == []

    def test_does_not_return_other_usina_models(self, tmp_path):
        from scripts.run_evaluation import _discover_models

        registry = ModelRegistry(tmp_path)
        registry.save(object(), "ANGRA_decision_tree")
        registry.save(object(), "ITAIPU_decision_tree")

        models = _discover_models("ANGRA", registry)
        assert all(m.startswith("ANGRA_") for m in models)
