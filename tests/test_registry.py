# tests/test_registry.py
from pathlib import Path

import pytest
from sklearn.tree import DecisionTreeRegressor

from src.models.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "models")


@pytest.fixture
def trained_dt() -> DecisionTreeRegressor:
    """Árvore mínima treinada para testes de persistência."""
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (50, 3))
    y = rng.uniform(0, 1, 50)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    return model


class TestModelRegistry:
    def test_creates_directory_on_init(self, tmp_path):
        models_dir = tmp_path / "nested" / "models"
        assert not models_dir.exists()
        ModelRegistry(models_dir)
        assert models_dir.exists()

    def test_save_creates_joblib_file(self, registry, trained_dt):
        path = registry.save(trained_dt, "test_model")
        assert path.exists()
        assert path.suffix == ".joblib"

    def test_save_returns_correct_path(self, registry, trained_dt):
        path = registry.save(trained_dt, "angra_dt")
        assert path.name == "angra_dt.joblib"

    def test_load_recovers_model(self, registry, trained_dt):
        registry.save(trained_dt, "my_model")
        loaded = registry.load("my_model")
        assert loaded is not None

    def test_loaded_model_has_same_params(self, registry, trained_dt):
        registry.save(trained_dt, "my_model")
        loaded = registry.load("my_model")
        assert loaded.get_params() == trained_dt.get_params()

    def test_loaded_model_predicts_identically(self, registry, trained_dt):
        import numpy as np

        registry.save(trained_dt, "my_model")
        loaded = registry.load("my_model")

        X_test = np.random.default_rng(0).uniform(0, 1, (10, 3))
        assert list(trained_dt.predict(X_test)) == list(loaded.predict(X_test))

    def test_load_raises_for_missing_model(self, registry):
        with pytest.raises(FileNotFoundError, match="Execute scripts/run_training.py"):
            registry.load("nao_existe")

    def test_list_models_empty_initially(self, registry):
        assert registry.list_models() == []

    def test_list_models_returns_names(self, registry, trained_dt):
        registry.save(trained_dt, "usina_a_dt")
        registry.save(trained_dt, "usina_b_rf")
        names = registry.list_models()
        assert "usina_a_dt" in names
        assert "usina_b_rf" in names

    def test_list_models_sorted(self, registry, trained_dt):
        registry.save(trained_dt, "z_model")
        registry.save(trained_dt, "a_model")
        assert registry.list_models() == ["a_model", "z_model"]

    def test_exists_true_after_save(self, registry, trained_dt):
        registry.save(trained_dt, "exists_test")
        assert registry.exists("exists_test") is True

    def test_exists_false_before_save(self, registry):
        assert registry.exists("not_saved") is False
