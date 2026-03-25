# tests/test_visualization.py
"""
Testes para os módulos de visualização.

matplotlib.use("Agg") é chamado antes de qualquer import de pyplot para
garantir que os testes rodem sem display (CI, ambientes headless).
Deve vir antes do import de matplotlib.pyplot.
"""
import matplotlib
matplotlib.use("Agg")  # noqa: E402 — deve ser antes de qualquer import pyplot
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.visualization.geographic import plot_combined, plot_plants, plot_stations
from src.visualization.performance import (
    plot_feature_importance,
    plot_metrics_comparison,
    plot_prediction_vs_real,
    plot_train_test_split,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Fecha todas as figuras matplotlib após cada teste para evitar memory leak."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def time_series():
    """Par de séries temporais (real e previsto) para testes de performance."""
    index = pd.date_range("2023-01-01", periods=48, freq="h")
    rng = np.random.default_rng(seed=0)
    y_real = pd.Series(rng.uniform(0, 1, 48), index=index, name="GERACAO")
    y_pred = pd.Series(rng.uniform(0, 1, 48), index=index, name="GERACAO_PREVISTA")
    return y_real, y_pred


@pytest.fixture
def importance_df():
    features = [f"feat_{i}" for i in range(15)]
    importances = np.linspace(0.15, 0.01, 15)
    return pd.DataFrame({"feature": features, "importance": importances})


@pytest.fixture
def metrics_df():
    return pd.DataFrame({
        "usina": ["USINA_A", "USINA_B", "USINA_C"],
        "decision_tree_r2": [0.85, 0.72, 0.91],
        "decision_tree_rmse": [0.05, 0.08, 0.03],
        "random_forest_r2": [0.88, 0.75, 0.93],
        "random_forest_rmse": [0.04, 0.07, 0.02],
    })


@pytest.fixture
def split_data(sample_df):
    n = len(sample_df)
    X_train = sample_df.iloc[: n // 2]
    X_test = sample_df.iloc[n // 2 :]
    return X_train, X_test


@pytest.fixture
def coords():
    rng = np.random.default_rng(seed=1)
    lats = rng.uniform(-30, 0, 5).tolist()
    lons = rng.uniform(-60, -40, 5).tolist()
    return lats, lons


# ---------------------------------------------------------------------------
# performance.py
# ---------------------------------------------------------------------------


class TestPlotPredictionVsReal:
    def test_returns_figure(self, time_series):
        y_real, y_pred = time_series
        fig = plot_prediction_vs_real(y_real, y_pred)
        assert isinstance(fig, Figure)

    def test_has_two_lines(self, time_series):
        y_real, y_pred = time_series
        fig = plot_prediction_vs_real(y_real, y_pred)
        ax = fig.axes[0]
        assert len(ax.lines) == 2

    def test_custom_title(self, time_series):
        y_real, y_pred = time_series
        fig = plot_prediction_vs_real(y_real, y_pred, title="Meu Título")
        assert fig.axes[0].get_title() == "Meu Título"

    def test_saves_to_file(self, tmp_path, time_series):
        y_real, y_pred = time_series
        path = str(tmp_path / "pred.png")
        plot_prediction_vs_real(y_real, y_pred, save_path=path)
        assert (tmp_path / "pred.png").exists()


class TestPlotFeatureImportance:
    def test_returns_figure(self, importance_df):
        fig = plot_feature_importance(importance_df)
        assert isinstance(fig, Figure)

    def test_respects_n_limit(self, importance_df):
        fig = plot_feature_importance(importance_df, n=5)
        ax = fig.axes[0]
        assert len(ax.patches) == 5

    def test_default_shows_10(self, importance_df):
        fig = plot_feature_importance(importance_df, n=10)
        ax = fig.axes[0]
        assert len(ax.patches) == 10

    def test_saves_to_file(self, tmp_path, importance_df):
        path = str(tmp_path / "importance.png")
        plot_feature_importance(importance_df, save_path=path)
        assert (tmp_path / "importance.png").exists()


class TestPlotMetricsComparison:
    def test_returns_figure(self, metrics_df):
        fig = plot_metrics_comparison(metrics_df, metric="r2")
        assert isinstance(fig, Figure)

    def test_raises_for_unknown_metric(self, metrics_df):
        with pytest.raises(ValueError, match="Nenhuma coluna"):
            plot_metrics_comparison(metrics_df, metric="f1_score")

    def test_bar_count_equals_usinas_times_models(self, metrics_df):
        fig = plot_metrics_comparison(metrics_df, metric="r2")
        ax = fig.axes[0]
        # 3 usinas × 2 modelos = 6 barras
        assert len(ax.patches) == 6

    def test_saves_to_file(self, tmp_path, metrics_df):
        path = str(tmp_path / "metrics.png")
        plot_metrics_comparison(metrics_df, save_path=path)
        assert (tmp_path / "metrics.png").exists()


class TestPlotTrainTestSplit:
    def test_returns_figure(self, split_data):
        X_train, X_test = split_data
        fig = plot_train_test_split(X_train, X_test)
        assert isinstance(fig, Figure)

    def test_has_two_bars(self, split_data):
        X_train, X_test = split_data
        fig = plot_train_test_split(X_train, X_test)
        ax = fig.axes[0]
        assert len(ax.patches) == 2

    def test_usina_name_in_title(self, split_data):
        X_train, X_test = split_data
        fig = plot_train_test_split(X_train, X_test, usina_name="ANGRA")
        assert "ANGRA" in fig.axes[0].get_title()

    def test_saves_to_file(self, tmp_path, split_data):
        X_train, X_test = split_data
        path = str(tmp_path / "split.png")
        plot_train_test_split(X_train, X_test, save_path=path)
        assert (tmp_path / "split.png").exists()


# ---------------------------------------------------------------------------
# geographic.py
# ---------------------------------------------------------------------------


class TestPlotPlants:
    def test_returns_figure(self, coords):
        lats, lons = coords
        fig = plot_plants(lats, lons)
        assert isinstance(fig, Figure)

    def test_custom_title(self, coords):
        lats, lons = coords
        fig = plot_plants(lats, lons, title="Minhas Plantas")
        assert "Minhas Plantas" in fig.axes[0].get_title()

    def test_saves_to_file(self, tmp_path, coords):
        lats, lons = coords
        path = str(tmp_path / "plants.png")
        plot_plants(lats, lons, save_path=path)
        assert (tmp_path / "plants.png").exists()


class TestPlotStations:
    def test_returns_figure(self, coords):
        lats, lons = coords
        fig = plot_stations(lats, lons)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_path, coords):
        lats, lons = coords
        path = str(tmp_path / "stations.png")
        plot_stations(lats, lons, save_path=path)
        assert (tmp_path / "stations.png").exists()


class TestPlotCombined:
    def test_returns_figure(self, coords):
        lats, lons = coords
        fig = plot_combined(lats, lons, lats, lons)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_path, coords):
        lats, lons = coords
        path = str(tmp_path / "combined.png")
        plot_combined(lats, lons, lats, lons, save_path=path)
        assert (tmp_path / "combined.png").exists()
