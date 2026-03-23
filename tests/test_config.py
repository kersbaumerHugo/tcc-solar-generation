# tests/test_config.py
from pathlib import Path
from types import MappingProxyType

import pytest

from src.config.settings import Config


def test_column_mapping_is_complete():
    """COLUMN_MAPPING deve conter as 10 colunas esperadas."""
    expected_values = {
        "GERACAO", "PRECIPITACAO", "PATM", "TEMPERATURA",
        "TEMPERATURA_ORVALHO", "UMIDADE", "VEL_VENTO",
        "COORD_N", "COORD_E", "RADIACAO",
    }
    assert set(Config.COLUMN_MAPPING.values()) == expected_values


def test_column_mapping_contains_radiation_key():
    assert Config.COLUMN_MAPPING["radiation"] == "RADIACAO"


def test_column_mapping_is_immutable():
    """MappingProxyType impede adição de novas chaves."""
    with pytest.raises(TypeError):
        Config.COLUMN_MAPPING["nova_chave"] = "VALOR"  # type: ignore


def test_features_to_drop_are_valid_columns():
    """Todas as colunas em FEATURES_TO_DROP devem existir no COLUMN_MAPPING."""
    mapped_columns = set(Config.COLUMN_MAPPING.values())
    for col in Config.FEATURES_TO_DROP:
        assert col in mapped_columns, f"'{col}' não existe no COLUMN_MAPPING"


def test_features_to_drop_is_tuple():
    """tuple impede Config.FEATURES_TO_DROP.append(...) acidental."""
    assert isinstance(Config.FEATURES_TO_DROP, tuple)


def test_features_to_drop_does_not_include_target():
    """TARGET_COLUMN não deve estar em FEATURES_TO_DROP — removê-la quebraria o modelo."""
    assert Config.TARGET_COLUMN not in Config.FEATURES_TO_DROP


def test_grids_are_mapping_proxy():
    """DT_GRID e RF_GRID devem ser imutáveis."""
    assert isinstance(Config.DT_GRID, MappingProxyType)
    assert isinstance(Config.RF_GRID, MappingProxyType)


def test_dt_grid_values_are_tuples():
    """Valores do grid devem ser tuples (imutáveis), não lists."""
    for key, value in Config.DT_GRID.items():
        assert isinstance(value, tuple), f"DT_GRID['{key}'] deveria ser tuple, é {type(value)}"


def test_rf_grid_values_are_tuples():
    for key, value in Config.RF_GRID.items():
        assert isinstance(value, tuple), f"RF_GRID['{key}'] deveria ser tuple, é {type(value)}"


def test_nan_threshold_is_between_0_and_1():
    assert 0.0 < Config.NAN_THRESHOLD < 1.0


def test_test_size_is_between_0_and_1():
    assert 0.0 < Config.TEST_SIZE < 1.0


def test_base_dir_is_a_path():
    assert isinstance(Config.BASE_DIR, Path)


def test_models_dir_is_relative_to_base():
    assert str(Config.MODELS_DIR).startswith(str(Config.BASE_DIR))


def test_dirs_are_relative_to_base():
    assert str(Config.DATA_DIR).startswith(str(Config.BASE_DIR))
    assert str(Config.FULL_DIR).startswith(str(Config.BASE_DIR))


def test_dt_grid_has_required_keys():
    assert "max_depth" in Config.DT_GRID
    assert "criterion" in Config.DT_GRID


def test_rf_grid_has_n_estimators():
    assert "n_estimators" in Config.RF_GRID


def test_random_state_not_in_grids():
    """random_state não é hiperparâmetro — deve ser passado ao estimador."""
    assert "random_state" not in Config.DT_GRID
    assert "random_state" not in Config.RF_GRID
