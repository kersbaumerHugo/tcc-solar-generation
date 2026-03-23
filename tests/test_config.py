# tests/test_config.py
from pathlib import Path

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
    """Chave 'radiation' deve mapear para 'RADIACAO'."""
    assert Config.COLUMN_MAPPING["radiation"] == "RADIACAO"


def test_features_to_drop_are_valid_columns():
    """Todas as colunas em FEATURES_TO_DROP devem existir no COLUMN_MAPPING."""
    mapped_columns = set(Config.COLUMN_MAPPING.values())
    for col in Config.FEATURES_TO_DROP:
        assert col in mapped_columns, f"'{col}' não existe no COLUMN_MAPPING"


def test_nan_threshold_is_between_0_and_1():
    assert 0.0 < Config.NAN_THRESHOLD < 1.0


def test_test_size_is_between_0_and_1():
    assert 0.0 < Config.TEST_SIZE < 1.0


def test_base_dir_is_a_path():
    assert isinstance(Config.BASE_DIR, Path)


def test_dirs_are_relative_to_base():
    """DATA_DIR e subdiretórios devem estar dentro de BASE_DIR."""
    assert str(Config.DATA_DIR).startswith(str(Config.BASE_DIR))
    assert str(Config.FULL_DIR).startswith(str(Config.BASE_DIR))


def test_dt_grid_has_required_keys():
    assert "max_depth" in Config.DT_GRID
    assert "criterion" in Config.DT_GRID


def test_rf_grid_has_n_estimators():
    assert "n_estimators" in Config.RF_GRID


def test_random_state_not_in_grids():
    """
    random_state não deve estar nos grids — deve ser passado ao estimador.
    Colocá-lo no grid é um anti-pattern: não é hiperparâmetro, é semente.
    """
    assert "random_state" not in Config.DT_GRID
    assert "random_state" not in Config.RF_GRID
