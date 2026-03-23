# tests/test_loaders.py
from pathlib import Path

import pandas as pd
import pytest

from src.data.base import BaseLoader
from src.data.loaders import FullDatasetLoader


# ---------------------------------------------------------------------------
# Testes do contrato BaseLoader
# ---------------------------------------------------------------------------

class TestBaseLoader:
    def test_raises_when_directory_does_not_exist(self, tmp_path):
        """BaseLoader deve lançar FileNotFoundError para diretórios inexistentes."""
        missing = tmp_path / "inexistente" / "full"
        with pytest.raises(FileNotFoundError, match="Diretório de dados não encontrado"):
            FullDatasetLoader(missing / "..")  # parent existe, mas "full/" não

    def test_stores_data_dir(self, tmp_path):
        full_dir = tmp_path / "full"
        full_dir.mkdir()
        loader = FullDatasetLoader(tmp_path)
        assert loader.data_dir == full_dir

    def test_is_abstract(self):
        """Não deve ser possível instanciar BaseLoader diretamente."""
        with pytest.raises(TypeError):
            BaseLoader(Path("."))  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Testes de FullDatasetLoader
# ---------------------------------------------------------------------------

@pytest.fixture
def full_data_dir(tmp_path) -> Path:
    """Cria estrutura data/full/ com dois CSVs de usina fake."""
    full_dir = tmp_path / "full"
    full_dir.mkdir()

    index = pd.date_range("2023-01-01 09:00", periods=5, freq="h")
    df = pd.DataFrame({"GERACAO": [1.0] * 5, "RADIACAO": [500.0] * 5}, index=index)

    df.to_csv(full_dir / "USINA_A_full.csv")
    df.to_csv(full_dir / "USINA_B_full.csv")

    return tmp_path


@pytest.fixture
def loader(full_data_dir) -> FullDatasetLoader:
    return FullDatasetLoader(full_data_dir)


class TestFullDatasetLoader:
    def test_load_returns_list(self, loader):
        result = loader.load()
        assert isinstance(result, list)

    def test_load_returns_correct_count(self, loader):
        result = loader.load()
        assert len(result) == 2

    def test_load_returns_tuples_of_name_and_dataframe(self, loader):
        result = loader.load()
        for name, df in result:
            assert isinstance(name, str)
            assert isinstance(df, pd.DataFrame)

    def test_load_strips_full_suffix_from_name(self, loader):
        result = loader.load()
        names = [name for name, _ in result]
        assert "USINA_A" in names
        assert "USINA_B" in names
        assert "USINA_A_full" not in names

    def test_load_results_are_sorted_by_filename(self, loader):
        result = loader.load()
        names = [name for name, _ in result]
        assert names == sorted(names)

    def test_load_ignores_corrupt_csv_and_continues(self, full_data_dir):
        """Arquivo corrompido não deve interromper o carregamento dos demais."""
        corrupt = full_data_dir / "full" / "USINA_C_full.csv"
        corrupt.write_text("isso não é um csv válido\x00\x00\x00")

        loader = FullDatasetLoader(full_data_dir)
        result = loader.load()

        names = [name for name, _ in result]
        assert "USINA_A" in names
        assert "USINA_B" in names

    def test_load_empty_directory_returns_empty_list(self, tmp_path):
        empty_full = tmp_path / "full"
        empty_full.mkdir()
        loader = FullDatasetLoader(tmp_path)
        assert loader.load() == []
