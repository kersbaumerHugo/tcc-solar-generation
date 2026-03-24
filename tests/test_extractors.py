# tests/test_extractors.py
import csv
from pathlib import Path

import pandas as pd
import pytest

from src.data.extractors.base import BaseExtractor
from src.data.extractors.climate import ClimateDataExtractor, _haversine_km
from src.data.extractors.generation import SolarGenerationExtractor


# ---------------------------------------------------------------------------
# Helpers para criar dados de teste
# ---------------------------------------------------------------------------

def _write_inmet_csv(path: Path, station_id: str, lat: float, lon: float) -> None:
    """
    Cria um CSV INMET mínimo válido para testes.

    O formato real do INMET usa ';' como separador, com metadados
    nas primeiras 8 linhas: cada linha tem [campo, valor, ...].
    Ex: "CÓDIGO (WMO):;A001;" → ao ler com delimiter=';', row[1] = 'A001'.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["REGIÃO:", "SUDESTE", ""],
        ["UF:", "SP", ""],
        ["ESTAÇÃO:", "CAMPINAS", ""],
        ["CÓDIGO (WMO):", station_id, ""],
        ["LATITUDE:", str(lat), ""],
        ["LONGITUDE:", str(lon), ""],
        ["ALTITUDE:", "640", ""],
        ["DATA DE FUNDAÇÃO:", "1920", ""],
        # headers (índice 8)
        [
            "Data", "Hora UTC", "RADIACAO GLOBAL (Kj/m²)",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
            "UMIDADE RELATIVA DO AR, HORARIA (%)", ""
        ],
        # dados (índice 9+)
        ["2023/01/01", "0000", "0", "25", "70", ""],
        ["2023/01/01", "0100", "0", "24", "72", ""],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows)


def _make_climate_dir(tmp_path: Path) -> Path:
    """Cria estrutura data/raw/climatico/2023/ com 2 estações."""
    clim_dir = tmp_path / "climatico"
    year_dir = clim_dir / "2023"
    _write_inmet_csv(year_dir / "A001_2023.csv", "A001", -22.9, -47.1)
    _write_inmet_csv(year_dir / "A002_2023.csv", "A002", -23.5, -46.6)
    return clim_dir


# ---------------------------------------------------------------------------
# BaseExtractor
# ---------------------------------------------------------------------------

class TestBaseExtractor:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseExtractor(Path("."))  # type: ignore[abstract]

    def test_raises_for_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dados brutos não encontrado"):
            SolarGenerationExtractor(tmp_path / "nao_existe")


# ---------------------------------------------------------------------------
# SolarGenerationExtractor
# ---------------------------------------------------------------------------

class TestSolarGenerationExtractor:
    def test_extract_returns_dataframe(self, tmp_path):
        """Diretório vazio deve retornar DataFrame vazio sem erro."""
        extractor = SolarGenerationExtractor(tmp_path)
        result = extractor.extract()
        assert isinstance(result, pd.DataFrame)

    def test_save_per_plant_creates_csvs(self, tmp_path):
        """save_per_plant deve criar um CSV por usina."""
        extractor = SolarGenerationExtractor(tmp_path)
        df = pd.DataFrame({
            "Sigla da Usina": ["USINA_A", "USINA_A", "USINA_B"],
            "Dia": ["2023-01-01", "2023-01-02", "2023-01-01"],
            "Hora": [9, 10, 9],
        })
        output = tmp_path / "geracao_csv"
        extractor.save_per_plant(df, output)

        assert (output / "USINA_A_geracao.csv").exists()
        assert (output / "USINA_B_geracao.csv").exists()

    def test_save_per_plant_sorts_by_dia(self, tmp_path):
        extractor = SolarGenerationExtractor(tmp_path)
        df = pd.DataFrame({
            "Sigla da Usina": ["USINA_A", "USINA_A"],
            "Dia": ["2023-01-02", "2023-01-01"],
        })
        output = tmp_path / "out"
        extractor.save_per_plant(df, output)
        saved = pd.read_csv(output / "USINA_A_geracao.csv")
        assert list(saved["Dia"]) == ["2023-01-01", "2023-01-02"]


# ---------------------------------------------------------------------------
# ClimateDataExtractor
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km((-22.9, -47.1), (-22.9, -47.1)) == pytest.approx(0.0)

    def test_known_distance(self):
        # São Paulo → Rio de Janeiro ≈ 360 km em linha reta
        sp = (-23.55, -46.63)
        rio = (-22.91, -43.17)
        dist = _haversine_km(sp, rio)
        assert 340 < dist < 380


class TestClimateDataExtractor:
    def test_get_station_metadata_returns_dataframe(self, tmp_path):
        clim_dir = _make_climate_dir(tmp_path)
        extractor = ClimateDataExtractor(clim_dir)
        meta = extractor.get_station_metadata()
        assert isinstance(meta, pd.DataFrame)

    def test_get_station_metadata_has_required_columns(self, tmp_path):
        clim_dir = _make_climate_dir(tmp_path)
        extractor = ClimateDataExtractor(clim_dir)
        meta = extractor.get_station_metadata()
        assert set(meta.columns) >= {"station_id", "lat", "lon"}

    def test_get_station_metadata_finds_both_stations(self, tmp_path):
        clim_dir = _make_climate_dir(tmp_path)
        extractor = ClimateDataExtractor(clim_dir)
        meta = extractor.get_station_metadata()
        assert set(meta["station_id"]) == {"A001", "A002"}

    def test_nearest_stations_returns_n_rows(self, tmp_path):
        clim_dir = _make_climate_dir(tmp_path)
        extractor = ClimateDataExtractor(clim_dir, num_nearest=1)
        meta = extractor.get_station_metadata()
        nearest = extractor.nearest_stations((-22.9, -47.1), meta)
        assert len(nearest) == 1

    def test_nearest_stations_closest_first(self, tmp_path):
        clim_dir = _make_climate_dir(tmp_path)
        extractor = ClimateDataExtractor(clim_dir, num_nearest=2)
        meta = extractor.get_station_metadata()
        # A001 está em (-22.9, -47.1) — deve ser a mais próxima de (-22.9, -47.1)
        nearest = extractor.nearest_stations((-22.9, -47.1), meta)
        assert nearest.iloc[0]["station_id"] == "A001"

    def test_nearest_stations_has_distance_column(self, tmp_path):
        clim_dir = _make_climate_dir(tmp_path)
        extractor = ClimateDataExtractor(clim_dir)
        meta = extractor.get_station_metadata()
        nearest = extractor.nearest_stations((-22.9, -47.1), meta)
        assert "distance_km" in nearest.columns


# ---------------------------------------------------------------------------
# ETLPipeline._normalize_day  (lógica crítica de negócio)
# ---------------------------------------------------------------------------

class TestNormalizeDay:
    """
    _normalize_day é o método mais frágil do legado. Testa os 3 formatos.
    """

    from src.data.pipeline import ETLPipeline

    def test_excel_serial_date(self):
        from src.data.pipeline import ETLPipeline
        # Dia 44927 em Excel = 2023-01-01
        result = ETLPipeline._normalize_day("44927")
        assert result == "2023-01-01"

    def test_iso_format(self):
        from src.data.pipeline import ETLPipeline
        assert ETLPipeline._normalize_day("2023-06-15") == "2023-06-15"

    def test_iso_with_time(self):
        from src.data.pipeline import ETLPipeline
        assert ETLPipeline._normalize_day("2023-06-15 00:00:00") == "2023-06-15"

    def test_br_format(self):
        from src.data.pipeline import ETLPipeline
        assert ETLPipeline._normalize_day("15/06/2023") == "2023-06-15"
