# src/data/extractors/climate.py
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from geopy import distance

from src.config.settings import Config
from src.data.extractors.base import BaseExtractor
from src.utils.logger import logger

# Linhas do cabeçalho dos CSVs do INMET (índices base 0)
_INMET_STATION_ID_ROW = 3
_INMET_LAT_ROW = 4
_INMET_LON_ROW = 5
_INMET_HEADERS_ROW = 8
_INMET_DATA_START_ROW = 9

Coords = Tuple[float, float]  # (latitude, longitude)


def _parse_inmet_header(csv_rows: List[List[str]]) -> Dict[str, Coords]:
    """
    Extrai id_estacao → (lat, lon) do cabeçalho de um CSV INMET.

    O INMET usa separador ';' e os metadados ficam nas primeiras linhas.
    """
    station_id = csv_rows[_INMET_STATION_ID_ROW][1]
    lat = float(csv_rows[_INMET_LAT_ROW][1].replace(",", "."))
    lon = float(csv_rows[_INMET_LON_ROW][1].replace(",", "."))
    return station_id, (lat, lon)


def _haversine_km(coord1: Coords, coord2: Coords) -> float:
    """Distância em km entre dois pontos (lat, lon) via geopy."""
    return distance.distance(coord1, coord2).km


class ClimateDataExtractor(BaseExtractor):
    """
    Extrai dados climáticos dos CSVs do INMET e associa às usinas.

    Para cada usina (identificada por coordenadas), encontra as
    N estações meteorológicas mais próximas e combina seus dados
    em um único DataFrame com sufixo numérico por estação.

    Melhoria sobre o legado (gerador_dados_climaticos.py):
    - Sem os.getcwd() nem paths Windows
    - _parse_inmet_header centraliza o parsing frágil do INMET
    - num_nearest_stations configurável via Config
    - Sem estado global no módulo
    - Erros por arquivo são isolados
    """

    def __init__(
        self,
        source_dir: Path,
        num_nearest: int = Config.NUM_NEAREST_STATIONS,
    ) -> None:
        super().__init__(source_dir)
        self.num_nearest = num_nearest

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """
        Não utilizado diretamente — use extract_for_plant() por usina,
        ou ETLPipeline.run() para o fluxo completo.

        Retorna todas as estações disponíveis como DataFrame de metadados.
        """
        return self.get_station_metadata()

    def get_station_metadata(self) -> pd.DataFrame:
        """
        Retorna DataFrame com id, lat, lon de todas as estações INMET
        encontradas recursivamente em source_dir.
        """
        records = []
        for year_dir in sorted(self.source_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for csv_path in sorted(year_dir.glob("*.csv")):
                try:
                    rows = self._read_csv_rows(csv_path)
                    station_id, (lat, lon) = _parse_inmet_header(rows)
                    records.append({"station_id": station_id, "lat": lat, "lon": lon})
                except Exception as e:
                    logger.debug(f"  Metadado ignorado {csv_path.name}: {e}")

        df = pd.DataFrame(records).drop_duplicates(subset="station_id").reset_index(drop=True)
        logger.info(f"Estações INMET encontradas: {len(df)}")
        return df

    def nearest_stations(
        self, plant_coords: Coords, station_meta: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Retorna as N estações mais próximas a plant_coords.

        Args:
            plant_coords: (lat, lon) da usina
            station_meta: DataFrame de get_station_metadata()
        """
        distances = station_meta.apply(
            lambda row: _haversine_km(plant_coords, (row["lat"], row["lon"])),
            axis=1,
        )
        return (
            station_meta.assign(distance_km=distances)
            .sort_values("distance_km")
            .head(self.num_nearest)
            .reset_index(drop=True)
        )

    def extract_for_stations(
        self, station_ids: List[str]
    ) -> pd.DataFrame:
        """
        Lê e combina dados climáticos das estações especificadas para
        todos os anos disponíveis em source_dir.

        Cada estação recebe um sufixo numérico nas colunas (0, 1, 2)
        para permitir a seleção posterior da melhor estação.

        Returns:
            DataFrame com índice datetime e colunas sufixadas por estação
        """
        combined: pd.DataFrame = pd.DataFrame()

        for year_dir in sorted(self.source_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            year_df: pd.DataFrame = pd.DataFrame()
            for position, station_id in enumerate(station_ids):
                station_df = self._load_station_year(year_dir, station_id, position)
                if station_df is None:
                    continue
                year_df = (
                    station_df
                    if year_df.empty
                    else year_df.merge(station_df, left_index=True, right_index=True)
                )

            if not year_df.empty:
                combined = pd.concat([combined, year_df])

        logger.debug(f"extract_for_stations: {len(combined)} linhas para {station_ids}")
        return combined

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _load_station_year(
        self, year_dir: Path, station_id: str, position: int
    ) -> pd.DataFrame | None:
        """Lê o CSV de uma estação em um ano específico."""
        for csv_path in year_dir.glob("*.csv"):
            try:
                rows = self._read_csv_rows(csv_path)
                if rows[_INMET_STATION_ID_ROW][1] != station_id:
                    continue
                return self._parse_inmet_data(rows, position)
            except Exception as e:
                logger.debug(f"  {csv_path.name} ignorado: {e}")
        return None

    @staticmethod
    def _read_csv_rows(path: Path) -> List[List[str]]:
        """Lê CSV com separador ';' e retorna lista de linhas."""
        with open(path, encoding="utf-8", errors="replace") as f:
            return list(csv.reader(f, delimiter=";"))

    @staticmethod
    def _parse_inmet_data(rows: List[List[str]], position: int) -> pd.DataFrame:
        """
        Converte as linhas brutas do INMET em DataFrame com índice datetime.

        Adiciona sufixo `position` em cada coluna para permitir merge
        de múltiplas estações sem colisão de nomes.
        """
        raw_headers = rows[_INMET_HEADERS_ROW]
        headers = [f"{h}{position}" for h in raw_headers]

        df = pd.DataFrame(rows[_INMET_DATA_START_ROW:], columns=headers)

        # Renomeia colunas de data/hora e cria índice datetime
        date_col = f"Data{position}"
        hour_col = f"Hora UTC{position}"
        datetime_col = f"Date{position}"

        df[hour_col] = df[hour_col].apply(
            lambda x: x[:2] + ":" + x[2:4] + ":00" if len(x) >= 4 else "00:00:00"
        )
        df[datetime_col] = pd.to_datetime(
            df[date_col] + " " + df[hour_col],
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce",
        )
        df = df.dropna(subset=[datetime_col])
        df = df.set_index(datetime_col)
        df = df.drop(columns=[date_col, hour_col], errors="ignore")
        # Renomeia RADIACAO GLOBAL para 'radiation{position}' (padrão interno)
        radiation_col = f"RADIACAO GLOBAL (Kj/m²){position}"
        if radiation_col in df.columns:
            df = df.rename(columns={radiation_col: f"radiation{position}"})

        return df
