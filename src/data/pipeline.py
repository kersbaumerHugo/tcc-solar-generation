# src/data/pipeline.py
import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import Config
from src.data.extractors.climate import ClimateDataExtractor
from src.data.extractors.generation import SolarGenerationExtractor
from src.utils.logger import logger

# Colunas do dataset de geração que não entram no dataset final
_GENERATION_COLS_TO_DROP = (
    "nomes_match", "nome_geracao", "nome_ceg", "nome_upper",
    "DatGeracaoConjuntoDados", "NomEmpreendimento", "IdeNucleoCEG",
    "CodCEG", "SigUFPrincipal", "SigTipoGeracao", "DscFaseUsina",
    "DscOrigemCombustivel", "DscFonteCombustivel", "DscTipoOutorga",
    "NomFonteCombustivel", "DatEntradaOperacao", "MdaPotenciaOutorgadaKw",
    "MdaPotenciaFiscalizadaKw", "MdaGarantiaFisicaKw", "DatInicioVigencia",
    "DatFimVigencia", "DscPropriRegimePariticipacao", "DscSubBacia",
    "DscMuninicpios", "Sigla da Usina", "Hora", "Dia", "IdcGeracaoQualificada",
)

# Colunas de horário máx/mín que não contribuem para o modelo horário
_CLIMATE_COLS_TO_DROP = (
    "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)",
    "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))",
    "VENTO, RAJADA MAXIMA (m/s)",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)",
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)",
    "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
)


class ETLPipeline:
    """
    Orquestra o pipeline completo de ETL do projeto:

    1. Extração: lê Excel de geração (ANEEL) + CSVs climáticos (INMET)
    2. Associação: encontra as N estações mais próximas a cada usina
    3. Merge: combina geração + clima no índice datetime
    4. Seleção: escolhe a melhor estação (menor taxa de NaN/zeros)
    5. Limpeza: remove colunas desnecessárias
    6. Saída: salva *_full.csv por usina

    Equivalente refatorado do merge_geracao_clima.py legado.

    Melhorias sobre o legado:
    - Sem os.getcwd() nem paths Windows
    - Cada método tem responsabilidade única e é testável
    - Logging estruturado em vez de print()
    - _select_best_station retorna -1 com log ao invés de variável muda
    - day_to_date encapsulado com tratamento de ValueError
    """

    def __init__(
        self,
        gen_extractor: SolarGenerationExtractor,
        clim_extractor: ClimateDataExtractor,
        output_dir: Path,
        # geracao_dir recebe os CSVs intermediários por usina (geração bruta
        # antes do merge). Separado de output_dir (que contém *_full.csv) para
        # manter rastreabilidade. Injetado aqui para respeitar DIP: o pipeline
        # não deve conhecer Config diretamente, quem chama decide o path.
        geracao_dir: Path = Config.GERACAO_DIR,
        nan_threshold: float = Config.NAN_THRESHOLD,
    ) -> None:
        self.gen_extractor = gen_extractor
        self.clim_extractor = clim_extractor
        self.output_dir = output_dir
        self.geracao_dir = geracao_dir
        self.nan_threshold = nan_threshold
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Executa o pipeline completo.

        Returns:
            DataFrame de resumo com uma linha por usina processada
            (qualidade dos dados, datas, estação escolhida).
        """
        logger.info("=" * 60)
        logger.info("ETL PIPELINE - Iniciando")
        logger.info("=" * 60)

        gen_df = self.gen_extractor.extract()
        if gen_df.empty:
            logger.error("Sem dados de geração. Abortando pipeline.")
            return pd.DataFrame()

        station_meta = self.clim_extractor.get_station_metadata()
        if station_meta.empty:
            logger.error("Sem metadados de estações climáticas. Abortando pipeline.")
            return pd.DataFrame()

        # Salva CSVs intermediários de geração por usina
        self.gen_extractor.save_per_plant(gen_df, self.geracao_dir)

        summaries = []
        plants = sorted(gen_df["Sigla da Usina"].unique())
        logger.info(f"Processando {len(plants)} usina(s)...")

        for usina in plants:
            logger.info(f"\n  Usina: {usina}")
            try:
                summary = self._process_plant(usina, gen_df, station_meta)
                if summary is not None:
                    summaries.append(summary)
            except Exception as e:
                logger.error(f"  Erro em {usina}: {e}", exc_info=True)

        df_summary = pd.DataFrame(summaries)
        if not df_summary.empty:
            summary_path = self.output_dir.parent / "info_geral.csv"
            df_summary.to_csv(summary_path, index=False)
            logger.info(f"\nResumo salvo em: {summary_path}")

        logger.info(f"ETL concluído: {len(summaries)}/{len(plants)} usinas processadas")
        return df_summary

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _process_plant(
        self,
        usina: str,
        gen_df: pd.DataFrame,
        station_meta: pd.DataFrame,
    ) -> Optional[dict]:
        """Processa uma usina: extrai coords, busca estações, merge, salva."""
        plant_df = gen_df[gen_df["Sigla da Usina"] == usina].copy()

        # Coordenadas da usina
        plant_coords = self._get_plant_coords(plant_df)
        if plant_coords is None:
            logger.warning(f"  {usina}: coordenadas não encontradas — ignorando")
            return None

        # Estações mais próximas
        nearest = self.clim_extractor.nearest_stations(plant_coords, station_meta)
        station_ids = nearest["station_id"].tolist()
        logger.debug(f"  Estações: {station_ids}")

        # Dados climáticos das estações
        clim_df = self.clim_extractor.extract_for_stations(station_ids)
        if clim_df.empty:
            logger.warning(f"  {usina}: sem dados climáticos — ignorando")
            return None

        # Prepara datetime no dataset de geração
        gen_prepared = self._prepare_generation_datetime(plant_df)
        if gen_prepared is None:
            logger.warning(f"  {usina}: falha no parse de datetime de geração — ignorando")
            return None

        # Merge geração + clima
        merged = clim_df.merge(gen_prepared, left_index=True, right_index=True)
        if merged.empty:
            logger.warning(f"  {usina}: merge resultou em DataFrame vazio — ignorando")
            return None

        # Converte strings para float (antes de resample/seleção de estação)
        merged = self._coerce_numeric(merged)

        # Seleciona melhor estação
        daily = merged.resample("D").sum(numeric_only=True)
        num_days = len(daily)
        best_station = self._select_best_station(daily, num_days)

        if best_station == "-1":
            logger.warning(f"  {usina}: nenhuma estação com qualidade suficiente — ignorando")
            return None

        merged = self._keep_best_station(merged, daily, best_station)

        # Limpeza de colunas desnecessárias — executada APÓS _keep_best_station
        # porque _parse_inmet_data sufixia todas as colunas climáticas com o
        # índice da estação (ex: "UMIDADE...0"). Se _drop_extra_columns fosse
        # chamado antes, os nomes em _CLIMATE_COLS_TO_DROP (sem sufixo) nunca
        # casariam e nenhuma coluna seria removida. Depois do keep, os sufixos
        # já foram retirados e os nomes batem com o tuple de drop.
        merged = self._drop_extra_columns(merged)

        # Salva
        out_path = self.output_dir / f"{usina}_full.csv"
        merged.to_csv(out_path)
        logger.info(f"  Salvo: {out_path.name} ({len(merged)} linhas)")

        return self._build_summary(usina, merged, daily, best_station, plant_coords)

    @staticmethod
    def _get_plant_coords(plant_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Extrai (lat, lon) da usina a partir do DataFrame de geração."""
        for lat_col, lon_col in (
            ("NumCoordNEmpreendimento", "NumCoordEEmpreendimento"),
        ):
            if lat_col in plant_df.columns and lon_col in plant_df.columns:
                try:
                    lat = float(str(plant_df[lat_col].iloc[0]).replace(",", "."))
                    lon = float(str(plant_df[lon_col].iloc[0]).replace(",", "."))
                    return (lat, lon)
                except (ValueError, IndexError):
                    pass
        return None

    @staticmethod
    def _prepare_generation_datetime(plant_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Cria coluna datetime a partir de Dia + Hora e define como índice.

        O campo 'Dia' pode vir em formatos diferentes:
        - Número de dias desde 1900 (formato Excel legado)
        - String "YYYY-MM-DD"
        - String "DD/MM/YYYY"
        """
        df = plant_df.copy()
        df["Hora"] = df["Hora"].apply(
            lambda x: f" {int(float(x)) % 24:02d}:00:00"
        )
        df["Dia"] = df["Dia"].apply(ETLPipeline._normalize_day)
        try:
            df["Date"] = pd.to_datetime(
                df["Dia"] + df["Hora"], format="%Y-%m-%d %H:%M:%S"
            )
        except Exception as e:
            logger.debug(f"  Falha no parse de datetime: {e}")
            return None

        df = df.set_index("Date")
        # Converte vírgulas → pontos em colunas string
        df = df.apply(lambda col: col.str.replace(",", ".") if col.dtype == object else col)
        return df

    @staticmethod
    def _normalize_day(day_val: str) -> str:
        """
        Normaliza o campo 'Dia' para formato YYYY-MM-DD.

        Três casos cobertos:
        1. Número inteiro (dias desde 01/01/1900, formato Excel)
        2. "YYYY-MM-DD" ou "YYYY-MM-DD HH:MM:SS" (já no formato certo)
        3. "DD/MM/YYYY" (formato alternativo do legado)
        """
        day_str = str(day_val).strip()

        # Caso 1: número Excel (dias desde 1900)
        try:
            day_int = int(float(day_str))
            if day_int < 50000:  # valores Excel razoáveis
                result = dt.datetime(1900, 1, 1) + dt.timedelta(days=day_int - 2)
                return result.strftime("%Y-%m-%d")
        except ValueError:
            pass

        # Caso 2: "YYYY-MM-DD ..." — trunca em 10 chars
        if len(day_str) >= 10 and day_str[4] == "-":
            return day_str[:10]

        # Caso 3: "DD/MM/YYYY"
        try:
            parsed = dt.datetime.strptime(day_str, "%d/%m/%Y")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            pass

        return day_str

    def _select_best_station(self, daily: pd.DataFrame, num_days: int) -> str:
        """
        Escolhe a estação com menor taxa de dados inválidos.

        Retorna o índice da melhor estação como string ("0", "1", "2")
        ou "-1" se nenhuma atender ao threshold.
        """
        for station_idx in range(Config.NUM_NEAREST_STATIONS):
            idx_str = str(station_idx)
            station_cols = [c for c in daily.columns if c.endswith(idx_str)]

            if not station_cols:
                continue

            acceptable = True
            for col in station_cols:
                base_col = col[:-1]  # remove sufixo numérico
                if "PRECIPITACAO" in base_col or "PRECIPITAÇÃO" in base_col:
                    continue
                zero_ratio = (daily[col] == 0).sum() / num_days
                nan_ratio = daily[col].isna().sum() / num_days
                if zero_ratio > self.nan_threshold or nan_ratio > self.nan_threshold:
                    acceptable = False
                    break

            if acceptable:
                return idx_str

        return "-1"

    @staticmethod
    def _keep_best_station(
        df: pd.DataFrame, daily: pd.DataFrame, best: str
    ) -> pd.DataFrame:
        """
        Renomeia as colunas da estação escolhida (remove sufixo)
        e descarta as colunas das outras estações.
        """
        station_cols = [c for c in df.columns if c[-1:].isdigit()]
        col_bases = list({c[:-1] for c in station_cols})

        for base in col_bases:
            for station_idx in range(Config.NUM_NEAREST_STATIONS):
                col = f"{base}{station_idx}"
                if col not in df.columns:
                    continue
                if str(station_idx) == best:
                    df = df.rename(columns={col: base})
                else:
                    df = df.drop(columns=[col])

        return df

    @staticmethod
    def _drop_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols_present = [c for c in _GENERATION_COLS_TO_DROP if c in df.columns]
        cols_present += [c for c in _CLIMATE_COLS_TO_DROP if c in df.columns]
        return df.drop(columns=cols_present, errors="ignore")

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Converte colunas de string para float, preservando NaN."""
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."),
                    errors="coerce",
                )
        return df

    @staticmethod
    def _build_summary(
        usina: str,
        df: pd.DataFrame,
        daily: pd.DataFrame,
        best_station: str,
        plant_coords: Tuple[float, float],
    ) -> dict:
        geracao_col = "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh"
        return {
            "Usina": usina,
            "Total de dias": len(daily),
            "Dias com Geração == 0": int((daily.get(geracao_col, pd.Series()) == 0).sum()),
            "Dias com Geração NaN": int(daily.get(geracao_col, pd.Series()).isna().sum()),
            "Start Date": daily.index.min(),
            "End Date": daily.index.max(),
            "Estação escolhida": best_station,
            "Latitude": plant_coords[0],
            "Longitude": plant_coords[1],
        }
