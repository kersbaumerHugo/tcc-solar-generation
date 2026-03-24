# src/data/extractors/generation.py
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import Config
from src.data.extractors.base import BaseExtractor
from src.utils.logger import logger

# Colunas lidas do Excel de geração (as demais são descartadas no merge)
_GENERATION_COLS = (
    "Sigla da Usina",
    "Hora",
    "Dia",
    "Fonte",
    "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh",
)

# Colunas de coordenadas que identificam a localização da usina
_COORD_COLS = ("NumCoordNEmpreendimento", "NumCoordEEmpreendimento")


class SolarGenerationExtractor(BaseExtractor):
    """
    Extrai dados de geração fotovoltaica dos arquivos Excel da ANEEL.

    Cada arquivo Excel contém dados horários de múltiplas usinas.
    O extrator lê a planilha correta (índice 8), filtra apenas usinas
    solares (Fonte == "Solar Fotovoltaica") e consolida todos os
    arquivos em um único DataFrame.

    Melhoria sobre o legado (gerador_csv_geracao.py):
    - Pathlib em vez de os.getcwd() + strings Windows (\\)
    - Sem estado global no nível do módulo
    - Cada arquivo com erro é ignorado individualmente (sem abortar)
    - sheet_name e header_row configuráveis via Config
    """

    def __init__(
        self,
        source_dir: Path,
        sheet_index: int = Config.EXCEL_SHEET_INDEX,
        header_row: int = Config.EXCEL_HEADER_ROW,
    ) -> None:
        super().__init__(source_dir)
        self.sheet_index = sheet_index
        self.header_row = header_row

    def extract(self) -> pd.DataFrame:
        """
        Lê todos os .xlsx de source_dir e retorna DataFrame consolidado
        com apenas as usinas fotovoltaicas.

        Returns:
            DataFrame com colunas: Sigla da Usina, Hora, Dia, Geração...,
            NumCoordNEmpreendimento, NumCoordEEmpreendimento
        """
        excel_files = sorted(self.source_dir.glob("*.xlsx"))
        logger.info(f"SolarGenerationExtractor: {len(excel_files)} arquivo(s) encontrado(s)")

        frames: List[pd.DataFrame] = []
        for path in excel_files:
            try:
                df = pd.read_excel(
                    path,
                    sheet_name=self.sheet_index,
                    header=self.header_row,
                    index_col=None,
                )
                df = df[df["Fonte"] == "Solar Fotovoltaica"]
                frames.append(df)
                logger.debug(f"  {path.name}: {len(df)} linhas (solar)")
            except Exception as e:
                logger.warning(f"  Ignorando {path.name}: {e}")

        if not frames:
            logger.warning("Nenhum dado de geração extraído.")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        logger.info(f"Extração concluída: {len(result)} linhas, {result['Sigla da Usina'].nunique()} usinas")
        return result

    def save_per_plant(self, df: pd.DataFrame, output_dir: Path) -> List[Path]:
        """
        Salva um CSV por usina em output_dir/{USINA}_geracao.csv.

        Args:
            df: DataFrame consolidado (resultado de extract())
            output_dir: diretório de saída (Config.GERACAO_DIR)

        Returns:
            Lista de Paths dos arquivos criados
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for usina in sorted(df["Sigla da Usina"].unique()):
            plant_df = df[df["Sigla da Usina"] == usina].sort_values("Dia")
            out_path = output_dir / f"{usina}_geracao.csv"
            plant_df.to_csv(out_path, index=False)
            saved.append(out_path)
            logger.debug(f"  Salvo: {out_path.name}")

        logger.info(f"save_per_plant: {len(saved)} arquivo(s) gerado(s) em {output_dir}")
        return saved
