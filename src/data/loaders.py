# src/data/loaders.py
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.data.base import BaseLoader
from src.utils.logger import logger


class FullDatasetLoader(BaseLoader):
    """
    Carrega os arquivos *_full.csv gerados pelo pipeline de ETL.

    Cada arquivo representa uma usina fotovoltaica: geração + dados climáticos
    já mergeados, limpos e prontos para feature engineering.

    Herda de BaseLoader:
    - Validação automática do diretório no __init__
    - Contrato .load() para uso polimórfico
    """

    def __init__(self, data_dir: Path) -> None:
        # data_dir aponta para o diretório pai (data/); o subdiretório "full/"
        # é resolvido internamente para manter a interface uniforme.
        full_dir = data_dir / "full"
        super().__init__(full_dir)

    def load(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Carrega todos os arquivos *_full.csv do diretório data/full/.

        Returns:
            Lista de (nome_usina, DataFrame) ordenada por nome de arquivo.
            Arquivos com erro de leitura são ignorados com log de aviso.
        """
        datasets: List[Tuple[str, pd.DataFrame]] = []

        csv_files = sorted(self.data_dir.glob("*_full.csv"))
        logger.info(f"Encontrados {len(csv_files)} arquivos em {self.data_dir}")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
                usina_name = file_path.stem.replace("_full", "")
                datasets.append((usina_name, df))
                logger.debug(f"Carregado: {usina_name} ({len(df)} linhas)")
            except Exception as e:
                logger.warning(f"Ignorando {file_path.name}: {e}")

        logger.info(f"Total carregado: {len(datasets)} datasets")
        return datasets
