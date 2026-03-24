# src/data/extractors/base.py
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseExtractor(ABC):
    """
    Contrato base para todos os extratores de dados brutos do projeto.

    Extratores são responsáveis pela etapa E do ETL: lêem dados brutos
    (Excel, CSV do INMET) e retornam DataFrames padronizados, sem
    fazer transformações de negócio.

    Separar extração de transformação (DataProcessor) respeita SRP e
    permite testar cada etapa de forma isolada.

    Uso esperado:
        extractor = SolarGenerationExtractor(Config.RAW_GERACAO_DIR)
        df = extractor.extract()
    """

    def __init__(self, source_dir: Path) -> None:
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Diretório de dados brutos não encontrado: {source_dir}\n"
                "Verifique se os dados foram baixados corretamente."
            )
        self.source_dir = source_dir

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Lê dados brutos e retorna DataFrame padronizado."""
        ...
