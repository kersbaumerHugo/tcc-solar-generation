# src/data/loaders.py
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from src.utils.logger import logger

class DataLoader:
    """Responsável por carregar dados dos arquivos CSV"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        logger.info(f"DataLoader inicializado com diretório: {data_dir}")
    
    def load_full_datasets(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Carrega todos os datasets _full.csv
        
        Returns:
            Lista de tuplas (nome_usina, dataframe)
        """
        full_dir = self.data_dir / "full"
        datasets = []
        
        logger.info(f"Carregando datasets de {full_dir}")
        
        for file_path in full_dir.glob("*_full.csv"):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
                usina_name = file_path.stem.replace("_full", "")
                datasets.append((usina_name, df))
                logger.debug(f"Dataset carregado: {usina_name} ({len(df)} linhas)")
            except Exception as e:
                logger.error(f"Erro ao carregar {file_path}: {e}")
        
        logger.info(f"Total de {len(datasets)} datasets carregados")
        return datasets