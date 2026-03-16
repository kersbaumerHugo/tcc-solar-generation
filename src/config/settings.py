from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Diretórios
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    GERACAO_DIR: Path = DATA_DIR / "geracao_csv"
    CLIMA_DIR: Path = DATA_DIR / "clima_csv"
    FULL_DIR: Path = DATA_DIR / "full"
    
    # Mapeamento de colunas
    COLUMN_MAPPING = {
        "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh": "GERACAO",
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "PRECIPITACAO",
        "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "TEMPERATURA",
        "UMIDADE RELATIVA DO AR, HORARIA (%)": "UMIDADE",
        "radiation": "RADIACAO"
    }
    
    # Hiperparâmetros
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    MIN_DATASET_SIZE: int = 10000