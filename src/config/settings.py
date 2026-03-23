from pathlib import Path
from types import MappingProxyType
from typing import Dict, Tuple


class Config:
    """
    Fonte única de verdade para todas as configurações do projeto.

    Classe simples (não dataclass): não precisamos de instâncias, apenas de
    constantes de classe acessíveis via Config.ATRIBUTO.

    Tipos imutáveis (tuple, MappingProxyType) evitam mutação acidental em runtime:
        Config.FEATURES_TO_DROP.append("X")  # ← TypeError: tuple não suporta append
    """

    # -------------------------------------------------------------------------
    # Diretórios
    # -------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    GERACAO_DIR: Path = DATA_DIR / "geracao_csv"
    CLIMA_DIR: Path = DATA_DIR / "clima_csv"
    FULL_DIR: Path = DATA_DIR / "full"
    RESULTS_DIR: Path = BASE_DIR / "results"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # -------------------------------------------------------------------------
    # Mapeamento de colunas (fonte única de verdade)
    # Chave: nome original no arquivo bruto → Valor: nome padronizado interno
    # MappingProxyType torna o dict imutável — nenhum módulo pode sobrescrever
    # -------------------------------------------------------------------------
    COLUMN_MAPPING: MappingProxyType = MappingProxyType(
        {
            "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh": "GERACAO",
            "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "PRECIPITACAO",
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "PATM",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "TEMPERATURA",
            "TEMPERATURA DO PONTO DE ORVALHO (°C)": "TEMPERATURA_ORVALHO",
            "UMIDADE RELATIVA DO AR, HORARIA (%)": "UMIDADE",
            "VENTO, VELOCIDADE HORARIA (m/s)": "VEL_VENTO",
            "NumCoordNEmpreendimento": "COORD_N",
            "NumCoordEEmpreendimento": "COORD_E",
            "radiation": "RADIACAO",
        }
    )

    # -------------------------------------------------------------------------
    # Features do modelo
    # -------------------------------------------------------------------------
    TARGET_COLUMN: str = "GERACAO"

    # tuple (imutável) em vez de list — evita Config.FEATURES_TO_DROP.append(...)
    FEATURES_TO_DROP: Tuple[str, ...] = (
        "VEL_VENTO",
        "COORD_N",
        "COORD_E",
        "TEMPERATURA_ORVALHO",
        "PATM",
        "PRECIPITACAO",
    )

    # -------------------------------------------------------------------------
    # Thresholds de qualidade de dados
    # -------------------------------------------------------------------------
    # Colunas com mais de NAN_THRESHOLD de valores ausentes são descartadas
    NAN_THRESHOLD: float = 0.75

    # Datasets com menos amostras que isso são ignorados no treinamento
    MIN_DATASET_SIZE: int = 10_000

    # -------------------------------------------------------------------------
    # Configuração de treinamento
    # -------------------------------------------------------------------------
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # -------------------------------------------------------------------------
    # Grids de hiperparâmetros
    # random_state é passado diretamente ao estimador, não ao grid
    # -------------------------------------------------------------------------
    DT_GRID: Dict[str, list] = {
        "max_features": [1.0],
        "max_depth": [5, 7, 10, 12, 15, 20],
        "criterion": ["squared_error"],
    }

    RF_GRID: Dict[str, list] = {
        "n_estimators": [1, 2, 5, 10, 15, 20],
        "max_features": [1.0],
        "max_depth": [5, 7, 10, 12, 15, 20],
        "criterion": ["squared_error"],
    }
