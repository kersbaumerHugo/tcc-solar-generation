from pathlib import Path
from types import MappingProxyType
from typing import Tuple


class Config:
    """
    Fonte única de verdade para todas as configurações do projeto.

    Classe simples (não dataclass): não precisamos de instâncias, apenas de
    constantes de classe acessíveis via Config.ATRIBUTO.

    Tipos imutáveis em todos os níveis:
    - Path: imutável por natureza
    - str, int, float: imutáveis por natureza
    - tuple: imutável (no lugar de list)
    - MappingProxyType: dict imutável em profundidade 1
      (valores internos são tuples, portanto totalmente imutável)
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
    MODELS_DIR: Path = BASE_DIR / "models"

    # Dados brutos (input do ETL)
    RAW_GERACAO_DIR: Path = DATA_DIR / "raw" / "geracao"
    RAW_CLIMA_DIR: Path = DATA_DIR / "raw" / "climatico"

    # -------------------------------------------------------------------------
    # Parâmetros de extração (ETL)
    # -------------------------------------------------------------------------
    # Arquivo Excel de geração: planilha 8 (base 0), header na linha 14 (base 0)
    EXCEL_SHEET_INDEX: int = 8
    EXCEL_HEADER_ROW: int = 14

    # Número de estações meteorológicas mais próximas a usar por usina
    NUM_NEAREST_STATIONS: int = 3

    # -------------------------------------------------------------------------
    # Mapeamento de colunas (fonte única de verdade)
    # Chave: nome original no arquivo bruto → Valor: nome padronizado interno
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

    # Colunas descartadas antes do treinamento (sem poder preditivo ou redundantes).
    # Aplicadas em DataProcessor.drop_irrelevant_features().
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
    NAN_THRESHOLD: float = 0.75
    MIN_DATASET_SIZE: int = 10_000

    # -------------------------------------------------------------------------
    # Configuração de treinamento
    # -------------------------------------------------------------------------
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # -------------------------------------------------------------------------
    # Grids de hiperparâmetros
    # Valores são tuples (imutáveis) — GridSearchCV aceita qualquer iterável.
    # MappingProxyType impede adição/remoção de chaves.
    # random_state é passado diretamente ao estimador, não ao grid.
    # -------------------------------------------------------------------------
    DT_GRID: MappingProxyType = MappingProxyType(
        {
            "max_features": (1.0,),
            "max_depth": (5, 7, 10, 12, 15, 20),
            "criterion": ("squared_error",),
        }
    )

    RF_GRID: MappingProxyType = MappingProxyType(
        {
            "n_estimators": (1, 2, 5, 10, 15, 20),
            "max_features": (1.0,),
            "max_depth": (5, 7, 10, 12, 15, 20),
            "criterion": ("squared_error",),
        }
    )
