# tests/conftest.py
"""
Fixtures compartilhadas entre todos os módulos de teste.

conftest.py é carregado automaticamente pelo pytest — não precisa de importação.
Fixtures definidas aqui ficam disponíveis para qualquer arquivo tests/test_*.py.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    DataFrame processado mínimo: colunas já renomeadas, índice datetime.

    Usado em testes de processor, trainer e evaluator.
    Cobre dois meses diferentes para evitar o bug do LabelBinarizer
    (uma única classe) e para testar features temporais corretamente.
    """
    index = pd.date_range("2023-01-15 09:00", periods=10, freq="h")
    index = index.append(pd.date_range("2023-02-15 09:00", periods=10, freq="h"))
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "GERACAO": rng.uniform(0.1, 1.0, 20),
            "RADIACAO": rng.uniform(100, 800, 20),
            "TEMPERATURA": rng.uniform(20, 35, 20),
            "UMIDADE": rng.uniform(40, 90, 20),
        },
        index=index,
    )


@pytest.fixture
def sample_df_with_nulls(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Variante do sample_df com NaN introduzidos em posições específicas.

    Útil para testar handle_missing_values sem depender de dados aleatórios.
    """
    df = sample_df.copy()
    df.loc[df.index[0], "RADIACAO"] = np.nan
    df.loc[df.index[1], "TEMPERATURA"] = np.nan
    df.loc[df.index[2], "UMIDADE"] = np.nan
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """DataFrame com nomes de colunas originais (antes do rename)."""
    index = pd.date_range("2023-01-01 09:00", periods=5, freq="h")
    return pd.DataFrame(
        {
            "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh": [1.0] * 5,
            "radiation": [500.0] * 5,
        },
        index=index,
    )
