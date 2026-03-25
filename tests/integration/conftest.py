# tests/integration/conftest.py
"""
Fixtures para testes de integração.

Diferente das fixtures unitárias (conftest.py raiz), aqui os dados
são maiores e mais realistas para exercitar o pipeline completo:
- 15.000 linhas (acima de MIN_DATASET_SIZE=10.000)
- Sazonalidade realista (radiação zerada à noite)
- NaN introduzidos propositalmente para testar limpeza
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def pipeline_df() -> pd.DataFrame:
    """
    DataFrame sintético pronto para entrar no pipeline de treinamento.

    scope="module": gerado uma vez por arquivo de teste — integração é
    mais custosa que unitário, vale o cache.

    Simula dados horários de 2 anos com padrão solar realista:
    - RADIACAO: 0 à noite (20h-8h), distribuição normal durante o dia
    - GERACAO: correlacionada com RADIACAO + ruído
    - NaN em ~5% das linhas de TEMPERATURA e UMIDADE
    """
    rng = np.random.default_rng(seed=42)
    n = 15_000
    index = pd.date_range("2021-01-01 00:00", periods=n, freq="h")

    hours = index.hour
    is_daytime = (hours >= 9) & (hours <= 19)

    radiacao = np.where(
        is_daytime,
        np.clip(rng.normal(500, 150, n), 0, 1200),
        0.0,
    )
    temperatura = rng.normal(25, 5, n)
    umidade = rng.uniform(40, 90, n)
    geracao = radiacao * 0.003 + rng.normal(0, 0.1, n)
    geracao = np.clip(geracao, 0, None)

    df = pd.DataFrame(
        {
            "GERACAO": geracao,
            "RADIACAO": radiacao,
            "TEMPERATURA": temperatura,
            "UMIDADE": umidade,
        },
        index=index,
    )

    # Introduz NaN realistas (~5%)
    nan_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[df.index[nan_idx[:len(nan_idx)//2]], "TEMPERATURA"] = np.nan
    df.loc[df.index[nan_idx[len(nan_idx)//2:]], "UMIDADE"] = np.nan

    return df


@pytest.fixture(scope="module")
def raw_df(pipeline_df) -> pd.DataFrame:
    """
    Versão do pipeline_df com nomes de colunas originais (pré-rename).
    Simula um CSV de entrada bruto antes do DataCleaner.
    """
    return pipeline_df.rename(columns={
        "GERACAO": "Geração no Centro de Gravidade - MW médios (Gp,j) - MWh",
        "RADIACAO": "radiation",
    })
