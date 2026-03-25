# src/visualization/performance.py
"""
Gráficos de performance dos modelos de predição de geração solar.

Todos os métodos retornam Figure para que o chamador decida se exibe
(plt.show()) ou salva (fig.savefig()). Nenhuma função chama plt.show()
diretamente — isso evita bloquear scripts batch e facilita testes.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_prediction_vs_real(
    y_real: pd.Series,
    y_pred: pd.Series,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota geração real vs prevista ao longo do tempo (plot2 do legado).

    Migrado de modelo_final.py: o legado filtrava a partir de "2023-02-10"
    e plotava diretamente. Aqui o chamador controla o slice temporal antes
    de passar os dados.

    Args:
        y_real:    Série com valores reais, índice datetime.
        y_pred:    Série com valores previstos, mesmo índice.
        title:     Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(y_real.index, y_real.values, label="Real", linewidth=1.2, color="steelblue")
    ax.plot(y_pred.index, y_pred.values, label="Previsto", linewidth=1.0,
            color="darkorange", linestyle="--")

    ax.set_title(title or "Geração Real vs Prevista")
    ax.set_xlabel("Data")
    ax.set_ylabel("Geração (MWh normalizado)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    n: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota importância das features como gráfico de barras horizontal.

    Args:
        importance_df: DataFrame com colunas "feature" e "importance",
                       ordenado por importância decrescente
                       (saída de FeatureImportanceAnalyzer.get_importance()).
        n:             Número máximo de features a exibir.
        title:         Título do gráfico.
        save_path:     Caminho para salvar a figura.

    Returns:
        Figure do matplotlib.
    """
    df = importance_df.head(n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.4)))

    bars = ax.barh(df["feature"], df["importance"], color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_title(title or f"Top {n} Features por Importância")
    ax.set_xlabel("Importância relativa")
    ax.set_xlim(0, df["importance"].max() * 1.15)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_comparison(
    df_results: pd.DataFrame,
    metric: str = "r2",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Compara a métrica escolhida entre modelos e usinas (bar chart agrupado).

    Args:
        df_results: DataFrame de saída do run_evaluation.py, com colunas
                    "usina", "{model}_r2", "{model}_rmse", etc.
        metric:     Sufixo da métrica a comparar (ex: "r2", "rmse", "mae").
        title:      Título do gráfico.
        save_path:  Caminho para salvar a figura.

    Returns:
        Figure do matplotlib.
    """
    # Descobre quais modelos estão no DataFrame pelo sufixo da métrica
    model_cols = [c for c in df_results.columns if c.endswith(f"_{metric}")]
    if not model_cols:
        raise ValueError(
            f"Nenhuma coluna com sufixo '_{metric}' encontrada em df_results.\n"
            f"Colunas disponíveis: {list(df_results.columns)}"
        )

    n_usinas = len(df_results)
    n_models = len(model_cols)
    x = np.arange(n_usinas)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_usinas * 1.2), 5))

    for i, col in enumerate(model_cols):
        model_name = col[: -len(f"_{metric}")]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_results[col], width, label=model_name)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_title(title or f"Comparação de {metric.upper()} por Usina e Modelo")
    ax.set_xlabel("Usina")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(df_results["usina"], rotation=30, ha="right", fontsize=8)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    usina_name: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Visualiza a divisão temporal treino/teste (plot1 do legado).

    Mostra dois intervalos: treino (azul) e teste (laranja) numa timeline,
    útil para verificar que a divisão preserva a ordem cronológica.

    Args:
        X_train:     Features de treino (índice datetime).
        X_test:      Features de teste (índice datetime).
        usina_name:  Nome da usina para o título.
        save_path:   Caminho para salvar a figura.

    Returns:
        Figure do matplotlib.
    """
    train_start = X_train.index.min()
    train_end = X_train.index.max()
    test_start = X_test.index.min()
    test_end = X_test.index.max()

    # mdates.date2num converte Timestamps para o float interno do matplotlib
    # (dias desde epoch). Misturar Timestamp (left) com int (width) no barh
    # produz um eixo sem escala de datas; date2num + xaxis_date() resolve.
    train_days = (train_end - train_start).days
    test_days = (test_end - test_start).days

    fig, ax = plt.subplots(figsize=(10, 2))

    ax.barh([0], train_days, left=mdates.date2num(train_start), color="steelblue", label="Treino")
    ax.barh([0], test_days, left=mdates.date2num(test_start), color="darkorange", label="Teste")
    ax.xaxis_date()

    ax.set_yticks([])
    ax.set_xlabel("Data")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        f"Divisão Treino / Teste — {usina_name}" if usina_name else "Divisão Treino / Teste"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
