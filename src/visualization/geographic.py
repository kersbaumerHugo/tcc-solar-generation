# src/visualization/geographic.py
"""
Visualizações geográficas de plantas solares e estações meteorológicas.

Usa Basemap (mpl_toolkits.basemap) quando disponível para adicionar o fundo
cartográfico do Brasil. Se Basemap não estiver instalado, gera um scatter plot
simples em coordenadas lat/lon — útil em ambientes sem a dependência pesada.

Basemap é uma biblioteca legada e de instalação complexa. A alternativa moderna
seria Cartopy, mas Basemap está nos requisitos originais do projeto (CLAUDE.md).
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

try:
    from mpl_toolkits.basemap import Basemap
    _BASEMAP_AVAILABLE = True
except ImportError:
    _BASEMAP_AVAILABLE = False


# Bounding box do Brasil em graus decimais
_BRAZIL_BOUNDS = {"llcrnrlon": -74, "llcrnrlat": -34, "urcrnrlon": -28, "urcrnrlat": 6}


def _setup_map(ax: plt.Axes) -> Optional["Basemap"]:
    """
    Configura o mapa de fundo no Axes fornecido.

    Retorna a instância de Basemap se disponível, None caso contrário.
    Sem Basemap, configura apenas os limites e rótulos de eixo como fallback.
    """
    if _BASEMAP_AVAILABLE:
        m = Basemap(
            projection="merc",
            ax=ax,
            **_BRAZIL_BOUNDS,
        )
        m.drawmapboundary(fill_color="#99ffff")
        m.drawcountries(linewidth=0.8)
        m.drawstates(color="gray", linewidth=0.5)
        m.fillcontinents(color="#cc9966", lake_color="#99ffff")
        return m

    # Fallback sem Basemap
    ax.set_xlim(_BRAZIL_BOUNDS["llcrnrlon"], _BRAZIL_BOUNDS["urcrnrlon"])
    ax.set_ylim(_BRAZIL_BOUNDS["llcrnrlat"], _BRAZIL_BOUNDS["urcrnrlat"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    return None


def plot_plants(
    lats: Sequence[float],
    lons: Sequence[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota a localização das plantas solares no mapa do Brasil.

    Args:
        lats:      Latitudes das plantas (graus decimais).
        lons:      Longitudes das plantas (graus decimais).
        title:     Título do gráfico. Se None, usa o padrão.
        save_path: Caminho para salvar a figura (ex: "output/plantas.png").
                   Se None, não salva.

    Returns:
        Figure do matplotlib (o chamador decide se exibe com plt.show()).
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    m = _setup_map(ax)

    if m is not None:
        x, y = m(list(lons), list(lats))
    else:
        x, y = list(lons), list(lats)

    ax.scatter(x, y, s=15, marker="o", color="black", zorder=5, label="Plantas solares")
    ax.set_title(title or f"Distribuição das {len(lats)} plantas pelo Brasil", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_stations(
    lats: Sequence[float],
    lons: Sequence[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota a localização das estações meteorológicas INMET no mapa do Brasil.

    Args:
        lats:      Latitudes das estações (graus decimais).
        lons:      Longitudes das estações (graus decimais).
        title:     Título do gráfico. Se None, usa o padrão.
        save_path: Caminho para salvar a figura.

    Returns:
        Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    m = _setup_map(ax)

    if m is not None:
        x, y = m(list(lons), list(lats))
    else:
        x, y = list(lons), list(lats)

    ax.scatter(x, y, s=10, marker="x", color="blue", zorder=5, label="Estações INMET")
    ax.set_title(title or f"Distribuição das {len(lats)} estações INMET pelo Brasil", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_combined(
    plant_lats: Sequence[float],
    plant_lons: Sequence[float],
    station_lats: Sequence[float],
    station_lons: Sequence[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota plantas solares e estações meteorológicas no mesmo mapa.

    Útil para verificar visualmente a proximidade entre cada planta e
    as estações INMET selecionadas pelo ETL (haversine nearest).

    Args:
        plant_lats / plant_lons:     Coordenadas das plantas.
        station_lats / station_lons: Coordenadas das estações INMET.
        title:     Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    m = _setup_map(ax)

    if m is not None:
        px, py = m(list(plant_lons), list(plant_lats))
        sx, sy = m(list(station_lons), list(station_lats))
    else:
        px, py = list(plant_lons), list(plant_lats)
        sx, sy = list(station_lons), list(station_lats)

    ax.scatter(sx, sy, s=8, marker="x", color="blue", zorder=4, label="Estações INMET")
    ax.scatter(px, py, s=20, marker="o", color="black", zorder=5, label="Plantas solares")

    n_plants = len(plant_lats)
    n_stations = len(station_lats)
    ax.set_title(
        title or f"{n_plants} plantas + {n_stations} estações INMET",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
