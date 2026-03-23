# src/data/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import pandas as pd


class BaseLoader(ABC):
    """
    Contrato base para todos os loaders de dados do projeto.

    Por que ABC?
    Quando tivermos loaders diferentes (CSV, Parquet, banco de dados),
    todos vão garantir a mesma interface: quem chama sabe que pode usar
    .load() em qualquer loader sem precisar saber como ele funciona por dentro.
    Isso é o Princípio da Inversão de Dependência (SOLID-D).

    Uso esperado:
        class MeuLoader(BaseLoader):
            def load(self) -> List[Tuple[str, pd.DataFrame]]:
                ...
    """

    def __init__(self, data_dir: Path) -> None:
        self._validate_dir(data_dir)
        self.data_dir = data_dir

    @abstractmethod
    def load(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Carrega dados e retorna lista de (identificador, DataFrame).

        O identificador é tipicamente o nome da usina ou da estação.
        """
        ...

    @staticmethod
    def _validate_dir(path: Path) -> None:
        """Garante que o diretório existe antes de tentar ler arquivos."""
        if not path.exists():
            raise FileNotFoundError(
                f"Diretório de dados não encontrado: {path}\n"
                "Verifique se os dados foram baixados e colocados no lugar correto."
            )
