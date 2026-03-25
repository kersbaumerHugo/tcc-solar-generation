# src/models/registry.py
from pathlib import Path
from typing import Any, List

import joblib

from src.utils.logger import logger


class ModelRegistry:
    """
    Persiste e recupera modelos treinados usando joblib.

    Por que joblib e não pickle?
    joblib é otimizado para objetos numpy/sklearn: comprime arrays grandes de
    forma eficiente e é mais rápido que pickle para RandomForest com muitas
    árvores. É a forma canônica de serializar modelos sklearn.

    Convenção de nomes: {usina}_{modelo}.joblib
    Exemplo: ANGRA_decision_tree.joblib
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ModelRegistry inicializado em: {models_dir}")

    def save(self, model: Any, name: str) -> Path:
        """
        Serializa e salva o modelo em disco.

        Args:
            model: Estimador sklearn treinado
            name: Identificador do modelo (ex: "ANGRA_decision_tree")

        Returns:
            Path do arquivo salvo
        """
        path = self.models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info(f"Modelo salvo: {path}")
        return path

    def load(self, name: str) -> Any:
        """
        Carrega e retorna um modelo previamente salvo.

        Args:
            name: Identificador do modelo (sem extensão)

        Raises:
            FileNotFoundError: Se o modelo não existir no registry
        """
        path = self.models_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo '{name}' não encontrado em {self.models_dir}\n"
                "Execute scripts/run_training.py primeiro."
            )
        model = joblib.load(path)
        logger.info(f"Modelo carregado: {path}")
        return model

    def list_models(self) -> List[str]:
        """Retorna os nomes de todos os modelos salvos (sem extensão)."""
        return sorted(p.stem for p in self.models_dir.glob("*.joblib"))

    def exists(self, name: str) -> bool:
        """Retorna True se o modelo existe no registry."""
        return (self.models_dir / f"{name}.joblib").exists()
