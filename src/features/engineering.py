# src/features/engineering.py
from typing import List

import pandas as pd

from src.utils.logger import logger


class FeatureImportanceAnalyzer:
    """
    Analisa e reporta a importância das features de modelos baseados em árvore.

    Por que feature importance?
    DecisionTree e RandomForest calculam a "impurity reduction" de cada feature
    ao longo de todas as divisões. Isso permite identificar quais variáveis
    climáticas (radiação, temperatura, umidade) mais contribuem para a previsão
    de geração — insight valioso para o TCC.

    Limitação: feature importance de RandomForest pode superestimar features
    com muitos valores únicos (alta cardinalidade). Para este projeto, todas as
    features são contínuas, então o bias é pequeno.
    """

    def get_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Retorna um DataFrame com a importância de cada feature, ordenado
        do mais para o menos importante.

        Args:
            model: Estimador sklearn com atributo feature_importances_
            feature_names: Nomes das colunas de X_train

        Returns:
            DataFrame com colunas ['feature', 'importance']
        """
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(
                f"{type(model).__name__} não possui feature_importances_. "
                "Use DecisionTreeRegressor ou RandomForestRegressor."
            )

        df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        logger.debug(f"Feature importance calculada para {len(df)} features")
        return df

    def top_features(self, model, feature_names: List[str], n: int = 10) -> pd.DataFrame:
        """
        Retorna as n features mais importantes.

        Args:
            n: Número de features a retornar (padrão: 10)
        """
        return self.get_importance(model, feature_names).head(n)

    def log_report(self, model, feature_names: List[str], model_name: str = "Model") -> None:
        """Loga as top 10 features mais importantes no formato tabular."""
        top = self.top_features(model, feature_names)
        logger.info(f"\nTop features — {model_name}:")
        for _, row in top.iterrows():
            bar = "█" * int(row["importance"] * 40)
            logger.info(f"  {row['feature']:<20} {row['importance']:.4f}  {bar}")
