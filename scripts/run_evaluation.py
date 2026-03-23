"""
Pipeline de avaliação de modelos treinados.

Carrega modelos salvos pelo run_training.py e gera um relatório de métricas
comparativo entre Decision Tree e Random Forest para cada usina.

Uso:
    python scripts/run_evaluation.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.config.settings import Config
from src.data.loaders import FullDatasetLoader
from src.data.processors import DataProcessor
from src.features.engineering import FeatureImportanceAnalyzer
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.trainer import ModelTrainer
from src.utils.logger import logger

MODEL_TYPES = ("decision_tree", "random_forest")


def main() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE DE AVALIAÇÃO - Previsão de Geração Solar")
    logger.info("=" * 60)

    registry = ModelRegistry(Config.MODELS_DIR)
    available = registry.list_models()

    if not available:
        logger.error(
            f"Nenhum modelo encontrado em {Config.MODELS_DIR}.\n"
            "Execute scripts/run_training.py primeiro."
        )
        return

    logger.info(f"{len(available)} modelo(s) disponível(is): {available}")

    loader = FullDatasetLoader(Config.DATA_DIR)
    datasets = loader.load()

    processor = DataProcessor()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    analyzer = FeatureImportanceAnalyzer()

    results = []

    for usina_name, df_raw in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Avaliando: {usina_name}")
        logger.info(f"{'='*60}")

        try:
            # Replicar o mesmo pipeline de pré-processamento do treinamento
            df = processor.rename_columns(df_raw)
            df = processor.add_temporal_features(df)
            df = processor.handle_missing_values(df)
            df = processor.drop_irrelevant_features(df)

            if len(df) < trainer.min_samples:
                logger.warning(f"Dataset muito pequeno ({len(df)} linhas) — ignorando")
                continue

            X_train, X_test, y_train, y_test = trainer.split_data(df)
            X_train = processor.normalize_data(X_train)
            X_test = processor.transform(X_test)

            feature_names = list(X_train.columns)
            row: dict = {"usina": usina_name, "n_test_samples": len(X_test)}

            for model_type in MODEL_TYPES:
                model_name = f"{usina_name}_{model_type}"

                if not registry.exists(model_name):
                    logger.warning(f"Modelo não encontrado: {model_name} — ignorando")
                    continue

                model = registry.load(model_name)
                metrics = evaluator.evaluate(model, X_test, y_test, model_type)
                analyzer.log_report(model, feature_names, model_type)

                prefix = model_type[:2]  # "dt" ou "ra" → mais curto na tabela
                for metric_name, value in metrics.items():
                    row[f"{prefix}_{metric_name}"] = value

            results.append(row)

        except Exception as e:
            logger.error(f"Erro ao avaliar {usina_name}: {e}", exc_info=True)
            continue

    if not results:
        logger.warning("Nenhuma usina avaliada com sucesso.")
        return

    df_results = pd.DataFrame(results)
    logger.info(f"\n{'='*60}\nRELATÓRIO FINAL\n{'='*60}")
    logger.info(f"\n{df_results.to_string()}")

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Config.RESULTS_DIR / "evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nRelatório salvo em: {output_path}")


if __name__ == "__main__":
    main()
