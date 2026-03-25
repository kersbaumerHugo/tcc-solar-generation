"""
Pipeline de treinamento dos modelos.

Equivalente refatorado do modelo_final.py legado.

Uso:
    python scripts/run_training.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.config.settings import Config
from src.data.loaders import FullDatasetLoader
from src.data.processors import DataProcessor
from src.data.validators.quality_checks import DataQualityChecker
from src.features.engineering import FeatureImportanceAnalyzer
from src.models.decision_tree import DecisionTreeStrategy
from src.models.evaluator import ModelEvaluator
from src.models.random_forest import RandomForestStrategy
from src.models.registry import ModelRegistry
from src.models.trainer import ModelTrainer
from src.utils.logger import logger


def main() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE DE TREINAMENTO - Previsão de Geração Solar")
    logger.info("=" * 60)

    results = []

    loader = FullDatasetLoader(Config.DATA_DIR)
    datasets = loader.load()

    processor = DataProcessor()
    checker = DataQualityChecker()
    trainer = ModelTrainer(
        strategies=[DecisionTreeStrategy(), RandomForestStrategy()]
    )
    evaluator = ModelEvaluator()
    analyzer = FeatureImportanceAnalyzer()
    registry = ModelRegistry(Config.MODELS_DIR)

    for usina_name, df_raw in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando: {usina_name}")
        logger.info(f"{'='*60}")

        try:
            # Validação de qualidade antes de qualquer transformação
            quality = checker.check(df_raw)
            if not quality.passed:
                logger.warning(f"Problemas de qualidade em {usina_name}:\n{quality}")

            # ETL + feature engineering
            df = processor.rename_columns(df_raw)
            df = processor.add_temporal_features(df)
            df = processor.handle_missing_values(df)
            df = processor.drop_irrelevant_features(df)

            if len(df) < trainer.min_samples:
                logger.warning(
                    f"Dataset muito pequeno: {len(df)} < {trainer.min_samples} — ignorando"
                )
                continue

            # Split ANTES da normalização (evita data leakage)
            X_train, X_test, y_train, y_test = trainer.split_data(df)
            X_train = processor.normalize_data(X_train)
            X_test = processor.transform(X_test)

            feature_names = list(X_train.columns)

            # Treina todas as estratégias de uma vez
            trained = trainer.train_all(X_train, y_train)

            for strategy_name, (model, _) in trained.items():
                metrics = evaluator.evaluate(model, X_test, y_test, strategy_name)
                analyzer.log_report(model, feature_names, strategy_name)
                model_key = f"{usina_name}_{strategy_name.lower().replace(' ', '_')}"
                registry.save(model, model_key)

                results.append(
                    {
                        "usina": usina_name,
                        "modelo": strategy_name,
                        "n_samples": len(df),
                        "n_features": len(feature_names),
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                    }
                )

            # Persiste o processor com o scaler já fitado — obrigatório para que
            # run_evaluation.py aplique a mesma escala usada no treinamento
            registry.save(processor, f"{usina_name}_processor")

        except Exception as e:
            logger.error(f"Erro ao processar {usina_name}: {e}", exc_info=True)
            continue

    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS FINAIS")
    logger.info("=" * 60)

    if not results:
        logger.warning("Nenhum dataset processado com sucesso. Verifique os logs acima.")
        return

    df_results = pd.DataFrame(results)
    logger.info(f"\n{df_results.to_string()}")

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Config.RESULTS_DIR / "training_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nResultados salvos em: {output_path}")
    logger.info(f"Modelos salvos em: {Config.MODELS_DIR}")


if __name__ == "__main__":
    main()
