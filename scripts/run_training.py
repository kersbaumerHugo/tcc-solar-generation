"""
Script principal de treinamento dos modelos.

Equivalente refatorado do modelo_final.py legado.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.config.settings import Config
from src.data.loaders import FullDatasetLoader
from src.data.processors import DataProcessor
from src.models.evaluator import ModelEvaluator
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
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    for usina_name, df_raw in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando: {usina_name}")
        logger.info(f"{'='*60}")

        try:
            df = processor.rename_columns(df_raw)
            df = processor.add_temporal_features(df)
            df = processor.handle_missing_values(df)

            if len(df) < trainer.min_samples:
                logger.warning(
                    f"Dataset muito pequeno: {len(df)} < {trainer.min_samples} — ignorando"
                )
                continue

            # Split ANTES da normalização: evita data leakage
            # (scaler deve aprender apenas com dados de treino)
            X_train, X_test, y_train, y_test = trainer.split_data(df)

            X_train = processor.normalize_data(X_train)
            X_test = processor.transform(X_test)

            dt_model, _ = trainer.train_decision_tree(X_train, y_train)
            dt_metrics = evaluator.evaluate(dt_model, X_test, y_test, "Decision Tree")

            rf_model, _ = trainer.train_random_forest(X_train, y_train)
            rf_metrics = evaluator.evaluate(rf_model, X_test, y_test, "Random Forest")

            results.append(
                {
                    "usina": usina_name,
                    "n_samples": len(df),
                    "dt_r2": dt_metrics["r2"],
                    "rf_r2": rf_metrics["r2"],
                }
            )

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
    logger.info(f"\n{df_results}")

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Config.RESULTS_DIR / "training_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
