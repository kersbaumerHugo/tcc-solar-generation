"""
Pipeline de avaliação de modelos treinados.

Carrega modelos e o DataProcessor (com scaler fitado) salvos pelo
run_training.py e gera um relatório comparativo de métricas.

Por que carregar o DataProcessor?
O MinMaxScaler aprende os parâmetros (min/max) dos dados de treino.
Usar um scaler novo ou re-fitar na avaliação produziria uma escala diferente,
corrompendo os resultados — o modelo foi treinado em uma escala, mas seria
avaliado em outra.

Os modelos disponíveis são descobertos automaticamente a partir do registry:
qualquer artefato com prefixo "{usina}_" e sufixo diferente de "_processor"
é tratado como modelo treinado. Isso elimina a necessidade de manter um dict
hardcoded (como MODEL_TYPES) que precisaria ser atualizado a cada novo algoritmo.

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


def _discover_models(usina: str, registry: ModelRegistry) -> list[str]:
    """
    Retorna os nomes dos modelos treinados disponíveis para uma usina.

    Filtra todos os artefatos do registry que começam com "{usina}_" e
    exclui o processor (que é infraestrutura, não um modelo de predição).
    """
    return [
        name for name in registry.list_models()
        if name.startswith(f"{usina}_") and not name.endswith("_processor")
    ]


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

    logger.info(f"{len(available)} artefato(s) disponível(is): {available}")

    loader = FullDatasetLoader(Config.DATA_DIR)
    datasets = loader.load()

    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    analyzer = FeatureImportanceAnalyzer()

    results = []

    for usina_name, df_raw in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Avaliando: {usina_name}")
        logger.info(f"{'='*60}")

        processor_name = f"{usina_name}_processor"
        if not registry.exists(processor_name):
            logger.warning(
                f"Processor não encontrado para {usina_name} — "
                "execute run_training.py para gerar os artefatos."
            )
            continue

        usina_models = _discover_models(usina_name, registry)
        if not usina_models:
            logger.warning(f"Nenhum modelo encontrado para {usina_name} — ignorando.")
            continue

        logger.info(f"Modelos encontrados: {usina_models}")

        try:
            # Carrega o processor com o scaler já fitado no treinamento.
            # NÃO instanciar um novo DataProcessor() aqui — o scaler seria diferente.
            processor: DataProcessor = registry.load(processor_name)

            df = processor.rename_columns(df_raw)
            df = processor.add_temporal_features(df)
            df = processor.handle_missing_values(df)
            df = processor.drop_irrelevant_features(df)

            if len(df) < trainer.min_samples:
                logger.warning(f"Dataset muito pequeno ({len(df)} linhas) — ignorando")
                continue

            X_train, X_test, y_train, y_test = trainer.split_data(df)
            # Usa a escala aprendida no treinamento — sem re-fit
            X_test = processor.transform(X_test)

            feature_names = list(X_test.columns)
            row: dict = {"usina": usina_name, "n_test_samples": len(X_test)}

            for model_name in usina_models:
                # model_type é o sufixo após "{usina}_" (ex: "decision_tree")
                model_type = model_name[len(f"{usina_name}_"):]

                model = registry.load(model_name)
                metrics = evaluator.evaluate(model, X_test, y_test, model_type)
                analyzer.log_report(model, feature_names, model_type)

                for metric_name, value in metrics.items():
                    row[f"{model_type}_{metric_name}"] = value

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
