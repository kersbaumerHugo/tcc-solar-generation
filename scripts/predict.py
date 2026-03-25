"""
Pipeline de predição de geração solar.

Carrega um modelo treinado e o DataProcessor persistido (com scaler fitado)
e aplica o pipeline completo sobre novos dados climáticos.

Por que carregar o processor persistido?
O MinMaxScaler aprende min/max dos dados de treino. Criar um novo scaler ou
re-fitar nos dados de predição produziria uma escala diferente da usada no
treinamento, corrompendo as predições silenciosamente.

Uso:
    python scripts/predict.py \\
        --usina ANGRA \\
        --model decision_tree \\
        --input data/full/ANGRA_full.csv \\
        --output results/ANGRA_predictions.csv

Se --output for omitido, imprime no stdout.
Se o CSV de entrada contiver a coluna GERACAO, ela é incluída no output
como GERACAO_REAL para comparação com GERACAO_PREVISTA.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.config.settings import Config
from src.data.processors import DataProcessor
from src.models.registry import ModelRegistry
from src.utils.logger import logger


def predict(
    usina: str,
    model_name: str,
    input_df: pd.DataFrame,
    registry: ModelRegistry,
) -> pd.DataFrame:
    """
    Aplica o pipeline de predição sobre um DataFrame de features climáticas.

    O fluxo é idêntico ao de treino, exceto que:
      - processor.transform() é usado (não fit_transform) — preserva a escala do treino.
      - O modelo não é retreinado, apenas carregado e aplicado.

    Args:
        usina:      Nome da usina (ex: "ANGRA") — deve corresponder a artefatos
                    no registry com prefixo "{usina}_".
        model_name: Sufixo do modelo no registry (ex: "decision_tree").
        input_df:   DataFrame com features climáticas e índice datetime.
        registry:   ModelRegistry de onde carregar processor e modelo.

    Returns:
        DataFrame com colunas GERACAO_PREVISTA e, se disponível no input,
        GERACAO_REAL (para comparação direta).

    Raises:
        FileNotFoundError: Se o processor ou o modelo não existirem no registry.
    """
    processor_key = f"{usina}_processor"
    full_model_key = f"{usina}_{model_name}"

    if not registry.exists(processor_key):
        raise FileNotFoundError(
            f"Processor não encontrado: '{processor_key}'.\n"
            "Execute scripts/run_training.py antes de predizer."
        )
    if not registry.exists(full_model_key):
        raise FileNotFoundError(
            f"Modelo não encontrado: '{full_model_key}'.\n"
            f"Modelos disponíveis: {registry.list_models()}"
        )

    processor: DataProcessor = registry.load(processor_key)
    model = registry.load(full_model_key)

    # Preserva GERACAO antes do pipeline de limpeza, se disponível
    geracao_real = (
        input_df[Config.TARGET_COLUMN].copy()
        if Config.TARGET_COLUMN in input_df.columns
        else None
    )

    # Pipeline de limpeza — identical ao de treino, sem normalize (scaler já fitado)
    df = processor.rename_columns(input_df)
    df = processor.add_temporal_features(df)
    df = processor.handle_missing_values(df)
    df = processor.drop_irrelevant_features(df)

    # Remove target se ainda presente após a limpeza
    X = df.drop(columns=[Config.TARGET_COLUMN], errors="ignore")

    # Normaliza com a escala DO TREINO — transform, não fit_transform
    X_scaled = processor.transform(X)

    preds = model.predict(X_scaled)
    result = pd.DataFrame({"GERACAO_PREVISTA": preds}, index=X_scaled.index)

    if geracao_real is not None:
        # Realinha pelo índice: handle_missing_values pode ter removido linhas
        result["GERACAO_REAL"] = geracao_real.reindex(result.index)

    logger.info(
        f"Predição concluída: {len(result)} amostras "
        f"[{result.index.min()} → {result.index.max()}]"
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predição de geração solar a partir de dados climáticos."
    )
    parser.add_argument(
        "--usina", required=True,
        help="Nome da usina (ex: ANGRA). Deve ter artefatos treinados no registry.",
    )
    parser.add_argument(
        "--model", default="decision_tree",
        dest="model_name",
        help="Sufixo do modelo no registry (padrão: decision_tree).",
    )
    parser.add_argument(
        "--input", required=True,
        type=Path,
        dest="input_path",
        help="CSV com dados climáticos para predição.",
    )
    parser.add_argument(
        "--output", default=None,
        type=Path,
        dest="output_path",
        help="CSV de saída com predições. Se omitido, imprime no stdout.",
    )
    parser.add_argument(
        "--models-dir", default=Config.MODELS_DIR,
        type=Path,
        dest="models_dir",
        help=f"Diretório do registry (padrão: {Config.MODELS_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input_path.exists():
        logger.error(f"Arquivo de input não encontrado: {args.input_path}")
        sys.exit(1)

    logger.info(f"Carregando input: {args.input_path}")
    input_df = pd.read_csv(args.input_path, index_col=0, parse_dates=True)

    registry = ModelRegistry(args.models_dir)

    try:
        result = predict(args.usina, args.model_name, input_df, registry)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_path)
        logger.info(f"Predições salvas em: {args.output_path}")
    else:
        print(result.to_string())


if __name__ == "__main__":
    main()
